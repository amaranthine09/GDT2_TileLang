"""Core attention: TileLang kernels and drivers (eq. 19 / 26, sink of eq. 27).

Five kernels.

    1. ``sparse_attn_fwd_kernel``     CSA forward  -> o, lse
    2. ``sparse_attn_bwd_q_kernel``   CSA backward -> dq, dkv_comp, dkv_win
    3. ``dense_attn_fwd_kernel``      HCA forward  -> o, lse
    4. ``dense_attn_bwd_q_kernel``    HCA backward -> dq, dkv_win
    5. ``dense_attn_bwd_kv_kernel``   HCA backward -> dkv_comp

CSA needs only one backward kernel because both of its KV gradients are
scattered from the query side; see "Backward decomposition" below for why the
key-parallel alternative is not worth writing.

What makes this attention unusual
---------------------------------
**Shared-KV MQA.** All ``n_h`` query heads read one KV stream, and each entry
is used as *both* key and value. Sharing is why a head tile can be made wide:
one loaded KV tile feeds every head in the tile, so the head axis buys
arithmetic intensity for free. Key-and-value-in-one is why the backward has to
sum two gradient paths into the same tensor -- miss the key path and the
gradient is silently wrong in a way that still trains, just worse.

**Head dim 512.** The output accumulator ``[M, c]`` lives in registers, so ``M``
-- the number of (token, head) rows a block owns -- is what the register file
is spent on: ``M * 512`` fp32 per block. ``M = 32`` at 256 threads is 64
registers a thread, which is the default here. Raising ``block_H`` raises
arithmetic intensity and register pressure together; that is the trade-off knob.

**Per-token top-k.** CSA queries each choose their own ``k`` entries, so a token
tile shares nothing and the sparse kernels run one token per block, tiling only
the head axis. HCA has no selection, so its queries *do* share a KV stream and
its kernels tile tokens as well.

**Sink.** ``exp(z'_h)`` joins the denominator only. It is folded in once at the
epilogue, after the online softmax has finished, by treating it as one final
element: rescale to ``max(running_max, z'_h)`` and add. Doing it there rather
than seeding the running max keeps the loop identical to ordinary flash
attention and makes the "attends to nothing" case fall out correctly -- the
output is zero and the LSE is ``z'_h``.

Masking
-------
Masked logits are set to :data:`~dsv4.compress.NEG_LARGE` and then *explicitly
zeroed* after the exponential rather than relying on ``exp`` to underflow. A
tile that is entirely masked would otherwise give ``max = NEG_LARGE`` and
``exp(NEG_LARGE - NEG_LARGE) = 1``, turning masked slots into weight-1 entries.
This is the one place where the usual "-inf just works" reasoning fails.

Backward decomposition
----------------------
Both variants recompute the logits from the saved ``lse``: because
``p = exp(q.k * scale - lse)`` needs nothing but the row's own statistics, a
block can reconstruct any probability it wants without seeing the rest of the
row. What differs is which axis each variant parallelises over, and that follows
from where the reuse is.

**CSA is parallelised over queries.** The tempting alternative -- invert the
top-k into "which queries chose this entry" and run one block per entry -- costs
about 128x more traffic. In the query direction all ``n_h`` heads share the one
gathered KV tile, so the head axis is free; in the key direction every
(entry, query) pair has to re-read ``q`` and ``do`` for all ``n_h`` heads, and
the head axis is paid for in full. The price of going query-parallel is that
``dkv_comp`` becomes a scatter and needs atomics -- but the atomic volume is
exactly the forward's gather volume, so it costs about one extra pass, not an
order of magnitude.

**HCA is parallelised over keys.** There is no gather, so one entry tile is read
by every query tile and the arithmetic intensity of the key direction is
``block_KV`` flops per element -- solidly compute-bound. ``dkv_comp`` is then
owned by exactly one block and needs no atomics at all.

The sliding window is local either way: a token is seen by at most ``n_win``
queries, so its gradient is accumulated atomically alongside ``dq`` in whichever
query-parallel kernel the variant already runs.

``dsink[h] = -sum_t exp(z'_h - lse[t,h]) * delta[t,h]`` needs only ``lse`` and
``delta``, so it is a two-line reduction over tensors the backward already has;
the drivers leave it to PyTorch rather than spending a kernel launch on it. The
same applies to ``delta = rowsum(do * o)`` itself.
"""

# NOTE: no `from __future__ import annotations` -- see dsv4/compress.py.

import tilelang
import tilelang.language as T
import torch

from .compress import NEG_LARGE, _tl_dtype
from .config import DEFAULT_KERNEL_CONFIG, KernelConfig

__all__ = [
    "sparse_attn_fwd_kernel",
    "sparse_attn_bwd_q_kernel",
    "dense_attn_fwd_kernel",
    "dense_attn_bwd_q_kernel",
    "dense_attn_bwd_kv_kernel",
    "sparse_attn_fwd",
    "sparse_attn_bwd",
    "dense_attn_fwd",
    "dense_attn_bwd",
]

_FAST_MATH = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}

# Anything at or below this is a masked slot, not a real logit. NEG_LARGE
# itself sits exactly at -1e30, so the test has to be inclusive.
MASK_TEST = NEG_LARGE / 2.0


# ---------------------------------------------------------------------------
# CSA: sparse, per-token gather
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[-2, -1], pass_configs=_FAST_MATH)
def sparse_attn_fwd_kernel(
    B: int,
    N: int,
    H: int,
    c: int,
    NB: int,
    k: int,
    scale: float,
    window: int = 128,
    input_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    block_H: int = 32,
    block_KV: int = 64,
    threads: int = 256,
    num_stages: int = 2,
):
    """CSA forward: gather ``k`` selected entries, add the window, add the sink.

    One block per (token, head tile). The token's ``k`` indices are walked in
    ``block_KV`` chunks; each chunk is gathered into shared memory, scored
    against the whole head tile with one GEMM, and folded into a running online
    softmax. The sliding window is then walked the same way, under the *same*
    running statistics -- the two branches share one softmax, so they must share
    one normalisation.

    Padding slots (``idx < 0``, emitted when a query has fewer than ``k`` legal
    candidates) gather zeros and are masked out of the logits, so they
    contribute nothing rather than duplicating entry 0.

    Args:
        B, N, H, c: batch, tokens, query heads, head dim.
        NB: number of compressed entries.
        k: selected entries per query.
        scale: logit scale.
        window: ``n_win``; 0 disables the window branch.
        input_dtype, accum_dtype: tensor / accumulator dtypes.
        block_H: heads per block -- also the register-pressure knob, see the
            module docstring.
        block_KV: KV entries per chunk.
        threads, num_stages: launch / pipelining config.

    Returns:
        A JIT kernel ``(q, kv_comp, kv_win, sink, idx) -> (o, lse)`` with ``o``
        of shape ``[B, N, H, c]`` and ``lse`` of ``[B, N, H]``. ``lse``
        includes the sink and is what the backward needs.
    """
    if H % block_H:
        raise ValueError(f"head count {H} must be a multiple of block_H {block_H}")
    n_kv = T.ceildiv(k, block_KV)
    n_win = T.ceildiv(window, block_KV) if window > 0 else 0

    @T.prim_func
    def kernel(
        q: T.Tensor((B, N, H, c), dtype=input_dtype),
        kv_comp: T.Tensor((B, NB, c), dtype=input_dtype),
        kv_win: T.Tensor((B, N, c), dtype=input_dtype),
        sink: T.Tensor((H,), dtype=accum_dtype),
        idx: T.Tensor((B, N, k), dtype="int32"),
        o: T.Tensor((B, N, H, c), dtype=input_dtype),
        lse: T.Tensor((B, N, H), dtype=accum_dtype),
    ):
        with T.Kernel(T.ceildiv(H, block_H), N, B, threads=threads) as (i_h, t, i_b):
            h0 = i_h * block_H

            q_sh = T.alloc_shared((block_H, c), dtype=input_dtype)
            kv_sh = T.alloc_shared((block_KV, c), dtype=input_dtype)
            idx_sh = T.alloc_shared((block_KV,), dtype="int32", scope="shared")
            ok_sh = T.alloc_shared((block_KV,), dtype=accum_dtype, scope="shared")

            z_f = T.alloc_fragment((block_H, block_KV), dtype=accum_dtype)
            p_in = T.alloc_fragment((block_H, block_KV), dtype=input_dtype)
            acc = T.alloc_fragment((block_H, c), dtype=accum_dtype)
            o_sh = T.alloc_shared((block_H, c), dtype=input_dtype)

            m_i = T.alloc_fragment((block_H,), dtype=accum_dtype)
            l_i = T.alloc_fragment((block_H,), dtype=accum_dtype)
            m_new = T.alloc_fragment((block_H,), dtype=accum_dtype)
            r_max = T.alloc_fragment((block_H,), dtype=accum_dtype)
            r_sum = T.alloc_fragment((block_H,), dtype=accum_dtype)
            rs = T.alloc_fragment((block_H,), dtype=accum_dtype)
            sk = T.alloc_fragment((block_H,), dtype=accum_dtype)

            T.copy(q[i_b, t, h0 : h0 + block_H, :], q_sh)
            T.copy(sink[h0 : h0 + block_H], sk)
            T.clear(acc)
            for i in T.Parallel(block_H):
                m_i[i] = NEG_LARGE
                l_i[i] = 0.0

            # ---- compressed branch: the k selected entries -----------------
            for i_kv in T.Pipelined(n_kv, num_stages=num_stages):
                s0 = i_kv * block_KV

                for j in T.Parallel(block_KV):
                    slot = s0 + j
                    sid = T.if_then_else(slot < k, idx[i_b, t, T.min(slot, k - 1)], -1)
                    idx_sh[j] = T.max(sid, 0)
                    ok_sh[j] = T.if_then_else((slot < k) & (sid >= 0), 1.0, 0.0)

                for j, d in T.Parallel(block_KV, c):
                    kv_sh[j, d] = T.if_then_else(
                        ok_sh[j] > 0.0, kv_comp[i_b, idx_sh[j], d], T.Cast(input_dtype, 0.0)
                    )

                T.gemm(q_sh, kv_sh, z_f, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(block_H, block_KV):
                    z_f[i, j] = T.if_then_else(ok_sh[j] > 0.0, z_f[i, j] * scale, NEG_LARGE)

                T.reduce_max(z_f, r_max, dim=-1, clear=True)
                for i in T.Parallel(block_H):
                    m_new[i] = T.max(m_i[i], r_max[i])
                    rs[i] = T.exp(m_i[i] - m_new[i])
                for i, j in T.Parallel(block_H, block_KV):
                    # Explicit zero, not underflow: a fully masked tile would
                    # otherwise give exp(NEG_LARGE - NEG_LARGE) = 1.
                    z_f[i, j] = T.if_then_else(
                        z_f[i, j] > MASK_TEST, T.exp(z_f[i, j] - m_new[i]), 0.0
                    )
                T.reduce_sum(z_f, r_sum, dim=-1, clear=True)

                for i, d in T.Parallel(block_H, c):
                    acc[i, d] = acc[i, d] * rs[i]
                for i in T.Parallel(block_H):
                    l_i[i] = l_i[i] * rs[i] + r_sum[i]
                    m_i[i] = m_new[i]
                for i, j in T.Parallel(block_H, block_KV):
                    p_in[i, j] = T.Cast(input_dtype, z_f[i, j])
                T.gemm(p_in, kv_sh, acc)

            # ---- sliding-window branch, same running statistics ------------
            for i_w in T.Pipelined(n_win, num_stages=num_stages):
                w0 = i_w * block_KV

                for j in T.Parallel(block_KV):
                    slot = w0 + j
                    jt = t - (window - 1) + slot
                    ok_sh[j] = T.if_then_else(
                        (slot < window) & (jt >= 0) & (jt <= t), 1.0, 0.0
                    )
                    idx_sh[j] = T.max(T.min(jt, N - 1), 0)

                for j, d in T.Parallel(block_KV, c):
                    kv_sh[j, d] = T.if_then_else(
                        ok_sh[j] > 0.0, kv_win[i_b, idx_sh[j], d], T.Cast(input_dtype, 0.0)
                    )

                T.gemm(q_sh, kv_sh, z_f, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(block_H, block_KV):
                    z_f[i, j] = T.if_then_else(ok_sh[j] > 0.0, z_f[i, j] * scale, NEG_LARGE)

                T.reduce_max(z_f, r_max, dim=-1, clear=True)
                for i in T.Parallel(block_H):
                    m_new[i] = T.max(m_i[i], r_max[i])
                    rs[i] = T.exp(m_i[i] - m_new[i])
                for i, j in T.Parallel(block_H, block_KV):
                    z_f[i, j] = T.if_then_else(
                        z_f[i, j] > MASK_TEST, T.exp(z_f[i, j] - m_new[i]), 0.0
                    )
                T.reduce_sum(z_f, r_sum, dim=-1, clear=True)

                for i, d in T.Parallel(block_H, c):
                    acc[i, d] = acc[i, d] * rs[i]
                for i in T.Parallel(block_H):
                    l_i[i] = l_i[i] * rs[i] + r_sum[i]
                    m_i[i] = m_new[i]
                for i, j in T.Parallel(block_H, block_KV):
                    p_in[i, j] = T.Cast(input_dtype, z_f[i, j])
                T.gemm(p_in, kv_sh, acc)

            # ---- epilogue: fold the sink into the denominator --------------
            for i in T.Parallel(block_H):
                m_new[i] = T.max(m_i[i], sk[i])
                rs[i] = T.exp(m_i[i] - m_new[i])
            for i, d in T.Parallel(block_H, c):
                acc[i, d] = acc[i, d] * rs[i]
            for i in T.Parallel(block_H):
                l_i[i] = l_i[i] * rs[i] + T.exp(sk[i] - m_new[i])
                m_i[i] = m_new[i]

            for i, d in T.Parallel(block_H, c):
                o_sh[i, d] = T.Cast(input_dtype, acc[i, d] / l_i[i])
            T.copy(o_sh, o[i_b, t, h0 : h0 + block_H, :])
            for i in T.Parallel(block_H):
                if h0 + i < H:
                    lse[i_b, t, h0 + i] = m_i[i] + T.log(l_i[i])

    return kernel


# ---------------------------------------------------------------------------
# HCA: dense over compressed entries
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[-2, -1], pass_configs=_FAST_MATH)
def dense_attn_fwd_kernel(
    B: int,
    N: int,
    H: int,
    c: int,
    NB: int,
    scale: float,
    compress_rate: int = 128,
    window: int = 128,
    input_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    block_T: int = 2,
    block_H: int = 16,
    block_KV: int = 64,
    threads: int = 256,
    num_stages: int = 2,
):
    """HCA forward: dense over all legal compressed entries, plus window, plus sink.

    Queries in a tile share the compressed stream, so unlike the sparse kernel
    this one tiles tokens as well as heads. Rows of the tile are (token, head)
    pairs flattened as ``r = it * block_H + ih``, which keeps a row's channels
    contiguous and lets the whole tile go through one GEMM.

    The causal boundary ``s < t // m'`` is the linear predicate
    ``(s + 1) * m' <= t``. Key tiles entirely above it are skipped; with
    ``m' = 128`` and a small ``block_T`` the surviving tiles are almost always
    fully unmasked, so the predicate costs nothing in practice.

    Returns:
        A JIT kernel ``(q, kv_comp, kv_win, sink) -> (o, lse)``.
    """
    if H % block_H:
        raise ValueError(f"head count {H} must be a multiple of block_H {block_H}")
    M = block_T * block_H
    # Union of the tile's per-token windows, not one window.
    win_span = window + block_T - 1
    n_win = T.ceildiv(win_span, block_KV) if window > 0 else 0

    @T.prim_func
    def kernel(
        q: T.Tensor((B, N, H, c), dtype=input_dtype),
        kv_comp: T.Tensor((B, NB, c), dtype=input_dtype),
        kv_win: T.Tensor((B, N, c), dtype=input_dtype),
        sink: T.Tensor((H,), dtype=accum_dtype),
        o: T.Tensor((B, N, H, c), dtype=input_dtype),
        lse: T.Tensor((B, N, H), dtype=accum_dtype),
    ):
        with T.Kernel(
            T.ceildiv(H, block_H), T.ceildiv(N, block_T), B, threads=threads
        ) as (i_h, i_t, i_b):
            h0 = i_h * block_H
            t0 = i_t * block_T

            q_sh = T.alloc_shared((M, c), dtype=input_dtype)
            kv_sh = T.alloc_shared((block_KV, c), dtype=input_dtype)
            ok_sh = T.alloc_shared((block_KV,), dtype=accum_dtype, scope="shared")
            idx_sh = T.alloc_shared((block_KV,), dtype="int32", scope="shared")

            z_f = T.alloc_fragment((M, block_KV), dtype=accum_dtype)
            p_in = T.alloc_fragment((M, block_KV), dtype=input_dtype)
            acc = T.alloc_fragment((M, c), dtype=accum_dtype)
            o_sh = T.alloc_shared((M, c), dtype=input_dtype)

            m_i = T.alloc_fragment((M,), dtype=accum_dtype)
            l_i = T.alloc_fragment((M,), dtype=accum_dtype)
            m_new = T.alloc_fragment((M,), dtype=accum_dtype)
            r_max = T.alloc_fragment((M,), dtype=accum_dtype)
            r_sum = T.alloc_fragment((M,), dtype=accum_dtype)
            rs = T.alloc_fragment((M,), dtype=accum_dtype)
            sk = T.alloc_fragment((M,), dtype=accum_dtype)

            for it in T.serial(block_T):
                T.copy(
                    q[i_b, T.min(t0 + it, N - 1), h0 : h0 + block_H, :],
                    q_sh[it * block_H : (it + 1) * block_H, :],
                )
            for r in T.Parallel(M):
                sk[r] = sink[h0 + r % block_H]
                m_i[r] = NEG_LARGE
                l_i[r] = 0.0
            T.clear(acc)

            # Only entries strictly older than the newest query in the tile.
            n_kv = T.ceildiv(
                T.max(T.min(t0 + block_T - 1, N - 1) // compress_rate, 0), block_KV
            ) + 1

            for i_kv in T.Pipelined(n_kv, num_stages=num_stages):
                s0 = i_kv * block_KV
                if s0 < NB:
                    for j in T.Parallel(block_KV):
                        ok_sh[j] = T.if_then_else(s0 + j < NB, 1.0, 0.0)
                        idx_sh[j] = T.min(s0 + j, NB - 1)
                    for j, d in T.Parallel(block_KV, c):
                        kv_sh[j, d] = T.if_then_else(
                            ok_sh[j] > 0.0, kv_comp[i_b, idx_sh[j], d], T.Cast(input_dtype, 0.0)
                        )

                    T.gemm(q_sh, kv_sh, z_f, transpose_B=True, clear_accum=True)
                    for r, j in T.Parallel(M, block_KV):
                        t = t0 + r // block_H
                        s = s0 + j
                        good = (ok_sh[j] > 0.0) & ((s + 1) * compress_rate <= t) & (t < N)
                        z_f[r, j] = T.if_then_else(good, z_f[r, j] * scale, NEG_LARGE)

                    T.reduce_max(z_f, r_max, dim=-1, clear=True)
                    for r in T.Parallel(M):
                        m_new[r] = T.max(m_i[r], r_max[r])
                        rs[r] = T.exp(m_i[r] - m_new[r])
                    for r, j in T.Parallel(M, block_KV):
                        z_f[r, j] = T.if_then_else(
                            z_f[r, j] > MASK_TEST, T.exp(z_f[r, j] - m_new[r]), 0.0
                        )
                    T.reduce_sum(z_f, r_sum, dim=-1, clear=True)

                    for r, d in T.Parallel(M, c):
                        acc[r, d] = acc[r, d] * rs[r]
                    for r in T.Parallel(M):
                        l_i[r] = l_i[r] * rs[r] + r_sum[r]
                        m_i[r] = m_new[r]
                    for r, j in T.Parallel(M, block_KV):
                        p_in[r, j] = T.Cast(input_dtype, z_f[r, j])
                    T.gemm(p_in, kv_sh, acc)

            # Window branch. Every token in the tile needs its own window, but
            # those windows are nested translates of each other, so their union
            # -- [t0 - window + 1, t0 + block_T - 1] -- is only block_T - 1
            # wider than one of them. Loading the union once and masking per
            # row keeps this to a single GEMM per chunk for the whole tile,
            # instead of one GEMM per token.
            w_base = t0 - (window - 1)
            for i_w in T.Pipelined(n_win, num_stages=num_stages):
                j0 = w_base + i_w * block_KV

                for j in T.Parallel(block_KV):
                    jt = j0 + j
                    ok_sh[j] = T.if_then_else((jt >= 0) & (jt < N), 1.0, 0.0)
                    idx_sh[j] = T.max(T.min(jt, N - 1), 0)
                for j, d in T.Parallel(block_KV, c):
                    kv_sh[j, d] = T.if_then_else(
                        ok_sh[j] > 0.0, kv_win[i_b, idx_sh[j], d], T.Cast(input_dtype, 0.0)
                    )

                T.gemm(q_sh, kv_sh, z_f, transpose_B=True, clear_accum=True)
                for r, j in T.Parallel(M, block_KV):
                    t = t0 + r // block_H
                    jt = j0 + j
                    good = (
                        (ok_sh[j] > 0.0)
                        & (t < N)
                        & (jt <= t)
                        & (jt > t - window)
                    )
                    z_f[r, j] = T.if_then_else(good, z_f[r, j] * scale, NEG_LARGE)

                T.reduce_max(z_f, r_max, dim=-1, clear=True)
                for r in T.Parallel(M):
                    m_new[r] = T.max(m_i[r], r_max[r])
                    rs[r] = T.exp(m_i[r] - m_new[r])
                for r, j in T.Parallel(M, block_KV):
                    z_f[r, j] = T.if_then_else(
                        z_f[r, j] > MASK_TEST, T.exp(z_f[r, j] - m_new[r]), 0.0
                    )
                T.reduce_sum(z_f, r_sum, dim=-1, clear=True)

                for r, d in T.Parallel(M, c):
                    acc[r, d] = acc[r, d] * rs[r]
                for r in T.Parallel(M):
                    l_i[r] = l_i[r] * rs[r] + r_sum[r]
                    m_i[r] = m_new[r]
                for r, j in T.Parallel(M, block_KV):
                    p_in[r, j] = T.Cast(input_dtype, z_f[r, j])
                T.gemm(p_in, kv_sh, acc)

            for r in T.Parallel(M):
                m_new[r] = T.max(m_i[r], sk[r])
                rs[r] = T.exp(m_i[r] - m_new[r])
            for r, d in T.Parallel(M, c):
                acc[r, d] = acc[r, d] * rs[r]
            for r in T.Parallel(M):
                l_i[r] = l_i[r] * rs[r] + T.exp(sk[r] - m_new[r])
                m_i[r] = m_new[r]

            for r, d in T.Parallel(M, c):
                o_sh[r, d] = T.Cast(input_dtype, acc[r, d] / l_i[r])
            for it in T.serial(block_T):
                if t0 + it < N:
                    T.copy(
                        o_sh[it * block_H : (it + 1) * block_H, :],
                        o[i_b, t0 + it, h0 : h0 + block_H, :],
                    )
            for r in T.Parallel(M):
                t = t0 + r // block_H
                if (t < N) & (h0 + r % block_H < H):
                    lse[i_b, t, h0 + r % block_H] = m_i[r] + T.log(l_i[r])

    return kernel


# ---------------------------------------------------------------------------
# Backward: CSA (query-parallel, atomic scatter into the compressed stream)
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[-1], pass_configs=_FAST_MATH)
def sparse_attn_bwd_q_kernel(
    B: int,
    N: int,
    H: int,
    c: int,
    NB: int,
    k: int,
    scale: float,
    window: int = 128,
    input_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    block_H: int = 32,
    block_KV: int = 64,
    threads: int = 256,
    num_stages: int = 2,
):
    """CSA backward, all of it: ``dq`` owned, ``dkv_comp`` and ``dkv_win`` scattered.

    Mirrors the forward's traversal exactly -- same token, same head tile, same
    chunking -- so every gathered KV tile is reused for all four products that
    need it::

        p    = exp(q.kv * scale - lse)              recomputed, not stored
        dz   = p * (do.kv - delta) * scale
        dq  += dz @ kv
        dkv += p^T @ do            (value path)
             + dz^T @ q            (key path)

    ``dz`` carries the ``scale`` factor so it can serve both ``dq`` and the key
    path without a second scaled copy.

    The two ``dkv`` terms are the consequence of each entry being key *and*
    value; dropping either leaves a gradient that is wrong but still descends,
    which is the kind of bug that shows up as a quality regression rather than a
    crash.

    ``dkv_comp`` and ``dkv_win`` are accumulated with atomics and must be zeroed
    by the caller. They are inputs rather than outputs for that reason -- the
    kernel adds into them, it does not define them.

    Args:
        B, N, H, c, NB, k, scale, window: as in the forward.
        input_dtype, accum_dtype, block_H, block_KV, threads, num_stages: as in
            the forward.

    Returns:
        A JIT kernel
        ``(q, kv_comp, kv_win, idx, do, lse, delta, dkv_comp, dkv_win) -> dq``.
    """
    if H % block_H:
        raise ValueError(f"head count {H} must be a multiple of block_H {block_H}")
    n_kv = T.ceildiv(k, block_KV)
    n_win = T.ceildiv(window, block_KV) if window > 0 else 0

    @T.prim_func
    def kernel(
        q: T.Tensor((B, N, H, c), dtype=input_dtype),
        kv_comp: T.Tensor((B, NB, c), dtype=input_dtype),
        kv_win: T.Tensor((B, N, c), dtype=input_dtype),
        idx: T.Tensor((B, N, k), dtype="int32"),
        do: T.Tensor((B, N, H, c), dtype=input_dtype),
        lse: T.Tensor((B, N, H), dtype=accum_dtype),
        delta: T.Tensor((B, N, H), dtype=accum_dtype),
        dkv_comp: T.Tensor((B, NB, c), dtype=accum_dtype),
        dkv_win: T.Tensor((B, N, c), dtype=accum_dtype),
        dq: T.Tensor((B, N, H, c), dtype=accum_dtype),
    ):
        with T.Kernel(T.ceildiv(H, block_H), N, B, threads=threads) as (i_h, t, i_b):
            h0 = i_h * block_H

            q_sh = T.alloc_shared((block_H, c), dtype=input_dtype)
            do_sh = T.alloc_shared((block_H, c), dtype=input_dtype)
            kv_sh = T.alloc_shared((block_KV, c), dtype=input_dtype)
            idx_sh = T.alloc_shared((block_KV,), dtype="int32", scope="shared")
            ok_sh = T.alloc_shared((block_KV,), dtype=accum_dtype, scope="shared")

            z_f = T.alloc_fragment((block_H, block_KV), dtype=accum_dtype)
            dov = T.alloc_fragment((block_H, block_KV), dtype=accum_dtype)
            pb = T.alloc_fragment((block_H, block_KV), dtype=input_dtype)
            dsb = T.alloc_fragment((block_H, block_KV), dtype=input_dtype)
            dq_acc = T.alloc_fragment((block_H, c), dtype=accum_dtype)
            dkv_t = T.alloc_fragment((block_KV, c), dtype=accum_dtype)
            dq_sh = T.alloc_shared((block_H, c), dtype=accum_dtype)

            lse_f = T.alloc_fragment((block_H,), dtype=accum_dtype)
            del_f = T.alloc_fragment((block_H,), dtype=accum_dtype)

            T.copy(q[i_b, t, h0 : h0 + block_H, :], q_sh)
            T.copy(do[i_b, t, h0 : h0 + block_H, :], do_sh)
            T.copy(lse[i_b, t, h0 : h0 + block_H], lse_f)
            T.copy(delta[i_b, t, h0 : h0 + block_H], del_f)
            T.clear(dq_acc)

            for i_kv in T.Pipelined(n_kv, num_stages=num_stages):
                s0 = i_kv * block_KV

                for j in T.Parallel(block_KV):
                    slot = s0 + j
                    sid = T.if_then_else(slot < k, idx[i_b, t, T.min(slot, k - 1)], -1)
                    idx_sh[j] = T.max(sid, 0)
                    ok_sh[j] = T.if_then_else((slot < k) & (sid >= 0), 1.0, 0.0)
                for j, d in T.Parallel(block_KV, c):
                    kv_sh[j, d] = T.if_then_else(
                        ok_sh[j] > 0.0, kv_comp[i_b, idx_sh[j], d], T.Cast(input_dtype, 0.0)
                    )

                T.gemm(q_sh, kv_sh, z_f, transpose_B=True, clear_accum=True)
                T.gemm(do_sh, kv_sh, dov, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(block_H, block_KV):
                    pv = T.if_then_else(
                        ok_sh[j] > 0.0, T.exp(z_f[i, j] * scale - lse_f[i]), 0.0
                    )
                    pb[i, j] = T.Cast(input_dtype, pv)
                    dsb[i, j] = T.Cast(input_dtype, pv * (dov[i, j] - del_f[i]) * scale)

                T.gemm(dsb, kv_sh, dq_acc)

                T.clear(dkv_t)
                T.gemm(pb, do_sh, dkv_t, transpose_A=True)
                T.gemm(dsb, q_sh, dkv_t, transpose_A=True)
                for j, d in T.Parallel(block_KV, c):
                    if ok_sh[j] > 0.0:
                        T.atomic_add(dkv_comp[i_b, idx_sh[j], d], dkv_t[j, d])

            for i_w in T.Pipelined(n_win, num_stages=num_stages):
                w0 = i_w * block_KV

                for j in T.Parallel(block_KV):
                    slot = w0 + j
                    jt = t - (window - 1) + slot
                    ok_sh[j] = T.if_then_else((slot < window) & (jt >= 0) & (jt <= t), 1.0, 0.0)
                    idx_sh[j] = T.max(T.min(jt, N - 1), 0)
                for j, d in T.Parallel(block_KV, c):
                    kv_sh[j, d] = T.if_then_else(
                        ok_sh[j] > 0.0, kv_win[i_b, idx_sh[j], d], T.Cast(input_dtype, 0.0)
                    )

                T.gemm(q_sh, kv_sh, z_f, transpose_B=True, clear_accum=True)
                T.gemm(do_sh, kv_sh, dov, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(block_H, block_KV):
                    pv = T.if_then_else(
                        ok_sh[j] > 0.0, T.exp(z_f[i, j] * scale - lse_f[i]), 0.0
                    )
                    pb[i, j] = T.Cast(input_dtype, pv)
                    dsb[i, j] = T.Cast(input_dtype, pv * (dov[i, j] - del_f[i]) * scale)

                T.gemm(dsb, kv_sh, dq_acc)

                T.clear(dkv_t)
                T.gemm(pb, do_sh, dkv_t, transpose_A=True)
                T.gemm(dsb, q_sh, dkv_t, transpose_A=True)
                for j, d in T.Parallel(block_KV, c):
                    if ok_sh[j] > 0.0:
                        T.atomic_add(dkv_win[i_b, idx_sh[j], d], dkv_t[j, d])

            T.copy(dq_acc, dq_sh)
            T.copy(dq_sh, dq[i_b, t, h0 : h0 + block_H, :])

    return kernel


# ---------------------------------------------------------------------------
# Backward: HCA (query side owned, key side owned, no atomics on the stream)
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[-1], pass_configs=_FAST_MATH)
def dense_attn_bwd_q_kernel(
    B: int,
    N: int,
    H: int,
    c: int,
    NB: int,
    scale: float,
    compress_rate: int = 128,
    window: int = 128,
    input_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    block_T: int = 2,
    block_H: int = 16,
    block_KV: int = 64,
    threads: int = 256,
    num_stages: int = 2,
):
    """HCA query-side backward: ``dq``, plus the window's scatter.

    ``dkv_comp`` is *not* produced here -- it is owned by
    :func:`dense_attn_bwd_kv_kernel`, which can accumulate it without atomics
    because HCA's compressed stream is shared rather than gathered. Only the
    sliding window, whose entries are per-token, is scattered atomically.

    Returns:
        A JIT kernel ``(q, kv_comp, kv_win, do, lse, delta, dkv_win) -> dq``.
    """
    if H % block_H:
        raise ValueError(f"head count {H} must be a multiple of block_H {block_H}")
    M = block_T * block_H
    win_span = window + block_T - 1
    n_win = T.ceildiv(win_span, block_KV) if window > 0 else 0

    @T.prim_func
    def kernel(
        q: T.Tensor((B, N, H, c), dtype=input_dtype),
        kv_comp: T.Tensor((B, NB, c), dtype=input_dtype),
        kv_win: T.Tensor((B, N, c), dtype=input_dtype),
        do: T.Tensor((B, N, H, c), dtype=input_dtype),
        lse: T.Tensor((B, N, H), dtype=accum_dtype),
        delta: T.Tensor((B, N, H), dtype=accum_dtype),
        dkv_win: T.Tensor((B, N, c), dtype=accum_dtype),
        dq: T.Tensor((B, N, H, c), dtype=accum_dtype),
    ):
        with T.Kernel(
            T.ceildiv(H, block_H), T.ceildiv(N, block_T), B, threads=threads
        ) as (i_h, i_t, i_b):
            h0 = i_h * block_H
            t0 = i_t * block_T

            q_sh = T.alloc_shared((M, c), dtype=input_dtype)
            do_sh = T.alloc_shared((M, c), dtype=input_dtype)
            kv_sh = T.alloc_shared((block_KV, c), dtype=input_dtype)
            idx_sh = T.alloc_shared((block_KV,), dtype="int32", scope="shared")
            ok_sh = T.alloc_shared((block_KV,), dtype=accum_dtype, scope="shared")

            z_f = T.alloc_fragment((M, block_KV), dtype=accum_dtype)
            dov = T.alloc_fragment((M, block_KV), dtype=accum_dtype)
            pb = T.alloc_fragment((M, block_KV), dtype=input_dtype)
            dsb = T.alloc_fragment((M, block_KV), dtype=input_dtype)
            dq_acc = T.alloc_fragment((M, c), dtype=accum_dtype)
            dkv_t = T.alloc_fragment((block_KV, c), dtype=accum_dtype)
            dq_sh = T.alloc_shared((M, c), dtype=accum_dtype)

            lse_f = T.alloc_fragment((M,), dtype=accum_dtype)
            del_f = T.alloc_fragment((M,), dtype=accum_dtype)

            for it in T.serial(block_T):
                T.copy(
                    q[i_b, T.min(t0 + it, N - 1), h0 : h0 + block_H, :],
                    q_sh[it * block_H : (it + 1) * block_H, :],
                )
                T.copy(
                    do[i_b, T.min(t0 + it, N - 1), h0 : h0 + block_H, :],
                    do_sh[it * block_H : (it + 1) * block_H, :],
                )
            for r in T.Parallel(M):
                tt = T.min(t0 + r // block_H, N - 1)
                lse_f[r] = lse[i_b, tt, h0 + r % block_H]
                del_f[r] = delta[i_b, tt, h0 + r % block_H]
            T.clear(dq_acc)

            n_kv = T.ceildiv(
                T.max(T.min(t0 + block_T - 1, N - 1) // compress_rate, 0), block_KV
            ) + 1

            for i_kv in T.Pipelined(n_kv, num_stages=num_stages):
                s0 = i_kv * block_KV
                if s0 < NB:
                    for j in T.Parallel(block_KV):
                        ok_sh[j] = T.if_then_else(s0 + j < NB, 1.0, 0.0)
                        idx_sh[j] = T.min(s0 + j, NB - 1)
                    for j, d in T.Parallel(block_KV, c):
                        kv_sh[j, d] = T.if_then_else(
                            ok_sh[j] > 0.0, kv_comp[i_b, idx_sh[j], d], T.Cast(input_dtype, 0.0)
                        )

                    T.gemm(q_sh, kv_sh, z_f, transpose_B=True, clear_accum=True)
                    T.gemm(do_sh, kv_sh, dov, transpose_B=True, clear_accum=True)
                    for r, j in T.Parallel(M, block_KV):
                        t = t0 + r // block_H
                        s = s0 + j
                        good = (ok_sh[j] > 0.0) & ((s + 1) * compress_rate <= t) & (t < N)
                        pv = T.if_then_else(
                            good, T.exp(z_f[r, j] * scale - lse_f[r]), 0.0
                        )
                        dsb[r, j] = T.Cast(input_dtype, pv * (dov[r, j] - del_f[r]) * scale)
                    T.gemm(dsb, kv_sh, dq_acc)

            w_base = t0 - (window - 1)
            for i_w in T.Pipelined(n_win, num_stages=num_stages):
                j0 = w_base + i_w * block_KV

                for j in T.Parallel(block_KV):
                    jt = j0 + j
                    ok_sh[j] = T.if_then_else((jt >= 0) & (jt < N), 1.0, 0.0)
                    idx_sh[j] = T.max(T.min(jt, N - 1), 0)
                for j, d in T.Parallel(block_KV, c):
                    kv_sh[j, d] = T.if_then_else(
                        ok_sh[j] > 0.0, kv_win[i_b, idx_sh[j], d], T.Cast(input_dtype, 0.0)
                    )

                T.gemm(q_sh, kv_sh, z_f, transpose_B=True, clear_accum=True)
                T.gemm(do_sh, kv_sh, dov, transpose_B=True, clear_accum=True)
                for r, j in T.Parallel(M, block_KV):
                    t = t0 + r // block_H
                    jt = j0 + j
                    good = (ok_sh[j] > 0.0) & (t < N) & (jt <= t) & (jt > t - window)
                    pv = T.if_then_else(good, T.exp(z_f[r, j] * scale - lse_f[r]), 0.0)
                    pb[r, j] = T.Cast(input_dtype, pv)
                    dsb[r, j] = T.Cast(input_dtype, pv * (dov[r, j] - del_f[r]) * scale)

                T.gemm(dsb, kv_sh, dq_acc)

                T.clear(dkv_t)
                T.gemm(pb, do_sh, dkv_t, transpose_A=True)
                T.gemm(dsb, q_sh, dkv_t, transpose_A=True)
                for j, d in T.Parallel(block_KV, c):
                    if ok_sh[j] > 0.0:
                        T.atomic_add(dkv_win[i_b, idx_sh[j], d], dkv_t[j, d])

            T.copy(dq_acc, dq_sh)
            for it in T.serial(block_T):
                if t0 + it < N:
                    T.copy(
                        dq_sh[it * block_H : (it + 1) * block_H, :],
                        dq[i_b, t0 + it, h0 : h0 + block_H, :],
                    )

    return kernel


# No `out_idx`: this kernel accumulates into `dkv_comp` with atomics, so the
# buffer has to arrive zeroed. Letting the JIT allocate it would hand us
# uninitialised memory to add into.
@tilelang.jit(pass_configs=_FAST_MATH)
def dense_attn_bwd_kv_kernel(
    B: int,
    N: int,
    H: int,
    c: int,
    NB: int,
    scale: float,
    compress_rate: int = 128,
    input_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    block_H: int = 16,
    block_KV: int = 64,
    block_T: int = 16,
    threads: int = 256,
    num_stages: int = 2,
):
    """HCA key-side backward: ``dkv_comp``.

    One block owns a ``[block_KV, c]`` slab of the compressed stream and sweeps
    the query tiles that can legally see it, accumulating both gradient paths in
    registers. The sweep starts at the first query that clears the causal
    boundary ``(s + 1) * m' <= t``; with ``m'`` at 128 that skips a large
    prefix for early entries.

    Arithmetic intensity is ``block_KV`` flops per loaded element, so this is
    comfortably compute-bound -- the reason HCA can afford the key direction
    while CSA cannot.

    HCA's compressed stream is short (``N / m'``), so parallelising over entries
    alone would leave only a few hundred blocks. The grid therefore carries a
    head-tile axis too, which makes each entry's gradient a sum over
    ``H / block_H`` disjoint partial sums and costs one atomic per
    (entry, channel) per head tile -- four writes at the published settings,
    against a GEMM that touched the whole query sequence.

    Returns:
        A JIT kernel ``(q, kv_comp, do, lse, delta, dkv_comp) -> None``.
        ``dkv_comp`` is accumulated in place and must be zeroed by the caller.
    """
    if H % block_H:
        raise ValueError(f"head count {H} must be a multiple of block_H {block_H}")
    M = block_T * block_H

    @T.prim_func
    def kernel(
        q: T.Tensor((B, N, H, c), dtype=input_dtype),
        kv_comp: T.Tensor((B, NB, c), dtype=input_dtype),
        do: T.Tensor((B, N, H, c), dtype=input_dtype),
        lse: T.Tensor((B, N, H), dtype=accum_dtype),
        delta: T.Tensor((B, N, H), dtype=accum_dtype),
        dkv_comp: T.Tensor((B, NB, c), dtype=accum_dtype),
    ):
        with T.Kernel(
            T.ceildiv(NB, block_KV), T.ceildiv(H, block_H), B, threads=threads
        ) as (i_s, i_h, i_b):
            s0 = i_s * block_KV
            h0 = i_h * block_H

            kv_sh = T.alloc_shared((block_KV, c), dtype=input_dtype)
            q_sh = T.alloc_shared((M, c), dtype=input_dtype)
            do_sh = T.alloc_shared((M, c), dtype=input_dtype)

            z_f = T.alloc_fragment((M, block_KV), dtype=accum_dtype)
            dov = T.alloc_fragment((M, block_KV), dtype=accum_dtype)
            pb = T.alloc_fragment((M, block_KV), dtype=input_dtype)
            dsb = T.alloc_fragment((M, block_KV), dtype=input_dtype)
            dkv_acc = T.alloc_fragment((block_KV, c), dtype=accum_dtype)
            dkv_sh = T.alloc_shared((block_KV, c), dtype=accum_dtype)

            lse_f = T.alloc_fragment((M,), dtype=accum_dtype)
            del_f = T.alloc_fragment((M,), dtype=accum_dtype)

            for j, d in T.Parallel(block_KV, c):
                kv_sh[j, d] = T.if_then_else(
                    s0 + j < NB, kv_comp[i_b, T.min(s0 + j, NB - 1), d],
                    T.Cast(input_dtype, 0.0),
                )
            T.clear(dkv_acc)

            # First query tile that clears the causal boundary for entry s0.
            t_lo = ((s0 + 1) * compress_rate) // block_T
            n_t = T.ceildiv(N, block_T)

            for i_t in T.Pipelined(n_t, num_stages=num_stages):
                t0 = i_t * block_T
                if (i_t >= t_lo) & (t0 < N):
                    for it in T.serial(block_T):
                        T.copy(
                            q[i_b, T.min(t0 + it, N - 1), h0 : h0 + block_H, :],
                            q_sh[it * block_H : (it + 1) * block_H, :],
                        )
                        T.copy(
                            do[i_b, T.min(t0 + it, N - 1), h0 : h0 + block_H, :],
                            do_sh[it * block_H : (it + 1) * block_H, :],
                        )
                    for r in T.Parallel(M):
                        tt = T.min(t0 + r // block_H, N - 1)
                        lse_f[r] = lse[i_b, tt, h0 + r % block_H]
                        del_f[r] = delta[i_b, tt, h0 + r % block_H]

                    T.gemm(q_sh, kv_sh, z_f, transpose_B=True, clear_accum=True)
                    T.gemm(do_sh, kv_sh, dov, transpose_B=True, clear_accum=True)
                    for r, j in T.Parallel(M, block_KV):
                        t = t0 + r // block_H
                        s = s0 + j
                        good = (
                            (s < NB) & ((s + 1) * compress_rate <= t) & (t < N)
                        )
                        pv = T.if_then_else(good, T.exp(z_f[r, j] * scale - lse_f[r]), 0.0)
                        pb[r, j] = T.Cast(input_dtype, pv)
                        dsb[r, j] = T.Cast(input_dtype, pv * (dov[r, j] - del_f[r]) * scale)

                    T.gemm(pb, do_sh, dkv_acc, transpose_A=True)
                    T.gemm(dsb, q_sh, dkv_acc, transpose_A=True)

            # Head tiles each own a partial sum over their own heads only, so
            # they are disjoint contributions to the same entry and must be
            # combined. One atomic per (entry, channel) per head tile is
            # H / block_H writes total -- negligible next to the GEMM traffic.
            T.copy(dkv_acc, dkv_sh)
            for j, d in T.Parallel(block_KV, c):
                if s0 + j < NB:
                    T.atomic_add(dkv_comp[i_b, s0 + j, d], dkv_sh[j, d])

    return kernel


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def attn_delta(do: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """``delta[t, h] = do[t, h] . o[t, h]``, the flash backward's row constant.

    A rowwise dot of two tensors the backward already holds. PyTorch does this
    at bandwidth, which is all a kernel could manage, so it stays here.

    Args:
        do, o: ``[B, N, H, c]``.

    Returns:
        ``[B, N, H]`` fp32.
    """
    return (do.float() * o.float()).sum(dim=-1)


def attn_dsink(sink: torch.Tensor, lse: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """``dsink[h] = -sum_{b,t} exp(z'_h - lse[b,t,h]) * delta[b,t,h]``.

    ``exp(z'_h - lse)`` is the probability mass head ``h`` parked on the sink
    instead of on any real entry, so this is minus that mass weighted by the
    row constant. It needs nothing the backward has not already computed.

    Args:
        sink: ``[H]`` sink logits.
        lse: ``[B, N, H]`` from the forward.
        delta: ``[B, N, H]`` from :func:`attn_delta`.

    Returns:
        ``[H]`` fp32.
    """
    return -(torch.exp(sink.float().view(1, 1, -1) - lse.float()) * delta).sum(dim=(0, 1))


def sparse_attn_fwd(
    q, kv_comp, kv_win, sink, idx, scale, window,
    config: KernelConfig = DEFAULT_KERNEL_CONFIG,
):
    """Run :func:`sparse_attn_fwd_kernel`.

    Args:
        q: ``[B, N, H, c]``, already normed and RoPE'd.
        kv_comp: ``[B, NB, c]`` compressed entries.
        kv_win: ``[B, N, c]`` uncompressed window entries.
        sink: ``[H]`` sink logits.
        idx: ``[B, N, k]`` int32 selected entries, ``-1`` padded.
        scale: logit scale.
        window: ``n_win``.
        config: tiling configuration.

    Returns:
        ``(o, lse)``.
    """
    B, N, H, c = q.shape
    NB = kv_comp.shape[1]
    k = idx.shape[-1]
    kernel = sparse_attn_fwd_kernel(
        B, N, H, c, NB, k, scale, window=window,
        input_dtype=_tl_dtype(q.dtype),
        block_H=config.block_H, block_KV=config.block_KV,
        threads=config.threads, num_stages=config.num_stages,
    )
    return kernel(
        q.contiguous(), kv_comp.contiguous(), kv_win.contiguous(),
        sink.float().contiguous(), idx.to(torch.int32).contiguous(),
    )


def sparse_attn_bwd(
    do, q, kv_comp, kv_win, sink, idx, scale, window, o, lse,
    config: KernelConfig = DEFAULT_KERNEL_CONFIG,
):
    """Run :func:`sparse_attn_bwd_q_kernel` and the two PyTorch reductions.

    Returns:
        ``(dq, dkv_comp, dkv_win, dsink)``, all fp32.
    """
    B, N, H, c = q.shape
    NB = kv_comp.shape[1]
    k = idx.shape[-1]
    delta = attn_delta(do, o)

    # Atomic destinations: zeroed, and passed in rather than allocated.
    dkv_comp = torch.zeros((B, NB, c), device=q.device, dtype=torch.float32)
    dkv_win = torch.zeros((B, N, c), device=q.device, dtype=torch.float32)

    kernel = sparse_attn_bwd_q_kernel(
        B, N, H, c, NB, k, scale, window=window,
        input_dtype=_tl_dtype(q.dtype),
        block_H=config.block_H, block_KV=config.block_KV,
        threads=config.threads, num_stages=config.num_stages,
    )
    dq = kernel(
        q.contiguous(), kv_comp.contiguous(), kv_win.contiguous(),
        idx.to(torch.int32).contiguous(), do.contiguous(),
        lse.float().contiguous(), delta.contiguous(), dkv_comp, dkv_win,
    )
    return dq, dkv_comp, dkv_win, attn_dsink(sink, lse, delta)


def dense_attn_fwd(
    q, kv_comp, kv_win, sink, scale, compress_rate, window,
    config: KernelConfig = DEFAULT_KERNEL_CONFIG,
):
    """Run :func:`dense_attn_fwd_kernel`.

    Returns:
        ``(o, lse)``.
    """
    B, N, H, c = q.shape
    NB = kv_comp.shape[1]
    kernel = dense_attn_fwd_kernel(
        B, N, H, c, NB, scale, compress_rate=compress_rate, window=window,
        input_dtype=_tl_dtype(q.dtype),
        block_T=config.block_T_attn, block_H=config.block_H,
        block_KV=config.block_KV,
        threads=config.threads, num_stages=config.num_stages,
    )
    return kernel(
        q.contiguous(), kv_comp.contiguous(), kv_win.contiguous(),
        sink.float().contiguous(),
    )


def dense_attn_bwd(
    do, q, kv_comp, kv_win, sink, scale, compress_rate, window, o, lse,
    config: KernelConfig = DEFAULT_KERNEL_CONFIG,
):
    """Run both HCA backward kernels and the two PyTorch reductions.

    The query-side and key-side kernels read the same inputs and write disjoint
    outputs, so they are independent and can overlap on separate streams.

    Returns:
        ``(dq, dkv_comp, dkv_win, dsink)``, all fp32.
    """
    B, N, H, c = q.shape
    NB = kv_comp.shape[1]
    delta = attn_delta(do, o)

    dkv_win = torch.zeros((B, N, c), device=q.device, dtype=torch.float32)
    dkv_comp = torch.zeros((B, NB, c), device=q.device, dtype=torch.float32)

    qk = dense_attn_bwd_q_kernel(
        B, N, H, c, NB, scale, compress_rate=compress_rate, window=window,
        input_dtype=_tl_dtype(q.dtype),
        block_T=config.block_T_attn, block_H=config.block_H,
        block_KV=config.block_KV,
        threads=config.threads, num_stages=config.num_stages,
    )
    dq = qk(
        q.contiguous(), kv_comp.contiguous(), kv_win.contiguous(), do.contiguous(),
        lse.float().contiguous(), delta.contiguous(), dkv_win,
    )

    kvk = dense_attn_bwd_kv_kernel(
        B, N, H, c, NB, scale, compress_rate=compress_rate,
        input_dtype=_tl_dtype(q.dtype),
        block_H=config.block_H, block_KV=config.block_KV,
        block_T=config.block_T,
        threads=config.threads, num_stages=config.num_stages,
    )
    kvk(
        q.contiguous(), kv_comp.contiguous(), do.contiguous(),
        lse.float().contiguous(), delta.contiguous(), dkv_comp,
    )

    return dq, dkv_comp, dkv_win, attn_dsink(sink, lse, delta)
