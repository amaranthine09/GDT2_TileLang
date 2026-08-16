"""Lightning indexer: TileLang kernels and drivers (eq. 13-17).

Three kernels.

    1. ``lightning_index_kernel``      qI,wI,KI -> I
    2. ``lightning_index_bwd_q_kernel`` dI,...  -> dqI, dwI
    3. ``lightning_index_bwd_k_kernel`` dI,...  -> dKI

The score of query token ``t`` against compressed block ``s`` is

    I[t, s] = sum_h w[t, h] * ReLU(q[t, h] . K[s])

which is *not* a plain GEMM: the ReLU sits between the contraction over ``c^I``
and the reduction over heads, so each head's dot product has to complete before
the head sum can start. Nor can the head weights be folded into ``q`` -- they
are signed and sit outside the ReLU, so ``w * ReLU(q.K) != ReLU((w q).K)``
whenever ``w`` is negative.

The kernels therefore run one GEMM per head into a ``[BT, BS]`` tile, rectify
it, and accumulate it into the score tile scaled by that head's weight. The
compressed-key tile is loaded once per head; with ``BT`` at 64 that reload is
amortised roughly 64 flops per byte, so this stays compute-bound rather than
turning into key-bandwidth.

Causality
---------
``s < Floor(t / m)`` is equivalent to the linear predicate ``(s + 1) * m <= t``,
which is what the kernels test -- it avoids an integer division per element and
lets whole tiles above the diagonal be skipped with one comparison.

``t`` there is the token's position in the *sequence*, which is not its row in
the tensor when the query axis has been chunked. Every kernel therefore takes a
runtime ``t_off`` and tests ``(s + 1) * m <= t_off + row``. Without it a chunk
starting at token 4096 would be masked as though it started at token 0 and would
see almost nothing -- and because the mask only ever removes candidates, the
mistake is invisible in the output shapes and silent in the loss.

Backward
--------
``dKI`` accumulates over every query token, so a query-parallel backward would
need global atomics on ``[NB, cI]``. Instead the backward is split the way
FlashAttention-2 splits its own: one kernel parallel over queries producing
``dqI`` and ``dwI``, one parallel over keys producing ``dKI``. Each output is
owned by exactly one block, so there are no atomics and no scratch buffer, at
the cost of recomputing the per-head logits twice. Recomputation is the right
trade regardless -- storing them would mean a ``[B, N, nhI, NB]`` tensor, which
at 1M context is orders of magnitude larger than the model.
"""

# NOTE: no `from __future__ import annotations` -- see dsv4/compress.py.

import tilelang
import tilelang.language as T

from .compress import NEG_LARGE, _tl_dtype
from .config import DEFAULT_KERNEL_CONFIG, KernelConfig

__all__ = [
    "lightning_index_kernel",
    "lightning_index_bwd_q_kernel",
    "lightning_index_bwd_k_kernel",
    "lightning_index_fwd",
    "lightning_index_bwd",
]

_FAST_MATH = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}


@tilelang.jit(out_idx=[-1], pass_configs=_FAST_MATH)
def lightning_index_kernel(
    B: int,
    N: int,
    NB: int,
    nhI: int,
    cI: int,
    compress_rate: int = 4,
    input_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    block_T: int = 64,
    block_S: int = 128,
    threads: int = 128,
    num_stages: int = 2,
):
    """Index scores for every (query token, compressed block) pair.

    One block owns a ``[block_T, block_S]`` tile of the score matrix and walks
    the ``nhI`` heads, issuing one GEMM each. Tiles that lie entirely above the
    causal diagonal are skipped without loading anything.

    Args:
        B, N, NB: batch, query tokens, compressed blocks.
        nhI, cI: indexer heads and head dim.
        compress_rate: ``m``, for the causal predicate.
        input_dtype, accum_dtype: tensor / accumulator dtypes.
        block_T, block_S: query and key tile sizes.
        threads, num_stages: launch / pipelining config.

    Returns:
        A JIT kernel ``(qI, wI, KI) -> I`` with ``qI`` of shape
        ``[B, N, nhI, cI]``, ``wI`` of ``[B, N, nhI]``, ``KI`` of
        ``[B, NB, cI]`` and ``I`` of ``[B, N, NB]``. Positions the causal mask
        kills are :data:`~dsv4.compress.NEG_LARGE`.
    """

    @T.prim_func
    def kernel(
        qI: T.Tensor((B, N, nhI, cI), dtype=input_dtype),
        wI: T.Tensor((B, N, nhI), dtype=accum_dtype),
        KI: T.Tensor((B, NB, cI), dtype=input_dtype),
        t_off: T.int32,
        I: T.Tensor((B, N, NB), dtype=accum_dtype),
    ):
        with T.Kernel(
            T.ceildiv(NB, block_S), T.ceildiv(N, block_T), B, threads=threads
        ) as (i_s, i_t, i_b):
            t0 = i_t * block_T
            s0 = i_s * block_S

            q_sh = T.alloc_shared((block_T, cI), dtype=input_dtype)
            k_sh = T.alloc_shared((block_S, cI), dtype=input_dtype)
            w_sh = T.alloc_shared((block_T, nhI), dtype=accum_dtype)
            a_f = T.alloc_fragment((block_T, block_S), dtype=accum_dtype)
            acc = T.alloc_fragment((block_T, block_S), dtype=accum_dtype)
            out = T.alloc_fragment((block_T, block_S), dtype=accum_dtype)

            T.clear(acc)

            # The whole tile is masked when even the newest query cannot reach
            # the oldest block: (s0 + 1) * m > t0 + block_T - 1.
            if (s0 + 1) * compress_rate < t_off + t0 + block_T:
                T.copy(KI[i_b, s0 : s0 + block_S, :], k_sh)
                T.copy(wI[i_b, t0 : t0 + block_T, :], w_sh)

                for h in T.Pipelined(nhI, num_stages=num_stages):
                    T.copy(qI[i_b, t0 : t0 + block_T, h, :], q_sh)
                    T.gemm(q_sh, k_sh, a_f, transpose_B=True, clear_accum=True)
                    for i, j in T.Parallel(block_T, block_S):
                        acc[i, j] += w_sh[i, h] * T.max(a_f[i, j], 0.0)

            for i, j in T.Parallel(block_T, block_S):
                t = t0 + i
                s = s0 + j
                ok = ((s + 1) * compress_rate <= t_off + t) & (t < N) & (s < NB)
                out[i, j] = T.if_then_else(ok, acc[i, j], NEG_LARGE)

            for i, j in T.Parallel(block_T, block_S):
                if (t0 + i < N) & (s0 + j < NB):
                    I[i_b, t0 + i, s0 + j] = out[i, j]

    return kernel


@tilelang.jit(out_idx=[-2, -1], pass_configs=_FAST_MATH)
def lightning_index_bwd_q_kernel(
    B: int,
    N: int,
    NB: int,
    nhI: int,
    cI: int,
    compress_rate: int = 4,
    input_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    block_T: int = 64,
    block_S: int = 128,
    threads: int = 128,
    num_stages: int = 2,
):
    """Query-side backward: ``dqI`` and ``dwI``.

    Parallel over query tiles, serial over heads and then key tiles, so each
    ``[block_T, cI]`` slab of ``dqI`` and each ``[block_T]`` slice of ``dwI`` is
    written by exactly one block::

        dw[t, h]    = sum_s dI[t, s] * ReLU(a[t, h, s])
        dr[t, h, s] = dI[t, s] * w[t, h] * [a[t, h, s] > 0]
        dq[t, h]    = sum_s dr[t, h, s] * K[s]

    Returns:
        A JIT kernel ``(qI, wI, KI, dI) -> (dqI, dwI)``.
    """

    @T.prim_func
    def kernel(
        qI: T.Tensor((B, N, nhI, cI), dtype=input_dtype),
        wI: T.Tensor((B, N, nhI), dtype=accum_dtype),
        KI: T.Tensor((B, NB, cI), dtype=input_dtype),
        dI: T.Tensor((B, N, NB), dtype=accum_dtype),
        t_off: T.int32,
        dqI: T.Tensor((B, N, nhI, cI), dtype=accum_dtype),
        dwI: T.Tensor((B, N, nhI), dtype=accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_T), B, threads=threads) as (i_t, i_b):
            t0 = i_t * block_T

            q_sh = T.alloc_shared((block_T, cI), dtype=input_dtype)
            k_sh = T.alloc_shared((block_S, cI), dtype=input_dtype)
            w_sh = T.alloc_shared((block_T, nhI), dtype=accum_dtype)
            di_sh = T.alloc_shared((block_T, block_S), dtype=accum_dtype)
            a_f = T.alloc_fragment((block_T, block_S), dtype=accum_dtype)
            dr_f = T.alloc_fragment((block_T, block_S), dtype=input_dtype)
            tmp = T.alloc_fragment((block_T, block_S), dtype=accum_dtype)
            dq_acc = T.alloc_fragment((block_T, cI), dtype=accum_dtype)
            dq_out = T.alloc_shared((block_T, cI), dtype=accum_dtype)
            dw_part = T.alloc_fragment((block_T,), dtype=accum_dtype)
            dw_acc = T.alloc_fragment((block_T,), dtype=accum_dtype)

            T.copy(wI[i_b, t0 : t0 + block_T, :], w_sh)

            # Only blocks strictly older than the tile's newest query matter.
            n_s = T.ceildiv(
                T.max(t_off + t0 + block_T - 1, 0), block_S * compress_rate
            ) + 1

            for h in T.serial(nhI):
                T.copy(qI[i_b, t0 : t0 + block_T, h, :], q_sh)
                T.clear(dq_acc)
                T.clear(dw_acc)

                for i_s in T.Pipelined(n_s, num_stages=num_stages):
                    s0 = i_s * block_S
                    if s0 < NB:
                        T.copy(KI[i_b, s0 : s0 + block_S, :], k_sh)
                        T.copy(dI[i_b, t0 : t0 + block_T, s0 : s0 + block_S], di_sh)
                        T.gemm(q_sh, k_sh, a_f, transpose_B=True, clear_accum=True)

                        for i, j in T.Parallel(block_T, block_S):
                            t = t0 + i
                            s = s0 + j
                            ok = (
                                ((s + 1) * compress_rate <= t_off + t)
                                & (t < N)
                                & (s < NB)
                            )
                            g = T.if_then_else(ok, di_sh[i, j], 0.0)
                            tmp[i, j] = g * T.max(a_f[i, j], 0.0)
                            dr_f[i, j] = T.Cast(
                                input_dtype,
                                g * w_sh[i, h] * T.if_then_else(a_f[i, j] > 0.0, 1.0, 0.0),
                            )

                        T.reduce_sum(tmp, dw_part, dim=-1, clear=True)
                        for i in T.Parallel(block_T):
                            dw_acc[i] += dw_part[i]

                        T.gemm(dr_f, k_sh, dq_acc)

                T.copy(dq_acc, dq_out)
                for i, d in T.Parallel(block_T, cI):
                    if t0 + i < N:
                        dqI[i_b, t0 + i, h, d] = dq_out[i, d]
                for i in T.Parallel(block_T):
                    if t0 + i < N:
                        dwI[i_b, t0 + i, h] = dw_acc[i]

    return kernel


@tilelang.jit(out_idx=[-1], pass_configs=_FAST_MATH)
def lightning_index_bwd_k_kernel(
    B: int,
    N: int,
    NB: int,
    nhI: int,
    cI: int,
    compress_rate: int = 4,
    input_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    block_T: int = 64,
    block_S: int = 128,
    threads: int = 128,
    num_stages: int = 2,
):
    """Key-side backward: ``dKI``.

    Parallel over key tiles, serial over heads and query tiles, so each
    ``[block_S, cI]`` slab of ``dKI`` is owned by one block::

        dK[s] = sum_t sum_h dr[t, h, s] * q[t, h]

    The query loop starts at the first tile that can legally see this key tile
    -- everything below ``(s0 + 1) * m`` is masked out for every query in it.

    Returns:
        A JIT kernel ``(qI, wI, KI, dI) -> dKI``.
    """

    @T.prim_func
    def kernel(
        qI: T.Tensor((B, N, nhI, cI), dtype=input_dtype),
        wI: T.Tensor((B, N, nhI), dtype=accum_dtype),
        KI: T.Tensor((B, NB, cI), dtype=input_dtype),
        dI: T.Tensor((B, N, NB), dtype=accum_dtype),
        t_off: T.int32,
        dKI: T.Tensor((B, NB, cI), dtype=accum_dtype),
    ):
        with T.Kernel(T.ceildiv(NB, block_S), B, threads=threads) as (i_s, i_b):
            s0 = i_s * block_S

            q_sh = T.alloc_shared((block_T, cI), dtype=input_dtype)
            k_sh = T.alloc_shared((block_S, cI), dtype=input_dtype)
            w_sh = T.alloc_shared((block_T, nhI), dtype=accum_dtype)
            di_sh = T.alloc_shared((block_T, block_S), dtype=accum_dtype)
            a_f = T.alloc_fragment((block_T, block_S), dtype=accum_dtype)
            dr_f = T.alloc_fragment((block_T, block_S), dtype=input_dtype)
            dk_acc = T.alloc_fragment((block_S, cI), dtype=accum_dtype)
            dk_out = T.alloc_shared((block_S, cI), dtype=accum_dtype)

            T.copy(KI[i_b, s0 : s0 + block_S, :], k_sh)
            T.clear(dk_acc)

            # First query tile that can see block s0 at all.
            t_lo = T.max((s0 + 1) * compress_rate - t_off, 0) // block_T
            n_t = T.ceildiv(N, block_T)

            for h in T.serial(nhI):
                for i_t in T.Pipelined(n_t, num_stages=num_stages):
                    t0 = i_t * block_T
                    if (i_t >= t_lo) & (t0 < N):
                        T.copy(qI[i_b, t0 : t0 + block_T, h, :], q_sh)
                        T.copy(wI[i_b, t0 : t0 + block_T, :], w_sh)
                        T.copy(dI[i_b, t0 : t0 + block_T, s0 : s0 + block_S], di_sh)
                        T.gemm(q_sh, k_sh, a_f, transpose_B=True, clear_accum=True)

                        for i, j in T.Parallel(block_T, block_S):
                            t = t0 + i
                            s = s0 + j
                            ok = (
                                ((s + 1) * compress_rate <= t_off + t)
                                & (t < N)
                                & (s < NB)
                            )
                            g = T.if_then_else(ok, di_sh[i, j], 0.0)
                            dr_f[i, j] = T.Cast(
                                input_dtype,
                                g * w_sh[i, h] * T.if_then_else(a_f[i, j] > 0.0, 1.0, 0.0),
                            )

                        T.gemm(dr_f, q_sh, dk_acc, transpose_A=True)

            T.copy(dk_acc, dk_out)
            for j, d in T.Parallel(block_S, cI):
                if s0 + j < NB:
                    dKI[i_b, s0 + j, d] = dk_out[j, d]

    return kernel


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def lightning_index_fwd(
    qI, wI, KI, compress_rate, t_offset: int = 0,
    config: KernelConfig = DEFAULT_KERNEL_CONFIG,
):
    """Run :func:`lightning_index_kernel`.

    Args:
        qI: ``[B, N, nhI, cI]``.
        wI: ``[B, N, nhI]``.
        KI: ``[B, NB, cI]``.
        compress_rate: ``m``.
        config: tiling configuration.

    Returns:
        ``[B, N, NB]`` fp32 scores.

    Note:
        The score matrix is ``N * NB`` per batch element, which is only
        tractable to materialise up to moderate context. Use
        :func:`dsv4.ops.index_and_select` instead for long sequences -- it
        chunks the query axis and collapses each chunk to its top-k before
        moving on, so peak memory is ``chunk * NB`` rather than ``N * NB``.
    """
    B, N, nhI, cI = qI.shape
    NB = KI.shape[1]
    kernel = lightning_index_kernel(
        B, N, NB, nhI, cI,
        compress_rate=compress_rate,
        input_dtype=_tl_dtype(qI.dtype),
        block_T=config.block_T,
        block_S=config.block_S,
        threads=config.threads,
        num_stages=config.num_stages,
    )
    return kernel(qI.contiguous(), wI.float().contiguous(), KI.contiguous(), int(t_offset))


def lightning_index_bwd(
    dI, qI, wI, KI, compress_rate, t_offset: int = 0,
    config: KernelConfig = DEFAULT_KERNEL_CONFIG,
):
    """Run both backward kernels.

    They write disjoint outputs from the same inputs, so they are independent
    and can overlap on separate streams.

    Returns:
        ``(dqI, dwI, dKI)``.
    """
    B, N, nhI, cI = qI.shape
    NB = KI.shape[1]
    common = dict(
        compress_rate=compress_rate,
        input_dtype=_tl_dtype(qI.dtype),
        block_T=config.block_T,
        block_S=config.block_S,
        threads=config.threads,
        num_stages=config.num_stages,
    )
    args = (
        qI.contiguous(), wI.float().contiguous(), KI.contiguous(),
        dI.float().contiguous(), int(t_offset),
    )

    dqI, dwI = lightning_index_bwd_q_kernel(B, N, NB, nhI, cI, **common)(*args)
    dKI = lightning_index_bwd_k_kernel(B, N, NB, nhI, cI, **common)(*args)
    return dqI.to(qI.dtype), dwI, dKI.to(KI.dtype)
