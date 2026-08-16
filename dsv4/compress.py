"""KV compression: TileLang kernels and drivers.

Four kernels, two per attention variant.

    1. ``hca_compress_kernel``      C,Z,B          -> CComp     (eq. 22/23)
    2. ``hca_compress_bwd_kernel``  dCComp,...     -> dC,dZ
    3. ``csa_compress_kernel``      Ca,Cb,Za,Zb,B  -> CComp     (eq. 11/12)
    4. ``csa_compress_bwd_kernel``  dCComp,...     -> dCa,dCb,dZa,dZb

Both variants pool a span of token rows into one entry with a softmax over the
span, taken **per channel**: with ``C, Z`` of shape ``[B, N, c]`` this is ``c``
independent length-``m`` softmaxes per block, not one softmax over channels.
That makes every channel an independent problem and the whole stage
embarrassingly parallel -- the only axis with a serial dependence is the short
reduction over ``m``.

Schedules
---------
The two variants sit at opposite ends of the ``m`` range and want different
schedules, which is why they are separate kernels rather than one parameterised
one:

``HCA`` (``m' = 128``)
    One block per compressed entry. The ``m`` axis is long enough to reduce
    directly, so the tile is transposed into ``[block_D, m]`` and reduced along
    the last axis -- the ordinary softmax layout.
``CSA`` (``m = 4``, ``2m = 8`` slots)
    A tile of 8 rows would leave the machine idle, so one block handles
    ``block_I`` consecutive entries and walks the 8 slots in a fully unrolled
    serial loop, with ``[block_I, block_D]`` accumulators.

Recomputation
-------------
Neither backward stores the pooling weights ``S``. They are the size of the
input, and recovering them costs one softmax over at most 128 values against a
re-read that the kernel is doing anyway -- the flash-attention trade, and at 1M
context the difference is a tensor the size of ``Z`` per layer.

Bias gradients
--------------
``dbias[r] = sum_{b,i} dZ[b, i*m + r]`` is a plain strided reduction over a
tensor the backward already had to write. Doing it inside the kernel would mean
either global atomics on every element (``B*N*c`` of them) or a per-program
workspace; doing it after, over the finished ``dZ``, is one extra streaming
read and hands the work to a reduction PyTorch already does at bandwidth. The
drivers therefore compute it with ``dZ.view(...).sum(...)``.
"""

# NOTE: deliberately no `from __future__ import annotations` -- TVMScript
# evaluates the `T.Tensor(...)` parameter annotations at definition time, and
# postponed annotations turn them into strings the parser cannot resolve.

import tilelang
import tilelang.language as T
import torch

from .config import DEFAULT_KERNEL_CONFIG, KernelConfig

__all__ = [
    "hca_compress_kernel",
    "hca_compress_bwd_kernel",
    "csa_compress_kernel",
    "csa_compress_bwd_kernel",
    "hca_compress_fwd",
    "hca_compress_bwd",
    "csa_compress_fwd",
    "csa_compress_bwd",
]

_FAST_MATH = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}

_TORCH_TO_TL = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}

# Large enough to zero an exp, small enough that subtracting a running max
# cannot overflow to +inf. Used instead of -inf so no NaN can appear from
# (-inf) - (-inf) when a whole span is masked.
NEG_LARGE = -1.0e30


def _tl_dtype(dtype):
    """Map a torch dtype to the TileLang dtype string."""
    try:
        return _TORCH_TO_TL[dtype]
    except KeyError:
        raise TypeError(f"unsupported dtype {dtype}; use float16, bfloat16 or float32") from None


def _check_dims(N, c, m, block_D):
    if N % m:
        raise ValueError(f"sequence length {N} must be a multiple of compress_rate {m}")
    if c % block_D:
        raise ValueError(f"head dim {c} must be a multiple of block_D {block_D}")


# ---------------------------------------------------------------------------
# HCA: non-overlapped, one stream
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[-1], pass_configs=_FAST_MATH)
def hca_compress_kernel(
    B: int,
    N: int,
    c: int,
    m: int,
    input_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    block_D: int = 64,
    threads: int = 128,
):
    """Pool every ``m`` token rows into one entry (eq. 22/23).

    ``S = softmax_j(Z_j + B_j)`` over the ``m`` rows of the block, per channel,
    then ``CComp = sum_j S_j * C_j``.

    The ``[m, block_D]`` loads are contiguous along ``block_D`` (the channel
    axis is innermost in ``[B, N, c]``), so they coalesce; the tile is then
    transposed in shared memory to ``[block_D, m]`` so the softmax reduces along
    the last axis.

    Args:
        B, N, c: batch, sequence length, head dim. ``N`` must be a multiple of
            ``m`` and ``c`` a multiple of ``block_D``.
        m: compression rate ``m'``.
        input_dtype, accum_dtype: tensor / accumulator dtypes.
        block_D: channels per block.
        threads: threads per block.

    Returns:
        A JIT kernel ``(C, Z, bias) -> CComp`` with ``C, Z`` of shape
        ``[B, N, c]``, ``bias`` of ``[m, c]`` and ``CComp`` of ``[B, N//m, c]``.
    """
    _check_dims(N, c, m, block_D)
    NB = N // m

    @T.prim_func
    def kernel(
        C: T.Tensor((B, N, c), dtype=input_dtype),
        Z: T.Tensor((B, N, c), dtype=input_dtype),
        bias: T.Tensor((m, c), dtype=accum_dtype),
        CComp: T.Tensor((B, NB, c), dtype=input_dtype),
    ):
        with T.Kernel(T.ceildiv(c, block_D), NB, B, threads=threads) as (i_d, i_blk, i_b):
            d0 = i_d * block_D
            t0 = i_blk * m

            z_sh = T.alloc_shared((m, block_D), dtype=input_dtype)
            c_sh = T.alloc_shared((m, block_D), dtype=input_dtype)
            b_sh = T.alloc_shared((m, block_D), dtype=accum_dtype)
            p = T.alloc_fragment((block_D, m), dtype=accum_dtype)
            mx = T.alloc_fragment((block_D,), dtype=accum_dtype)
            sm = T.alloc_fragment((block_D,), dtype=accum_dtype)
            acc = T.alloc_fragment((block_D,), dtype=accum_dtype)
            out = T.alloc_fragment((block_D,), dtype=accum_dtype)

            T.copy(Z[i_b, t0 : t0 + m, d0 : d0 + block_D], z_sh)
            T.copy(C[i_b, t0 : t0 + m, d0 : d0 + block_D], c_sh)
            T.copy(bias[0:m, d0 : d0 + block_D], b_sh)

            for d, j in T.Parallel(block_D, m):
                p[d, j] = z_sh[j, d] + b_sh[j, d]
            T.reduce_max(p, mx, dim=-1, clear=True)

            for d, j in T.Parallel(block_D, m):
                p[d, j] = T.exp(p[d, j] - mx[d])
            T.reduce_sum(p, sm, dim=-1, clear=True)

            # Reuse p for the weighted values: one fewer live tile.
            for d, j in T.Parallel(block_D, m):
                p[d, j] = p[d, j] * c_sh[j, d]
            T.reduce_sum(p, acc, dim=-1, clear=True)

            for d in T.Parallel(block_D):
                out[d] = acc[d] / sm[d]
            T.copy(out, CComp[i_b, i_blk, d0 : d0 + block_D])

    return kernel


@tilelang.jit(out_idx=[-2, -1], pass_configs=_FAST_MATH)
def hca_compress_bwd_kernel(
    B: int,
    N: int,
    c: int,
    m: int,
    input_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    block_D: int = 64,
    threads: int = 128,
):
    """Backward of :func:`hca_compress_kernel`.

    The softmax Jacobian collapses because the pooled output is exactly the
    weighted mean the Jacobian's second term needs::

        dC_j = S_j * dO
        dZ_j = dC_j * (C_j - CComp)

    so no second reduction is required -- ``CComp`` from the forward *is* the
    ``sum_l S_l C_l`` term. ``S`` is recomputed here rather than saved.

    Args:
        B, N, c, m, input_dtype, accum_dtype, block_D, threads: as in the
            forward.

    Returns:
        A JIT kernel ``(C, Z, bias, CComp, dCComp) -> (dC, dZ)``. ``dbias`` is
        the caller's strided sum over ``dZ``; see the module docstring.
    """
    _check_dims(N, c, m, block_D)
    NB = N // m

    @T.prim_func
    def kernel(
        C: T.Tensor((B, N, c), dtype=input_dtype),
        Z: T.Tensor((B, N, c), dtype=input_dtype),
        bias: T.Tensor((m, c), dtype=accum_dtype),
        CComp: T.Tensor((B, NB, c), dtype=input_dtype),
        dCComp: T.Tensor((B, NB, c), dtype=accum_dtype),
        dC: T.Tensor((B, N, c), dtype=accum_dtype),
        dZ: T.Tensor((B, N, c), dtype=accum_dtype),
    ):
        with T.Kernel(T.ceildiv(c, block_D), NB, B, threads=threads) as (i_d, i_blk, i_b):
            d0 = i_d * block_D
            t0 = i_blk * m

            z_sh = T.alloc_shared((m, block_D), dtype=input_dtype)
            c_sh = T.alloc_shared((m, block_D), dtype=input_dtype)
            b_sh = T.alloc_shared((m, block_D), dtype=accum_dtype)
            p = T.alloc_fragment((block_D, m), dtype=accum_dtype)
            mx = T.alloc_fragment((block_D,), dtype=accum_dtype)
            sm = T.alloc_fragment((block_D,), dtype=accum_dtype)
            do_f = T.alloc_fragment((block_D,), dtype=accum_dtype)
            oc_f = T.alloc_fragment((block_D,), dtype=accum_dtype)
            dc_t = T.alloc_fragment((block_D, m), dtype=accum_dtype)
            dz_t = T.alloc_fragment((block_D, m), dtype=accum_dtype)
            dc_sh = T.alloc_shared((m, block_D), dtype=accum_dtype)
            dz_sh = T.alloc_shared((m, block_D), dtype=accum_dtype)

            T.copy(Z[i_b, t0 : t0 + m, d0 : d0 + block_D], z_sh)
            T.copy(C[i_b, t0 : t0 + m, d0 : d0 + block_D], c_sh)
            T.copy(bias[0:m, d0 : d0 + block_D], b_sh)
            T.copy(dCComp[i_b, i_blk, d0 : d0 + block_D], do_f)
            T.copy(CComp[i_b, i_blk, d0 : d0 + block_D], oc_f)

            # Recompute S.
            for d, j in T.Parallel(block_D, m):
                p[d, j] = z_sh[j, d] + b_sh[j, d]
            T.reduce_max(p, mx, dim=-1, clear=True)
            for d, j in T.Parallel(block_D, m):
                p[d, j] = T.exp(p[d, j] - mx[d])
            T.reduce_sum(p, sm, dim=-1, clear=True)

            for d, j in T.Parallel(block_D, m):
                s = p[d, j] / sm[d]
                dc_t[d, j] = s * do_f[d]
                dz_t[d, j] = dc_t[d, j] * (c_sh[j, d] - oc_f[d])

            for j, d in T.Parallel(m, block_D):
                dc_sh[j, d] = dc_t[d, j]
                dz_sh[j, d] = dz_t[d, j]

            T.copy(dc_sh, dC[i_b, t0 : t0 + m, d0 : d0 + block_D])
            T.copy(dz_sh, dZ[i_b, t0 : t0 + m, d0 : d0 + block_D])

    return kernel


# ---------------------------------------------------------------------------
# CSA: overlapped, two streams
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[-1], pass_configs=_FAST_MATH)
def csa_compress_kernel(
    B: int,
    N: int,
    c: int,
    m: int,
    input_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    block_D: int = 64,
    block_I: int = 32,
    threads: int = 128,
):
    """Pool every ``m`` token rows into one entry, two overlapped streams.

    Entry ``i`` draws ``m`` rows of stream *a* from its own span and ``m`` rows
    of stream *b* from the span *before* it, under one softmax over all ``2m``
    (eq. 11/12). Entry 0 has no predecessor: its *b* logits are forced to
    :data:`NEG_LARGE` and its *b* values to zero, which the softmax turns into
    zero weight.

    ``m`` is 4 in every published configuration, so the ``2m`` axis is walked as
    an unrolled serial loop and ``block_I`` entries are processed per block to
    keep the tiles a useful size.

    The *b*-stream tile starts one span earlier than the *a*-stream tile, so for
    the first block its first rows are out of bounds. They are filled by a
    predicated element-wise gather rather than a bulk copy -- still coalesced
    along the channel axis, and it keeps the boundary out of the index
    arithmetic.

    Args:
        B, N, c: batch, sequence length, head dim.
        m: compression rate.
        input_dtype, accum_dtype: tensor / accumulator dtypes.
        block_D: channels per block.
        block_I: compressed entries per block.
        threads: threads per block.

    Returns:
        A JIT kernel ``(Ca, Cb, Za, Zb, bias_a, bias_b) -> CComp``.
    """
    _check_dims(N, c, m, block_D)
    NB = N // m
    R = block_I * m  # token rows held per block, per stream

    @T.prim_func
    def kernel(
        Ca: T.Tensor((B, N, c), dtype=input_dtype),
        Cb: T.Tensor((B, N, c), dtype=input_dtype),
        Za: T.Tensor((B, N, c), dtype=input_dtype),
        Zb: T.Tensor((B, N, c), dtype=input_dtype),
        bias_a: T.Tensor((m, c), dtype=accum_dtype),
        bias_b: T.Tensor((m, c), dtype=accum_dtype),
        CComp: T.Tensor((B, NB, c), dtype=input_dtype),
    ):
        with T.Kernel(
            T.ceildiv(c, block_D), T.ceildiv(NB, block_I), B, threads=threads
        ) as (i_d, i_g, i_b):
            d0 = i_d * block_D
            g0 = i_g * block_I  # first compressed entry handled here

            za_sh = T.alloc_shared((R, block_D), dtype=accum_dtype)
            ca_sh = T.alloc_shared((R, block_D), dtype=accum_dtype)
            zb_sh = T.alloc_shared((R, block_D), dtype=accum_dtype)
            cb_sh = T.alloc_shared((R, block_D), dtype=accum_dtype)
            ba_sh = T.alloc_shared((m, block_D), dtype=accum_dtype)
            bb_sh = T.alloc_shared((m, block_D), dtype=accum_dtype)

            mx = T.alloc_fragment((block_I, block_D), dtype=accum_dtype)
            sm = T.alloc_fragment((block_I, block_D), dtype=accum_dtype)
            acc = T.alloc_fragment((block_I, block_D), dtype=accum_dtype)
            out = T.alloc_shared((block_I, block_D), dtype=input_dtype)

            T.copy(bias_a[0:m, d0 : d0 + block_D], ba_sh)
            T.copy(bias_b[0:m, d0 : d0 + block_D], bb_sh)

            # Stream a: rows [g0*m, g0*m + R). Guarded -- the last group can
            # run past NB*m when NB is not a multiple of block_I.
            for r, d in T.Parallel(R, block_D):
                t = g0 * m + r
                za_sh[r, d] = T.if_then_else(
                    t < N, T.Cast(accum_dtype, Za[i_b, T.min(t, N - 1), d0 + d]), NEG_LARGE
                )
                ca_sh[r, d] = T.if_then_else(
                    t < N, T.Cast(accum_dtype, Ca[i_b, T.min(t, N - 1), d0 + d]), 0.0
                )

            # Stream b: rows [(g0-1)*m, (g0-1)*m + R). Negative for the first
            # group; those slots are the padded predecessor of entry 0.
            for r, d in T.Parallel(R, block_D):
                t = (g0 - 1) * m + r
                safe = T.max(T.min(t, N - 1), 0)
                ok = (t >= 0) & (t < N)
                zb_sh[r, d] = T.if_then_else(
                    ok, T.Cast(accum_dtype, Zb[i_b, safe, d0 + d]), NEG_LARGE
                )
                cb_sh[r, d] = T.if_then_else(
                    ok, T.Cast(accum_dtype, Cb[i_b, safe, d0 + d]), 0.0
                )

            # Joint softmax over the 2m slots, then the weighted sum. Two
            # unrolled passes: max, then exp/sum/accumulate fused.
            for ii, d in T.Parallel(block_I, block_D):
                mx[ii, d] = NEG_LARGE
            for j in T.serial(m):
                for ii, d in T.Parallel(block_I, block_D):
                    mx[ii, d] = T.max(mx[ii, d], za_sh[ii * m + j, d] + ba_sh[j, d])
                    mx[ii, d] = T.max(mx[ii, d], zb_sh[ii * m + j, d] + bb_sh[j, d])

            for ii, d in T.Parallel(block_I, block_D):
                sm[ii, d] = 0.0
                acc[ii, d] = 0.0
            for j in T.serial(m):
                for ii, d in T.Parallel(block_I, block_D):
                    pa = T.exp(za_sh[ii * m + j, d] + ba_sh[j, d] - mx[ii, d])
                    pb = T.exp(zb_sh[ii * m + j, d] + bb_sh[j, d] - mx[ii, d])
                    sm[ii, d] += pa + pb
                    acc[ii, d] += pa * ca_sh[ii * m + j, d] + pb * cb_sh[ii * m + j, d]

            for ii, d in T.Parallel(block_I, block_D):
                out[ii, d] = T.Cast(input_dtype, acc[ii, d] / sm[ii, d])

            for ii, d in T.Parallel(block_I, block_D):
                if g0 + ii < NB:
                    CComp[i_b, g0 + ii, d0 + d] = out[ii, d]

    return kernel


# No `out_idx`: stream *b*'s last span has no consumer (it would belong to entry
# NB, which does not exist), so no block ever writes those rows. A JIT-allocated
# output would leave them as uninitialised memory, which then flows straight
# into `dbias_b`. The caller passes zeroed buffers instead.
@tilelang.jit(pass_configs=_FAST_MATH)
def csa_compress_bwd_kernel(
    B: int,
    N: int,
    c: int,
    m: int,
    input_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    block_D: int = 64,
    block_I: int = 32,
    threads: int = 128,
):
    """Backward of :func:`csa_compress_kernel`.

    Same collapsed softmax Jacobian as the HCA backward, then a scatter: slot
    ``j`` of entry ``i`` belongs to stream *a* at token ``i*m + j`` for the
    first ``m`` slots and to stream *b* at token ``(i-1)*m + j`` for the last
    ``m``.

    Because stream *a* row ``t`` feeds only entry ``t // m`` and stream *b* row
    ``t`` only entry ``t // m + 1``, every destination is written by exactly one
    slot of exactly one entry. The scatter needs no atomics and no
    read-modify-write, which is the practical payoff of the overlapped scheme
    being a shift rather than a true overlap.

    Returns:
        A JIT kernel
        ``(Ca, Cb, Za, Zb, bias_a, bias_b, CComp, dCComp,
        dCa, dCb, dZa, dZb) -> None``. The four gradients are written in place
        and **must be zeroed by the caller**: stream *b*'s final span is read by
        no entry, so those rows are never assigned. The bias gradients are the
        caller's strided sums over ``dZa`` and ``dZb``.
    """
    _check_dims(N, c, m, block_D)
    NB = N // m
    R = block_I * m

    @T.prim_func
    def kernel(
        Ca: T.Tensor((B, N, c), dtype=input_dtype),
        Cb: T.Tensor((B, N, c), dtype=input_dtype),
        Za: T.Tensor((B, N, c), dtype=input_dtype),
        Zb: T.Tensor((B, N, c), dtype=input_dtype),
        bias_a: T.Tensor((m, c), dtype=accum_dtype),
        bias_b: T.Tensor((m, c), dtype=accum_dtype),
        CComp: T.Tensor((B, NB, c), dtype=input_dtype),
        dCComp: T.Tensor((B, NB, c), dtype=accum_dtype),
        dCa: T.Tensor((B, N, c), dtype=accum_dtype),
        dCb: T.Tensor((B, N, c), dtype=accum_dtype),
        dZa: T.Tensor((B, N, c), dtype=accum_dtype),
        dZb: T.Tensor((B, N, c), dtype=accum_dtype),
    ):
        with T.Kernel(
            T.ceildiv(c, block_D), T.ceildiv(NB, block_I), B, threads=threads
        ) as (i_d, i_g, i_b):
            d0 = i_d * block_D
            g0 = i_g * block_I

            za_sh = T.alloc_shared((R, block_D), dtype=accum_dtype)
            ca_sh = T.alloc_shared((R, block_D), dtype=accum_dtype)
            zb_sh = T.alloc_shared((R, block_D), dtype=accum_dtype)
            cb_sh = T.alloc_shared((R, block_D), dtype=accum_dtype)
            ba_sh = T.alloc_shared((m, block_D), dtype=accum_dtype)
            bb_sh = T.alloc_shared((m, block_D), dtype=accum_dtype)

            mx = T.alloc_fragment((block_I, block_D), dtype=accum_dtype)
            sm = T.alloc_fragment((block_I, block_D), dtype=accum_dtype)
            do_f = T.alloc_fragment((block_I, block_D), dtype=accum_dtype)
            oc_f = T.alloc_fragment((block_I, block_D), dtype=accum_dtype)

            T.copy(bias_a[0:m, d0 : d0 + block_D], ba_sh)
            T.copy(bias_b[0:m, d0 : d0 + block_D], bb_sh)

            for r, d in T.Parallel(R, block_D):
                t = g0 * m + r
                za_sh[r, d] = T.if_then_else(
                    t < N, T.Cast(accum_dtype, Za[i_b, T.min(t, N - 1), d0 + d]), NEG_LARGE
                )
                ca_sh[r, d] = T.if_then_else(
                    t < N, T.Cast(accum_dtype, Ca[i_b, T.min(t, N - 1), d0 + d]), 0.0
                )
            for r, d in T.Parallel(R, block_D):
                t = (g0 - 1) * m + r
                safe = T.max(T.min(t, N - 1), 0)
                ok = (t >= 0) & (t < N)
                zb_sh[r, d] = T.if_then_else(
                    ok, T.Cast(accum_dtype, Zb[i_b, safe, d0 + d]), NEG_LARGE
                )
                cb_sh[r, d] = T.if_then_else(
                    ok, T.Cast(accum_dtype, Cb[i_b, safe, d0 + d]), 0.0
                )

            for ii, d in T.Parallel(block_I, block_D):
                i_glb = T.min(g0 + ii, NB - 1)
                do_f[ii, d] = dCComp[i_b, i_glb, d0 + d]
                oc_f[ii, d] = T.Cast(accum_dtype, CComp[i_b, i_glb, d0 + d])

            # Recompute the joint softmax.
            for ii, d in T.Parallel(block_I, block_D):
                mx[ii, d] = NEG_LARGE
            for j in T.serial(m):
                for ii, d in T.Parallel(block_I, block_D):
                    mx[ii, d] = T.max(mx[ii, d], za_sh[ii * m + j, d] + ba_sh[j, d])
                    mx[ii, d] = T.max(mx[ii, d], zb_sh[ii * m + j, d] + bb_sh[j, d])
            for ii, d in T.Parallel(block_I, block_D):
                sm[ii, d] = 0.0
            for j in T.serial(m):
                for ii, d in T.Parallel(block_I, block_D):
                    sm[ii, d] += T.exp(za_sh[ii * m + j, d] + ba_sh[j, d] - mx[ii, d])
                    sm[ii, d] += T.exp(zb_sh[ii * m + j, d] + bb_sh[j, d] - mx[ii, d])

            # Scatter. Each destination row is touched by one slot only.
            for j in T.serial(m):
                for ii, d in T.Parallel(block_I, block_D):
                    ta = g0 * m + ii * m + j
                    if ta < N:
                        sa = T.exp(za_sh[ii * m + j, d] + ba_sh[j, d] - mx[ii, d]) / sm[ii, d]
                        gc = sa * do_f[ii, d]
                        dCa[i_b, ta, d0 + d] = gc
                        dZa[i_b, ta, d0 + d] = gc * (ca_sh[ii * m + j, d] - oc_f[ii, d])

                    tb = (g0 - 1) * m + ii * m + j
                    if (tb >= 0) & (tb < N) & (g0 + ii < NB):
                        sb = T.exp(zb_sh[ii * m + j, d] + bb_sh[j, d] - mx[ii, d]) / sm[ii, d]
                        gc = sb * do_f[ii, d]
                        dCb[i_b, tb, d0 + d] = gc
                        dZb[i_b, tb, d0 + d] = gc * (cb_sh[ii * m + j, d] - oc_f[ii, d])

    return kernel


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def hca_compress_fwd(C, Z, bias, config: KernelConfig = DEFAULT_KERNEL_CONFIG):
    """Run :func:`hca_compress_kernel`.

    Args:
        C, Z: ``[B, N, c]``.
        bias: ``[m, c]`` fp32.
        config: tiling configuration.

    Returns:
        ``[B, N // m, c]`` compressed entries.
    """
    B, N, c = C.shape
    m = bias.shape[0]
    kernel = hca_compress_kernel(
        B, N, c, m,
        input_dtype=_tl_dtype(C.dtype),
        block_D=config.block_D,
        threads=config.threads,
    )
    return kernel(C.contiguous(), Z.contiguous(), bias.float().contiguous())


def hca_compress_bwd(C, Z, bias, CComp, dCComp, config: KernelConfig = DEFAULT_KERNEL_CONFIG):
    """Run :func:`hca_compress_bwd_kernel` and reduce ``dbias``.

    Returns:
        ``(dC, dZ, dbias)``.
    """
    B, N, c = C.shape
    m = bias.shape[0]
    kernel = hca_compress_bwd_kernel(
        B, N, c, m,
        input_dtype=_tl_dtype(C.dtype),
        block_D=config.block_D,
        threads=config.threads,
    )
    dC, dZ = kernel(
        C.contiguous(), Z.contiguous(), bias.float().contiguous(),
        CComp.contiguous(), dCComp.float().contiguous(),
    )
    dbias = dZ.view(B, N // m, m, c).sum(dim=(0, 1))
    return dC.to(C.dtype), dZ.to(Z.dtype), dbias


def csa_compress_fwd(Ca, Cb, Za, Zb, bias_a, bias_b, config: KernelConfig = DEFAULT_KERNEL_CONFIG):
    """Run :func:`csa_compress_kernel`.

    Returns:
        ``[B, N // m, c]`` compressed entries.
    """
    B, N, c = Ca.shape
    m = bias_a.shape[0]
    kernel = csa_compress_kernel(
        B, N, c, m,
        input_dtype=_tl_dtype(Ca.dtype),
        block_D=config.block_D,
        block_I=max(1, config.block_T // m),
        threads=config.threads,
    )
    return kernel(
        Ca.contiguous(), Cb.contiguous(), Za.contiguous(), Zb.contiguous(),
        bias_a.float().contiguous(), bias_b.float().contiguous(),
    )


def csa_compress_bwd(
    Ca, Cb, Za, Zb, bias_a, bias_b, CComp, dCComp,
    config: KernelConfig = DEFAULT_KERNEL_CONFIG,
):
    """Run :func:`csa_compress_bwd_kernel` and reduce both bias gradients.

    Returns:
        ``(dCa, dCb, dZa, dZb, dbias_a, dbias_b)``.
    """
    B, N, c = Ca.shape
    m = bias_a.shape[0]
    NB = N // m
    kernel = csa_compress_bwd_kernel(
        B, N, c, m,
        input_dtype=_tl_dtype(Ca.dtype),
        block_D=config.block_D,
        block_I=max(1, config.block_T // m),
        threads=config.threads,
    )
    # Zeroed, not empty: stream b's last span is written by no block, because
    # the entry that would consume it does not exist. See the kernel.
    dCa, dCb, dZa, dZb = (
        torch.zeros((B, N, c), device=Ca.device, dtype=torch.float32) for _ in range(4)
    )
    kernel(
        Ca.contiguous(), Cb.contiguous(), Za.contiguous(), Zb.contiguous(),
        bias_a.float().contiguous(), bias_b.float().contiguous(),
        CComp.contiguous(), dCComp.float().contiguous(),
        dCa, dCb, dZa, dZb,
    )
    dbias_a = dZa.view(B, NB, m, c).sum(dim=(0, 1))
    dbias_b = dZb.view(B, NB, m, c).sum(dim=(0, 1))
    return (
        dCa.to(Ca.dtype), dCb.to(Cb.dtype), dZa.to(Za.dtype), dZb.to(Zb.dtype),
        dbias_a, dbias_b,
    )
