"""Forward pass: TileLang kernels and the driver that runs them.

Seven kernels form the training-time pipeline; an eighth handles single-token
decoding. Naming follows the maths in :mod:`gdn2.reference`.

    1. ``cumsum_kernel``       g            -> G      (local cumsum, log2 space)
    2. ``intra_diag_kernel``   q,k,b,G      -> Aqk, Akk diagonal sub-blocks
    3. ``inter_blocks_kernel`` q,k,b,G      -> Aqk, Akk off-diagonal sub-blocks
    4. ``solve_kernel``        Akk          -> Ai = (I + Akk)^-1
    5. ``wy_kernel``           k,v,b,w,G,Ai -> W, U, KG
    6. ``state_kernel``        KG,W,U,G     -> h (per-chunk states), Un, S_final
    7. ``output_kernel``       q,G,Aqk,h,Un -> o

Kernels 2 and 3 both write ``Aqk`` and ``Akk`` in place over disjoint column
ranges, so they must run in that order.

All decay arithmetic happens in **log2 space** (``G`` holds
``log2(e) * cumsum(g)``) so the kernels can use the faster ``exp2``.

Overflow safety
---------------
The WY form needs ``k_s / gamma_s``, and ``1 / gamma_s`` overflows fp32 as soon
as a channel's cumulative decay within a chunk exceeds ~88 nats -- which real
gates do reach. Every block of the ``[BS, BS]`` matrices is therefore rescaled
by the cumulative decay at the first row of its row-sub-block, so both
exponentials are ``exp2`` of a non-positive number. Underflow to zero is the
correct answer there, since the true product is itself vanishing.

That rescaling has no valid reference row on the diagonal sub-blocks, so those
are computed token-parallel in kernel 2 with the exact pairwise difference
``exp2(G_r - G_s)``, which is non-positive for ``r >= s`` by construction.

The module ends with :func:`chunk_gdn2_fwd`, which chains the kernels, and
:func:`gdn2_decode_step` for single-token inference.
"""

# NOTE: deliberately no `from __future__ import annotations` here. TVMScript
# evaluates the `T.Tensor(...)` parameter annotations of a `T.prim_func` at
# definition time; postponed annotations turn them into strings the parser
# cannot resolve against the enclosing scope.

import tilelang
import tilelang.language as T
import torch

from .config import DEFAULT_KERNEL_CONFIG, GDN2Config

__all__ = [
    "chunk_gdn2_fwd",
    "gdn2_decode_step",
    "cumsum_kernel",
    "intra_diag_kernel",
    "inter_blocks_kernel",
    "solve_kernel",
    "wy_kernel",
    "state_kernel",
    "output_kernel",
    "decode_step_kernel",
]

LOG2E = 1.4426950408889634

_FAST_MATH = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}


@tilelang.jit(out_idx=[-1], pass_configs=_FAST_MATH)
def cumsum_kernel(
    B: int,
    S: int,
    H: int,
    DK: int,
    gate_dtype: str = "float32",
    chunk_size: int = 64,
    threads: int = 128,
):
    """Chunk-local cumulative sum of the log decay, rescaled to log2 space.

    ``G[b, t, h, c] = log2(e) * sum_{i in chunk(t), i <= t} g[b, i, h, c]``.

    The scan runs along the token axis with an explicit log-step (Hillis-Steele)
    sweep rather than ``T.cumsum(dim=0)``: that path lowers to ``CumSum2D`` with
    ``Axis=0``, which addresses ``src[col * W + gRow]`` while iterating ``col``
    up to ``W`` -- out of bounds unless the tile is square, and our tiles are
    ``[BS, DK]``. ``log2(BS)`` passes over a tile this small costs nothing.

    Args:
        B, S, H, DK: batch, sequence length, heads, key dim. ``S`` must be a
            multiple of ``chunk_size``.
        gate_dtype: dtype of ``g`` and ``G``; fp32 is strongly recommended.
        chunk_size: chunk length ``BS``; must be a power of two.
        threads: threads per block.

    Returns:
        A JIT kernel ``(g) -> G``, both ``[B, S, H, DK]``.
    """
    BS = chunk_size
    if BS & (BS - 1):
        raise ValueError(f"chunk_size {BS} must be a power of two")
    n_steps = BS.bit_length() - 1
    shape = (B, S, H, DK)

    @T.prim_func
    def kernel(
        g: T.Tensor(shape, dtype=gate_dtype),
        G: T.Tensor(shape, dtype=gate_dtype),
    ):
        with T.Kernel(T.ceildiv(S, BS), B * H, threads=threads) as (i_s, i_bh):
            i_b, i_h = i_bh // H, i_bh % H
            cur = T.alloc_shared((BS, DK), dtype=gate_dtype)
            nxt = T.alloc_shared((BS, DK), dtype=gate_dtype)

            T.copy(g[i_b, i_s * BS : (i_s + 1) * BS, i_h, :], cur)
            for t, c in T.Parallel(BS, DK):
                cur[t, c] = cur[t, c] * LOG2E

            for step in T.serial(n_steps):
                off = T.shift_left(1, step)
                for t, c in T.Parallel(BS, DK):
                    # Index is clamped as well as predicated: correct either
                    # way, and never reads outside the tile.
                    prev = cur[T.max(t - off, 0), c]
                    nxt[t, c] = cur[t, c] + T.if_then_else(t >= off, prev, 0.0)
                T.copy(nxt, cur)

            T.copy(cur, G[i_b, i_s * BS : (i_s + 1) * BS, i_h, :])

    return kernel


@tilelang.jit(out_idx=[-2, -1], pass_configs=_FAST_MATH)
def intra_diag_kernel(
    B: int,
    S: int,
    H: int,
    DK: int,
    scale: float,
    input_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    gate_dtype: str = "float32",
    chunk_size: int = 64,
    sub_chunk_size: int = 16,
    block_H: int = 1,
    threads: int = 64,
    num_stages: int = 1,
):
    """Diagonal sub-blocks of the intra-chunk matrices, one block per token.

    For token ``t`` this walks every earlier token ``j`` inside ``t``'s
    *sub-chunk* and reduces over the channel axis directly, so the decay enters
    as the exact ``exp2(G_t - G_j)`` (non-positive, hence safe) rather than as a
    ratio of two exponentials. Parallelising over tokens rather than over chunks
    is what keeps these serial-looking reductions cheap.

    Both matrices are written as full ``[BS]``-wide rows, zero outside the
    diagonal sub-block; :func:`inter_blocks_kernel` fills the off-diagonal
    columns afterwards, so the two kernels must run in this order.

    Args:
        B, S, H, DK: problem dims.
        scale: query scale, folded into ``Aqk`` here.
        input_dtype, accum_dtype, gate_dtype: tensor / accumulator dtypes.
        chunk_size: chunk length ``BS``.
        sub_chunk_size: sub-chunk length ``BC``.
        block_H: heads handled per block.
        threads, num_stages: launch / pipelining config.

    Returns:
        A JIT kernel ``(q, k, b, G) -> (Aqk, Akk)``, both ``[B, S, H, BS]``.
        ``Aqk`` includes its diagonal (a token reads the state after its own
        write); ``Akk`` is strictly lower.
    """
    BS, BC = chunk_size, sub_chunk_size
    qk_shape = (B, S, H, DK)
    a_shape = (B, S, H, BS)

    @T.prim_func
    def kernel(
        q: T.Tensor(qk_shape, dtype=input_dtype),
        k: T.Tensor(qk_shape, dtype=input_dtype),
        b: T.Tensor(qk_shape, dtype=input_dtype),
        G: T.Tensor(qk_shape, dtype=gate_dtype),
        Aqk: T.Tensor(a_shape, dtype=input_dtype),
        Akk: T.Tensor(a_shape, dtype=accum_dtype),
    ):
        with T.Kernel(B * S, T.ceildiv(H, block_H), threads=threads) as (i_bs, i_h):
            i_b, t = i_bs // S, i_bs % S
            # First token of this token's sub-chunk, and how many terms to sum.
            t_sub = (t // BS) * BS + ((t % BS) // BC) * BC
            n_terms = t + 1 - t_sub

            q_i = T.alloc_shared((block_H, DK), dtype=input_dtype)
            k_i = T.alloc_shared((block_H, DK), dtype=input_dtype)
            b_i = T.alloc_shared((block_H, DK), dtype=input_dtype)
            G_i = T.alloc_shared((block_H, DK), dtype=gate_dtype)
            k_j = T.alloc_shared((block_H, DK), dtype=input_dtype)
            G_j = T.alloc_shared((block_H, DK), dtype=gate_dtype)

            qi_f = T.alloc_fragment((block_H, DK), dtype=accum_dtype)
            ei_f = T.alloc_fragment((block_H, DK), dtype=accum_dtype)
            kj_f = T.alloc_fragment((block_H, DK), dtype=accum_dtype)
            aqk_prod = T.alloc_shared((block_H, DK), dtype=accum_dtype)
            akk_prod = T.alloc_shared((block_H, DK), dtype=accum_dtype)
            aqk_sum = T.alloc_fragment((block_H,), dtype=accum_dtype)
            akk_sum = T.alloc_fragment((block_H,), dtype=accum_dtype)
            aqk_row = T.alloc_shared((block_H, BS), dtype=input_dtype)
            akk_row = T.alloc_shared((block_H, BS), dtype=accum_dtype)

            T.copy(q[i_b, t, i_h * block_H : (i_h + 1) * block_H, :], q_i)
            T.copy(k[i_b, t, i_h * block_H : (i_h + 1) * block_H, :], k_i)
            T.copy(b[i_b, t, i_h * block_H : (i_h + 1) * block_H, :], b_i)
            T.copy(G[i_b, t, i_h * block_H : (i_h + 1) * block_H, :], G_i)

            T.disable_warp_group_reg_alloc()
            for h, c in T.Parallel(block_H, DK):
                qi_f[h, c] = q_i[h, c] * scale
                ei_f[h, c] = k_i[h, c] * b_i[h, c]  # erase row: b * k

            T.clear(aqk_row)
            T.clear(akk_row)

            for d in T.Pipelined(n_terms, num_stages=num_stages):
                j = t_sub + d
                T.copy(k[i_b, j, i_h * block_H : (i_h + 1) * block_H, :], k_j)
                T.copy(G[i_b, j, i_h * block_H : (i_h + 1) * block_H, :], G_j)
                for h, c in T.Parallel(block_H, DK):
                    # G_i <= G_j for j <= t, so the exponent is <= 0.
                    kj_f[h, c] = k_j[h, c] * T.exp2(G_i[h, c] - G_j[h, c])
                    aqk_prod[h, c] = qi_f[h, c] * kj_f[h, c]
                    akk_prod[h, c] = ei_f[h, c] * kj_f[h, c]

                T.reduce_sum(aqk_prod, aqk_sum, dim=-1, clear=True)
                T.reduce_sum(akk_prod, akk_sum, dim=-1, clear=True)

                T.copy(aqk_sum, aqk_row[:, j % BS])
                if j < t:  # Akk is strictly lower triangular
                    T.copy(akk_sum, akk_row[:, j % BS])

            T.copy(aqk_row, Aqk[i_b, t, i_h * block_H : (i_h + 1) * block_H, :])
            T.copy(akk_row, Akk[i_b, t, i_h * block_H : (i_h + 1) * block_H, :])

    return kernel


@tilelang.jit(pass_configs=_FAST_MATH)
def inter_blocks_kernel(
    B: int,
    S: int,
    H: int,
    DK: int,
    scale: float,
    input_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    gate_dtype: str = "float32",
    chunk_size: int = 64,
    sub_chunk_size: int = 16,
    block_DK: int = 32,
    threads: int = 128,
    num_stages: int = 2,
):
    """Off-diagonal sub-blocks of ``Aqk`` and ``Akk``, on tensor cores.

    One block per (row-sub-block ``i``, chunk, head), looping over the column
    sub-blocks ``j < i``. Every entry of such a pair shares one reference row --
    the cumulative decay at row-sub-block ``i``'s first token -- which splits
    the ``exp2(G_r - G_s)`` factor into two GEMM operands::

        row[r, c] = f_r[c] * exp2(G_r[c] - ref[c])    r is at/after the ref row
        col[s, c] = k_s[c] * exp2(ref[c] - G_s[c])    s is strictly before it

    Both exponents are non-positive, so neither operand can overflow; where they
    underflow to zero the true product is vanishing anyway. Sub-block 0 has no
    columns to its left, so the grid starts at ``i = 1``.

    Writes ``Aqk`` and ``Akk`` in place, touching only columns
    ``[0, i * BC)`` -- the diagonal sub-blocks stay as
    :func:`intra_diag_kernel` left them.

    Args:
        B, S, H, DK: problem dims.
        scale: query scale, folded into ``Aqk``.
        input_dtype, accum_dtype, gate_dtype: tensor / accumulator dtypes.
        chunk_size, sub_chunk_size: ``BS`` and ``BC``; ``BS`` must be a
            multiple of ``BC``, with at least two sub-chunks per chunk.
        block_DK: channel tile for the reduction over ``DK``.
        threads, num_stages: launch / pipelining config.

    Returns:
        A JIT kernel ``(q, k, b, G, Aqk, Akk) -> None``; the last two are
        updated in place.
    """
    BS, BC = chunk_size, sub_chunk_size
    if BS % BC != 0:
        raise ValueError(f"chunk_size {BS} must be divisible by sub_chunk_size {BC}")
    NSUB = BS // BC
    if NSUB < 2:
        raise ValueError(f"need at least 2 sub-chunks per chunk, got {NSUB}")

    qk_shape = (B, S, H, DK)
    a_shape = (B, S, H, BS)

    @T.prim_func
    def kernel(
        q: T.Tensor(qk_shape, dtype=input_dtype),
        k: T.Tensor(qk_shape, dtype=input_dtype),
        b: T.Tensor(qk_shape, dtype=input_dtype),
        G: T.Tensor(qk_shape, dtype=gate_dtype),
        Aqk: T.Tensor(a_shape, dtype=input_dtype),
        Akk: T.Tensor(a_shape, dtype=accum_dtype),
    ):
        with T.Kernel(NSUB - 1, T.ceildiv(S, BS), B * H, threads=threads) as (i_sub0, i_s, i_bh):
            i_b, i_h = i_bh // H, i_bh % H
            i_sub = i_sub0 + 1  # sub-block 0 has nothing to its left
            t0 = i_s * BS
            r0 = t0 + i_sub * BC  # first token of the row sub-block

            q_r = T.alloc_shared((BC, block_DK), dtype=input_dtype)
            k_r = T.alloc_shared((BC, block_DK), dtype=input_dtype)
            b_r = T.alloc_shared((BC, block_DK), dtype=input_dtype)
            G_r = T.alloc_shared((BC, block_DK), dtype=gate_dtype)
            k_c = T.alloc_shared((BC, block_DK), dtype=input_dtype)
            G_c = T.alloc_shared((BC, block_DK), dtype=gate_dtype)
            # 1-D shared buffers use static "shared": tilelang can emit
            # misaligned accesses for very small "shared.dyn" allocations.
            ref = T.alloc_shared((block_DK,), dtype=gate_dtype, scope="shared")
            row_q = T.alloc_shared((BC, block_DK), dtype=accum_dtype)
            row_e = T.alloc_shared((BC, block_DK), dtype=accum_dtype)
            col_k = T.alloc_shared((BC, block_DK), dtype=accum_dtype)
            aqk_f = T.alloc_fragment((BC, BC), dtype=accum_dtype)
            akk_f = T.alloc_fragment((BC, BC), dtype=accum_dtype)
            aqk_o = T.alloc_fragment((BC, BC), dtype=accum_dtype)

            for j in T.serial(i_sub):
                c0 = t0 + j * BC  # first token of the column sub-block
                T.clear(aqk_f)
                T.clear(akk_f)

                for i_k in T.Pipelined(T.ceildiv(DK, block_DK), num_stages=num_stages):
                    lo, hi = i_k * block_DK, (i_k + 1) * block_DK
                    T.copy(G[i_b, r0, i_h, lo:hi], ref)
                    T.copy(q[i_b, r0 : r0 + BC, i_h, lo:hi], q_r)
                    T.copy(k[i_b, r0 : r0 + BC, i_h, lo:hi], k_r)
                    T.copy(b[i_b, r0 : r0 + BC, i_h, lo:hi], b_r)
                    T.copy(G[i_b, r0 : r0 + BC, i_h, lo:hi], G_r)
                    T.copy(k[i_b, c0 : c0 + BC, i_h, lo:hi], k_c)
                    T.copy(G[i_b, c0 : c0 + BC, i_h, lo:hi], G_c)

                    for r, c in T.Parallel(BC, block_DK):
                        decay = T.exp2(G_r[r, c] - ref[c])
                        row_q[r, c] = q_r[r, c] * decay
                        row_e[r, c] = k_r[r, c] * b_r[r, c] * decay
                    for u, c in T.Parallel(BC, block_DK):
                        col_k[u, c] = k_c[u, c] * T.exp2(ref[c] - G_c[u, c])

                    T.gemm(row_q, col_k, aqk_f, transpose_B=True)
                    T.gemm(row_e, col_k, akk_f, transpose_B=True)

                for r, u in T.Parallel(BC, BC):
                    aqk_o[r, u] = aqk_f[r, u] * scale
                T.copy(aqk_o, Aqk[i_b, r0 : r0 + BC, i_h, j * BC : (j + 1) * BC])
                T.copy(akk_f, Akk[i_b, r0 : r0 + BC, i_h, j * BC : (j + 1) * BC])

    return kernel


@tilelang.jit(out_idx=[-1], pass_configs=_FAST_MATH)
def solve_kernel(
    B: int,
    S: int,
    H: int,
    input_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    chunk_size: int = 64,
    threads: int = 128,
    num_stages: int = 1,
):
    """Invert the unit lower triangular ``I + Akk`` by forward substitution.

    Row ``r`` of the inverse depends only on rows before it::

        Ai[r, :] = e_r - sum_{s < r} Akk[r, s] Ai[s, :]

    which is one rank-1 update per row against the rows already solved. Rows 0
    and 1 need no substitution -- for a strictly lower ``L``, the first two rows
    of ``(I + L)^{-1} - I`` are just ``-L`` -- so the loop starts at 2.

    This is the serial stage of the pipeline. A blocked variant (invert
    ``BC x BC`` diagonal blocks, then merge with small GEMMs) trades these
    ``BS - 2`` steps for ``BC``-wide ones plus tensor-core work, and is the
    first thing to try if this kernel shows up in a profile.

    Args:
        B, S, H: problem dims.
        input_dtype: dtype of the emitted ``Ai`` (it feeds GEMMs downstream).
        accum_dtype: dtype of ``Akk`` and of the substitution arithmetic.
        chunk_size: chunk length ``BS``.
        threads, num_stages: launch / pipelining config.

    Returns:
        A JIT kernel ``(Akk) -> Ai``, both ``[B, S, H, BS]``.
    """
    BS = chunk_size
    a_shape = (B, S, H, BS)

    @T.prim_func
    def kernel(
        Akk: T.Tensor(a_shape, dtype=accum_dtype),
        Ai: T.Tensor(a_shape, dtype=input_dtype),
    ):
        with T.Kernel(T.ceildiv(S, BS), B * H, threads=threads) as (i_s, i_bh):
            i_b, i_h = i_bh // H, i_bh % H
            t0 = i_s * BS

            ai_s = T.alloc_shared((BS, BS), dtype=accum_dtype)
            mul_s = T.alloc_shared((BS, BS), dtype=accum_dtype)
            row_s = T.alloc_shared((BS,), dtype=accum_dtype, scope="shared")
            red_f = T.alloc_fragment((BS,), dtype=accum_dtype)
            out_s = T.alloc_shared((BS, BS), dtype=input_dtype)

            # Track (I + Akk)^{-1} - I; the identity goes back on at the end so
            # that every row read during substitution is the true inverse row.
            T.copy(Akk[i_b, t0 : t0 + BS, i_h, :], ai_s)
            for r, c in T.Parallel(BS, BS):
                ai_s[r, c] = T.if_then_else(r > c, -ai_s[r, c], 0.0)

            for r in T.Pipelined(2, BS, num_stages=num_stages):
                T.copy(Akk[i_b, t0 + r, i_h, :], row_s)
                for c in T.Parallel(BS):
                    row_s[c] = T.if_then_else(c < r, -row_s[c], 0.0)
                for c0, c1 in T.Parallel(BS, BS):
                    mul_s[c0, c1] = row_s[c0] * ai_s[c0, c1]
                T.reduce_sum(mul_s, red_f, dim=0, clear=True)
                for c in T.Parallel(BS):
                    row_s[c] = row_s[c] + red_f[c]
                for c0, c1 in T.Parallel(BS, BS):
                    ai_s[c0, c1] = T.if_then_else(c0 == r, row_s[c1], ai_s[c0, c1])

            for r, c in T.Parallel(BS, BS):
                out_s[r, c] = ai_s[r, c] + T.if_then_else(r == c, 1.0, 0.0)
            T.copy(out_s, Ai[i_b, t0 : t0 + BS, i_h, :])

    return kernel


@tilelang.jit(out_idx=[-3, -2, -1], pass_configs=_FAST_MATH)
def wy_kernel(
    B: int,
    S: int,
    H: int,
    DK: int,
    DV: int,
    input_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    gate_dtype: str = "float32",
    chunk_size: int = 64,
    block_DK: int = 64,
    block_DV: int = 64,
    threads: int = 128,
    num_stages: int = 2,
):
    """Apply the inverse to build the WY factors.

    ``W = Ai (b * k * gamma)``, ``U = Ai (w * v)``, and ``KG = k * gamma_C /
    gamma_r`` (the key rows already carrying the residual decay to the end of
    the chunk, ready for the state update).

    Args:
        B, S, H, DK, DV: problem dims.
        input_dtype, accum_dtype, gate_dtype: tensor / accumulator dtypes.
        chunk_size: chunk length ``BS``.
        block_DK, block_DV: channel tiles.
        threads, num_stages: launch / pipelining config.

    Returns:
        A JIT kernel ``(k, v, b, w, G, Ai) -> (W, U, KG)`` with ``W, KG`` of
        shape ``[B, S, H, DK]`` and ``U`` of shape ``[B, S, H, DV]``.
    """
    BS = chunk_size
    k_shape = (B, S, H, DK)
    v_shape = (B, S, H, DV)
    a_shape = (B, S, H, BS)

    @T.prim_func
    def kernel(
        k: T.Tensor(k_shape, dtype=input_dtype),
        v: T.Tensor(v_shape, dtype=input_dtype),
        b: T.Tensor(k_shape, dtype=input_dtype),
        w: T.Tensor(v_shape, dtype=input_dtype),
        G: T.Tensor(k_shape, dtype=gate_dtype),
        Ai: T.Tensor(a_shape, dtype=input_dtype),
        W: T.Tensor(k_shape, dtype=input_dtype),
        U: T.Tensor(v_shape, dtype=input_dtype),
        KG: T.Tensor(k_shape, dtype=input_dtype),
    ):
        with T.Kernel(T.ceildiv(S, BS), B * H, threads=threads) as (i_s, i_bh):
            i_b, i_h = i_bh // H, i_bh % H
            t0 = i_s * BS

            ai_s = T.alloc_shared((BS, BS), dtype=input_dtype)
            k_s = T.alloc_shared((BS, block_DK), dtype=input_dtype)
            b_s = T.alloc_shared((BS, block_DK), dtype=input_dtype)
            v_s = T.alloc_shared((BS, block_DV), dtype=input_dtype)
            w_s = T.alloc_shared((BS, block_DV), dtype=input_dtype)
            G_s = T.alloc_shared((BS, block_DK), dtype=gate_dtype)
            G_last = T.alloc_shared((block_DK,), dtype=gate_dtype, scope="shared")
            e_s = T.alloc_shared((BS, block_DK), dtype=input_dtype)
            z_s = T.alloc_shared((BS, block_DV), dtype=input_dtype)
            kg_s = T.alloc_shared((BS, block_DK), dtype=input_dtype)
            out_k = T.alloc_fragment((BS, block_DK), dtype=accum_dtype)
            out_v = T.alloc_fragment((BS, block_DV), dtype=accum_dtype)

            T.copy(Ai[i_b, t0 : t0 + BS, i_h, :], ai_s)

            for i_v in T.Pipelined(T.ceildiv(DV, block_DV), num_stages=num_stages):
                lo, hi = i_v * block_DV, (i_v + 1) * block_DV
                T.copy(v[i_b, t0 : t0 + BS, i_h, lo:hi], v_s)
                T.copy(w[i_b, t0 : t0 + BS, i_h, lo:hi], w_s)
                for t, c in T.Parallel(BS, block_DV):
                    z_s[t, c] = v_s[t, c] * w_s[t, c]
                T.gemm(ai_s, z_s, out_v, clear_accum=True)
                T.copy(out_v, U[i_b, t0 : t0 + BS, i_h, lo:hi])

            for i_k in T.Pipelined(T.ceildiv(DK, block_DK), num_stages=num_stages):
                lo, hi = i_k * block_DK, (i_k + 1) * block_DK
                T.copy(k[i_b, t0 : t0 + BS, i_h, lo:hi], k_s)
                T.copy(b[i_b, t0 : t0 + BS, i_h, lo:hi], b_s)
                T.copy(G[i_b, t0 : t0 + BS, i_h, lo:hi], G_s)
                T.copy(G[i_b, t0 + BS - 1, i_h, lo:hi], G_last)

                for t, c in T.Parallel(BS, block_DK):
                    e_s[t, c] = k_s[t, c] * b_s[t, c] * T.exp2(G_s[t, c])
                    kg_s[t, c] = k_s[t, c] * T.exp2(G_last[c] - G_s[t, c])
                T.gemm(ai_s, e_s, out_k, clear_accum=True)
                T.copy(out_k, W[i_b, t0 : t0 + BS, i_h, lo:hi])
                T.copy(kg_s, KG[i_b, t0 : t0 + BS, i_h, lo:hi])

    return kernel


@tilelang.jit(out_idx=[-3, -2, -1], pass_configs=_FAST_MATH)
def state_kernel(
    B: int,
    S: int,
    H: int,
    DK: int,
    DV: int,
    input_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    gate_dtype: str = "float32",
    state_dtype: str = "float32",
    chunk_size: int = 64,
    use_initial_state: bool = True,
    block_DV: int = 64,
    threads: int = 128,
    num_stages: int = 2,
):
    """Sequential scan over chunks: the only serial stage of the pipeline.

    Per chunk ``n``, with ``S_n`` the state entering the chunk::

        Un      = U_n - W_n S_n
        S_{n+1} = diag(gamma_C) S_n + KG_n^T Un

    ``h[:, n]`` records ``S_n`` (the state *before* the chunk's own writes) so
    that :func:`output_kernel` can run fully in parallel afterwards. Blocks are
    parallel over ``(batch, head, DV tile)``.

    Args:
        B, S, H, DK, DV: problem dims.
        input_dtype, accum_dtype, gate_dtype, state_dtype: dtypes. ``h`` is
            emitted in ``input_dtype`` since it feeds GEMMs.
        chunk_size: chunk length ``BS``.
        use_initial_state: read ``initial_state`` instead of starting at zero.
        block_DV: value tile per block.
        threads, num_stages: launch / pipelining config.

    Returns:
        A JIT kernel ``(KG, W, U, G, initial_state) -> (h, Un, final_state)``
        with ``h`` of shape ``[B, NC, H, DK, DV]``, ``Un`` of shape
        ``[B, S, H, DV]`` and ``final_state`` of shape ``[B, H, DK, DV]``.
    """
    BS = chunk_size
    NC = (S + BS - 1) // BS
    k_shape = (B, S, H, DK)
    v_shape = (B, S, H, DV)
    h_shape = (B, NC, H, DK, DV)
    st_shape = (B, H, DK, DV)

    @T.prim_func
    def kernel(
        KG: T.Tensor(k_shape, dtype=input_dtype),
        W: T.Tensor(k_shape, dtype=input_dtype),
        U: T.Tensor(v_shape, dtype=input_dtype),
        G: T.Tensor(k_shape, dtype=gate_dtype),
        initial_state: T.Tensor(st_shape, dtype=state_dtype),
        h: T.Tensor(h_shape, dtype=input_dtype),
        Un: T.Tensor(v_shape, dtype=input_dtype),
        final_state: T.Tensor(st_shape, dtype=state_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV), B * H, threads=threads) as (i_v, i_bh):
            i_b, i_h = i_bh // H, i_bh % H
            lo, hi = i_v * block_DV, (i_v + 1) * block_DV

            st_s = T.alloc_shared((DK, block_DV), dtype=input_dtype)
            st_f = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            kg_s = T.alloc_shared((BS, DK), dtype=input_dtype)
            w_s = T.alloc_shared((BS, DK), dtype=input_dtype)
            u_s = T.alloc_shared((BS, block_DV), dtype=input_dtype)
            un_f = T.alloc_fragment((BS, block_DV), dtype=accum_dtype)
            un_s = T.alloc_shared((BS, block_DV), dtype=input_dtype)
            g_last = T.alloc_shared((DK,), dtype=gate_dtype, scope="shared")

            if use_initial_state:
                T.copy(initial_state[i_b, i_h, :, lo:hi], st_f)
            else:
                T.clear(st_f)
            T.copy(st_f, st_s)

            for i_s in T.Pipelined(T.ceildiv(S, BS), num_stages=num_stages):
                t0 = i_s * BS
                # Snapshot the incoming state before this chunk writes to it.
                T.copy(st_s, h[i_b, i_s, i_h, :, lo:hi])

                # Un = U - W @ S
                T.copy(W[i_b, t0 : t0 + BS, i_h, :], w_s)
                T.gemm(w_s, st_s, un_f, clear_accum=True)
                T.copy(U[i_b, t0 : t0 + BS, i_h, lo:hi], u_s)
                for t, c in T.Parallel(BS, block_DV):
                    un_f[t, c] = u_s[t, c] - un_f[t, c]
                T.copy(un_f, un_s)
                T.copy(un_s, Un[i_b, t0 : t0 + BS, i_h, lo:hi])

                # S <- diag(gamma_C) S + KG^T Un
                T.copy(G[i_b, t0 + BS - 1, i_h, :], g_last)
                for c, d in T.Parallel(DK, block_DV):
                    st_f[c, d] = st_f[c, d] * T.exp2(g_last[c])
                T.copy(KG[i_b, t0 : t0 + BS, i_h, :], kg_s)
                T.gemm(kg_s, un_s, st_f, transpose_A=True)
                T.copy(st_f, st_s)

            T.copy(st_f, final_state[i_b, i_h, :, lo:hi])

    return kernel


@tilelang.jit(out_idx=[-1], pass_configs=_FAST_MATH)
def output_kernel(
    B: int,
    S: int,
    H: int,
    DK: int,
    DV: int,
    scale: float,
    input_dtype: str = "bfloat16",
    output_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    gate_dtype: str = "float32",
    chunk_size: int = 64,
    block_DK: int = 64,
    block_DV: int = 64,
    threads: int = 128,
    num_stages: int = 2,
):
    """Combine the cross-chunk and intra-chunk contributions.

    ``o = (scale * q * gamma) h_n + tril(Aqk) Un``. Fully parallel over
    ``(chunk, head, DV tile)``.

    Args:
        B, S, H, DK, DV: problem dims.
        scale: query scale (``Aqk`` already carries it; this applies it to the
            cross-chunk term).
        input_dtype, output_dtype, accum_dtype, gate_dtype: dtypes.
        chunk_size: chunk length ``BS``.
        block_DK, block_DV: channel tiles.
        threads, num_stages: launch / pipelining config.

    Returns:
        A JIT kernel ``(q, G, Aqk, h, Un) -> o`` with ``o`` of shape
        ``[B, S, H, DV]``.
    """
    BS = chunk_size
    NC = (S + BS - 1) // BS
    k_shape = (B, S, H, DK)
    v_shape = (B, S, H, DV)
    a_shape = (B, S, H, BS)
    h_shape = (B, NC, H, DK, DV)

    @T.prim_func
    def kernel(
        q: T.Tensor(k_shape, dtype=input_dtype),
        G: T.Tensor(k_shape, dtype=gate_dtype),
        Aqk: T.Tensor(a_shape, dtype=input_dtype),
        h: T.Tensor(h_shape, dtype=input_dtype),
        Un: T.Tensor(v_shape, dtype=input_dtype),
        o: T.Tensor(v_shape, dtype=output_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV), T.ceildiv(S, BS), B * H, threads=threads) as (i_v, i_s, i_bh):
            i_b, i_h = i_bh // H, i_bh % H
            t0 = i_s * BS
            vlo, vhi = i_v * block_DV, (i_v + 1) * block_DV

            q_s = T.alloc_shared((BS, block_DK), dtype=input_dtype)
            G_s = T.alloc_shared((BS, block_DK), dtype=gate_dtype)
            qg_s = T.alloc_shared((BS, block_DK), dtype=input_dtype)
            h_s = T.alloc_shared((block_DK, block_DV), dtype=input_dtype)
            a_s = T.alloc_shared((BS, BS), dtype=input_dtype)
            un_s = T.alloc_shared((BS, block_DV), dtype=input_dtype)
            o_f = T.alloc_fragment((BS, block_DV), dtype=accum_dtype)

            T.clear(o_f)
            for i_k in T.Pipelined(T.ceildiv(DK, block_DK), num_stages=num_stages):
                klo, khi = i_k * block_DK, (i_k + 1) * block_DK
                T.copy(q[i_b, t0 : t0 + BS, i_h, klo:khi], q_s)
                T.copy(G[i_b, t0 : t0 + BS, i_h, klo:khi], G_s)
                T.copy(h[i_b, i_s, i_h, klo:khi, vlo:vhi], h_s)
                for t, c in T.Parallel(BS, block_DK):
                    qg_s[t, c] = q_s[t, c] * scale * T.exp2(G_s[t, c])
                T.gemm(qg_s, h_s, o_f)

            T.copy(Aqk[i_b, t0 : t0 + BS, i_h, :], a_s)
            T.copy(Un[i_b, t0 : t0 + BS, i_h, vlo:vhi], un_s)
            for r, c in T.Parallel(BS, BS):
                a_s[r, c] = T.if_then_else(r < c, 0, a_s[r, c])
            T.gemm(a_s, un_s, o_f)

            T.copy(o_f, o[i_b, t0 : t0 + BS, i_h, vlo:vhi])

    return kernel


@tilelang.jit(out_idx=[-1], pass_configs=_FAST_MATH)
def decode_step_kernel(
    B: int,
    H: int,
    DK: int,
    DV: int,
    scale: float,
    input_dtype: str = "bfloat16",
    output_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    gate_dtype: str = "float32",
    state_dtype: str = "float32",
    block_DV: int = 64,
    threads: int = 128,
):
    """Single-token GDN-2 update for autoregressive decoding.

    Applies the recurrence directly -- no chunking, no WY form -- and updates
    ``state`` in place::

        S <- diag(exp(g)) S
        u  = w * v - S^T (b * k)
        S <- S + k u^T
        o  = scale * S^T q

    Args:
        B, H, DK, DV: problem dims (one token).
        scale: query scale.
        input_dtype, output_dtype, accum_dtype, gate_dtype, state_dtype: dtypes.
        block_DV: value tile per block.
        threads: threads per block.

    Returns:
        A JIT kernel ``(q, k, v, g, b, w, state) -> o``. Inputs are
        ``[B, H, D]``-shaped, ``state`` is ``[B, H, DK, DV]`` and is mutated.
    """
    k_shape = (B, H, DK)
    v_shape = (B, H, DV)
    st_shape = (B, H, DK, DV)

    @T.prim_func
    def kernel(
        q: T.Tensor(k_shape, dtype=input_dtype),
        k: T.Tensor(k_shape, dtype=input_dtype),
        v: T.Tensor(v_shape, dtype=input_dtype),
        g: T.Tensor(k_shape, dtype=gate_dtype),
        b: T.Tensor(k_shape, dtype=input_dtype),
        w: T.Tensor(v_shape, dtype=input_dtype),
        state: T.Tensor(st_shape, dtype=state_dtype),
        o: T.Tensor(v_shape, dtype=output_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV), B * H, threads=threads) as (i_v, i_bh):
            i_b, i_h = i_bh // H, i_bh % H
            lo, hi = i_v * block_DV, (i_v + 1) * block_DV

            st_f = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            q_s = T.alloc_shared((DK,), dtype=input_dtype, scope="shared")
            k_s = T.alloc_shared((DK,), dtype=input_dtype, scope="shared")
            b_s = T.alloc_shared((DK,), dtype=input_dtype, scope="shared")
            g_s = T.alloc_shared((DK,), dtype=gate_dtype, scope="shared")
            v_s = T.alloc_shared((block_DV,), dtype=input_dtype, scope="shared")
            w_s = T.alloc_shared((block_DV,), dtype=input_dtype, scope="shared")
            prod = T.alloc_shared((DK, block_DV), dtype=accum_dtype)
            red = T.alloc_fragment((block_DV,), dtype=accum_dtype)
            u_s = T.alloc_shared((block_DV,), dtype=accum_dtype, scope="shared")

            T.copy(q[i_b, i_h, :], q_s)
            T.copy(k[i_b, i_h, :], k_s)
            T.copy(b[i_b, i_h, :], b_s)
            T.copy(g[i_b, i_h, :], g_s)
            T.copy(v[i_b, i_h, lo:hi], v_s)
            T.copy(w[i_b, i_h, lo:hi], w_s)
            T.copy(state[i_b, i_h, :, lo:hi], st_f)

            # decay, then read out along the key axis
            for c, d in T.Parallel(DK, block_DV):
                st_f[c, d] = st_f[c, d] * T.exp(g_s[c])
                prod[c, d] = st_f[c, d] * b_s[c] * k_s[c]
            T.reduce_sum(prod, red, dim=0, clear=True)

            for d in T.Parallel(block_DV):
                u_s[d] = w_s[d] * v_s[d] - red[d]

            # rank-1 write, then read the output
            for c, d in T.Parallel(DK, block_DV):
                st_f[c, d] = st_f[c, d] + k_s[c] * u_s[d]
                prod[c, d] = st_f[c, d] * q_s[c] * scale
            T.reduce_sum(prod, red, dim=0, clear=True)

            T.copy(st_f, state[i_b, i_h, :, lo:hi])
            T.copy(red, o[i_b, i_h, lo:hi])

    return kernel


_TORCH_TO_TL = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


def _tl_dtype(dtype: torch.dtype) -> str:
    """Map a torch dtype to the TileLang dtype string."""
    try:
        return _TORCH_TO_TL[dtype]
    except KeyError:
        raise TypeError(f"unsupported dtype {dtype}; use float16, bfloat16 or float32") from None


def _pad_to_chunk(x: torch.Tensor, total: int) -> torch.Tensor:
    """Right-pad the sequence axis of ``[B, S, H, D]`` with zeros to ``total``."""
    if x.shape[1] == total:
        return x.contiguous()
    pad = x.new_zeros(x.shape[0], total - x.shape[1], *x.shape[2:])
    return torch.cat([x, pad], dim=1).contiguous()


def chunk_gdn2_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    config: GDN2Config = DEFAULT_KERNEL_CONFIG,
    return_intermediates: bool = False,
):
    """Run the seven-kernel TileLang forward pipeline.

    Sequence lengths that are not a multiple of ``chunk_size`` are zero-padded.
    Padding is inert: ``g = 0`` leaves the decay at 1 while ``b = w = 0`` makes
    the update a no-op, so the carried state is unaffected.

    Args:
        q, k: ``[B, S, H, DK]``, contiguous, fp16/bf16/fp32.
        v: ``[B, S, H, DV]``.
        g: ``[B, S, H, DK]`` fp32 log decay, ``<= 0``.
        b: ``[B, S, H, DK]`` erase gate.
        w: ``[B, S, H, DV]`` write gate.
        scale: query scale.
        initial_state: ``[B, H, DK, DV]`` or ``None``.
        output_final_state: also return the state after the last real token.
        config: tiling configuration.
        return_intermediates: also return the tensors the backward re-uses,
            avoiding a full recompute in ``chunk_gdn2_bwd``.

    Returns:
        ``(o, final_state)``, or ``(o, final_state, intermediates)``.
    """
    
    B, S, H, DK = q.shape
    DV = v.shape[-1]
    BS, BC = config.chunk_size, config.sub_chunk_size
    S_pad = ((S + BS - 1) // BS) * BS

    in_dt = _tl_dtype(q.dtype)
    st_dt = _tl_dtype(config.state_dtype)

    q, k, b_ = (_pad_to_chunk(x, S_pad) for x in (q, k, b))
    v, w_ = (_pad_to_chunk(x, S_pad) for x in (v, w))
    g = _pad_to_chunk(g.float(), S_pad)

    G = cumsum_kernel(B, S_pad, H, DK, chunk_size=BS, threads=config.threads)(g)

    # Kernels 2 and 3 fill disjoint column ranges of the same two matrices:
    # kernel 2 the diagonal sub-blocks, kernel 3 everything left of them.
    Aqk, Akk = intra_diag_kernel(
        B, S_pad, H, DK, scale,
        input_dtype=in_dt, chunk_size=BS, sub_chunk_size=BC,
    )(q, k, b_, G)

    inter_blocks_kernel(
        B, S_pad, H, DK, scale,
        input_dtype=in_dt, chunk_size=BS, sub_chunk_size=BC,
        block_DK=min(config.block_DK, DK), threads=config.threads,
        num_stages=config.num_stages,
    )(q, k, b_, G, Aqk, Akk)

    Ai = solve_kernel(
        B, S_pad, H, input_dtype=in_dt, chunk_size=BS, threads=config.threads
    )(Akk)

    W, U, KG = wy_kernel(
        B, S_pad, H, DK, DV,
        input_dtype=in_dt, chunk_size=BS,
        block_DK=min(config.block_DK, DK), block_DV=min(config.block_DV, DV),
        threads=config.threads, num_stages=config.num_stages,
    )(k, v, b_, w_, G, Ai)

    if initial_state is None:
        st_in = torch.empty(B, H, DK, DV, dtype=config.state_dtype, device=q.device)
    else:
        st_in = initial_state.to(config.state_dtype).contiguous()

    h, Un, final_state = state_kernel(
        B, S_pad, H, DK, DV,
        input_dtype=in_dt, state_dtype=st_dt, chunk_size=BS,
        use_initial_state=initial_state is not None,
        block_DV=min(config.block_DV, DV), threads=config.threads,
        num_stages=config.num_stages,
    )(KG, W, U, G, st_in)

    o = output_kernel(
        B, S_pad, H, DK, DV, scale,
        input_dtype=in_dt, output_dtype=in_dt, chunk_size=BS,
        block_DK=min(config.block_DK, DK), block_DV=min(config.block_DV, DV),
        threads=config.threads, num_stages=config.num_stages,
    )(q, G, Aqk, h, Un)

    # The padded tail leaves the state untouched, so final_state needs no fixup.
    out = (o[:, :S], final_state if output_final_state else None)
    if return_intermediates:
        return (*out, dict(G=G, Aqk=Aqk, Ai=Ai, W=W, U=U, KG=KG, h=h, Un=Un,
                           q=q, k=k, v=v, b=b_, w=w_, S_pad=S_pad))
    return out


def gdn2_decode_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    state: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """One autoregressive step, updating ``state`` in place.

    Args:
        q, k, b: ``[B, H, DK]``.
        v, w: ``[B, H, DV]``.
        g: ``[B, H, DK]`` fp32 log decay.
        state: ``[B, H, DK, DV]``, mutated.
        scale: query scale; defaults to ``DK ** -0.5``.

    Returns:
        ``[B, H, DV]`` output for this token.
    """
    
    B, H, DK = q.shape
    DV = v.shape[-1]
    scale = DK**-0.5 if scale is None else scale

    if q.device.type != "cuda":
        S = state
        S.mul_(g.float().exp().unsqueeze(-1))
        read = torch.einsum("bhkd,bhk->bhd", S, (b * k).float())
        u = (w * v).float() - read
        S.add_(k.float().unsqueeze(-1) * u.unsqueeze(-2))
        return torch.einsum("bhkd,bhk->bhd", S, q.float() * scale).to(v.dtype)

    kernel = decode_step_kernel(
        B, H, DK, DV, scale,
        input_dtype=_tl_dtype(q.dtype), output_dtype=_tl_dtype(v.dtype),
        state_dtype=_tl_dtype(state.dtype), block_DV=min(64, DV),
    )
    return kernel(q, k, v, g.float(), b, w, state)
