"""Backward pass: TileLang kernels and the driver that runs them.

Each kernel implements one stage of :mod:`gdn2.reference`, which is derived and
checked against autograd there; read that module first, then this.

    1. ``dstate_kernel``      reverse scan  -> dUn, dh, dS0
    2. ``dwy_a_kernel``       dW, dKtail, dQg, dGC_state
    3. ``daqk_kernel``        dAqk = tril(dO Un^T)
    4. ``dwy_b_kernel``       dZ, dE, dTm
    5. ``dintra_diag_kernel`` diagonal sub-blocks of the four pairwise grads
    6. ``dinter_row_kernel``  off-diagonal, row side  -> dEt_tm, dq_aqk
    7. ``dinter_col_kernel``  off-diagonal, col side  -> dk_tm, dk_aqk
    8. ``dassemble_kernel``   dq, dk, dv, db, dw, dg

Kernel 5 writes the four pairwise gradients; 6 and 7 accumulate into them.
They touch disjoint arrays (6 owns the row grads, 7 the column grads) and each
block owns a disjoint sub-block of rows, so the read-modify-writes never race.
Ordering: 5 before 6 and 7; 6 and 7 are independent of each other.

As in the forward, ``G`` is log2-space cumulative decay and every exponential
is ``exp2`` of a non-positive number.
"""

# NOTE: no `from __future__ import annotations` -- see gdn2/kernels.py.

import tilelang
import tilelang.language as T
import torch

from .config import DEFAULT_KERNEL_CONFIG, GDN2Config
from .forward import _pad_to_chunk, _tl_dtype, chunk_gdn2_fwd

__all__ = [
    "chunk_gdn2_bwd",
    "dstate_kernel",
    "dwy_a_kernel",
    "daqk_kernel",
    "dwy_b_kernel",
    "dintra_diag_kernel",
    "dinter_row_kernel",
    "dinter_col_kernel",
    "dassemble_kernel",
]

_FAST_MATH = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}


@tilelang.jit(out_idx=[-3, -2, -1], pass_configs=_FAST_MATH)
def dstate_kernel(
    B: int,
    S: int,
    H: int,
    DK: int,
    DV: int,
    scale: float,
    input_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    gate_dtype: str = "float32",
    state_dtype: str = "float32",
    chunk_size: int = 64,
    use_dht: bool = True,
    block_DV: int = 64,
    threads: int = 128,
    num_stages: int = 2,
):
    """Reverse scan over chunks -- the mirror of ``state_kernel``.

    Walking chunks backwards with ``dS`` the gradient of the state *leaving*
    the chunk::

        dUn = Ktail dS + Aqk^T dO
        dS  = diag(gamma_C) dS + Qg^T dO - W^T dUn

    ``dh[n]`` records the incoming ``dS`` so the parallel stages downstream can
    use it, exactly as the forward stores ``h[n]``.

    Args:
        B, S, H, DK, DV: problem dims.
        scale: query scale, applied when forming ``Qg`` inline.
        input_dtype, accum_dtype, gate_dtype, state_dtype: dtypes.
        chunk_size: chunk length ``BS``.
        use_dht: seed the scan from ``dht`` rather than zeros.
        block_DV: value tile per block.
        threads, num_stages: launch / pipelining config.

    Returns:
        A JIT kernel ``(KG, W, q, G, Aqk, dO, dht) -> (dUn, dh, dS0)``.
    """
    BS = chunk_size
    NC = (S + BS - 1) // BS
    k_shape = (B, S, H, DK)
    v_shape = (B, S, H, DV)
    a_shape = (B, S, H, BS)
    h_shape = (B, NC, H, DK, DV)
    st_shape = (B, H, DK, DV)

    @T.prim_func
    def kernel(
        KG: T.Tensor(k_shape, dtype=input_dtype),
        W: T.Tensor(k_shape, dtype=input_dtype),
        q: T.Tensor(k_shape, dtype=input_dtype),
        G: T.Tensor(k_shape, dtype=gate_dtype),
        Aqk: T.Tensor(a_shape, dtype=input_dtype),
        dO: T.Tensor(v_shape, dtype=input_dtype),
        dht: T.Tensor(st_shape, dtype=state_dtype),
        dUn: T.Tensor(v_shape, dtype=input_dtype),
        dh: T.Tensor(h_shape, dtype=input_dtype),
        dS0: T.Tensor(st_shape, dtype=state_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV), B * H, threads=threads) as (i_v, i_bh):
            i_b, i_h = i_bh // H, i_bh % H
            lo, hi = i_v * block_DV, (i_v + 1) * block_DV

            ds_s = T.alloc_shared((DK, block_DV), dtype=input_dtype)
            ds_f = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            kg_s = T.alloc_shared((BS, DK), dtype=input_dtype)
            w_s = T.alloc_shared((BS, DK), dtype=input_dtype)
            qg_s = T.alloc_shared((BS, DK), dtype=input_dtype)
            q_s = T.alloc_shared((BS, DK), dtype=input_dtype)
            G_s = T.alloc_shared((BS, DK), dtype=gate_dtype)
            a_s = T.alloc_shared((BS, BS), dtype=input_dtype)
            do_s = T.alloc_shared((BS, block_DV), dtype=input_dtype)
            dun_f = T.alloc_fragment((BS, block_DV), dtype=accum_dtype)
            dun_s = T.alloc_shared((BS, block_DV), dtype=input_dtype)
            g_last = T.alloc_shared((DK,), dtype=gate_dtype, scope="shared")

            if use_dht:
                T.copy(dht[i_b, i_h, :, lo:hi], ds_f)
            else:
                T.clear(ds_f)
            T.copy(ds_f, ds_s)

            for i_r in T.Pipelined(T.ceildiv(S, BS), num_stages=num_stages):
                i_s = (S // BS) - 1 - i_r  # walk chunks in reverse
                t0 = i_s * BS
                # Snapshot the gradient entering from the right.
                T.copy(ds_s, dh[i_b, i_s, i_h, :, lo:hi])

                # dUn = Ktail dS + Aqk^T dO
                T.copy(KG[i_b, t0 : t0 + BS, i_h, :], kg_s)
                T.gemm(kg_s, ds_s, dun_f, clear_accum=True)
                T.copy(Aqk[i_b, t0 : t0 + BS, i_h, :], a_s)
                T.copy(dO[i_b, t0 : t0 + BS, i_h, lo:hi], do_s)
                for r, c in T.Parallel(BS, BS):
                    a_s[r, c] = T.if_then_else(r < c, 0, a_s[r, c])
                T.gemm(a_s, do_s, dun_f, transpose_A=True)
                T.copy(dun_f, dun_s)
                T.copy(dun_s, dUn[i_b, t0 : t0 + BS, i_h, lo:hi])

                # dS <- diag(gamma_C) dS + Qg^T dO - W^T dUn
                T.copy(G[i_b, t0 + BS - 1, i_h, :], g_last)
                for c, d in T.Parallel(DK, block_DV):
                    ds_f[c, d] = ds_f[c, d] * T.exp2(g_last[c])
                T.copy(q[i_b, t0 : t0 + BS, i_h, :], q_s)
                T.copy(G[i_b, t0 : t0 + BS, i_h, :], G_s)
                for r, c in T.Parallel(BS, DK):
                    qg_s[r, c] = q_s[r, c] * scale * T.exp2(G_s[r, c])
                T.gemm(qg_s, do_s, ds_f, transpose_A=True)
                T.copy(W[i_b, t0 : t0 + BS, i_h, :], w_s)
                for r, c in T.Parallel(BS, DK):
                    w_s[r, c] = -w_s[r, c]
                T.gemm(w_s, dun_s, ds_f, transpose_A=True)
                T.copy(ds_f, ds_s)

            T.copy(ds_f, dS0[i_b, i_h, :, lo:hi])

    return kernel


@tilelang.jit(out_idx=[-4, -3, -2, -1], pass_configs=_FAST_MATH)
def dwy_a_kernel(
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
    """Gradients of the three ``[BS, DK]`` factors that meet the state.

    ``dW = -dUn h^T``, ``dKtail = Un dh^T``, ``dQg = dO h^T`` -- all reductions
    over ``DV``, so one block owns a ``DK`` tile and sweeps ``DV``. That sweep
    also makes this the natural place to accumulate the state half of the
    chunk-final decay gradient, ``dGC_state[c] = sum_d dh[c,d] gamma_C[c] h[c,d]``.

    Args:
        B, S, H, DK, DV: problem dims.
        input_dtype, accum_dtype, gate_dtype: dtypes.
        chunk_size: chunk length ``BS``.
        block_DK, block_DV: channel tiles.
        threads, num_stages: launch / pipelining config.

    Returns:
        A JIT kernel ``(Un, dUn, dO, h, dh, G) -> (dW, dKtail, dQg, dGC_state)``.
    """
    BS = chunk_size
    NC = (S + BS - 1) // BS
    k_shape = (B, S, H, DK)
    v_shape = (B, S, H, DV)
    h_shape = (B, NC, H, DK, DV)
    gc_shape = (B, NC, H, DK)

    @T.prim_func
    def kernel(
        Un: T.Tensor(v_shape, dtype=input_dtype),
        dUn: T.Tensor(v_shape, dtype=input_dtype),
        dO: T.Tensor(v_shape, dtype=input_dtype),
        h: T.Tensor(h_shape, dtype=input_dtype),
        dh: T.Tensor(h_shape, dtype=input_dtype),
        G: T.Tensor(k_shape, dtype=gate_dtype),
        dW: T.Tensor(k_shape, dtype=input_dtype),
        dKtail: T.Tensor(k_shape, dtype=input_dtype),
        dQg: T.Tensor(k_shape, dtype=input_dtype),
        dGC_state: T.Tensor(gc_shape, dtype=accum_dtype),
    ):
        with T.Kernel(T.ceildiv(DK, block_DK), T.ceildiv(S, BS), B * H, threads=threads) as (i_k, i_s, i_bh):
            i_b, i_h = i_bh // H, i_bh % H
            t0 = i_s * BS
            klo, khi = i_k * block_DK, (i_k + 1) * block_DK

            un_s = T.alloc_shared((BS, block_DV), dtype=input_dtype)
            dun_s = T.alloc_shared((BS, block_DV), dtype=input_dtype)
            do_s = T.alloc_shared((BS, block_DV), dtype=input_dtype)
            h_s = T.alloc_shared((block_DK, block_DV), dtype=input_dtype)
            dh_s = T.alloc_shared((block_DK, block_DV), dtype=input_dtype)
            dw_f = T.alloc_fragment((BS, block_DK), dtype=accum_dtype)
            dkt_f = T.alloc_fragment((BS, block_DK), dtype=accum_dtype)
            dqg_f = T.alloc_fragment((BS, block_DK), dtype=accum_dtype)
            gc_prod = T.alloc_shared((block_DK, block_DV), dtype=accum_dtype)
            gc_part = T.alloc_fragment((block_DK,), dtype=accum_dtype)
            gc_acc = T.alloc_fragment((block_DK,), dtype=accum_dtype)
            g_last = T.alloc_shared((block_DK,), dtype=gate_dtype, scope="shared")

            T.clear(dw_f)
            T.clear(dkt_f)
            T.clear(dqg_f)
            T.clear(gc_acc)
            T.copy(G[i_b, t0 + BS - 1, i_h, klo:khi], g_last)

            for i_v in T.Pipelined(T.ceildiv(DV, block_DV), num_stages=num_stages):
                vlo, vhi = i_v * block_DV, (i_v + 1) * block_DV
                T.copy(Un[i_b, t0 : t0 + BS, i_h, vlo:vhi], un_s)
                T.copy(dUn[i_b, t0 : t0 + BS, i_h, vlo:vhi], dun_s)
                T.copy(dO[i_b, t0 : t0 + BS, i_h, vlo:vhi], do_s)
                T.copy(h[i_b, i_s, i_h, klo:khi, vlo:vhi], h_s)
                T.copy(dh[i_b, i_s, i_h, klo:khi, vlo:vhi], dh_s)

                T.gemm(dun_s, h_s, dw_f, transpose_B=True)
                T.gemm(un_s, dh_s, dkt_f, transpose_B=True)
                T.gemm(do_s, h_s, dqg_f, transpose_B=True)

                for c, d in T.Parallel(block_DK, block_DV):
                    gc_prod[c, d] = dh_s[c, d] * h_s[c, d] * T.exp2(g_last[c])
                T.reduce_sum(gc_prod, gc_part, dim=-1, clear=True)
                for c in T.Parallel(block_DK):
                    gc_acc[c] = gc_acc[c] + gc_part[c]

            for r, c in T.Parallel(BS, block_DK):
                dw_f[r, c] = -dw_f[r, c]  # dW = -dUn h^T
            T.copy(dw_f, dW[i_b, t0 : t0 + BS, i_h, klo:khi])
            T.copy(dkt_f, dKtail[i_b, t0 : t0 + BS, i_h, klo:khi])
            T.copy(dqg_f, dQg[i_b, t0 : t0 + BS, i_h, klo:khi])
            T.copy(gc_acc, dGC_state[i_b, i_s, i_h, klo:khi])

    return kernel


@tilelang.jit(out_idx=[-1], pass_configs=_FAST_MATH)
def daqk_kernel(
    B: int,
    S: int,
    H: int,
    DV: int,
    input_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    chunk_size: int = 64,
    block_DV: int = 64,
    threads: int = 128,
    num_stages: int = 2,
):
    """``dAqk = tril(dO Un^T)``, reduced over ``DV``.

    Args:
        B, S, H, DV: problem dims.
        input_dtype, accum_dtype: dtypes.
        chunk_size: chunk length ``BS``.
        block_DV: value tile.
        threads, num_stages: launch / pipelining config.

    Returns:
        A JIT kernel ``(dO, Un) -> dAqk`` with ``dAqk`` of shape
        ``[B, S, H, BS]``, lower triangular including the diagonal.
    """
    BS = chunk_size
    v_shape = (B, S, H, DV)
    a_shape = (B, S, H, BS)

    @T.prim_func
    def kernel(
        dO: T.Tensor(v_shape, dtype=input_dtype),
        Un: T.Tensor(v_shape, dtype=input_dtype),
        dAqk: T.Tensor(a_shape, dtype=accum_dtype),
    ):
        with T.Kernel(T.ceildiv(S, BS), B * H, threads=threads) as (i_s, i_bh):
            i_b, i_h = i_bh // H, i_bh % H
            t0 = i_s * BS

            do_s = T.alloc_shared((BS, block_DV), dtype=input_dtype)
            un_s = T.alloc_shared((BS, block_DV), dtype=input_dtype)
            acc = T.alloc_fragment((BS, BS), dtype=accum_dtype)

            T.clear(acc)
            for i_v in T.Pipelined(T.ceildiv(DV, block_DV), num_stages=num_stages):
                vlo, vhi = i_v * block_DV, (i_v + 1) * block_DV
                T.copy(dO[i_b, t0 : t0 + BS, i_h, vlo:vhi], do_s)
                T.copy(Un[i_b, t0 : t0 + BS, i_h, vlo:vhi], un_s)
                T.gemm(do_s, un_s, acc, transpose_B=True)

            for r, c in T.Parallel(BS, BS):
                acc[r, c] = T.if_then_else(r < c, 0, acc[r, c])
            T.copy(acc, dAqk[i_b, t0 : t0 + BS, i_h, :])

    return kernel


@tilelang.jit(out_idx=[-3, -2, -1], pass_configs=_FAST_MATH)
def dwy_b_kernel(
    B: int,
    S: int,
    H: int,
    DK: int,
    DV: int,
    input_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    chunk_size: int = 64,
    block_DK: int = 64,
    block_DV: int = 64,
    threads: int = 128,
    num_stages: int = 2,
):
    """Push through ``Ai`` and the matrix inverse.

    ``dZ = Ai^T dUn`` and ``dE = Ai^T dW``. The inverse would nominally need
    ``dTm = -Ai^T dAi Ai^T``, but ``dAi = dUn Z^T + dW E^T`` makes that collapse
    to ``dTm = -strict_tril(dZ U^T + dE W^T)`` -- two GEMMs, no second solve.

    Args:
        B, S, H, DK, DV: problem dims.
        input_dtype, accum_dtype: dtypes.
        chunk_size: chunk length ``BS``.
        block_DK, block_DV: channel tiles.
        threads, num_stages: launch / pipelining config.

    Returns:
        A JIT kernel ``(Ai, dUn, dW, U, W) -> (dZ, dE, dTm)``.
    """
    BS = chunk_size
    k_shape = (B, S, H, DK)
    v_shape = (B, S, H, DV)
    a_shape = (B, S, H, BS)

    @T.prim_func
    def kernel(
        Ai: T.Tensor(a_shape, dtype=input_dtype),
        dUn: T.Tensor(v_shape, dtype=input_dtype),
        dW: T.Tensor(k_shape, dtype=input_dtype),
        U: T.Tensor(v_shape, dtype=input_dtype),
        W: T.Tensor(k_shape, dtype=input_dtype),
        dZ: T.Tensor(v_shape, dtype=input_dtype),
        dE: T.Tensor(k_shape, dtype=input_dtype),
        dTm: T.Tensor(a_shape, dtype=accum_dtype),
    ):
        with T.Kernel(T.ceildiv(S, BS), B * H, threads=threads) as (i_s, i_bh):
            i_b, i_h = i_bh // H, i_bh % H
            t0 = i_s * BS

            ai_s = T.alloc_shared((BS, BS), dtype=input_dtype)
            dun_s = T.alloc_shared((BS, block_DV), dtype=input_dtype)
            u_s = T.alloc_shared((BS, block_DV), dtype=input_dtype)
            dw_s = T.alloc_shared((BS, block_DK), dtype=input_dtype)
            w_s = T.alloc_shared((BS, block_DK), dtype=input_dtype)
            dz_f = T.alloc_fragment((BS, block_DV), dtype=accum_dtype)
            de_f = T.alloc_fragment((BS, block_DK), dtype=accum_dtype)
            dz_s = T.alloc_shared((BS, block_DV), dtype=input_dtype)
            de_s = T.alloc_shared((BS, block_DK), dtype=input_dtype)
            dtm_f = T.alloc_fragment((BS, BS), dtype=accum_dtype)

            T.copy(Ai[i_b, t0 : t0 + BS, i_h, :], ai_s)
            T.clear(dtm_f)

            for i_v in T.Pipelined(T.ceildiv(DV, block_DV), num_stages=num_stages):
                vlo, vhi = i_v * block_DV, (i_v + 1) * block_DV
                T.copy(dUn[i_b, t0 : t0 + BS, i_h, vlo:vhi], dun_s)
                T.gemm(ai_s, dun_s, dz_f, transpose_A=True, clear_accum=True)
                T.copy(dz_f, dz_s)
                T.copy(dz_s, dZ[i_b, t0 : t0 + BS, i_h, vlo:vhi])
                T.copy(U[i_b, t0 : t0 + BS, i_h, vlo:vhi], u_s)
                T.gemm(dz_s, u_s, dtm_f, transpose_B=True)

            for i_k in T.Pipelined(T.ceildiv(DK, block_DK), num_stages=num_stages):
                klo, khi = i_k * block_DK, (i_k + 1) * block_DK
                T.copy(dW[i_b, t0 : t0 + BS, i_h, klo:khi], dw_s)
                T.gemm(ai_s, dw_s, de_f, transpose_A=True, clear_accum=True)
                T.copy(de_f, de_s)
                T.copy(de_s, dE[i_b, t0 : t0 + BS, i_h, klo:khi])
                T.copy(W[i_b, t0 : t0 + BS, i_h, klo:khi], w_s)
                T.gemm(de_s, w_s, dtm_f, transpose_B=True)

            for r, c in T.Parallel(BS, BS):
                dtm_f[r, c] = T.if_then_else(r > c, -dtm_f[r, c], 0)
            T.copy(dtm_f, dTm[i_b, t0 : t0 + BS, i_h, :])

    return kernel


@tilelang.jit(out_idx=[-4, -3, -2, -1], pass_configs=_FAST_MATH)
def dintra_diag_kernel(
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
    """Diagonal sub-blocks of the pairwise backward, one block per token.

    ``Tm`` and ``Aqk`` are both ``M[r, s] = <X_r, Y_s exp2(G_r - G_s)>``, so
    both contribute a row gradient and a column gradient::

        dX_r[c] = sum_s dM[r, s] Y_s[c] exp2(G_r - G_s)
        dY_s[c] = sum_r dM[r, s] X_r[c] exp2(G_r - G_s)

    For token ``t`` the row sum walks ``j <= t`` and the column sum walks
    ``r >= t``, both restricted to ``t``'s sub-chunk. Every exponent is then
    non-positive and evaluated exactly, which is why the diagonal blocks cannot
    use the reference-row factorisation the off-diagonal ones do.

    Args:
        B, S, H, DK: problem dims.
        scale: query scale, folded into the ``Aqk`` gradients.
        input_dtype, accum_dtype, gate_dtype: dtypes.
        chunk_size, sub_chunk_size: ``BS`` and ``BC``.
        block_H: heads per block.
        threads, num_stages: launch / pipelining config.

    Returns:
        A JIT kernel ``(q, k, b, G, dTm, dAqk) ->
        (dEt_tm, dk_tm, dq_aqk, dk_aqk)``, each ``[B, S, H, DK]``.
    """
    BS, BC = chunk_size, sub_chunk_size
    k_shape = (B, S, H, DK)
    a_shape = (B, S, H, BS)

    @T.prim_func
    def kernel(
        q: T.Tensor(k_shape, dtype=input_dtype),
        k: T.Tensor(k_shape, dtype=input_dtype),
        b: T.Tensor(k_shape, dtype=input_dtype),
        G: T.Tensor(k_shape, dtype=gate_dtype),
        dTm: T.Tensor(a_shape, dtype=accum_dtype),
        dAqk: T.Tensor(a_shape, dtype=accum_dtype),
        dEt_tm: T.Tensor(k_shape, dtype=accum_dtype),
        dk_tm: T.Tensor(k_shape, dtype=accum_dtype),
        dq_aqk: T.Tensor(k_shape, dtype=accum_dtype),
        dk_aqk: T.Tensor(k_shape, dtype=accum_dtype),
    ):
        with T.Kernel(B * S, T.ceildiv(H, block_H), threads=threads) as (i_bs, i_h):
            i_b, t = i_bs // S, i_bs % S
            t_sub = (t // BS) * BS + ((t % BS) // BC) * BC  # first token of the sub-chunk
            t_end = t_sub + BC  # one past the last
            n_row = t + 1 - t_sub  # partners j <= t
            n_col = t_end - t  # partners r >= t

            G_i = T.alloc_shared((block_H, DK), dtype=gate_dtype)
            q_i = T.alloc_shared((block_H, DK), dtype=input_dtype)
            k_i = T.alloc_shared((block_H, DK), dtype=input_dtype)
            b_i = T.alloc_shared((block_H, DK), dtype=input_dtype)
            G_j = T.alloc_shared((block_H, DK), dtype=gate_dtype)
            q_j = T.alloc_shared((block_H, DK), dtype=input_dtype)
            k_j = T.alloc_shared((block_H, DK), dtype=input_dtype)
            b_j = T.alloc_shared((block_H, DK), dtype=input_dtype)

            et_i = T.alloc_fragment((block_H, DK), dtype=accum_dtype)
            acc_et = T.alloc_fragment((block_H, DK), dtype=accum_dtype)
            acc_ktm = T.alloc_fragment((block_H, DK), dtype=accum_dtype)
            acc_q = T.alloc_fragment((block_H, DK), dtype=accum_dtype)
            acc_kq = T.alloc_fragment((block_H, DK), dtype=accum_dtype)
            dtm_v = T.alloc_shared((block_H,), dtype=accum_dtype, scope="shared")
            daq_v = T.alloc_shared((block_H,), dtype=accum_dtype, scope="shared")

            T.copy(G[i_b, t, i_h * block_H : (i_h + 1) * block_H, :], G_i)
            T.copy(q[i_b, t, i_h * block_H : (i_h + 1) * block_H, :], q_i)
            T.copy(k[i_b, t, i_h * block_H : (i_h + 1) * block_H, :], k_i)
            T.copy(b[i_b, t, i_h * block_H : (i_h + 1) * block_H, :], b_i)

            T.disable_warp_group_reg_alloc()
            for hh, c in T.Parallel(block_H, DK):
                et_i[hh, c] = k_i[hh, c] * b_i[hh, c]
            T.clear(acc_et)
            T.clear(acc_ktm)
            T.clear(acc_q)
            T.clear(acc_kq)

            # Row side: this token is r, partners j <= t. Exponent G_t - G_j <= 0.
            for d in T.Pipelined(n_row, num_stages=num_stages):
                j = t_sub + d
                T.copy(k[i_b, j, i_h * block_H : (i_h + 1) * block_H, :], k_j)
                T.copy(G[i_b, j, i_h * block_H : (i_h + 1) * block_H, :], G_j)
                for hh in T.Parallel(block_H):
                    dtm_v[hh] = dTm[i_b, t, i_h * block_H + hh, j % BS]
                    daq_v[hh] = dAqk[i_b, t, i_h * block_H + hh, j % BS] * scale
                for hh, c in T.Parallel(block_H, DK):
                    dec = k_j[hh, c] * T.exp2(G_i[hh, c] - G_j[hh, c])
                    acc_q[hh, c] = acc_q[hh, c] + daq_v[hh] * dec
                    if j < t:  # dTm is strictly lower
                        acc_et[hh, c] = acc_et[hh, c] + dtm_v[hh] * dec

            # Column side: this token is s, partners r >= t. Exponent G_r - G_t <= 0.
            for d in T.Pipelined(n_col, num_stages=num_stages):
                r = t + d
                T.copy(q[i_b, r, i_h * block_H : (i_h + 1) * block_H, :], q_j)
                T.copy(k[i_b, r, i_h * block_H : (i_h + 1) * block_H, :], k_j)
                T.copy(b[i_b, r, i_h * block_H : (i_h + 1) * block_H, :], b_j)
                T.copy(G[i_b, r, i_h * block_H : (i_h + 1) * block_H, :], G_j)
                for hh in T.Parallel(block_H):
                    dtm_v[hh] = dTm[i_b, r, i_h * block_H + hh, t % BS]
                    daq_v[hh] = dAqk[i_b, r, i_h * block_H + hh, t % BS] * scale
                for hh, c in T.Parallel(block_H, DK):
                    dec = T.exp2(G_j[hh, c] - G_i[hh, c])
                    acc_kq[hh, c] = acc_kq[hh, c] + daq_v[hh] * q_j[hh, c] * dec
                    if r > t:
                        acc_ktm[hh, c] = acc_ktm[hh, c] + dtm_v[hh] * k_j[hh, c] * b_j[hh, c] * dec

            hs = i_h * block_H  # head slice base
            T.copy(acc_et, dEt_tm[i_b, t, hs : hs + block_H, :])
            T.copy(acc_ktm, dk_tm[i_b, t, hs : hs + block_H, :])
            T.copy(acc_q, dq_aqk[i_b, t, hs : hs + block_H, :])
            T.copy(acc_kq, dk_aqk[i_b, t, hs : hs + block_H, :])

    return kernel


@tilelang.jit(pass_configs=_FAST_MATH)
def dinter_row_kernel(
    B: int, S: int, H: int, DK: int, scale: float,
    input_dtype: str = "bfloat16", accum_dtype: str = "float32", gate_dtype: str = "float32",
    chunk_size: int = 64, sub_chunk_size: int = 16,
    block_DK: int = 32, threads: int = 128, num_stages: int = 2,
):
    """Off-diagonal row gradients: ``dEt_tm`` and ``dq_aqk``.

    One block per (row sub-block ``i``, chunk, head), summing over column
    sub-blocks ``j < i``. Both matrices share the column operand ``k``, so one
    rescaled ``col_k`` feeds both GEMMs::

        dX_r[c] = exp2(G_r - ref) * sum_j (dM[i,j] @ col_k_j)[r, c]
        col_k_j[s, c] = k_s[c] exp2(ref[c] - G_s[c])

    Accumulates into the arrays :func:`dintra_diag_kernel` wrote. Blocks own
    disjoint row sub-blocks, so the read-modify-write is race-free.

    Args:
        B, S, H, DK: problem dims.
        scale: query scale, folded into ``dq_aqk``.
        input_dtype, accum_dtype, gate_dtype: dtypes.
        chunk_size, sub_chunk_size: ``BS`` and ``BC``.
        block_DK: channel tile.
        threads, num_stages: launch / pipelining config.

    Returns:
        A JIT kernel ``(k, G, dTm, dAqk, dEt_tm, dq_aqk) -> None``; the last two
        are updated in place.
    """
    BS, BC = chunk_size, sub_chunk_size
    NSUB = BS // BC
    if BS % BC != 0 or NSUB < 2:
        raise ValueError(f"need chunk_size divisible by sub_chunk_size with >= 2 sub-chunks, got {BS}/{BC}")
    k_shape = (B, S, H, DK)
    a_shape = (B, S, H, BS)

    @T.prim_func
    def kernel(
        k: T.Tensor(k_shape, dtype=input_dtype),
        G: T.Tensor(k_shape, dtype=gate_dtype),
        dTm: T.Tensor(a_shape, dtype=accum_dtype),
        dAqk: T.Tensor(a_shape, dtype=accum_dtype),
        dEt_tm: T.Tensor(k_shape, dtype=accum_dtype),
        dq_aqk: T.Tensor(k_shape, dtype=accum_dtype),
    ):
        with T.Kernel(NSUB - 1, T.ceildiv(S, BS), B * H, threads=threads) as (i_sub0, i_s, i_bh):
            i_b, i_h = i_bh // H, i_bh % H
            i_sub = i_sub0 + 1
            t0 = i_s * BS
            r0 = t0 + i_sub * BC

            ref = T.alloc_shared((block_DK,), dtype=gate_dtype, scope="shared")
            G_r = T.alloc_shared((BC, block_DK), dtype=gate_dtype)
            G_c = T.alloc_shared((BC, block_DK), dtype=gate_dtype)
            k_c = T.alloc_shared((BC, block_DK), dtype=input_dtype)
            col_k = T.alloc_shared((BC, block_DK), dtype=accum_dtype)
            dtm_s = T.alloc_shared((BC, BC), dtype=accum_dtype)
            daq_s = T.alloc_shared((BC, BC), dtype=accum_dtype)
            acc_et = T.alloc_fragment((BC, block_DK), dtype=accum_dtype)
            acc_q = T.alloc_fragment((BC, block_DK), dtype=accum_dtype)
            out = T.alloc_fragment((BC, block_DK), dtype=accum_dtype)

            for i_k in T.Pipelined(T.ceildiv(DK, block_DK), num_stages=num_stages):
                lo, hi = i_k * block_DK, (i_k + 1) * block_DK
                T.copy(G[i_b, r0, i_h, lo:hi], ref)
                T.copy(G[i_b, r0 : r0 + BC, i_h, lo:hi], G_r)
                T.clear(acc_et)
                T.clear(acc_q)

                for j in T.serial(i_sub):
                    c0 = t0 + j * BC
                    T.copy(k[i_b, c0 : c0 + BC, i_h, lo:hi], k_c)
                    T.copy(G[i_b, c0 : c0 + BC, i_h, lo:hi], G_c)
                    for u, c in T.Parallel(BC, block_DK):
                        col_k[u, c] = k_c[u, c] * T.exp2(ref[c] - G_c[u, c])
                    T.copy(dTm[i_b, r0 : r0 + BC, i_h, j * BC : (j + 1) * BC], dtm_s)
                    T.copy(dAqk[i_b, r0 : r0 + BC, i_h, j * BC : (j + 1) * BC], daq_s)
                    T.gemm(dtm_s, col_k, acc_et)
                    T.gemm(daq_s, col_k, acc_q)

                for r, c in T.Parallel(BC, block_DK):
                    out[r, c] = dEt_tm[i_b, r0 + r, i_h, lo + c] + acc_et[r, c] * T.exp2(G_r[r, c] - ref[c])
                T.copy(out, dEt_tm[i_b, r0 : r0 + BC, i_h, lo:hi])
                for r, c in T.Parallel(BC, block_DK):
                    out[r, c] = dq_aqk[i_b, r0 + r, i_h, lo + c] + scale * acc_q[r, c] * T.exp2(G_r[r, c] - ref[c])
                T.copy(out, dq_aqk[i_b, r0 : r0 + BC, i_h, lo:hi])

    return kernel


@tilelang.jit(pass_configs=_FAST_MATH)
def dinter_col_kernel(
    B: int, S: int, H: int, DK: int, scale: float,
    input_dtype: str = "bfloat16", accum_dtype: str = "float32", gate_dtype: str = "float32",
    chunk_size: int = 64, sub_chunk_size: int = 16,
    block_DK: int = 32, threads: int = 128, num_stages: int = 2,
):
    """Off-diagonal column gradients: ``dk_tm`` and ``dk_aqk``.

    Mirror of :func:`dinter_row_kernel` with the roles swapped -- one block per
    (column sub-block ``j``, chunk, head), summing over row sub-blocks
    ``i > j``. The reference row belongs to ``i``, so unlike the row kernel it
    moves with the loop and the ``exp2(ref - G_s)`` factor is applied inside::

        dY_s[c] = sum_i exp2(ref_i - G_s) * (dM[i,j]^T @ row_X_i)[s, c]

    Args:
        B, S, H, DK: problem dims.
        scale: query scale, folded into ``dk_aqk``.
        input_dtype, accum_dtype, gate_dtype: dtypes.
        chunk_size, sub_chunk_size: ``BS`` and ``BC``.
        block_DK: channel tile.
        threads, num_stages: launch / pipelining config.

    Returns:
        A JIT kernel ``(q, k, b, G, dTm, dAqk, dk_tm, dk_aqk) -> None``; the
        last two are updated in place.
    """
    BS, BC = chunk_size, sub_chunk_size
    NSUB = BS // BC
    if BS % BC != 0 or NSUB < 2:
        raise ValueError(f"need chunk_size divisible by sub_chunk_size with >= 2 sub-chunks, got {BS}/{BC}")
    k_shape = (B, S, H, DK)
    a_shape = (B, S, H, BS)

    @T.prim_func
    def kernel(
        q: T.Tensor(k_shape, dtype=input_dtype),
        k: T.Tensor(k_shape, dtype=input_dtype),
        b: T.Tensor(k_shape, dtype=input_dtype),
        G: T.Tensor(k_shape, dtype=gate_dtype),
        dTm: T.Tensor(a_shape, dtype=accum_dtype),
        dAqk: T.Tensor(a_shape, dtype=accum_dtype),
        dk_tm: T.Tensor(k_shape, dtype=accum_dtype),
        dk_aqk: T.Tensor(k_shape, dtype=accum_dtype),
    ):
        with T.Kernel(NSUB - 1, T.ceildiv(S, BS), B * H, threads=threads) as (j_sub, i_s, i_bh):
            i_b, i_h = i_bh // H, i_bh % H
            t0 = i_s * BS
            c0 = t0 + j_sub * BC

            ref = T.alloc_shared((block_DK,), dtype=gate_dtype, scope="shared")
            G_c = T.alloc_shared((BC, block_DK), dtype=gate_dtype)
            G_r = T.alloc_shared((BC, block_DK), dtype=gate_dtype)
            q_r = T.alloc_shared((BC, block_DK), dtype=input_dtype)
            k_r = T.alloc_shared((BC, block_DK), dtype=input_dtype)
            b_r = T.alloc_shared((BC, block_DK), dtype=input_dtype)
            row_e = T.alloc_shared((BC, block_DK), dtype=accum_dtype)
            row_q = T.alloc_shared((BC, block_DK), dtype=accum_dtype)
            dtm_s = T.alloc_shared((BC, BC), dtype=accum_dtype)
            daq_s = T.alloc_shared((BC, BC), dtype=accum_dtype)
            tmp_e = T.alloc_fragment((BC, block_DK), dtype=accum_dtype)
            tmp_q = T.alloc_fragment((BC, block_DK), dtype=accum_dtype)
            acc_e = T.alloc_fragment((BC, block_DK), dtype=accum_dtype)
            acc_q = T.alloc_fragment((BC, block_DK), dtype=accum_dtype)
            out = T.alloc_fragment((BC, block_DK), dtype=accum_dtype)

            for i_k in T.Pipelined(T.ceildiv(DK, block_DK), num_stages=num_stages):
                lo, hi = i_k * block_DK, (i_k + 1) * block_DK
                T.copy(G[i_b, c0 : c0 + BC, i_h, lo:hi], G_c)
                T.clear(acc_e)
                T.clear(acc_q)

                for d in T.serial(NSUB - 1 - j_sub):
                    i_row = j_sub + 1 + d
                    r0 = t0 + i_row * BC
                    T.copy(G[i_b, r0, i_h, lo:hi], ref)
                    T.copy(G[i_b, r0 : r0 + BC, i_h, lo:hi], G_r)
                    T.copy(q[i_b, r0 : r0 + BC, i_h, lo:hi], q_r)
                    T.copy(k[i_b, r0 : r0 + BC, i_h, lo:hi], k_r)
                    T.copy(b[i_b, r0 : r0 + BC, i_h, lo:hi], b_r)
                    for r, c in T.Parallel(BC, block_DK):
                        dec = T.exp2(G_r[r, c] - ref[c])
                        row_e[r, c] = k_r[r, c] * b_r[r, c] * dec
                        row_q[r, c] = q_r[r, c] * dec
                    T.copy(dTm[i_b, r0 : r0 + BC, i_h, j_sub * BC : (j_sub + 1) * BC], dtm_s)
                    T.copy(dAqk[i_b, r0 : r0 + BC, i_h, j_sub * BC : (j_sub + 1) * BC], daq_s)
                    T.gemm(dtm_s, row_e, tmp_e, transpose_A=True, clear_accum=True)
                    T.gemm(daq_s, row_q, tmp_q, transpose_A=True, clear_accum=True)
                    # The reference moves with i, so fold its factor in here.
                    for u, c in T.Parallel(BC, block_DK):
                        col_dec = T.exp2(ref[c] - G_c[u, c])
                        acc_e[u, c] = acc_e[u, c] + tmp_e[u, c] * col_dec
                        acc_q[u, c] = acc_q[u, c] + tmp_q[u, c] * col_dec

                for u, c in T.Parallel(BC, block_DK):
                    out[u, c] = dk_tm[i_b, c0 + u, i_h, lo + c] + acc_e[u, c]
                T.copy(out, dk_tm[i_b, c0 : c0 + BC, i_h, lo:hi])
                for u, c in T.Parallel(BC, block_DK):
                    out[u, c] = dk_aqk[i_b, c0 + u, i_h, lo + c] + scale * acc_q[u, c]
                T.copy(out, dk_aqk[i_b, c0 : c0 + BC, i_h, lo:hi])

    return kernel


@tilelang.jit(out_idx=[-6, -5, -4, -3, -2, -1], pass_configs=_FAST_MATH)
def dassemble_kernel(
    B: int, S: int, H: int, DK: int, DV: int, scale: float,
    input_dtype: str = "bfloat16", accum_dtype: str = "float32", gate_dtype: str = "float32",
    chunk_size: int = 64, block_DK: int = 64, block_DV: int = 64,
    threads: int = 128, num_stages: int = 2,
):
    """Collect every partial into the input gradients.

    All the elementwise chain rules, plus the reverse cumulative sum that turns
    ``dG`` into ``dg`` (``g_i`` feeds every ``G_r`` with ``r >= i``). The decay
    gradient collects from all seven paths that read ``G``; the chunk-final
    ``G_C`` terms land on the last row.

    Args:
        B, S, H, DK, DV: problem dims.
        scale: query scale.
        input_dtype, accum_dtype, gate_dtype: dtypes.
        chunk_size: chunk length ``BS``; must be a power of two.
        block_DK, block_DV: channel tiles.
        threads, num_stages: launch / pipelining config.

    Returns:
        A JIT kernel ``(q, k, b, v, w, G, dE, dEt_tm, dk_tm, dq_aqk, dk_aqk,
        dQg, dKtail, dZ, dGC_state) -> (dq, dk, db, dv, dw, dg)``.
    """
    BS = chunk_size
    if BS & (BS - 1):
        raise ValueError(f"chunk_size {BS} must be a power of two")
    n_steps = BS.bit_length() - 1
    NC = (S + BS - 1) // BS
    k_shape = (B, S, H, DK)
    v_shape = (B, S, H, DV)
    gc_shape = (B, NC, H, DK)

    @T.prim_func
    def kernel(
        q: T.Tensor(k_shape, dtype=input_dtype),
        k: T.Tensor(k_shape, dtype=input_dtype),
        b: T.Tensor(k_shape, dtype=input_dtype),
        v: T.Tensor(v_shape, dtype=input_dtype),
        w: T.Tensor(v_shape, dtype=input_dtype),
        G: T.Tensor(k_shape, dtype=gate_dtype),
        dE: T.Tensor(k_shape, dtype=input_dtype),
        dEt_tm: T.Tensor(k_shape, dtype=accum_dtype),
        dk_tm: T.Tensor(k_shape, dtype=accum_dtype),
        dq_aqk: T.Tensor(k_shape, dtype=accum_dtype),
        dk_aqk: T.Tensor(k_shape, dtype=accum_dtype),
        dQg: T.Tensor(k_shape, dtype=input_dtype),
        dKtail: T.Tensor(k_shape, dtype=input_dtype),
        dZ: T.Tensor(v_shape, dtype=input_dtype),
        dGC_state: T.Tensor(gc_shape, dtype=accum_dtype),
        dq: T.Tensor(k_shape, dtype=input_dtype),
        dk: T.Tensor(k_shape, dtype=input_dtype),
        db: T.Tensor(k_shape, dtype=input_dtype),
        dv: T.Tensor(v_shape, dtype=input_dtype),
        dw: T.Tensor(v_shape, dtype=input_dtype),
        dg: T.Tensor(k_shape, dtype=gate_dtype),
    ):
        with T.Kernel(T.ceildiv(DK, block_DK), T.ceildiv(S, BS), B * H, threads=threads) as (i_k, i_s, i_bh):
            i_b, i_h = i_bh // H, i_bh % H
            t0 = i_s * BS
            lo, hi = i_k * block_DK, (i_k + 1) * block_DK

            q_s = T.alloc_shared((BS, block_DK), dtype=input_dtype)
            k_s = T.alloc_shared((BS, block_DK), dtype=input_dtype)
            b_s = T.alloc_shared((BS, block_DK), dtype=input_dtype)
            G_s = T.alloc_shared((BS, block_DK), dtype=gate_dtype)
            gl = T.alloc_shared((block_DK,), dtype=gate_dtype, scope="shared")
            dE_s = T.alloc_shared((BS, block_DK), dtype=input_dtype)
            dQg_s = T.alloc_shared((BS, block_DK), dtype=input_dtype)
            dKt_s = T.alloc_shared((BS, block_DK), dtype=input_dtype)
            dEt = T.alloc_fragment((BS, block_DK), dtype=accum_dtype)
            dG = T.alloc_shared((BS, block_DK), dtype=gate_dtype)
            nxt = T.alloc_shared((BS, block_DK), dtype=gate_dtype)
            ktdk = T.alloc_shared((BS, block_DK), dtype=accum_dtype)
            gc_sum = T.alloc_fragment((block_DK,), dtype=accum_dtype)
            out = T.alloc_fragment((BS, block_DK), dtype=accum_dtype)

            T.copy(q[i_b, t0 : t0 + BS, i_h, lo:hi], q_s)
            T.copy(k[i_b, t0 : t0 + BS, i_h, lo:hi], k_s)
            T.copy(b[i_b, t0 : t0 + BS, i_h, lo:hi], b_s)
            T.copy(G[i_b, t0 : t0 + BS, i_h, lo:hi], G_s)
            T.copy(G[i_b, t0 + BS - 1, i_h, lo:hi], gl)
            T.copy(dE[i_b, t0 : t0 + BS, i_h, lo:hi], dE_s)
            T.copy(dQg[i_b, t0 : t0 + BS, i_h, lo:hi], dQg_s)
            T.copy(dKtail[i_b, t0 : t0 + BS, i_h, lo:hi], dKt_s)

            for r, c in T.Parallel(BS, block_DK):
                gam = T.exp2(G_s[r, c])
                tail = T.exp2(gl[c] - G_s[r, c])
                et = k_s[r, c] * b_s[r, c]
                etm = dEt_tm[i_b, t0 + r, i_h, lo + c]
                ktm = dk_tm[i_b, t0 + r, i_h, lo + c]
                qaq = dq_aqk[i_b, t0 + r, i_h, lo + c]
                kaq = dk_aqk[i_b, t0 + r, i_h, lo + c]
                # dEt = dE * gamma + dEt_tm
                dEt[r, c] = dE_s[r, c] * gam + etm
                ktdk[r, c] = k_s[r, c] * tail * dKt_s[r, c]  # Ktail * dKtail
                dG[r, c] = (
                    et * gam * dE_s[r, c]
                    + et * etm - k_s[r, c] * ktm
                    + q_s[r, c] * qaq - k_s[r, c] * kaq
                    + q_s[r, c] * gam * scale * dQg_s[r, c]
                    - ktdk[r, c]
                )
                out[r, c] = qaq + scale * gam * dQg_s[r, c]
            T.copy(out, dq[i_b, t0 : t0 + BS, i_h, lo:hi])

            for r, c in T.Parallel(BS, block_DK):
                out[r, c] = (
                    dEt[r, c] * b_s[r, c]
                    + dk_tm[i_b, t0 + r, i_h, lo + c]
                    + dk_aqk[i_b, t0 + r, i_h, lo + c]
                    + dKt_s[r, c] * T.exp2(gl[c] - G_s[r, c])
                )
            T.copy(out, dk[i_b, t0 : t0 + BS, i_h, lo:hi])
            for r, c in T.Parallel(BS, block_DK):
                out[r, c] = dEt[r, c] * k_s[r, c]
            T.copy(out, db[i_b, t0 : t0 + BS, i_h, lo:hi])

            # G_C is the last row's cumulative decay, so both of its gradient
            # terms land there.
            T.reduce_sum(ktdk, gc_sum, dim=0, clear=True)
            for c in T.Parallel(block_DK):
                dG[BS - 1, c] = dG[BS - 1, c] + gc_sum[c] + dGC_state[i_b, i_s, i_h, lo + c]

            # dg = reverse cumulative sum of dG: g_i feeds every G_r with r >= i.
            for step in T.serial(n_steps):
                off = T.shift_left(1, step)
                for r, c in T.Parallel(BS, block_DK):
                    nxtv = dG[T.min(r + off, BS - 1), c]
                    nxt[r, c] = dG[r, c] + T.if_then_else(r + off < BS, nxtv, 0.0)
                T.copy(nxt, dG)
            T.copy(dG, dg[i_b, t0 : t0 + BS, i_h, lo:hi])

            # dv = dZ * w, dw = dZ * v -- independent of the DK tiling, so only
            # the first tile does them.
            if i_k == 0:
                v_s = T.alloc_shared((BS, block_DV), dtype=input_dtype)
                w_s = T.alloc_shared((BS, block_DV), dtype=input_dtype)
                dZ_s = T.alloc_shared((BS, block_DV), dtype=input_dtype)
                ov = T.alloc_fragment((BS, block_DV), dtype=accum_dtype)
                for i_v in T.Pipelined(T.ceildiv(DV, block_DV), num_stages=num_stages):
                    vlo, vhi = i_v * block_DV, (i_v + 1) * block_DV
                    T.copy(v[i_b, t0 : t0 + BS, i_h, vlo:vhi], v_s)
                    T.copy(w[i_b, t0 : t0 + BS, i_h, vlo:vhi], w_s)
                    T.copy(dZ[i_b, t0 : t0 + BS, i_h, vlo:vhi], dZ_s)
                    for r, c in T.Parallel(BS, block_DV):
                        ov[r, c] = dZ_s[r, c] * w_s[r, c]
                    T.copy(ov, dv[i_b, t0 : t0 + BS, i_h, vlo:vhi])
                    for r, c in T.Parallel(BS, block_DV):
                        ov[r, c] = dZ_s[r, c] * v_s[r, c]
                    T.copy(ov, dw[i_b, t0 : t0 + BS, i_h, vlo:vhi])

    return kernel


def chunk_gdn2_bwd(
    q, k, v, g, b, w, scale, initial_state, do, dht,
    config: GDN2Config = DEFAULT_KERNEL_CONFIG,
):
    """Run the eight-kernel TileLang backward pipeline.

    Mirrors :func:`gdn2.backward.chunk_gdn2_bwd_torch` stage for stage; that
    module carries the derivation. The forward is recomputed first, because the
    backward needs every WY intermediate and recomputing them is cheaper than
    keeping them resident across the step.

    Args:
        q, k, v, g, b, w: forward inputs, ``[B, S, H, D]``.
        scale: query scale.
        initial_state: ``[B, H, DK, DV]`` or ``None``.
        do: ``[B, S, H, DV]`` output gradient.
        dht: ``[B, H, DK, DV]`` final-state gradient, or ``None``.
        config: tiling configuration.

    Returns:
        ``(dq, dk, dv, dg, db, dw, dinitial_state)``.
    """

    B, S, H, DK = q.shape
    DV = v.shape[-1]
    BS, BC = config.chunk_size, config.sub_chunk_size
    in_dt = _tl_dtype(q.dtype)
    st_dt = _tl_dtype(config.state_dtype)

    _, _, f = chunk_gdn2_fwd(q, k, v, g, b, w, scale, initial_state, False, config,
                             return_intermediates=True)
    S_pad = f["S_pad"]
    do_p = _pad_to_chunk(do, S_pad)
    dev, tl_kw = q.device, dict(threads=config.threads, num_stages=config.num_stages)
    bdk, bdv = min(config.block_DK, DK), min(config.block_DV, DV)

    dht_in = (dht.to(config.state_dtype).contiguous() if dht is not None
              else torch.empty(B, H, DK, DV, dtype=config.state_dtype, device=dev))

    dUn, dh, dS0 = dstate_kernel(
        B, S_pad, H, DK, DV, scale, input_dtype=in_dt, state_dtype=st_dt,
        chunk_size=BS, use_dht=dht is not None, block_DV=bdv, **tl_kw,
    )(f["KG"], f["W"], f["q"], f["G"], f["Aqk"], do_p, dht_in)

    dW, dKtail, dQg, dGC = dwy_a_kernel(
        B, S_pad, H, DK, DV, input_dtype=in_dt, chunk_size=BS,
        block_DK=bdk, block_DV=bdv, **tl_kw,
    )(f["Un"], dUn, do_p, f["h"], dh, f["G"])

    dAqk = daqk_kernel(
        B, S_pad, H, DV, input_dtype=in_dt, chunk_size=BS, block_DV=bdv, **tl_kw,
    )(do_p, f["Un"])

    dZ, dE, dTm = dwy_b_kernel(
        B, S_pad, H, DK, DV, input_dtype=in_dt, chunk_size=BS,
        block_DK=bdk, block_DV=bdv, **tl_kw,
    )(f["Ai"], dUn, dW, f["U"], f["W"])

    # Kernel 5 writes the four pairwise gradients; 6 and 7 accumulate into
    # disjoint halves of them, so they must follow it.
    dEt_tm, dk_tm, dq_aqk, dk_aqk = dintra_diag_kernel(
        B, S_pad, H, DK, scale, input_dtype=in_dt, chunk_size=BS, sub_chunk_size=BC,
    )(f["q"], f["k"], f["b"], f["G"], dTm, dAqk)

    dinter_row_kernel(
        B, S_pad, H, DK, scale, input_dtype=in_dt, chunk_size=BS,
        sub_chunk_size=BC, block_DK=min(32, DK), **tl_kw,
    )(f["k"], f["G"], dTm, dAqk, dEt_tm, dq_aqk)

    dinter_col_kernel(
        B, S_pad, H, DK, scale, input_dtype=in_dt, chunk_size=BS,
        sub_chunk_size=BC, block_DK=min(32, DK), **tl_kw,
    )(f["q"], f["k"], f["b"], f["G"], dTm, dAqk, dk_tm, dk_aqk)

    dq, dk, db, dv, dw, dg = dassemble_kernel(
        B, S_pad, H, DK, DV, scale, input_dtype=in_dt, chunk_size=BS,
        block_DK=bdk, block_DV=bdv, **tl_kw,
    )(f["q"], f["k"], f["b"], f["v"], f["w"], f["G"], dE, dEt_tm, dk_tm,
      dq_aqk, dk_aqk, dQg, dKtail, dZ, dGC)

    trim = lambda x: x[:, :S]
    return (*(trim(t) for t in (dq, dk, dv, dg, db, dw)),
            dS0 if initial_state is not None else None)
