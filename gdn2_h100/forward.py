"""H100-tuned forward kernels.

Same maths as :mod:`gdn2.forward` -- validated against the same oracles in
:mod:`gdn2.reference` -- with three changes that matter on Hopper.

**1. The triangular inverse runs on tensor cores.**
The baseline ``solve_kernel`` does ``BS - 2`` *dependent* forward-substitution
steps, each an elementwise multiply over ``[BS, BS]`` plus a 64-way reduction
plus a barrier. At ``BS = 64`` that is 62 serialised round trips through shared
memory and it does no tensor-core work at all.

Because ``Akk`` is strictly lower triangular it is nilpotent, so the Neumann
series terminates exactly::

    (I + T)^-1 = sum_{j=0}^{BS-1} (-T)^j

and repeated squaring evaluates it in ``log2(BS)`` steps::

    P_1 = I,  Q_1 = -T
    P_2m = P_m + Q_m P_m,   Q_2m = Q_m^2

Six steps of two GEMMs each, instead of 62 dependent reductions. It is only
worth doing if it is numerically safe, and it is: the decay keeps ``|T|`` under
~0.2 even with perfectly correlated keys, ``||(I+T)^-1||`` stays at 1, and the
squared iterates never grow (peak ``|Q| < 0.2``). Measured against a float64
solve, both methods land at ~2e-9 -- identical. ``tests/test_h100.py`` pins
this.

**2. ``Ai`` never reaches HBM.**
It is produced by the solve and consumed only by the WY kernel, so the fused
kernel keeps it in shared memory. That removes a ``[B, S, H, BS]`` write and
read -- at ``B=2, S=8192, H=16`` roughly 270 MB of traffic per forward.

**3. Per-kernel tiling.**
Each kernel takes its own :class:`~gdn2_h100.config.TileSpec` rather than one
global setting, and every factory is autotunable -- see
:mod:`gdn2_h100.tuning`.

Everything else -- the log2-space decay, the per-sub-block reference rows, the
token-parallel diagonal -- is unchanged from the baseline, deliberately: that
part is verified and is not what is slow.
"""

# NOTE: no `from __future__ import annotations` -- TVMScript evaluates
# T.Tensor(...) parameter annotations at definition time.

import tilelang
import tilelang.language as T
import torch

from gdn2.forward import (
    LOG2E,
    _pad_to_chunk,
    _tl_dtype,
    cumsum_kernel,
    inter_blocks_kernel,
    intra_diag_kernel,
    output_kernel,
    state_kernel,
)

from .config import H100_PASS_CONFIGS, H100Config, TileSpec

__all__ = ["solve_wy_kernel", "neumann_wy_stages", "chunk_gdn2_fwd_h100"]


@tilelang.jit(out_idx=[-3, -2, -1], pass_configs=H100_PASS_CONFIGS)
def solve_wy_kernel(
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
    threads: int = 256,
    num_stages: int = 3,
):
    """Invert ``I + Akk`` and apply it, in one kernel.

    Replaces the baseline's ``solve_kernel`` + ``wy_kernel`` pair. The inverse
    is built by Neumann squaring (``log2(BS)`` tensor-core steps rather than
    ``BS - 2`` serial substitutions) and stays in shared memory, so the
    ``[B, S, H, BS]`` round trip for ``Ai`` disappears.

    Emits ``W = Ai (b * k * gamma)``, ``U = Ai (w * v)`` and
    ``KG = k * gamma_C / gamma_r``, exactly as the baseline pair does.

    Args:
        B, S, H, DK, DV: problem dims.
        input_dtype, accum_dtype, gate_dtype: tensor / accumulator dtypes.
        chunk_size: chunk length ``BS``; must be a power of two.
        block_DK, block_DV: channel tiles.
        threads: threads per block; 256 gives the compiler two warp groups to
            specialise into producer and consumer.
        num_stages: software pipeline depth.

    Returns:
        A JIT kernel ``(k, v, b, w, G, Akk) -> (W, U, KG)``.
    """
    BS = chunk_size
    if BS & (BS - 1):
        raise ValueError(f"chunk_size {BS} must be a power of two")
    n_steps = BS.bit_length() - 1  # Neumann doubling reaches BS terms exactly

    k_shape = (B, S, H, DK)
    v_shape = (B, S, H, DV)
    a_shape = (B, S, H, BS)

    @T.prim_func
    def kernel(
        k: T.Tensor(k_shape, dtype=input_dtype),  # type: ignore
        v: T.Tensor(v_shape, dtype=input_dtype),  # type: ignore
        b: T.Tensor(k_shape, dtype=input_dtype),  # type: ignore
        w: T.Tensor(v_shape, dtype=input_dtype),  # type: ignore
        G: T.Tensor(k_shape, dtype=gate_dtype),  # type: ignore
        Akk: T.Tensor(a_shape, dtype=accum_dtype),  # type: ignore
        W: T.Tensor(k_shape, dtype=input_dtype),  # type: ignore
        U: T.Tensor(v_shape, dtype=input_dtype),  # type: ignore
        KG: T.Tensor(k_shape, dtype=input_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(S, BS), B * H, threads=threads) as (i_s, i_bh):
            i_b, i_h = i_bh // H, i_bh % H
            t0 = i_s * BS

            # --- Neumann inverse, in shared memory --------------------------
            # The iterates are held in fp32 on purpose. Squaring six times in
            # bf16 costs four orders of magnitude (measured: 5e-4 relative vs
            # 1e-9 in fp32), and Ai feeds the state scan where errors compound
            # across chunks. The GEMMs are therefore TF32, which is still far
            # cheaper than 62 dependent shared-memory reductions. Only the
            # final operand is rounded to bf16 -- exactly what the baseline
            # does when it stores Ai.
            P_s = T.alloc_shared((BS, BS), dtype=accum_dtype)
            Q_s = T.alloc_shared((BS, BS), dtype=accum_dtype)
            Ai_s = T.alloc_shared((BS, BS), dtype=input_dtype)
            qp = T.alloc_fragment((BS, BS), dtype=accum_dtype)
            qq = T.alloc_fragment((BS, BS), dtype=accum_dtype)

            # P <- I, Q <- -T  (T strictly lower, so Q is nilpotent)
            T.copy(Akk[i_b, t0 : t0 + BS, i_h, :], qp)
            for r, c in T.Parallel(BS, BS):
                Q_s[r, c] = T.if_then_else(r > c, -qp[r, c], 0)
                P_s[r, c] = T.if_then_else(r == c, 1, 0)

            for _ in T.serial(n_steps):
                # Both GEMMs read the old P and Q before either is overwritten.
                T.gemm(Q_s, P_s, qp, clear_accum=True)
                T.gemm(Q_s, Q_s, qq, clear_accum=True)
                for r, c in T.Parallel(BS, BS):
                    qp[r, c] = qp[r, c] + P_s[r, c]
                T.copy(qp, P_s)
                T.copy(qq, Q_s)

            T.copy(P_s, Ai_s)  # one rounding, at the end

            # --- apply it: W = Ai E, U = Ai Z, and the tail-decayed keys -----
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

            for i_v in T.Pipelined(T.ceildiv(DV, block_DV), num_stages=num_stages):
                lo, hi = i_v * block_DV, (i_v + 1) * block_DV
                T.copy(v[i_b, t0 : t0 + BS, i_h, lo:hi], v_s)
                T.copy(w[i_b, t0 : t0 + BS, i_h, lo:hi], w_s)
                for t, c in T.Parallel(BS, block_DV):
                    z_s[t, c] = v_s[t, c] * w_s[t, c]
                T.gemm(Ai_s, z_s, out_v, clear_accum=True)
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
                T.gemm(Ai_s, e_s, out_k, clear_accum=True)
                T.copy(out_k, W[i_b, t0 : t0 + BS, i_h, lo:hi])
                T.copy(kg_s, KG[i_b, t0 : t0 + BS, i_h, lo:hi])

    return kernel


def neumann_wy_stages(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: float,
    config: H100Config,
) -> dict:
    """Kernels 1-4 of the H100 pipeline: everything before the state scan.

    Mirrors :func:`gdn2.forward.wy_stages` but runs the fused
    :func:`solve_wy_kernel` in place of the separate solve and WY kernels, so
    ``Ai`` never leaves shared memory.

    Args:
        q, k, v, g, b, w: forward inputs, ``[B, S, H, D]``.
        scale: query scale.
        config: per-kernel tiling.

    Returns:
        A dict with the padded inputs and ``G, Aqk, W, U, KG``, plus ``S_pad``.
    """
    B, S, H, DK = q.shape
    DV = v.shape[-1]
    BS, BC = config.chunk_size, config.sub_chunk_size
    S_pad = ((S + BS - 1) // BS) * BS
    in_dt = _tl_dtype(q.dtype)

    q, k, b_ = (_pad_to_chunk(x, S_pad) for x in (q, k, b))
    v, w_ = (_pad_to_chunk(x, S_pad) for x in (v, w))
    g = _pad_to_chunk(g.float(), S_pad)

    G = cumsum_kernel(B, S_pad, H, DK, chunk_size=BS,
                      threads=config.intra_diag.threads)(g)

    diag, inter = config.intra_diag.clamp(DK, DV), config.inter_blocks.clamp(DK, DV)
    Aqk, Akk = intra_diag_kernel(
        B, S_pad, H, DK, scale, input_dtype=in_dt, chunk_size=BS,
        sub_chunk_size=BC, threads=diag.threads, num_stages=diag.num_stages,
    )(q, k, b_, G)

    inter_blocks_kernel(
        B, S_pad, H, DK, scale, input_dtype=in_dt, chunk_size=BS,
        sub_chunk_size=BC, block_DK=inter.block_DK, threads=inter.threads,
        num_stages=inter.num_stages,
    )(q, k, b_, G, Aqk, Akk)

    wy = config.wy.clamp(DK, DV)
    W, U, KG = solve_wy_kernel(
        B, S_pad, H, DK, DV, input_dtype=in_dt, chunk_size=BS,
        block_DK=wy.block_DK, block_DV=wy.block_DV,
        threads=wy.threads, num_stages=wy.num_stages,
    )(k, v, b_, w_, G, Akk)

    return dict(G=G, Aqk=Aqk, W=W, U=U, KG=KG,
                q=q, k=k, v=v, b=b_, w=w_, S_pad=S_pad)


def chunk_gdn2_fwd_h100(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    config: H100Config | None = None,
):
    """H100 forward: six kernels instead of the baseline's seven.

    Args:
        q, k, v, g, b, w, scale: as in :func:`gdn2.forward.chunk_gdn2_fwd`.
        initial_state: ``[B, H, DK, DV]`` or ``None``.
        output_final_state: also return the state after the last real token.
        config: per-kernel tiling; defaults to :class:`H100Config`.

    Returns:
        ``(o, final_state)`` with ``o`` of shape ``[B, S, H, DV]``.
    """
    config = config or H100Config()
    B, S, H, DK = q.shape
    DV = v.shape[-1]
    BS = config.chunk_size
    in_dt, st_dt = _tl_dtype(q.dtype), _tl_dtype(config.state_dtype)

    f = neumann_wy_stages(q, k, v, g, b, w, scale, config)
    S_pad = f["S_pad"]

    if initial_state is None:
        st_in = torch.empty(B, H, DK, DV, dtype=config.state_dtype, device=q.device)
    else:
        st_in = initial_state.to(config.state_dtype).contiguous()

    st = config.state.clamp(DK, DV)
    h, Un, final_state = state_kernel(
        B, S_pad, H, DK, DV, input_dtype=in_dt, state_dtype=st_dt, chunk_size=BS,
        use_initial_state=initial_state is not None, block_DV=st.block_DV,
        threads=st.threads, num_stages=st.num_stages,
    )(f["KG"], f["W"], f["U"], f["G"], st_in)

    out = config.output.clamp(DK, DV)
    o = output_kernel(
        B, S_pad, H, DK, DV, scale, input_dtype=in_dt, output_dtype=in_dt,
        chunk_size=BS, block_DK=out.block_DK, block_DV=out.block_DV,
        threads=out.threads, num_stages=out.num_stages,
    )(f["q"], f["G"], f["Aqk"], h, Un)

    return o[:, :S], (final_state if output_final_state else None)
