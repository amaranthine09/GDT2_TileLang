"""Inference: prefill reuses the chunkwise forward, decoding gets its own kernels.

Decoding is memory bound, not compute bound. Each step touches the whole
``[DK, DV]`` state -- 64 KB per head at 128x128 in fp32 -- and does only rank-1
work with it, so the state traffic *is* the cost. Two kernels, differing only
in how many tokens they amortise that traffic over:

``decode_step_kernel``
    One token. State in, state out.

``decode_multi_kernel``
    ``n`` tokens in a single launch, with the state held in registers across
    all of them. One state read and one state write instead of ``n`` of each,
    which is the whole point: at ``n = 8`` it turns 16 state round trips into
    2. Use it for speculative decoding, chunked prefill of short tails, and any
    time more than one token is available at once.

Both apply the recurrence directly rather than the chunkwise WY form -- with a
handful of tokens there is no intra-chunk structure worth exploiting, and the
WY machinery would cost more than it saves.

Prefill is the other half. It reuses the chunkwise kernels for the compute,
but not the training forward's *plumbing*: that stores the per-chunk states
``h`` (``[B, NC, H, DK, DV]``, hundreds of MB on a long prompt) so the output
kernel can run parallel over chunks and so the backward can reuse them. Serving
needs neither. ``prefill_scan_kernel`` fuses the state scan with the output and
never materialises ``h``; the cost is that it is sequential over chunks, so it
trades chunk parallelism for the allocation. With a real serving batch there is
plenty of parallelism left in ``B * H * DV-tiles``.

The two entry points:

    gdn2_prefill(q, k, v, g, b, w, state)  ->  (o, state)
    gdn2_decode (q, k, v, g, b, w, state)  ->  o   (state updated in place)

``gdn2_prefill`` also routes any ragged tail through the decode kernel rather
than padding the last chunk, so a 1000-token prompt does not pay for 1024.
"""

# NOTE: no `from __future__ import annotations` -- see gdn2/forward.py.

import tilelang
import tilelang.language as T
import torch

from .config import DEFAULT_KERNEL_CONFIG, GDN2Config
from .forward import _tl_dtype, wy_stages

__all__ = [
    "gdn2_prefill",
    "gdn2_decode",
    "gdn2_decode_step",
    "prefill_scan_kernel",
    "decode_step_kernel",
    "decode_multi_kernel",
]

_FAST_MATH = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}


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


@tilelang.jit(out_idx=[-1], pass_configs=_FAST_MATH)
def decode_multi_kernel(
    B: int,
    H: int,
    DK: int,
    DV: int,
    n_steps: int,
    scale: float,
    input_dtype: str = "bfloat16",
    output_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    gate_dtype: str = "float32",
    state_dtype: str = "float32",
    block_DV: int = 64,
    threads: int = 128,
):
    """``n_steps`` decode steps in one launch, state resident in registers.

    The recurrence is sequential, so the tokens cannot be parallelised -- but
    they can share one trip to memory for the state, which is what dominates.
    The state is loaded once, stepped ``n_steps`` times entirely in registers,
    and written back once.

    Each step is the plain recurrence::

        S <- diag(exp(g_t)) S
        u  = w_t * v_t - S^T (b_t * k_t)
        S <- S + k_t u^T
        o  = scale * S^T q_t

    Args:
        B, H, DK, DV: problem dims.
        n_steps: tokens per launch. Compile-time so the loop can unroll; the
            driver caches one kernel per value it sees.
        scale: query scale.
        input_dtype, output_dtype, accum_dtype, gate_dtype, state_dtype: dtypes.
        block_DV: value tile per block.
        threads: threads per block.

    Returns:
        A JIT kernel ``(q, k, v, g, b, w, state) -> o`` with the sequence
        tensors shaped ``[B, n_steps, H, D]``, ``state`` of shape
        ``[B, H, DK, DV]`` (mutated), and ``o`` of shape ``[B, n_steps, H, DV]``.
    """
    k_shape = (B, n_steps, H, DK)
    v_shape = (B, n_steps, H, DV)
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

            # One read of the state for the whole batch of steps.
            T.copy(state[i_b, i_h, :, lo:hi], st_f)

            for t in T.serial(n_steps):
                T.copy(q[i_b, t, i_h, :], q_s)
                T.copy(k[i_b, t, i_h, :], k_s)
                T.copy(b[i_b, t, i_h, :], b_s)
                T.copy(g[i_b, t, i_h, :], g_s)
                T.copy(v[i_b, t, i_h, lo:hi], v_s)
                T.copy(w[i_b, t, i_h, lo:hi], w_s)

                # decay, then the gated read along the key axis
                for c, d in T.Parallel(DK, block_DV):
                    st_f[c, d] = st_f[c, d] * T.exp(g_s[c])
                    prod[c, d] = st_f[c, d] * b_s[c] * k_s[c]
                T.reduce_sum(prod, red, dim=0, clear=True)

                for d in T.Parallel(block_DV):
                    u_s[d] = w_s[d] * v_s[d] - red[d]

                # rank-1 write, then read this token's output
                for c, d in T.Parallel(DK, block_DV):
                    st_f[c, d] = st_f[c, d] + k_s[c] * u_s[d]
                    prod[c, d] = st_f[c, d] * q_s[c] * scale
                T.reduce_sum(prod, red, dim=0, clear=True)
                T.copy(red, o[i_b, t, i_h, lo:hi])

            # ... and one write back.
            T.copy(st_f, state[i_b, i_h, :, lo:hi])

    return kernel


def gdn2_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    state: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Decode ``n`` tokens, updating ``state`` in place.

    Dispatches to :func:`decode_multi_kernel` on CUDA, or the reference
    recurrence elsewhere. ``n = 1`` is the ordinary autoregressive step; larger
    ``n`` amortises the state traffic and is what you want for speculative
    decoding or a short prefill tail.

    Args:
        q, k, b: ``[B, n, H, DK]``.
        v, w: ``[B, n, H, DV]``.
        g: ``[B, n, H, DK]`` fp32 log decay, ``<= 0``.
        state: ``[B, H, DK, DV]``, mutated in place.
        scale: query scale; defaults to ``DK ** -0.5``.

    Returns:
        ``[B, n, H, DV]`` outputs for the decoded tokens.
    """
    B, n, H, DK = q.shape
    DV = v.shape[-1]
    scale = DK**-0.5 if scale is None else scale

    if q.device.type != "cuda":
        outs = []
        for t in range(n):
            state.mul_(g[:, t].float().exp().unsqueeze(-1))
            read = torch.einsum("bhkd,bhk->bhd", state, (b[:, t] * k[:, t]).float())
            u = (w[:, t] * v[:, t]).float() - read
            state.add_(k[:, t].float().unsqueeze(-1) * u.unsqueeze(-2))
            outs.append(torch.einsum("bhkd,bhk->bhd", state, q[:, t].float() * scale))
        return torch.stack(outs, dim=1).to(v.dtype)

    kernel = decode_multi_kernel(
        B, H, DK, DV, n, scale,
        input_dtype=_tl_dtype(q.dtype), output_dtype=_tl_dtype(v.dtype),
        state_dtype=_tl_dtype(state.dtype), block_DV=min(64, DV),
    )
    return kernel(q.contiguous(), k.contiguous(), v.contiguous(),
                  g.float().contiguous(), b.contiguous(), w.contiguous(), state)


@tilelang.jit(out_idx=[-2, -1], pass_configs=_FAST_MATH)
def prefill_scan_kernel(
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
    state_dtype: str = "float32",
    chunk_size: int = 64,
    use_initial_state: bool = True,
    block_DV: int = 64,
    threads: int = 128,
    num_stages: int = 2,
):
    """State scan and output in one pass, without storing the per-chunk states.

    The training forward splits this in two: ``state_kernel`` scans and records
    ``h[n]`` for every chunk, then ``output_kernel`` runs parallel over chunks
    reading ``h``. That parallelism is worth an ``[B, NC, H, DK, DV]``
    allocation when a backward is coming. For inference it is not, so this
    kernel keeps the state in registers and emits each chunk's output as it
    goes::

        Un = U - W S
        o  = Qg S + tril(Aqk) Un
        S  = diag(gamma_C) S + KG^T Un

    Blocks are parallel over ``(batch, head, DV tile)`` and sequential over
    chunks.

    Args:
        B, S, H, DK, DV: problem dims.
        scale: query scale, applied when forming ``Qg`` inline.
        input_dtype, output_dtype, accum_dtype, gate_dtype, state_dtype: dtypes.
        chunk_size: chunk length ``BS``.
        use_initial_state: read ``initial_state`` instead of starting at zero.
        block_DV: value tile per block.
        threads, num_stages: launch / pipelining config.

    Returns:
        A JIT kernel ``(q, G, Aqk, W, U, KG, initial_state) ->
        (o, final_state)``.
    """
    BS = chunk_size
    k_shape = (B, S, H, DK)
    v_shape = (B, S, H, DV)
    a_shape = (B, S, H, BS)
    st_shape = (B, H, DK, DV)

    @T.prim_func
    def kernel(
        q: T.Tensor(k_shape, dtype=input_dtype),
        G: T.Tensor(k_shape, dtype=gate_dtype),
        Aqk: T.Tensor(a_shape, dtype=input_dtype),
        W: T.Tensor(k_shape, dtype=input_dtype),
        U: T.Tensor(v_shape, dtype=input_dtype),
        KG: T.Tensor(k_shape, dtype=input_dtype),
        initial_state: T.Tensor(st_shape, dtype=state_dtype),
        o: T.Tensor(v_shape, dtype=output_dtype),
        final_state: T.Tensor(st_shape, dtype=state_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV), B * H, threads=threads) as (i_v, i_bh):
            i_b, i_h = i_bh // H, i_bh % H
            lo, hi = i_v * block_DV, (i_v + 1) * block_DV

            st_s = T.alloc_shared((DK, block_DV), dtype=input_dtype)
            st_f = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            w_s = T.alloc_shared((BS, DK), dtype=input_dtype)
            kg_s = T.alloc_shared((BS, DK), dtype=input_dtype)
            q_s = T.alloc_shared((BS, DK), dtype=input_dtype)
            G_s = T.alloc_shared((BS, DK), dtype=gate_dtype)
            qg_s = T.alloc_shared((BS, DK), dtype=input_dtype)
            u_s = T.alloc_shared((BS, block_DV), dtype=input_dtype)
            a_s = T.alloc_shared((BS, BS), dtype=input_dtype)
            un_f = T.alloc_fragment((BS, block_DV), dtype=accum_dtype)
            un_s = T.alloc_shared((BS, block_DV), dtype=input_dtype)
            o_f = T.alloc_fragment((BS, block_DV), dtype=accum_dtype)
            g_last = T.alloc_shared((DK,), dtype=gate_dtype, scope="shared")

            if use_initial_state:
                T.copy(initial_state[i_b, i_h, :, lo:hi], st_f)
            else:
                T.clear(st_f)
            T.copy(st_f, st_s)

            for i_s in T.Pipelined(T.ceildiv(S, BS), num_stages=num_stages):
                t0 = i_s * BS

                # Un = U - W S, using the state entering this chunk.
                T.copy(W[i_b, t0 : t0 + BS, i_h, :], w_s)
                T.gemm(w_s, st_s, un_f, clear_accum=True)
                T.copy(U[i_b, t0 : t0 + BS, i_h, lo:hi], u_s)
                for t, c in T.Parallel(BS, block_DV):
                    un_f[t, c] = u_s[t, c] - un_f[t, c]
                T.copy(un_f, un_s)

                # o = Qg S + tril(Aqk) Un -- emitted now, so h is never stored.
                T.copy(q[i_b, t0 : t0 + BS, i_h, :], q_s)
                T.copy(G[i_b, t0 : t0 + BS, i_h, :], G_s)
                for t, c in T.Parallel(BS, DK):
                    qg_s[t, c] = q_s[t, c] * scale * T.exp2(G_s[t, c])
                T.gemm(qg_s, st_s, o_f, clear_accum=True)
                T.copy(Aqk[i_b, t0 : t0 + BS, i_h, :], a_s)
                for r, c in T.Parallel(BS, BS):
                    a_s[r, c] = T.if_then_else(r < c, 0, a_s[r, c])
                T.gemm(a_s, un_s, o_f)
                T.copy(o_f, o[i_b, t0 : t0 + BS, i_h, lo:hi])

                # S <- diag(gamma_C) S + KG^T Un
                T.copy(G[i_b, t0 + BS - 1, i_h, :], g_last)
                for c, d in T.Parallel(DK, block_DV):
                    st_f[c, d] = st_f[c, d] * T.exp2(g_last[c])
                T.copy(KG[i_b, t0 : t0 + BS, i_h, :], kg_s)
                T.gemm(kg_s, un_s, st_f, transpose_A=True)
                T.copy(st_f, st_s)

            T.copy(st_f, final_state[i_b, i_h, :, lo:hi])

    return kernel


@torch.no_grad()
def gdn2_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    state: torch.Tensor | None = None,
    scale: float | None = None,
    config: GDN2Config = DEFAULT_KERNEL_CONFIG,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefill a prompt, returning its outputs and the state to decode from.

    Whole chunks go through the chunkwise kernels with the scan fused into the
    output, so the per-chunk states are never materialised. Any ragged tail --
    a prompt length that is not a multiple of ``chunk_size`` -- goes through
    the decode kernel instead of padding the last chunk, so a 1000-token prompt
    does not pay for 1024.

    Call it repeatedly, passing the returned state back in, to prefill a long
    prompt in segments.

    Args:
        q, k, b: ``[B, S, H, DK]``.
        v, w: ``[B, S, H, DV]``.
        g: ``[B, S, H, DK]`` fp32 log decay, ``<= 0``.
        state: ``[B, H, DK, DV]`` state to continue from, or ``None``.
        scale: query scale; defaults to ``DK ** -0.5``.
        config: tiling configuration.

    Returns:
        ``(o, state)`` with ``o`` of shape ``[B, S, H, DV]`` and ``state`` the
        recurrent state after the last token.
    """
    B, S, H, DK = q.shape
    DV = v.shape[-1]
    BS = config.chunk_size
    scale = DK**-0.5 if scale is None else scale
    n_full = (S // BS) * BS

    if state is None:
        state = torch.zeros(B, H, DK, DV, dtype=config.state_dtype, device=q.device)
    else:
        state = state.to(config.state_dtype).contiguous()

    if q.device.type != "cuda":
        from .reference import chunk_gdn2_torch
        outs = []
        if n_full:
            o1, state = chunk_gdn2_torch(
                *(t[:, :n_full] for t in (q, k, v, g, b, w)), scale, state, True,
                chunk_size=BS, sub_chunk_size=config.sub_chunk_size,
            )
            outs.append(o1)
        if S > n_full:
            outs.append(gdn2_decode(*(t[:, n_full:] for t in (q, k, v, g, b, w)), state, scale))
        return torch.cat(outs, dim=1), state

    outs = []
    if n_full:
        f = wy_stages(*(t[:, :n_full] for t in (q, k, v, g, b, w)), scale, config)
        o1, state = prefill_scan_kernel(
            B, n_full, H, DK, DV, scale,
            input_dtype=_tl_dtype(q.dtype), output_dtype=_tl_dtype(v.dtype),
            state_dtype=_tl_dtype(state.dtype), chunk_size=BS,
            use_initial_state=True, block_DV=min(config.block_DV, DV),
            threads=config.threads, num_stages=config.num_stages,
        )(f["q"], f["G"], f["Aqk"], f["W"], f["U"], f["KG"], state)
        outs.append(o1)
    if S > n_full:
        # Ragged tail: the decode kernel, rather than padding a whole chunk.
        outs.append(gdn2_decode(*(t[:, n_full:] for t in (q, k, v, g, b, w)), state, scale))
    return torch.cat(outs, dim=1), state
