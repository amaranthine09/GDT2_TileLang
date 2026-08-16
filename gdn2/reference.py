"""Reference implementations of Gated DeltaNet-2 (GDN-2).

GDN-2 (Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention,
arXiv:2605.22791) generalises KDA / Gated DeltaNet by splitting the single
scalar delta gate into a channel-wise *erase* gate on the key axis and a
channel-wise *write* gate on the value axis.

Recurrence (per batch, per head, ``t = 1..T``)::

    S_t = (I - k_t (b_t * k_t)^T) diag(a_t) S_{t-1} + k_t (w_t * v_t)^T
    o_t = S_t^T q_t

with

    S_t in R^{DK x DV}   recurrent state
    q_t, k_t in R^{DK}   query / key (L2-normalised per head by the layer)
    v_t in R^{DV}        value
    a_t in (0, 1]^{DK}   channel-wise decay, ``a_t = exp(g_t)``
    b_t in [0, 1]^{DK}   channel-wise erase gate  (key side)
    w_t in [0, 1]^{DV}   channel-wise write gate  (value side)

Setting ``b_t = w_t = beta_t * 1`` recovers KDA; additionally collapsing
``a_t`` to a scalar recovers Gated DeltaNet.

Two references live here:

``recurrent_gdn2``
    The literal token-by-token recurrence. Slow, but it is the definition, so
    it is what the kernels are ultimately validated against.

``chunk_gdn2_torch``
    The chunkwise WY form the forward kernels implement, in plain
    differentiable PyTorch.

``chunk_gdn2_bwd_torch``
    The chunkwise backward, staged exactly as the backward kernels run it. Its
    derivation is written out below; read it before :mod:`gdn2.backward`.

Nothing here is meant to be fast. These are the oracles the kernels are tested
against, and the CPU fallback.
"""

from __future__ import annotations

import torch

__all__ = [
    "recurrent_gdn2",
    "chunk_gdn2_torch",
    "chunk_gdn2_bwd_torch",
    "forward_intermediates",
    "pairwise_bwd",
]


def recurrent_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Token-by-token GDN-2 recurrence -- the definition, in fp32.

    Args:
        q, k: ``[B, T, H, DK]`` query / key.
        v: ``[B, T, H, DV]`` value.
        g: ``[B, T, H, DK]`` per-channel log decay, ``<= 0`` (natural log).
        b: ``[B, T, H, DK]`` per-channel erase gate.
        w: ``[B, T, H, DV]`` per-channel write gate.
        scale: query scale; defaults to ``DK ** -0.5``.
        initial_state: ``[B, H, DK, DV]`` incoming state, or ``None`` for zeros.
        output_final_state: also return the state after the last token.

    Returns:
        ``(o, final_state)`` with ``o`` of shape ``[B, T, H, DV]`` and
        ``final_state`` of shape ``[B, H, DK, DV]`` (or ``None``).
    """
    B, T, H, DK = q.shape
    DV = v.shape[-1]
    scale = DK**-0.5 if scale is None else scale

    dtype = torch.float32
    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    g, b, w = g.to(dtype), b.to(dtype), w.to(dtype)

    if initial_state is None:
        S = q.new_zeros(B, H, DK, DV)
    else:
        S = initial_state.to(dtype).clone()

    o = q.new_empty(B, T, H, DV)
    for t in range(T):
        # S_bar = diag(a_t) S_{t-1}: decay acts on the key axis.
        S = S * g[:, t].exp().unsqueeze(-1)
        # u_t = (w_t * v_t) - S_bar^T (b_t * k_t)
        read = torch.einsum("bhkd,bhk->bhd", S, b[:, t] * k[:, t])
        u = w[:, t] * v[:, t] - read
        # S_t = S_bar + k_t u_t^T
        S = S + k[:, t].unsqueeze(-1) * u.unsqueeze(-2)
        o[:, t] = torch.einsum("bhkd,bhk->bhd", S, q[:, t] * scale)

    return o, (S if output_final_state else None)


def _chunk_local_cumsum(g: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Cumulative sum of ``g`` restricted to each chunk. ``[B, T, H, D]``."""
    B, T, H, D = g.shape
    return g.view(B, T // chunk_size, chunk_size, H, D).cumsum(dim=2).view(B, T, H, D)


def _strict_tril_kkt(
    e: torch.Tensor,
    k: torch.Tensor,
    G: torch.Tensor,
    sub_chunk_size: int,
) -> torch.Tensor:
    """``T[r, s] = <e_r * exp(G_r), k_s * exp(-G_s)>`` for ``r > s``, else 0.

    Computed the same way the kernels do it, so that the reference shares the
    kernels' numerics: never materialise ``exp(-G)`` (which overflows whenever a
    channel decays fast), but rescale each 2-D block by the cumulative decay at
    the first row of its row-sub-block. Both exponents are then ``<= 0`` for
    every off-diagonal block, so the factorisation cannot overflow.

    Diagonal sub-blocks have no single valid reference row, so they are done
    with the exact pairwise difference ``exp(G_r - G_s)``. That costs an extra
    ``[BC, BC, D]`` intermediate, which is why the sub-chunk is small.

    Args:
        e: ``[..., BS, D]`` left operand rows (already gated).
        k: ``[..., BS, D]`` right operand rows.
        G: ``[..., BS, D]`` within-chunk cumulative log decay (natural log).
        sub_chunk_size: block size ``BC`` used for the reference rescaling.

    Returns:
        ``[..., BS, BS]`` strictly lower triangular matrix.
    """
    *lead, BS, D = e.shape
    BC = sub_chunk_size
    nsub = BS // BC

    e_s = e.reshape(*lead, nsub, BC, D)
    k_s = k.reshape(*lead, nsub, BC, D)
    G_s = G.reshape(*lead, nsub, BC, D)

    out = e.new_zeros(*lead, nsub, BC, nsub, BC)
    tri = torch.ones(BC, BC, dtype=torch.bool, device=e.device).tril(-1)

    for i in range(nsub):
        ref = G_s[..., i, 0:1, :]  # [..., 1, D] decay at the first row of block i
        row = e_s[..., i, :, :] * (G_s[..., i, :, :] - ref).exp()
        for j in range(i):
            col = k_s[..., j, :, :] * (ref - G_s[..., j, :, :]).exp()
            out[..., i, :, j, :] = row @ col.transpose(-1, -2)
        # Diagonal block: exact pairwise decay difference, no factorisation.
        dG = (G_s[..., i, :, None, :] - G_s[..., i, None, :, :]).clamp_max(0)
        diag = torch.einsum("...rd,...sd,...rsd->...rs", e_s[..., i, :, :], k_s[..., i, :, :], dG.exp())
        out[..., i, :, i, :] = diag.masked_fill(~tri, 0.0)

    return out.reshape(*lead, BS, BS)


def chunk_gdn2_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    sub_chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunkwise WY form of GDN-2 in pure PyTorch (differentiable).

    Mathematically identical to :func:`recurrent_gdn2` but expressed as the
    matrix algebra the TileLang kernels execute:

    Substituting ``S_t = diag(gamma_t) S_hat_t`` with
    ``gamma_t = exp(cumsum(g))`` turns the recurrence into a plain delta rule,
    which unrolls over a chunk into a WY representation::

        Tm   = strict_tril(E K_bar^T)         E_r    = b_r * k_r * gamma_r
        Ai   = (I + Tm)^{-1}                  K_bar_s = k_s / gamma_s
        Wm   = Ai E                           Um     = Ai (w * v)
        Un   = Um - Wm S                      (S = state at chunk start)
        O    = Q_gamma S + tril(Aqk) Un       Q_gamma_r = scale * q_r * gamma_r
        S'   = diag(gamma_C) S + K_tail^T Un  K_tail_r  = k_r * gamma_C / gamma_r

    The only sequential part is the state scan over chunks.

    Args:
        q, k, v, g, b, w, scale, initial_state, output_final_state: as in
            :func:`recurrent_gdn2`.
        chunk_size: chunk length ``BS``; ``T`` must be divisible by it.
        sub_chunk_size: ``BC`` used for the overflow-safe decay rescaling.

    Returns:
        ``(o, final_state)``, same shapes as :func:`recurrent_gdn2`.
    """
    B, T, H, DK = q.shape
    DV = v.shape[-1]
    BS = chunk_size
    if T % BS != 0:
        raise ValueError(f"sequence length {T} must be divisible by chunk_size {BS}")
    if BS % sub_chunk_size != 0:
        raise ValueError(f"chunk_size {BS} must be divisible by sub_chunk_size {sub_chunk_size}")
    scale = DK**-0.5 if scale is None else scale
    NC = T // BS

    f32 = torch.float32
    q, k, v = q.to(f32), k.to(f32), v.to(f32)
    g, b, w = g.to(f32), b.to(f32), w.to(f32)

    G = _chunk_local_cumsum(g, BS)

    # [B, T, H, D] -> [B, H, NC, BS, D]
    def blk(x: torch.Tensor) -> torch.Tensor:
        return x.view(B, NC, BS, H, x.shape[-1]).permute(0, 3, 1, 2, 4)

    qb, kb, vb = blk(q), blk(k), blk(v)
    Gb, bb_, wb = blk(G), blk(b), blk(w)

    gamma = Gb.exp()  # [B, H, NC, BS, DK], <= 1
    E = bb_ * kb * gamma  # erase rows, decay absorbed
    Z = wb * vb  # write rows

    Tm = _strict_tril_kkt(bb_ * kb, kb, Gb, sub_chunk_size)
    eye = torch.eye(BS, dtype=f32, device=q.device).expand_as(Tm)
    Ai = torch.linalg.solve_triangular(eye + Tm, eye, upper=False, unitriangular=True)

    Wm = Ai @ E  # [B, H, NC, BS, DK]
    Um = Ai @ Z  # [B, H, NC, BS, DV]

    # Intra-chunk attention, same rescaling trick as the KKT matrix.
    Aqk = _strict_tril_kkt(qb, kb, Gb, sub_chunk_size)
    # _strict_tril_kkt drops the diagonal; o_t reads the state *after* its own
    # write, so the diagonal r == s belongs in Aqk.
    Aqk = Aqk + torch.diag_embed(torch.einsum("...rd,...rd->...r", qb, kb))
    Aqk = Aqk * scale

    gamma_last = gamma[:, :, :, -1]  # [B, H, NC, DK]
    K_tail = kb * (Gb[:, :, :, -1:] - Gb).exp()  # k_r * gamma_C / gamma_r

    S = q.new_zeros(B, H, DK, DV) if initial_state is None else initial_state.to(f32).clone()

    o = q.new_empty(B, H, NC, BS, DV)
    for n in range(NC):
        Un = Um[:, :, n] - Wm[:, :, n] @ S
        o[:, :, n] = (qb[:, :, n] * gamma[:, :, n] * scale) @ S + Aqk[:, :, n] @ Un
        S = gamma_last[:, :, n].unsqueeze(-1) * S + K_tail[:, :, n].transpose(-1, -2) @ Un

    o = o.permute(0, 2, 3, 1, 4).reshape(B, T, H, DV)
    return o, (S if output_final_state else None)


# ---------------------------------------------------------------------------
# Backward
#
# Derivation
# ----------
# Write the chunk forward as (dropping the chunk index; ``S`` is the state
# entering the chunk, ``gamma_r = exp(G_r)``, ``Et_r = b_r * k_r``)::
# 
#     E     = Et * gamma                Z    = w * v
#     Tm    = strict_tril(E Kbar^T)     Ai   = (I + Tm)^{-1}
#     W     = Ai E                      U    = Ai Z
#     Un    = U - W S
#     O     = Qg S + Aqk Un             Qg_r = scale * q_r * gamma_r
#     S'    = diag(gamma_C) S + Ktail^T Un
# 
# Reverse-mode, in the order the kernels compute it::
# 
#     dUn   = Ktail dS' + Aqk^T dO
#     dS    = diag(gamma_C) dS' + Qg^T dO - W^T dUn        (reverse scan)
#     dW    = -dUn S^T      dKtail = Un dS'^T     dQg = dO S^T
#     dAqk  = tril(dO Un^T)
#     dZ    = Ai^T dUn      dE     = Ai^T dW
# 
# The inverse contributes ``dTm = -Ai^T dAi Ai^T``, and since
# ``dAi = dU Z^T + dW E^T`` this collapses to two plain GEMMs with no second
# inversion::
# 
#     dTm   = -strict_tril(dZ U^T + dE W^T)
# 
# ``Tm`` and ``Aqk`` are both of the form ``M[r,s] = <X_r, Y_s * exp(G_r - G_s)>``,
# so they share one backward (:func:`pairwise_bwd`) producing a row gradient and
# a column gradient. The decay gradient collects from every term that touches
# ``G``; ``g`` is then a reverse cumulative sum of ``dG`` within the chunk.
# 
# Every exponential here is ``exp`` of a non-positive number, using the same
# per-sub-block reference rows as the forward. Nothing forms ``1 / gamma``.
#
# ---------------------------------------------------------------------------


def forward_intermediates(q, k, v, g, b, w, scale, initial_state, chunk_size, sub_chunk_size):
    """Recompute everything the backward needs, in ``[B, H, NC, BS, D]`` layout.

    Returns a dict of forward intermediates keyed by the names used in the
    module docstring, plus the per-chunk pre-states ``h`` and ``Un``.
    """
    B, T, H, DK = q.shape
    DV = v.shape[-1]
    BS, BC = chunk_size, sub_chunk_size
    NC = T // BS

    G = _chunk_local_cumsum(g, BS)

    def blk(x):
        return x.view(B, NC, BS, H, x.shape[-1]).permute(0, 3, 1, 2, 4)

    qb, kb, vb, Gb, bb, wb = (blk(x) for x in (q, k, v, G, b, w))
    gamma = Gb.exp()
    Et = bb * kb  # erase rows before the decay is folded in
    E = Et * gamma
    Z = wb * vb

    Tm = _strict_tril_kkt(Et, kb, Gb, BC)
    eye = torch.eye(BS, dtype=Tm.dtype, device=Tm.device).expand_as(Tm)
    Ai = torch.linalg.solve_triangular(eye + Tm, eye, upper=False, unitriangular=True)
    W, U = Ai @ E, Ai @ Z

    Aqk = _strict_tril_kkt(qb, kb, Gb, BC)
    Aqk = (Aqk + torch.diag_embed(torch.einsum("...rd,...rd->...r", qb, kb))) * scale
    Ktail = kb * (Gb[..., -1:, :] - Gb).exp()

    S = q.new_zeros(B, H, DK, DV) if initial_state is None else initial_state.float().clone()
    h = q.new_empty(B, H, NC, DK, DV)
    Un = q.new_empty(B, H, NC, BS, DV)
    for n in range(NC):
        h[:, :, n] = S
        Un[:, :, n] = U[:, :, n] - W[:, :, n] @ S
        S = gamma[:, :, n, -1].unsqueeze(-1) * S + Ktail[:, :, n].transpose(-1, -2) @ Un[:, :, n]

    return dict(
        G=Gb, gamma=gamma, Et=Et, E=E, Ai=Ai, W=W, U=U, Aqk=Aqk, Ktail=Ktail,
        h=h, Un=Un, qb=qb, kb=kb, vb=vb, bb=bb, wb=wb,
        B=B, H=H, NC=NC, BS=BS, BC=BC, DK=DK, DV=DV,
    )


def pairwise_bwd(dM, X, Y, G, sub_chunk_size, inclusive):
    """Backward of ``M[r, s] = <X_r, Y_s * exp(G_r - G_s)>`` over ``r >= s``.

    Both ``Tm`` (strictly lower) and ``Aqk`` (lower, diagonal included) have
    this shape, so they share one routine::

        dX_r[c] = sum_s dM[r, s] Y_s[c] exp(G_r - G_s)
        dY_s[c] = sum_r dM[r, s] X_r[c] exp(G_r - G_s)

    Off-diagonal sub-blocks factor through the reference row at the start of
    the row sub-block, exactly as in the forward: the row side carries
    ``exp(G_r - ref)`` and the column side ``exp(ref - G_s)``, both bounded by
    1. Diagonal sub-blocks have no valid reference row and use the exact
    pairwise difference instead.

    Args:
        dM: ``[..., BS, BS]`` upstream gradient, already masked.
        X: ``[..., BS, D]`` row operand.
        Y: ``[..., BS, D]`` column operand.
        G: ``[..., BS, D]`` within-chunk cumulative log decay.
        sub_chunk_size: ``BC``.
        inclusive: whether ``r == s`` participates.

    Returns:
        ``(dX, dY)``, both ``[..., BS, D]``.
    """
    *lead, BS, D = X.shape
    BC = sub_chunk_size
    nsub = BS // BC

    Xs = X.reshape(*lead, nsub, BC, D)
    Ys = Y.reshape(*lead, nsub, BC, D)
    Gs = G.reshape(*lead, nsub, BC, D)
    dMs = dM.reshape(*lead, nsub, BC, nsub, BC)

    dX = torch.zeros_like(X).reshape(*lead, nsub, BC, D)
    dY = torch.zeros_like(Y).reshape(*lead, nsub, BC, D)
    tri = torch.ones(BC, BC, dtype=torch.bool, device=X.device).tril(0 if inclusive else -1)

    for i in range(nsub):
        ref = Gs[..., i, 0:1, :]
        row_decay = (Gs[..., i, :, :] - ref).exp()  # <= 1
        for j in range(i):
            col_decay = (ref - Gs[..., j, :, :]).exp()  # <= 1
            block = dMs[..., i, :, j, :]
            dX[..., i, :, :] += row_decay * (block @ (Ys[..., j, :, :] * col_decay))
            dY[..., j, :, :] += col_decay * (block.transpose(-1, -2) @ (Xs[..., i, :, :] * row_decay))
        # Diagonal sub-block: exact pairwise decay, no factorisation. The
        # exponent is <= 0 for every entry we keep; clamping stops the
        # discarded upper triangle from producing inf and then 0 * inf = NaN.
        dec = (Gs[..., i, :, None, :] - Gs[..., i, None, :, :]).clamp_max(0).exp()
        block = dMs[..., i, :, i, :] * tri
        dX[..., i, :, :] += torch.einsum("...rs,...sd,...rsd->...rd", block, Ys[..., i, :, :], dec)
        dY[..., i, :, :] += torch.einsum("...rs,...rd,...rsd->...sd", block, Xs[..., i, :, :], dec)

    return dX.reshape(*lead, BS, D), dY.reshape(*lead, BS, D)


def chunk_gdn2_bwd_torch(
    q, k, v, g, b, w, scale, initial_state, do, dht,
    chunk_size: int = 64, sub_chunk_size: int = 16,
):
    """Full chunkwise GDN-2 backward.

    Args:
        q, k, v, g, b, w: forward inputs, ``[B, T, H, D]``.
        scale: query scale.
        initial_state: ``[B, H, DK, DV]`` or ``None``.
        do: ``[B, T, H, DV]`` gradient of the output.
        dht: ``[B, H, DK, DV]`` gradient of the final state, or ``None``.
        chunk_size, sub_chunk_size: ``BS`` and ``BC``.

    Returns:
        ``(dq, dk, dv, dg, db, dw, dinitial_state)``.
    """
    f = forward_intermediates(q, k, v, g, b, w, scale, initial_state, chunk_size, sub_chunk_size)
    B, H, NC, BS, DK, DV = f["B"], f["H"], f["NC"], f["BS"], f["DK"], f["DV"]
    qb, kb, vb, bb, wb = f["qb"], f["kb"], f["vb"], f["bb"], f["wb"]
    Gb, gamma, Aqk, Ai = f["G"], f["gamma"], f["Aqk"], f["Ai"]
    W, U, Un, h, Ktail, Et, E = f["W"], f["U"], f["Un"], f["h"], f["Ktail"], f["Et"], f["E"]
    dob = do.view(B, NC, BS, H, DV).permute(0, 3, 1, 2, 4).float()

    # --- B1: reverse scan over chunks (the one sequential stage) ------------
    dS = q.new_zeros(B, H, DK, DV) if dht is None else dht.float().clone()
    dUn = torch.empty_like(Un)
    dh = torch.empty_like(h)  # dh[n] = gradient of the state *leaving* chunk n
    for n in range(NC - 1, -1, -1):
        dh[:, :, n] = dS
        dUn[:, :, n] = Ktail[:, :, n] @ dS + Aqk[:, :, n].transpose(-1, -2) @ dob[:, :, n]
        Qg_n = qb[:, :, n] * gamma[:, :, n] * scale
        dS = (
            gamma[:, :, n, -1].unsqueeze(-1) * dS
            + Qg_n.transpose(-1, -2) @ dob[:, :, n]
            - W[:, :, n].transpose(-1, -2) @ dUn[:, :, n]
        )
    dS0 = dS

    # --- B2: per-chunk WY backward -----------------------------------------
    dW = -dUn @ h.transpose(-1, -2)
    dKtail = Un @ dh.transpose(-1, -2)
    dQg = dob @ h.transpose(-1, -2)
    lower = torch.ones(BS, BS, dtype=torch.bool, device=q.device).tril()
    dAqk = (dob @ Un.transpose(-1, -2)) * lower
    dZ = Ai.transpose(-1, -2) @ dUn
    dE = Ai.transpose(-1, -2) @ dW
    dTm = -(dZ @ U.transpose(-1, -2) + dE @ W.transpose(-1, -2)) * lower.tril(-1)

    # --- B3: through the two intra-chunk matrices ---------------------------
    dEt_tm, dk_tm = pairwise_bwd(dTm, Et, kb, Gb, f["BC"], inclusive=False)
    dq_aqk, dk_aqk = pairwise_bwd(dAqk * scale, qb, kb, Gb, f["BC"], inclusive=True)

    # --- B4: assemble -------------------------------------------------------
    dEt = dE * gamma + dEt_tm
    Qg = qb * gamma * scale
    # Every path that reads G contributes; signs follow exp(G_r - G_s).
    dG = (
        E * dE
        + Et * dEt_tm - kb * dk_tm
        + qb * dq_aqk - kb * dk_aqk
        + Qg * dQg - Ktail * dKtail
    )
    # G_C is the last row's cumulative decay, so its gradient lands there.
    dG[..., -1, :] += (Ktail * dKtail).sum(-2) + (dh * gamma[..., -1, :].unsqueeze(-1) * h).sum(-1)

    dq = dq_aqk + scale * gamma * dQg
    dk = dEt * bb + dk_tm + dk_aqk + dKtail * (Gb[..., -1:, :] - Gb).exp()
    db = dEt * kb
    dv = dZ * wb
    dw = dZ * vb
    dg = dG.flip(-2).cumsum(-2).flip(-2)  # g_i affects every G_r with r >= i

    def flat(x):
        return x.permute(0, 2, 3, 1, 4).reshape(B, NC * BS, H, -1)

    return (*(flat(t) for t in (dq, dk, dv, dg, db, dw)), dS0)
