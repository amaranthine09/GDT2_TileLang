"""Reference implementations of DeepSeek-V4 hybrid attention.

DeepSeek-V4 (*Towards Highly Efficient Million-Token Context Intelligence*,
arXiv:2606.19348) replaces the flat KV cache with two compressed attention
variants that are interleaved through the stack:

``CSA`` -- Compressed Sparse Attention
    Pools every ``m`` tokens into one KV entry (overlapped, two-stream), scores
    those entries with a lightning indexer, and attends to only the top ``k``.
``HCA`` -- Heavily Compressed Attention
    Pools every ``m' >> m`` tokens into one entry (non-overlapped, one stream)
    and attends to all of them densely.

Both share the same core attention: shared-KV MQA (one compressed stream read
by every query head, each entry acting as *both* key and value), a sliding
window of uncompressed entries under the same softmax, a learnable per-head
attention sink in the denominator, per-head RMSNorm, and partial RoPE.

Pipeline, per layer::

    h --> C, Z          projections                       (eq. 9/10, 20/21)
      --> CComp         softmax-weighted pooling          (eq. 11/12, 22/23)
      --> KIComp        same pooling, indexer width       (CSA only)
      --> I             lightning index scores            (eq. 16)
      --> idx           top-k selection                   (eq. 17)
      --> o             shared-KV MQA + window + sink     (eq. 19/26, 27)
      --> o_hat         grouped output projection

Everything here is written for clarity and checked against autograd; it is the
oracle the TileLang kernels in :mod:`dsv4.compress`, :mod:`dsv4.indexer` and
:mod:`dsv4.core_attn` are validated against, and the CPU fallback.

Underdetermined details
-----------------------
Three things the report does not pin down numerically. Each is a named
parameter here with the default written out, so a checkpoint that disagrees can
be matched without editing kernels:

1. **Output un-rotation position.** The report says RoPE is applied "with
   position :math:`-i`" to the core attention outputs, having just used ``i``
   for both the head index and the compressed-block index. The stated purpose
   -- making each entry's contribution depend on *the distance between the
   query and the KV entry* -- forces the query position, so that is what
   :func:`unrope_output` uses.
2. **Position tag of a compressed entry.** A pooled entry covers a span of
   tokens, so RoPE needs a representative position for it. ``rope_block_pos``
   selects one; the default is the block's last source token, which puts the
   compressed and sliding-window branches on the same token scale (they share a
   softmax, so they must agree).
3. **Sliding-window projection.** "We additionally produce :math:`n_win`
   uncompressed KV entries" reads as a projection of its own, which is the
   default. ``window_shares_kv_proj`` instead aliases the compression stream's
   pre-pooling entries.
"""

from __future__ import annotations

import torch

__all__ = [
    "rms_norm",
    "rope_cos_sin",
    "apply_partial_rope",
    "unrope_output",
    "csa_compress",
    "hca_compress",
    "csa_compress_bwd",
    "hca_compress_bwd",
    "lightning_index",
    "lightning_index_bwd",
    "topk_select",
    "core_attention",
    "core_attention_bwd",
    "grouped_out_proj",
    "block_positions",
]

def _f(x: torch.Tensor) -> torch.Tensor:
    """Cast to the accumulation dtype.

    fp32 in production, but a caller running in fp64 -- the gradient tests --
    stays in fp64, so the oracle can be checked against autograd to fp64
    precision instead of being capped at fp32 roundoff.
    """
    return x if x.dtype == torch.float64 else x.float()


NEG_INF = float("-inf")

# A score at or below this never came from a real candidate. The reference masks
# with -inf and the kernels with -1e30; both fail this test, so one predicate
# serves either producer.
NO_CANDIDATE = -1.0e30


# ---------------------------------------------------------------------------
# Norm and positional encoding
# ---------------------------------------------------------------------------


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm over the last axis, computed in fp32.

    The report applies this to each query head and to the single compressed KV
    head just before core attention, to keep the logits from exploding -- with
    ``head_dim`` 512 and no other normalisation the dot products are large.

    Args:
        x: ``[..., D]``.
        weight: optional learnable gain ``[D]``.
        eps: variance epsilon.

    Returns:
        Tensor shaped like ``x``, in ``x``'s dtype.
    """
    dtype = x.dtype
    xf = _f(x)
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        out = out * _f(weight)
    return out.to(dtype)


def rope_cos_sin(
    positions: torch.Tensor,
    rope_dim: int,
    theta: float = 10000.0,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine / sine tables for RoPE at arbitrary (possibly negative) positions.

    Positions are taken as a float tensor rather than an arange so the same
    routine serves the ``-t`` output un-rotation and the fractional block
    centres ``rope_block_pos="center"`` produces.

    Args:
        positions: any shape ``[...]``, float or integer.
        rope_dim: number of channels RoPE covers; must be even.
        theta: RoPE base.
        dtype: output dtype for the tables; defaults to the accumulation dtype
            of ``positions``.

    Returns:
        ``(cos, sin)``, each ``[..., rope_dim // 2]``.

    Note:
        The *angle* is always formed in fp64, whatever ``dtype`` asks for, and
        only the cosine and sine are cast down.

        This matters at the context lengths V4 targets. ``inv_freq`` is not
        exactly representable in fp32, and the angle is ``position *
        inv_freq``, so the absolute angle error grows with position. At
        ``rope_dim = 64``, ``theta = 10000`` and position ``10**6``, carrying
        the whole computation in fp32 costs about ``0.022`` radians of angle
        and about ``0.016`` absolute on the cosine -- on a quantity bounded in
        ``[-1, 1]``, and it gets worse further into the context. Cheap to
        avoid, and invisible if you do not: short-context tests will not show
        it.

        Casting the *results* down is free, since they are bounded. The table
        is only ``[N, rope_dim / 2]``, so the fp64 arithmetic costs nothing
        worth measuring.
    """
    if rope_dim % 2:
        raise ValueError(f"rope_dim {rope_dim} must be even")
    half = rope_dim // 2
    inv_freq = theta ** (
        -torch.arange(half, device=positions.device, dtype=torch.float64) / half
    )
    angles = positions.to(torch.float64).unsqueeze(-1) * inv_freq
    dtype = _f(positions).dtype if dtype is None else dtype
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_partial_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
) -> torch.Tensor:
    """Rotate the trailing ``2 * cos.shape[-1]`` channels of ``x``.

    Channels before that are passed through untouched -- this is the "partial
    RoPE" of the report, which rotates only the last 64 dimensions of every
    query vector and KV entry.

    Args:
        x: ``[..., D]``.
        cos, sin: ``[..., rope_dim // 2]``, broadcastable against ``x``'s
            leading axes.
        interleaved: pair channel ``2i`` with ``2i+1`` when ``True``; pair ``i``
            with ``i + rope_dim/2`` when ``False`` (the HuggingFace
            ``rotate_half`` convention, and the default).

    Returns:
        Tensor shaped like ``x``.
    """
    half = cos.shape[-1]
    rope_dim = 2 * half
    D = x.shape[-1]
    if rope_dim > D:
        raise ValueError(f"rope_dim {rope_dim} exceeds channel count {D}")

    passthrough, rot = x[..., : D - rope_dim], x[..., D - rope_dim :]
    dtype = x.dtype
    rot = _f(rot)
    cos, sin = _f(cos), _f(sin)

    if interleaved:
        even, odd = rot[..., 0::2], rot[..., 1::2]
        out_even = even * cos - odd * sin
        out_odd = even * sin + odd * cos
        rotated = torch.stack((out_even, out_odd), dim=-1).flatten(-2)
    else:
        lo, hi = rot[..., :half], rot[..., half:]
        rotated = torch.cat((lo * cos - hi * sin, lo * sin + hi * cos), dim=-1)

    return torch.cat((passthrough, rotated.to(dtype)), dim=-1)


def unrope_output(
    o: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
    theta: float = 10000.0,
    interleaved: bool = False,
) -> torch.Tensor:
    """Apply RoPE at ``-position`` to the trailing ``rope_dim`` channels.

    Because each compressed entry is used as key *and* value, the raw attention
    output is a weighted sum of position-rotated vectors and therefore carries
    absolute position. Rotating by the negative query position turns every
    entry's contribution into a function of its distance from the query.

    Args:
        o: ``[B, N, H, D]`` core attention output.
        positions: ``[B, N]`` or ``[N]`` query positions.
        rope_dim, theta, interleaved: as in :func:`apply_partial_rope`.

    Returns:
        Tensor shaped like ``o``.
    """
    if positions.dim() == 1:
        positions = positions.unsqueeze(0).expand(o.shape[0], -1)
    # Negate before any narrowing cast: rope_cos_sin forms the angle in fp64,
    # and routing an integer position through fp32 first would throw that away.
    cos, sin = rope_cos_sin(-positions, rope_dim, theta, dtype=_f(o).dtype)
    return apply_partial_rope(o, cos.unsqueeze(2), sin.unsqueeze(2), interleaved=interleaved)


def block_positions(
    num_blocks: int,
    compress_rate: int,
    mode: str = "last",
    device: torch.device | None = None,
) -> torch.Tensor:
    """Representative token position of each compressed block.

    A compressed entry pools a span of tokens but needs a single position for
    RoPE. The compressed and sliding-window branches share one softmax, so this
    has to be on the token scale, not the block scale.

    Args:
        num_blocks: how many compressed entries.
        compress_rate: ``m`` (CSA) or ``m'`` (HCA).
        mode: ``"last"`` (default) tags block ``s`` with its newest source
            token ``m(s+1)-1``; ``"first"`` uses ``ms``; ``"center"`` uses the
            span midpoint, which is fractional for even ``m``.
        device: device for the result.

    Returns:
        ``[num_blocks]`` float tensor.
    """
    # fp64: these reach 10**6 at 1M context, where an fp32 RoPE angle has
    # already drifted measurably. See rope_cos_sin.
    s = torch.arange(num_blocks, device=device, dtype=torch.float64)
    if mode == "last":
        return s * compress_rate + (compress_rate - 1)
    if mode == "first":
        return s * compress_rate
    if mode == "center":
        return s * compress_rate + (compress_rate - 1) / 2.0
    raise ValueError(f"unknown rope_block_pos {mode!r}; expected last/first/center")


# ---------------------------------------------------------------------------
# Compression  (eq. 11/12 for CSA, eq. 22/23 for HCA)
# ---------------------------------------------------------------------------


def hca_compress(
    C: torch.Tensor,
    Z: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Non-overlapped compression: every ``m'`` tokens become one entry.

    Per channel independently::

        S_{m'i .. m'(i+1)-1} = softmax(Z_{m'i .. m'(i+1)-1} + B)
        CComp_i              = sum_j S_j * C_j

    The softmax runs over the ``m'`` tokens of the block, *not* over channels,
    so this is ``c`` independent length-``m'`` softmaxes per block.

    Args:
        C: ``[B, N, c]`` KV entries, ``H W^KV``.
        Z: ``[B, N, c]`` compression logits, ``H W^Z``.
        bias: ``[m', c]`` learnable positional bias ``B``.

    Returns:
        ``(CComp, S)`` with ``CComp`` of shape ``[B, NB, c]``, ``NB = N // m'``,
        and ``S`` of shape ``[B, NB, m', c]`` -- the pooling weights, saved for
        the backward pass.
    """
    B, N, c = C.shape
    m = bias.shape[0]
    if N % m:
        raise ValueError(f"sequence length {N} must be a multiple of compress_rate {m}")
    NB = N // m

    Zb = _f(Z.view(B, NB, m, c)) + _f(bias)
    S = torch.softmax(Zb, dim=2)
    CComp = (S * _f(C.view(B, NB, m, c))).sum(dim=2)
    return CComp.to(C.dtype), S


def csa_compress(
    Ca: torch.Tensor,
    Cb: torch.Tensor,
    Za: torch.Tensor,
    Zb: torch.Tensor,
    bias_a: torch.Tensor,
    bias_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Overlapped two-stream compression: every ``m`` tokens become one entry.

    Block ``i`` pools ``m`` rows of stream *a* taken from its own span and
    ``m`` rows of stream *b* taken from the span before it, under a *single*
    softmax over all ``2m`` of them (per channel)::

        [S^a_{mi..m(i+1)-1}; S^b_{m(i-1)..mi-1}]
            = softmax([Z^a_{mi..m(i+1)-1} + B^a; Z^b_{m(i-1)..mi-1} + B^b])
        CComp_i = sum_j S^a_j * C^a_j + sum_j S^b_j * C^b_j

    Each entry therefore draws on ``2m`` rows, but consecutive entries reuse
    rows (block ``i``'s *b*-span is block ``i-1``'s *a*-span), so the sequence
    still shrinks by exactly ``m``. Block 0 has no predecessor: its *b* logits
    are ``-inf`` and its *b* values zero, which the softmax handles by giving
    the *b* half zero weight.

    Note that stream *a* index ``j`` feeds only block ``j // m`` and stream *b*
    index ``j`` only block ``j // m + 1``, so the backward scatter has no
    read-modify-write conflicts.

    Args:
        Ca, Cb: ``[B, N, c]`` KV entries for the two streams.
        Za, Zb: ``[B, N, c]`` compression logits for the two streams.
        bias_a, bias_b: ``[m, c]`` learnable positional biases.

    Returns:
        ``(CComp, S)`` with ``CComp`` of shape ``[B, NB, c]``, ``NB = N // m``,
        and ``S`` of shape ``[B, NB, 2m, c]`` holding the joint weights --
        first ``m`` along axis 2 are stream *a*, last ``m`` are stream *b*.
    """
    B, N, c = Ca.shape
    m = bias_a.shape[0]
    if N % m:
        raise ValueError(f"sequence length {N} must be a multiple of compress_rate {m}")
    NB = N // m

    za = _f(Za.view(B, NB, m, c)) + _f(bias_a)
    ca = _f(Ca.view(B, NB, m, c))

    # Stream b, shifted one block later. Block 0 reads the pad slot.
    zb_blocks = _f(Zb.view(B, NB, m, c)) + _f(bias_b)
    cb_blocks = _f(Cb.view(B, NB, m, c))
    pad_z = torch.full((B, 1, m, c), NEG_INF, device=Za.device, dtype=za.dtype)
    pad_c = torch.zeros((B, 1, m, c), device=Ca.device, dtype=ca.dtype)
    zb = torch.cat((pad_z, zb_blocks[:, :-1]), dim=1)
    cb = torch.cat((pad_c, cb_blocks[:, :-1]), dim=1)

    S = torch.softmax(torch.cat((za, zb), dim=2), dim=2)
    CComp = (S * torch.cat((ca, cb), dim=2)).sum(dim=2)
    return CComp.to(Ca.dtype), S


def hca_compress_bwd(
    dCComp: torch.Tensor,
    C: torch.Tensor,
    S: torch.Tensor,
    CComp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Explicit backward of :func:`hca_compress`; what the kernel implements.

    With ``o = sum_j S_j c_j`` and ``S = softmax_j(z + B)`` taken per channel::

        dC_j = S_j * dO
        dS_j = dO * C_j
        dZ_j = S_j * (dS_j - sum_l S_l dS_l)
             = S_j * dO * (C_j - o)          since sum_l S_l C_l = o

    so the softmax Jacobian collapses to a single fused multiply -- no second
    reduction is needed beyond the pooled output the forward already produced.

    Args:
        dCComp: ``[B, NB, c]`` gradient of the pooled entries.
        C: ``[B, N, c]`` forward input.
        S: ``[B, NB, m', c]`` pooling weights from the forward.
        CComp: ``[B, NB, c]`` pooled entries from the forward.

    Returns:
        ``(dC, dZ, dbias)`` shaped ``[B, N, c]``, ``[B, N, c]``, ``[m', c]``.
    """
    B, NB, m, c = S.shape
    dO = _f(dCComp).unsqueeze(2)  # [B, NB, 1, c]
    Cb = _f(C.view(B, NB, m, c))

    dC = S * dO
    dZ = dC * (Cb - _f(CComp).unsqueeze(2))
    dbias = dZ.sum(dim=(0, 1))
    return (
        dC.reshape(B, NB * m, c).to(C.dtype),
        dZ.reshape(B, NB * m, c).to(C.dtype),
        dbias,
    )


def csa_compress_bwd(
    dCComp: torch.Tensor,
    Ca: torch.Tensor,
    Cb: torch.Tensor,
    S: torch.Tensor,
    CComp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Explicit backward of :func:`csa_compress`; what the kernel implements.

    Identical algebra to :func:`hca_compress_bwd` -- one joint softmax over
    ``2m`` slots -- followed by a scatter that sends the first ``m`` slots back
    to stream *a* at block ``i`` and the last ``m`` to stream *b* at block
    ``i-1``. Block 0's *b* half has zero weight and contributes nothing.

    Args:
        dCComp: ``[B, NB, c]`` gradient of the pooled entries.
        Ca, Cb: ``[B, N, c]`` forward inputs.
        S: ``[B, NB, 2m, c]`` joint pooling weights.
        CComp: ``[B, NB, c]`` pooled entries.

    Returns:
        ``(dCa, dCb, dZa, dZb, dbias_a, dbias_b)``.
    """
    B, NB, two_m, c = S.shape
    m = two_m // 2
    dO = _f(dCComp).unsqueeze(2)

    ca = _f(Ca.view(B, NB, m, c))
    cb_blocks = _f(Cb.view(B, NB, m, c))
    pad = torch.zeros((B, 1, m, c), device=Cb.device, dtype=cb_blocks.dtype)
    cb_shift = torch.cat((pad, cb_blocks[:, :-1]), dim=1)
    values = torch.cat((ca, cb_shift), dim=2)

    dV = S * dO
    dZ = dV * (values - _f(CComp).unsqueeze(2))

    dCa, dCb_shift = dV[:, :, :m], dV[:, :, m:]
    dZa, dZb_shift = dZ[:, :, :m], dZ[:, :, m:]

    dbias_a = dZa.sum(dim=(0, 1))
    # Block 0's b-slots are the pad; they carry zero weight, hence zero grad,
    # but slice them off explicitly rather than relying on that.
    dbias_b = dZb_shift[:, 1:].sum(dim=(0, 1))

    # Un-shift: block i's b-half belongs to source block i-1.
    zeros_tail = torch.zeros((B, 1, m, c), device=Cb.device, dtype=dV.dtype)
    dCb = torch.cat((dCb_shift[:, 1:], zeros_tail), dim=1)
    dZb = torch.cat((dZb_shift[:, 1:], zeros_tail), dim=1)

    return (
        dCa.reshape(B, NB * m, c).to(Ca.dtype),
        dCb.reshape(B, NB * m, c).to(Cb.dtype),
        dZa.reshape(B, NB * m, c).to(Ca.dtype),
        dZb.reshape(B, NB * m, c).to(Cb.dtype),
        dbias_a,
        dbias_b,
    )


# ---------------------------------------------------------------------------
# Lightning indexer  (eq. 13-17)
# ---------------------------------------------------------------------------


def lightning_index(
    qI: torch.Tensor,
    wI: torch.Tensor,
    KI: torch.Tensor,
    compress_rate: int,
    causal: bool = True,
    t_offset: int = 0,
) -> torch.Tensor:
    """Index scores between every query token and every compressed block.

    ``I_{t,s} = sum_h w^I_{t,h} * ReLU(q^I_{t,h} . K^IComp_s)``

    The ReLU sits *inside* the head sum, so this is not a plain GEMM: the
    contraction over ``c^I`` has to finish per head before the head reduction
    starts. The head weights ``w^I`` are signed, so a large score can come from
    a negative weight on a small activation -- do not fold them into ``q^I``
    and hope to reuse a single GEMM.

    Args:
        qI: ``[B, N, nhI, cI]`` indexer queries.
        wI: ``[B, N, nhI]`` per-head weights.
        KI: ``[B, NB, cI]`` compressed indexer keys.
        compress_rate: ``m``; sets the causal boundary ``s < t // m``.
        causal: apply that boundary, masking with ``-inf``.
        t_offset: position of row 0 in the full sequence. Non-zero when the
            query axis has been chunked -- the boundary is absolute, so it has
            to be measured from the token's real position, not its row.

    Returns:
        ``[B, N, NB]`` scores. Masked entries are ``-inf`` so top-k cannot pick
        them.
    """
    B, N, nhI, cI = qI.shape
    NB = KI.shape[1]

    a = torch.einsum("bthd,bsd->bths", _f(qI), _f(KI))
    I = torch.einsum("bths,bth->bts", torch.relu(a), _f(wI))

    if causal:
        t = torch.arange(N, device=qI.device) + t_offset
        s = torch.arange(NB, device=qI.device)
        allowed = s.unsqueeze(0) < (t // compress_rate).unsqueeze(1)
        I = I.masked_fill(~allowed.unsqueeze(0), NEG_INF)
    return I


def lightning_index_bwd(
    dI: torch.Tensor,
    qI: torch.Tensor,
    wI: torch.Tensor,
    KI: torch.Tensor,
    compress_rate: int,
    causal: bool = True,
    t_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Explicit backward of :func:`lightning_index`.

    Recomputes the per-head logits ``a = q^I . K`` rather than storing the
    ``[B, N, nhI, NB]`` tensor, which at 1M context would be far larger than
    the model. Only the ReLU sign is needed, and recomputation is one extra
    GEMM against many gigabytes of traffic::

        dw_{t,h}   = sum_s dI_{t,s} * ReLU(a_{t,h,s})
        dr_{t,h,s} = dI_{t,s} * w_{t,h} * [a_{t,h,s} > 0]
        dq_{t,h}   = sum_s dr_{t,h,s} * K_s
        dK_s       = sum_t sum_h dr_{t,h,s} * q_{t,h}

    Args:
        dI: ``[B, N, NB]`` gradient of the scores. Entries the causal mask
            killed are zeroed here before use.
        qI, wI, KI, compress_rate, causal: as in :func:`lightning_index`.

    Returns:
        ``(dqI, dwI, dKI)``.
    """
    B, N, nhI, cI = qI.shape
    NB = KI.shape[1]

    dI = _f(dI)
    if causal:
        t = torch.arange(N, device=qI.device) + t_offset
        s = torch.arange(NB, device=qI.device)
        allowed = (s.unsqueeze(0) < (t // compress_rate).unsqueeze(1)).unsqueeze(0)
        dI = dI.masked_fill(~allowed, 0.0)

    a = torch.einsum("bthd,bsd->bths", _f(qI), _f(KI))
    r = torch.relu(a)

    dwI = torch.einsum("bts,bths->bth", dI, r)
    dr = dI.unsqueeze(2) * _f(wI).unsqueeze(-1) * (a > 0).to(a.dtype)
    dqI = torch.einsum("bths,bsd->bthd", dr, _f(KI))
    dKI = torch.einsum("bths,bthd->bsd", dr, _f(qI))

    return dqI.to(qI.dtype), dwI.to(wI.dtype), dKI.to(KI.dtype)


def topk_select(I: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k compressed blocks per query token (eq. 17).

    Selection is a hard, non-differentiable gather; gradients reach the indexer
    only through whatever auxiliary loss trains it, never through this.

    Rows with fewer than ``k`` legal candidates -- every query with
    ``t // m < k`` -- are padded with ``-1``, which the attention kernels treat
    as an empty slot. Padding rather than clamping matters: repeating a real
    index would double-count that entry in the softmax.

    Args:
        I: ``[B, N, NB]`` index scores, masked with ``-inf``.
        k: how many to keep.

    Returns:
        ``[B, N, k]`` int32 block indices, ``-1`` where there was no candidate.
        Indices are returned sorted ascending so the attention kernel walks the
        KV cache monotonically.
    """
    B, N, NB = I.shape
    kk = min(k, NB)
    scores, idx = torch.topk(_f(I), kk, dim=-1)
    # The kernels mask with a large finite negative rather than -inf (see
    # dsv4.core_attn on why), so test the magnitude, not isneginf.
    idx = idx.to(torch.int32).masked_fill(scores <= NO_CANDIDATE, -1)

    if kk < k:
        pad = torch.full((B, N, k - kk), -1, dtype=torch.int32, device=I.device)
        idx = torch.cat((idx, pad), dim=-1)

    # Sort ascending with the -1 pads pushed to the end.
    order = torch.where(idx < 0, torch.iinfo(torch.int32).max, idx).argsort(dim=-1)
    return torch.gather(idx, -1, order)


# ---------------------------------------------------------------------------
# Core attention  (eq. 19 / 26, with the sink of eq. 27)
# ---------------------------------------------------------------------------


def _chosen_mask(idx: torch.Tensor, NB: int) -> torch.Tensor:
    """``[B, N, NB]`` bool: which compressed blocks a query actually selected.

    The ``-1`` pads have to scatter somewhere harmless. Routing them to a
    scratch column ``NB`` that is then sliced off is not the same as clamping
    them to 0 -- clamping would mark block 0 selected for every padded row.
    """
    B, N, _ = idx.shape
    scratch = torch.zeros((B, N, NB + 1), dtype=torch.bool, device=idx.device)
    tgt = torch.where(idx >= 0, idx.long(), torch.full_like(idx, NB, dtype=torch.long))
    scratch.scatter_(2, tgt, torch.ones_like(tgt, dtype=torch.bool))
    return scratch[..., :NB]


def _window_gather(kv_win: torch.Tensor, N: int, window: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialise the per-token sliding window and its validity mask.

    Slot ``w`` of token ``t`` holds token ``j = t - (window - 1) + w``, so the
    newest token sits in the last slot. Only used by the reference; the kernel
    walks the window in place.

    Returns:
        ``(kv, valid)`` with ``kv`` of shape ``[B, N, window, c]`` and ``valid``
        of shape ``[N, window]``.
    """
    B, _, c = kv_win.shape
    device = kv_win.device
    t = torch.arange(N, device=device).unsqueeze(1)
    w = torch.arange(window, device=device).unsqueeze(0)
    j = t - (window - 1) + w
    valid = (j >= 0) & (j <= t)
    gathered = kv_win.gather(
        1, j.clamp(min=0).reshape(1, -1, 1).expand(B, -1, c)
    ).view(B, N, window, c)
    return gathered * valid.view(1, N, window, 1).to(gathered.dtype), valid


def core_attention(
    q: torch.Tensor,
    kv_comp: torch.Tensor,
    kv_win: torch.Tensor | None,
    sink: torch.Tensor,
    idx: torch.Tensor | None,
    compress_rate: int,
    window: int,
    scale: float,
    return_stats: bool = False,
):
    """Shared-KV MQA over compressed entries plus a sliding window, with sink.

    Every query head reads the *same* KV stream, and each entry is used as both
    key and value -- so a compressed entry's gradient is the sum of a "key"
    term and a "value" term, which the backward must not miss.

    The two branches share one softmax. Denominator carries the per-head sink
    ``exp(z'_h)`` (eq. 27), which lets a head route mass nowhere and hold its
    total attention below 1::

        s_{h,t,j} = exp(z_{h,t,j}) / (sum_l exp(z_{h,t,l}) + exp(z'_h))

    Causality. A query at ``t`` may read compressed blocks ``s < t // m``
    strictly -- its own block is excluded, since that block pools tokens at and
    after ``t``. The window covers ``[t - window + 1, t]`` and is what gives the
    query access to its own neighbourhood.

    Args:
        q: ``[B, N, H, c]`` queries, already normed and RoPE'd.
        kv_comp: ``[B, NB, c]`` compressed entries, already normed and RoPE'd.
        kv_win: ``[B, N, c]`` uncompressed window entries, or ``None`` to run
            compressed-only.
        sink: ``[H]`` learnable sink logits.
        idx: ``[B, N, k]`` selected block indices for CSA, ``-1`` padded; or
            ``None`` for HCA, which attends densely.
        compress_rate: ``m`` or ``m'``, for the causal boundary.
        window: ``n_win``; 0 disables the window branch.
        scale: logit scale.
        return_stats: also return ``(lse, max)``, the softmax statistics the
            backward needs.

    Returns:
        ``[B, N, H, c]`` output, or ``(o, lse, mx)`` when ``return_stats``.
    """
    B, N, H, c = q.shape
    NB = kv_comp.shape[1]
    device, dtype = q.device, q.dtype
    qf = _f(q)

    logits, values = [], []

    # --- compressed branch -------------------------------------------------
    zc = torch.einsum("bthd,bsd->bths", qf, _f(kv_comp)) * scale
    t_ar = torch.arange(N, device=device)
    s_ar = torch.arange(NB, device=device)
    allowed = (s_ar.unsqueeze(0) < (t_ar // compress_rate).unsqueeze(1)).unsqueeze(0)  # [1,N,NB]

    if idx is not None:
        allowed = allowed & _chosen_mask(idx, NB)

    logits.append(zc.masked_fill(~allowed.unsqueeze(2), NEG_INF))
    values.append(_f(kv_comp).unsqueeze(1).expand(B, N, NB, c))

    # --- sliding-window branch --------------------------------------------
    if kv_win is not None and window > 0:
        win_kv, win_valid = _window_gather(kv_win, N, window)
        zw = torch.einsum("bthd,btwd->bthw", qf, _f(win_kv)) * scale
        logits.append(zw.masked_fill(~win_valid.view(1, N, 1, window), NEG_INF))
        values.append(_f(win_kv))

    z = torch.cat(logits, dim=-1)  # [B, N, H, L]
    v = torch.cat(values, dim=2)  # [B, N, L, c]

    # Softmax with the sink folded into the denominator. The running max has to
    # cover the sink logit too, or exp(z' - mx) can overflow for a large sink.
    sink_f = _f(sink).view(1, 1, H, 1)
    mx = torch.maximum(z.max(dim=-1, keepdim=True).values, sink_f)
    mx = torch.where(torch.isneginf(mx), torch.zeros_like(mx), mx)

    p = torch.exp(z - mx)
    denom = p.sum(dim=-1, keepdim=True) + torch.exp(sink_f - mx)
    o = torch.einsum("bthl,btld->bthd", p / denom, v)

    if return_stats:
        lse = (mx + denom.log()).squeeze(-1)  # [B, N, H]
        return o.to(dtype), lse, mx.squeeze(-1)
    return o.to(dtype)


def core_attention_bwd(
    do: torch.Tensor,
    q: torch.Tensor,
    kv_comp: torch.Tensor,
    kv_win: torch.Tensor | None,
    sink: torch.Tensor,
    idx: torch.Tensor | None,
    compress_rate: int,
    window: int,
    scale: float,
    o: torch.Tensor,
    lse: torch.Tensor,
):
    """Explicit backward of :func:`core_attention`; what the kernel implements.

    Standard flash-attention algebra with two wrinkles.

    *Sink.* Adding ``exp(z')`` to the denominator changes the values of the
    probabilities but not the *form* of the logit gradient::

        dz_l  = s_l * ((do . v_l) - Delta),    Delta = do . o
        dz'_h = -s_0 * Delta,                  s_0 = exp(z'_h - lse)

    The ``Delta = do . o`` identity survives because ``sum_l s_l v_l = o`` still
    holds -- the sink contributes no value, only denominator. So a flash kernel
    needs no structural change; it just seeds the denominator and picks up one
    extra scalar gradient per head.

    *Shared key/value.* Each entry is key and value at once, so::

        dkv_j = sum_h s_{h,j} * do_h        (value path)
              + sum_h dz_{h,j} * scale * q_h  (key path)

    Both terms land on the same tensor and must be summed.

    Args:
        do: ``[B, N, H, c]`` output gradient.
        q, kv_comp, kv_win, sink, idx, compress_rate, window, scale: forward
            arguments.
        o: ``[B, N, H, c]`` forward output.
        lse: ``[B, N, H]`` log-sum-exp including the sink, from the forward.

    Returns:
        ``(dq, dkv_comp, dkv_win, dsink)``; ``dkv_win`` is ``None`` when the
        forward ran without a window.
    """
    B, N, H, c = q.shape
    NB = kv_comp.shape[1]
    device = q.device
    qf, dof = _f(q), _f(do)

    zc = torch.einsum("bthd,bsd->bths", qf, _f(kv_comp)) * scale
    t_ar = torch.arange(N, device=device)
    s_ar = torch.arange(NB, device=device)
    allowed = (s_ar.unsqueeze(0) < (t_ar // compress_rate).unsqueeze(1)).unsqueeze(0)
    if idx is not None:
        allowed = allowed & _chosen_mask(idx, NB)

    lse_e = _f(lse).unsqueeze(-1)
    pc = torch.exp(zc - lse_e).masked_fill(~allowed.unsqueeze(2), 0.0)  # [B,N,H,NB]
    delta = (dof * _f(o)).sum(dim=-1)  # [B, N, H]

    # value path + key path, compressed branch
    dkv_comp = torch.einsum("bths,bthd->bsd", pc, dof)
    dzc = pc * (torch.einsum("bthd,bsd->bths", dof, _f(kv_comp)) - delta.unsqueeze(-1))
    dq = torch.einsum("bths,bsd->bthd", dzc, _f(kv_comp)) * scale
    dkv_comp = dkv_comp + torch.einsum("bths,bthd->bsd", dzc, qf) * scale

    dkv_win = None
    if kv_win is not None and window > 0:
        win_kv, win_valid = _window_gather(kv_win, N, window)
        zw = torch.einsum("bthd,btwd->bthw", qf, _f(win_kv)) * scale
        pw = torch.exp(zw - lse_e).masked_fill(~win_valid.view(1, N, 1, window), 0.0)

        dzw = pw * (torch.einsum("bthd,btwd->bthw", dof, _f(win_kv)) - delta.unsqueeze(-1))
        dq = dq + torch.einsum("bthw,btwd->bthd", dzw, _f(win_kv)) * scale

        dwin_slots = torch.einsum("bthw,bthd->btwd", pw, dof) + torch.einsum(
            "bthw,bthd->btwd", dzw, qf
        ) * scale
        # Scatter slots back onto tokens: slot w of token t is token
        # t - (window - 1) + w, so each token collects from every t that saw it.
        dkv_win = torch.zeros((B, N, c), device=device, dtype=dwin_slots.dtype)
        t_i = torch.arange(N, device=device).unsqueeze(1)
        w_i = torch.arange(window, device=device).unsqueeze(0)
        j = t_i - (window - 1) + w_i
        keep = (j >= 0) & (j <= t_i)
        src = (dwin_slots * keep.view(1, N, window, 1)).reshape(B, N * window, c)
        dkv_win.index_add_(1, j.clamp(min=0).reshape(-1), src)
        dkv_win = dkv_win.to(kv_win.dtype)

    # Sink: exp(z' - lse) is the probability mass the head parked on nothing.
    s0 = torch.exp(_f(sink).view(1, 1, H) - _f(lse))
    dsink = (-s0 * delta).sum(dim=(0, 1))

    return dq.to(q.dtype), dkv_comp.to(kv_comp.dtype), dkv_win, dsink


# ---------------------------------------------------------------------------
# Grouped output projection
# ---------------------------------------------------------------------------


def grouped_out_proj(
    o: torch.Tensor,
    w_group: torch.Tensor,
    w_out: torch.Tensor,
) -> torch.Tensor:
    """Two-stage output projection.

    With ``c * n_h`` at 32k-64k, a single ``[c n_h, d]`` matrix dominates the
    layer's parameters. Splitting the heads into ``g`` groups, projecting each
    group down to ``d_g``, then projecting the concatenation to ``d``, costs
    ``g * (c n_h / g) * d_g + g d_g d`` instead of ``c n_h d`` -- about half the
    parameters at the published settings, and the first stage is block-diagonal
    so it batches as a single bmm.

    Args:
        o: ``[B, N, H, c]`` attention output, after the ``-t`` un-rotation.
        w_group: ``[g, c * H // g, d_g]`` per-group down-projection.
        w_out: ``[g * d_g, d]`` final projection.

    Returns:
        ``[B, N, d]``.
    """
    B, N, H, c = o.shape
    g = w_group.shape[0]
    if H % g:
        raise ValueError(f"head count {H} must be divisible by group count {g}")

    grouped = o.reshape(B, N, g, (H // g) * c).permute(2, 0, 1, 3).reshape(g, B * N, -1)
    inter = torch.bmm(grouped, w_group)  # [g, B*N, d_g]
    inter = inter.permute(1, 0, 2).reshape(B, N, -1)
    return inter @ w_out


def sliding_window_attention(
    q: torch.Tensor,
    kv_win: torch.Tensor,
    sink: torch.Tensor,
    window: int,
    scale: float,
) -> torch.Tensor:
    """The pure sliding-window layer V4-Flash uses for its first two blocks.

    Exactly :func:`core_attention` with the compressed branch removed.
    """
    B, N, H, c = q.shape
    win_kv, win_valid = _window_gather(kv_win, N, window)
    z = torch.einsum("bthd,btwd->bthw", _f(q), _f(win_kv)) * scale
    z = z.masked_fill(~win_valid.view(1, N, 1, window), NEG_INF)

    sink_f = _f(sink).view(1, 1, H, 1)
    mx = torch.maximum(z.max(dim=-1, keepdim=True).values, sink_f)
    p = torch.exp(z - mx)
    denom = p.sum(dim=-1, keepdim=True) + torch.exp(sink_f - mx)
    return torch.einsum("bthw,btwd->bthd", p / denom, _f(win_kv)).to(q.dtype)
