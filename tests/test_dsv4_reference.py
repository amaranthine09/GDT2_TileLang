"""The DSV4 reference, checked against autograd in fp64.

Every kernel in :mod:`dsv4` implements a backward that is written out by hand,
and every one of those is validated against the explicit backward in
:mod:`dsv4.reference`. So this module is the bottom of the stack: if these
formulas are wrong, nothing above them can be right.

The reference promotes to fp64 whenever its inputs are fp64, which is why these
compare at ``1e-12`` rather than at fp32 tolerance. An error in a gradient
formula that only shows up in the third digit would pass a loose check and then
train slightly wrong forever.
"""

from __future__ import annotations

import pytest
import torch

from dsv4 import reference as R

D = torch.float64
TOL = dict(rtol=1e-11, atol=1e-11)


def _rand(*shape, grad=True):
    return torch.randn(*shape, dtype=D, requires_grad=grad)


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


def test_hca_compress_backward_matches_autograd():
    B, N, c, m = 2, 24, 5, 4
    C, Z = _rand(B, N, c), _rand(B, N, c)
    bias = _rand(m, c)

    CComp, S = R.hca_compress(C, Z, bias)
    g = torch.randn_like(CComp)
    CComp.backward(g)

    dC, dZ, dbias = R.hca_compress_bwd(g, C.detach(), S, CComp.detach())
    torch.testing.assert_close(dC, C.grad, **TOL)
    torch.testing.assert_close(dZ, Z.grad, **TOL)
    torch.testing.assert_close(dbias, bias.grad, **TOL)


def test_csa_compress_backward_matches_autograd():
    B, N, c, m = 2, 24, 5, 4
    Ca, Cb, Za, Zb = (_rand(B, N, c) for _ in range(4))
    ba, bb = _rand(m, c), _rand(m, c)

    CComp, S = R.csa_compress(Ca, Cb, Za, Zb, ba, bb)
    g = torch.randn_like(CComp)
    CComp.backward(g)

    dCa, dCb, dZa, dZb, dba, dbb = R.csa_compress_bwd(
        g, Ca.detach(), Cb.detach(), S, CComp.detach()
    )
    for got, ref in ((dCa, Ca), (dCb, Cb), (dZa, Za), (dZb, Zb)):
        torch.testing.assert_close(got, ref.grad, **TOL)
    torch.testing.assert_close(dba, ba.grad, **TOL)
    torch.testing.assert_close(dbb, bb.grad, **TOL)


def test_csa_compression_shrinks_by_exactly_m():
    """The two streams overlap, but consecutive entries reuse rows.

    Each entry pools ``2m`` source rows, which invites the reading that the
    sequence shrinks by ``2m``. It does not: entry ``i``'s *b*-span is entry
    ``i-1``'s *a*-span, so the stride is still ``m``.
    """
    B, N, c, m = 1, 32, 3, 4
    args = [torch.randn(B, N, c, dtype=D) for _ in range(4)]
    CComp, S = R.csa_compress(*args, torch.zeros(m, c, dtype=D), torch.zeros(m, c, dtype=D))
    assert CComp.shape == (B, N // m, c)
    assert S.shape == (B, N // m, 2 * m, c)


def test_csa_first_block_ignores_its_missing_predecessor():
    """Entry 0's *b* half is padding and must carry exactly zero weight."""
    B, N, c, m = 1, 16, 3, 4
    Ca, Cb, Za, Zb = (torch.randn(B, N, c, dtype=D) for _ in range(4))
    ba, bb = torch.zeros(m, c, dtype=D), torch.zeros(m, c, dtype=D)
    _, S = R.csa_compress(Ca, Cb, Za, Zb, ba, bb)
    torch.testing.assert_close(S[:, 0, m:], torch.zeros_like(S[:, 0, m:]), **TOL)
    # ...and the surviving half is a proper distribution on its own.
    torch.testing.assert_close(
        S[:, 0, :m].sum(dim=1), torch.ones(B, c, dtype=D), **TOL
    )


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------


def test_lightning_index_backward_matches_autograd():
    B, N, nhI, cI, m = 2, 24, 3, 6, 4
    NB = N // m
    qI, KI = _rand(B, N, nhI, cI), _rand(B, NB, cI)
    wI = _rand(B, N, nhI)

    I = R.lightning_index(qI, wI, KI, m)
    g = torch.randn_like(I)
    torch.where(torch.isneginf(I), torch.zeros_like(I), I).backward(g)

    dq, dw, dk = R.lightning_index_bwd(g, qI.detach(), wI.detach(), KI.detach(), m)
    torch.testing.assert_close(dq, qI.grad, **TOL)
    torch.testing.assert_close(dw, wI.grad, **TOL)
    torch.testing.assert_close(dk, KI.grad, **TOL)


def test_lightning_index_head_weights_are_signed():
    """``w * ReLU(q.K)`` is not ``ReLU((w q).K)``.

    Folding the head weight into the query would be a tempting way to turn the
    indexer into one GEMM. It is wrong for negative weights, and the whole point
    of the weights is that they can be negative.
    """
    B, N, nhI, cI, m = 1, 8, 1, 4, 4
    qI = torch.randn(B, N, nhI, cI, dtype=D)
    KI = torch.randn(B, N // m, cI, dtype=D)
    wI = -torch.ones(B, N, nhI, dtype=D)

    correct = R.lightning_index(qI, wI, KI, m)
    folded = R.lightning_index(qI * wI.unsqueeze(-1), torch.ones_like(wI), KI, m)
    live = ~torch.isneginf(correct)
    assert live.any()
    assert not torch.allclose(correct[live], folded[live])


def test_lightning_index_t_offset_matches_a_full_pass():
    """Chunking the query axis must not move the causal boundary.

    The boundary is a function of the token's position in the sequence, not of
    its row in the tensor. A chunk that forgets its offset masks as though it
    started at token 0 -- which only ever *removes* candidates, so the output
    still has the right shape and the loss still goes down.
    """
    B, N, nhI, cI, m = 1, 32, 2, 4, 4
    NB = N // m
    qI = torch.randn(B, N, nhI, cI, dtype=D)
    wI = torch.randn(B, N, nhI, dtype=D)
    KI = torch.randn(B, NB, cI, dtype=D)

    full = R.lightning_index(qI, wI, KI, m)
    chunk = 8
    parts = [
        R.lightning_index(qI[:, t0 : t0 + chunk], wI[:, t0 : t0 + chunk], KI, m, t_offset=t0)
        for t0 in range(0, N, chunk)
    ]
    torch.testing.assert_close(torch.cat(parts, dim=1), full, **TOL)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_topk_pads_rather_than_repeating():
    """A query with too few candidates gets ``-1``, never a duplicate index.

    Clamping a padding slot to 0 would make entry 0 appear twice in the softmax
    and be counted twice -- a silent numerical error, not a crash.
    """
    I = torch.tensor([[[1.0, 5.0, 3.0, float("-inf")], [float("-inf")] * 4]], dtype=D)
    idx = R.topk_select(I, 4)
    assert idx.shape == (1, 2, 4)
    # Three real candidates, so the fourth slot pads; sorted ascending with the
    # pad pushed to the end.
    assert idx[0, 0].tolist() == [0, 1, 2, -1]
    assert idx[0, 1].tolist() == [-1, -1, -1, -1]


def test_topk_accepts_the_kernel_sentinel():
    """Kernels mask with -1e30, not -inf; both must read as 'no candidate'."""
    I = torch.tensor([[[2.0, R.NO_CANDIDATE, 1.0]]], dtype=D)
    assert R.topk_select(I, 3)[0, 0].tolist() == [0, 2, -1]


# ---------------------------------------------------------------------------
# Core attention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tag,use_idx,window",
    [
        ("dense+window", False, 5),
        ("dense,no window", False, 0),
        ("sparse+window", True, 5),
    ],
)
def test_core_attention_backward_matches_autograd(tag, use_idx, window):
    B, N, H, c, m, k = 2, 16, 3, 8, 4, 3
    NB = N // m
    q, kvc = _rand(B, N, H, c), _rand(B, NB, c)
    kvw = _rand(B, N, c) if window else None
    sink = _rand(H)
    scale = c**-0.5

    idx = None
    if use_idx:
        I = R.lightning_index(
            torch.randn(B, N, 2, 4, dtype=D), torch.randn(B, N, 2, dtype=D),
            torch.randn(B, NB, 4, dtype=D), m,
        )
        idx = R.topk_select(I, k)

    o, lse, _ = R.core_attention(
        q, kvc, kvw, sink, idx, m, window, scale, return_stats=True
    )
    do = torch.randn_like(o)
    o.backward(do)

    dq, dkvc, dkvw, dsink = R.core_attention_bwd(
        do, q.detach(), kvc.detach(), None if kvw is None else kvw.detach(),
        sink.detach(), idx, m, window, scale, o.detach(), lse.detach(),
    )
    torch.testing.assert_close(dq, q.grad, **TOL)
    torch.testing.assert_close(dkvc, kvc.grad, **TOL)
    torch.testing.assert_close(dsink, sink.grad, **TOL)
    if kvw is not None:
        torch.testing.assert_close(dkvw, kvw.grad, **TOL)


def test_shared_key_value_gradient_has_both_paths():
    """Each entry is key *and* value, so its gradient is a sum of two terms.

    Dropping the key path leaves a gradient that still points broadly downhill,
    so training does not diverge -- it just quietly converges worse. This pins
    the value path alone as insufficient.
    """
    B, N, H, c, m = 1, 8, 2, 6, 4
    q, kvc = _rand(B, N, H, c), _rand(B, N // m, c)
    sink = torch.zeros(H, dtype=D)
    scale = c**-0.5

    o, lse, _ = R.core_attention(q, kvc, None, sink, None, m, 0, scale, return_stats=True)
    do = torch.randn_like(o)
    o.backward(do)

    _, dkvc, _, _ = R.core_attention_bwd(
        do, q.detach(), kvc.detach(), None, sink, None, m, 0, scale, o.detach(), lse.detach()
    )
    torch.testing.assert_close(dkvc, kvc.grad, **TOL)

    # Value path only, as an ablation: p^T do, without the dz^T q term.
    p = torch.exp(
        torch.einsum("bthd,bsd->bths", q.detach(), kvc.detach()) * scale - lse.unsqueeze(-1)
    )
    t = torch.arange(N)
    s = torch.arange(N // m)
    p = p * (s.view(1, 1, 1, -1) < (t // m).view(1, -1, 1, 1))
    value_only = torch.einsum("bths,bthd->bsd", p, do)
    assert not torch.allclose(value_only, kvc.grad, rtol=1e-3, atol=1e-3)


def test_sink_withholds_attention_mass():
    """A large sink logit should pull the output towards zero.

    The sink joins the denominator but contributes no value, so raising it lets
    a head route mass nowhere. With no sink the probabilities sum to 1.
    """
    B, N, H, c, m = 1, 8, 2, 6, 4
    q = torch.randn(B, N, H, c, dtype=D)
    kvc = torch.randn(B, N // m, c, dtype=D)
    scale = c**-0.5

    small = R.core_attention(q, kvc, None, torch.full((H,), -30.0, dtype=D), None, m, 0, scale)
    large = R.core_attention(q, kvc, None, torch.full((H,), 30.0, dtype=D), None, m, 0, scale)
    assert large.norm() < small.norm() * 1e-6

    # With a negligible sink the row is an ordinary softmax.
    _, lse, _ = R.core_attention(
        q, kvc, None, torch.full((H,), -300.0, dtype=D), None, m, 0, scale,
        return_stats=True,
    )
    z = torch.einsum("bthd,bsd->bths", q, kvc) * scale
    t = torch.arange(N)
    s = torch.arange(N // m)
    allowed = (s.view(1, -1) < (t // m).view(-1, 1)).view(1, N, 1, -1)
    ref = torch.logsumexp(z.masked_fill(~allowed, float("-inf")), dim=-1)
    live = allowed.any(-1).squeeze(-1)
    torch.testing.assert_close(lse[live], ref[live], rtol=1e-9, atol=1e-9)


def test_query_with_no_candidates_outputs_zero():
    """Token 0 can see no compressed block and, without a window, nothing at all.

    The result has to be exactly zero with ``lse`` equal to the sink logit --
    not NaN, which is what a masked-tile maximum of ``-inf`` would produce.
    """
    B, N, H, c, m = 1, 8, 2, 6, 4
    q = torch.randn(B, N, H, c, dtype=D)
    kvc = torch.randn(B, N // m, c, dtype=D)
    sink = torch.tensor([0.5, -1.25], dtype=D)

    o, lse, _ = R.core_attention(q, kvc, None, sink, None, m, 0, c**-0.5, return_stats=True)
    assert torch.isfinite(o).all() and torch.isfinite(lse).all()
    torch.testing.assert_close(o[:, 0], torch.zeros_like(o[:, 0]), **TOL)
    torch.testing.assert_close(lse[:, 0], sink.expand(B, H), **TOL)


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------


def test_partial_rope_leaves_the_nope_lanes_alone():
    x = torch.randn(2, 3, 10, dtype=D)
    cos, sin = R.rope_cos_sin(torch.arange(3), 4, dtype=D)
    y = R.apply_partial_rope(x, cos.unsqueeze(0), sin.unsqueeze(0))
    torch.testing.assert_close(y[..., :6], x[..., :6], **TOL)
    assert not torch.allclose(y[..., 6:], x[..., 6:])


def test_rope_logit_depends_only_on_distance():
    c, rd = 12, 6
    q, kv = torch.randn(1, 1, 1, c, dtype=D), torch.randn(1, 1, 1, c, dtype=D)

    def rope_at(x, p):
        cos, sin = R.rope_cos_sin(torch.tensor([float(p)], dtype=D), rd, dtype=D)
        return R.apply_partial_rope(x, cos.view(1, 1, 1, -1), sin.view(1, 1, 1, -1))

    def logit(t, p):
        return (rope_at(q, t)[..., c - rd :] * rope_at(kv, p)[..., c - rd :]).sum()

    torch.testing.assert_close(logit(5, 2), logit(9, 6), **TOL)
    assert not torch.allclose(logit(5, 2), logit(5, 3))


def test_output_unrotation_recovers_relative_position():
    """The countermeasure of section 2.3.3, in one line.

    An entry cached at absolute position ``p``, un-rotated by the query position
    ``t``, must look exactly like an entry at relative position ``p - t``.
    """
    c, rd = 12, 6
    kv = torch.randn(1, 1, 1, c, dtype=D)

    def rope_at(x, p):
        cos, sin = R.rope_cos_sin(torch.tensor([float(p)], dtype=D), rd, dtype=D)
        return R.apply_partial_rope(x, cos.view(1, 1, 1, -1), sin.view(1, 1, 1, -1))

    got = R.unrope_output(rope_at(kv, 7), torch.tensor([[3]]), rd)
    torch.testing.assert_close(got, rope_at(kv, 4), **TOL)


def test_rope_angle_survives_million_token_positions():
    """The angle is built in fp64 because an fp32 one drifts at long context.

    ``inv_freq`` is not exactly representable in fp32 and the angle scales with
    position, so the error grows as the context does. At the published
    ``rope_dim = 64`` and position 1e6 an all-fp32 pipeline is off by ~0.016 on
    a cosine -- nothing crashes, and no short-context test would catch it.
    """
    rope_dim, theta, half = 64, 10000.0, 32
    pos = torch.tensor([1_000_000])

    cos, _ = R.rope_cos_sin(pos, rope_dim, theta=theta, dtype=torch.float64)

    inv64 = theta ** (-torch.arange(half, dtype=torch.float64) / half)
    exact = (pos.double().unsqueeze(-1) * inv64).cos()
    torch.testing.assert_close(cos, exact, rtol=1e-12, atol=1e-12)

    inv32 = theta ** (-torch.arange(half, dtype=torch.float32) / half)
    all_fp32 = (pos.float().unsqueeze(-1) * inv32).cos()
    assert (all_fp32.double() - exact).abs().max() > 1e-2


def test_block_positions_put_entries_on_the_token_scale():
    """Compressed and window entries share a softmax, so they share a scale."""
    m = 4
    last = R.block_positions(3, m, "last")
    torch.testing.assert_close(last, torch.tensor([3.0, 7.0, 11.0], dtype=D))
    torch.testing.assert_close(
        R.block_positions(3, m, "first"), torch.tensor([0.0, 4.0, 8.0], dtype=D)
    )
    torch.testing.assert_close(
        R.block_positions(3, m, "center"), torch.tensor([1.5, 5.5, 9.5], dtype=D)
    )


def test_grouped_out_proj_matches_the_explicit_two_stage_form():
    B, N, H, c, g, dg, d = 2, 5, 8, 4, 4, 6, 10
    o = torch.randn(B, N, H, c, dtype=D)
    wg = torch.randn(g, (H // g) * c, dg, dtype=D)
    wo = torch.randn(g * dg, d, dtype=D)

    parts = [
        o[:, :, i * (H // g) : (i + 1) * (H // g), :].reshape(B, N, -1) @ wg[i]
        for i in range(g)
    ]
    torch.testing.assert_close(
        R.grouped_out_proj(o, wg, wo), torch.cat(parts, dim=-1) @ wo, **TOL
    )
