"""Layer-level behaviour of the DSV4 attention modules, on CPU.

These run through the PyTorch backend, so they exercise the maths and the
plumbing but not the kernels -- see ``test_dsv4_kernel_parse.py`` for what can
be checked of those without a GPU.

The properties here are the ones that are cheap to get subtly wrong and
expensive to notice: causality, the equivalence of the chunked selection path to
the unchunked one, and the degeneracies that should collapse one variant onto
another.
"""

from __future__ import annotations

import pytest
import torch

from dsv4 import (
    CSAAttention,
    CSAConfig,
    DSV4Config,
    HCAAttention,
    HCAConfig,
    LayerKind,
    SWAAttention,
    V4_FLASH,
    V4_PRO,
    build_attention_stack,
)
from dsv4 import ops
from dsv4 import reference as R

D = torch.float64


def csa_cfg(**kw):
    base = dict(
        hidden_size=64, head_dim=16, num_heads=4, q_lora_rank=32, rope_dim=8,
        window_size=8, num_out_groups=2, out_group_dim=16,
        compress_rate=4, top_k=3, indexer_heads=2, indexer_dim=8,
    )
    base.update(kw)
    return CSAConfig(**base)


def hca_cfg(**kw):
    base = dict(
        hidden_size=64, head_dim=16, num_heads=4, q_lora_rank=32, rope_dim=8,
        window_size=8, num_out_groups=2, out_group_dim=16, compress_rate=8,
    )
    base.update(kw)
    return HCAConfig(**base)


def _layer(cls, cfg, seed=0):
    torch.manual_seed(seed)
    return cls(cfg).to(D)


# ---------------------------------------------------------------------------
# Shapes and gradients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,cfg",
    [(CSAAttention, csa_cfg()), (HCAAttention, hca_cfg()), (SWAAttention, hca_cfg())],
)
def test_layer_forward_and_backward(cls, cfg):
    layer = _layer(cls, cfg)
    x = torch.randn(2, 32, cfg.hidden_size, dtype=D, requires_grad=True)

    out = layer(x)
    assert out.shape == (2, 32, cfg.hidden_size)
    assert torch.isfinite(out).all()

    out.square().mean().backward()
    assert torch.isfinite(x.grad).all()

    indexer = {
        "w_aki", "w_bki", "w_azi", "w_bzi", "bias_ai", "bias_bi", "w_iuq", "w_w",
    }
    for name, p in layer.named_parameters():
        if name in indexer:
            continue  # see test_indexer_gets_no_gradient_from_the_output
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} gradient is not finite"


def test_indexer_gets_no_gradient_from_the_output():
    """Top-k is a hard gather, so the main loss cannot train the indexer.

    This is the architecture, not an oversight -- but it is a trap: an optimiser
    built over ``layer.parameters()`` will silently leave the indexer at its
    initialisation, the model will still train, and it will simply be selecting
    compressed entries at random. The fix is an auxiliary loss; the point of
    pinning it here is that the failure is invisible otherwise.
    """
    cfg = csa_cfg()
    layer = _layer(CSAAttention, cfg)
    x = torch.randn(2, 32, cfg.hidden_size, dtype=D)
    layer(x).square().mean().backward()

    indexer = ["w_aki", "w_bki", "w_azi", "w_bzi", "bias_ai", "bias_bi", "w_iuq", "w_w"]
    for name in indexer:
        assert getattr(layer, name).grad is None, f"{name} unexpectedly got a gradient"

    # The auxiliary loss is what reaches them.
    layer.zero_grad(set_to_none=True)
    pos = torch.arange(32)
    _, cq = layer._queries(x, pos)
    scores = layer.index_scores(x, cq, pos)
    target = torch.rand(2, 32, scores.shape[-1], dtype=D)
    ops.indexer_kl_loss(scores, target, cfg.compress_rate).backward()
    for name in indexer:
        g = getattr(layer, name).grad
        assert g is not None and torch.isfinite(g).all(), f"{name} still untrained"


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,cfg",
    [(CSAAttention, csa_cfg()), (HCAAttention, hca_cfg()), (SWAAttention, hca_cfg())],
)
def test_output_is_strictly_causal(cls, cfg):
    """Perturbing token ``j`` must leave every output before ``j`` bit-identical.

    Compressed attention has three places causality can leak, and all of them
    are off-by-one shaped: a query may read compressed blocks ``s < t // m``
    strictly (its *own* block pools tokens at and after it), the CSA *b* stream
    reaches one span further back than the *a* stream, and the window is
    inclusive of ``t`` at one end only. Selection is data-dependent on top of
    that. Comparing exactly, not approximately, is deliberate -- a leak of one
    token would still look tiny under a tolerance.
    """
    layer = _layer(cls, cfg)
    torch.manual_seed(1)
    x = torch.randn(1, 32, cfg.hidden_size, dtype=D)

    base = layer(x)
    for j in (5, 12, 20, 31):
        bumped = x.clone()
        bumped[:, j] += 1.0
        got = layer(bumped)
        assert torch.equal(got[:, :j], base[:, :j]), f"future token {j} leaked backwards"
        assert not torch.allclose(got[:, j], base[:, j]), f"token {j} ignored its own input"


def test_compressed_entry_never_summarises_the_future():
    """Entry ``s`` may only be read by queries strictly past its newest token.

    Entry ``s`` pools tokens up to ``m(s+1) - 1``. A query at ``t`` reads
    ``s < t // m``, so the newest token it can reach through the compressed
    branch is ``m * (t // m) - 1 <= t - 1``. The window is what gives it access
    to its own neighbourhood.
    """
    N, m = 32, 4
    for t in range(N):
        for s in range(N // m):
            reachable = s < t // m
            newest_token_in_block = m * (s + 1) - 1
            if reachable:
                assert newest_token_in_block < t


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_chunked_selection_matches_a_single_pass():
    """``index_and_select`` must agree with scoring the whole sequence at once."""
    torch.manual_seed(2)
    B, N, nhI, cI, m, k = 1, 64, 2, 8, 4, 5
    NB = N // m
    qI = torch.randn(B, N, nhI, cI, dtype=D)
    wI = torch.randn(B, N, nhI, dtype=D)
    KI = torch.randn(B, NB, cI, dtype=D)

    whole = R.topk_select(R.lightning_index(qI, wI, KI, m), k)
    for chunk in (8, 16, 64, 128):
        got = ops.index_and_select(qI, wI, KI, m, k, chunk=chunk, backend="torch")
        assert torch.equal(got, whole), f"chunk={chunk} disagreed"


def test_selecting_everything_reproduces_dense_attention():
    """With ``k >= NB``, CSA must collapse exactly onto HCA's core attention.

    Top-k then keeps every legal candidate and pads the rest, so the sparse path
    and the dense path are attending to the same set. If padding slots were
    being counted -- or clamped onto entry 0 -- this is where it shows.
    """
    torch.manual_seed(3)
    B, N, H, c, m = 2, 24, 3, 8, 4
    NB = N // m
    q = torch.randn(B, N, H, c, dtype=D)
    kv = torch.randn(B, NB, c, dtype=D)
    win = torch.randn(B, N, c, dtype=D)
    sink = torch.randn(H, dtype=D)
    scale = c**-0.5

    full = torch.full((B, N, NB), 0.0, dtype=D)
    idx = R.topk_select(R.lightning_index(
        torch.randn(B, N, 2, 4, dtype=D), torch.randn(B, N, 2, dtype=D),
        torch.randn(B, NB, 4, dtype=D), m,
    ), NB)

    sparse = R.core_attention(q, kv, win, sink, idx, m, 6, scale)
    dense = R.core_attention(q, kv, win, sink, None, m, 6, scale)
    torch.testing.assert_close(sparse, dense, rtol=1e-11, atol=1e-11)


def test_zero_window_removes_the_local_branch():
    """``window_size = 0`` is the compressed-only ablation, not a crash."""
    cfg = hca_cfg(window_size=0)
    layer = _layer(HCAAttention, cfg)
    out = layer(torch.randn(1, 32, cfg.hidden_size, dtype=D))
    assert out.shape == (1, 32, cfg.hidden_size)
    assert torch.isfinite(out).all()


def test_swa_layer_rejects_a_shared_kv_projection():
    with pytest.raises(ValueError, match="no compression stream"):
        SWAAttention(hca_cfg(window_shares_kv_proj=True))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_published_stacks_interleave_as_documented():
    """V4-Flash: two SWA then CSA/HCA alternating. V4-Pro: two HCA then the same."""
    flash = [V4_FLASH.layer_kind(i) for i in range(V4_FLASH.num_layers)]
    assert flash[:2] == [LayerKind.SWA, LayerKind.SWA]
    assert flash[2:8] == [
        LayerKind.CSA, LayerKind.HCA, LayerKind.CSA,
        LayerKind.HCA, LayerKind.CSA, LayerKind.HCA,
    ]
    assert len(flash) == 43

    pro = [V4_PRO.layer_kind(i) for i in range(V4_PRO.num_layers)]
    assert pro[:2] == [LayerKind.HCA, LayerKind.HCA]
    assert pro[2] == LayerKind.CSA
    assert len(pro) == 61


def test_published_hyperparameters_match_the_report():
    assert (V4_FLASH.hidden_size, V4_FLASH.num_layers) == (4096, 43)
    assert (V4_FLASH.csa.compress_rate, V4_FLASH.csa.top_k) == (4, 512)
    assert (V4_FLASH.csa.indexer_heads, V4_FLASH.csa.indexer_dim) == (64, 128)
    assert V4_FLASH.hca.compress_rate == 128
    assert (V4_FLASH.csa.num_heads, V4_FLASH.csa.head_dim) == (64, 512)
    assert V4_FLASH.csa.q_lora_rank == 1024
    assert (V4_FLASH.csa.num_out_groups, V4_FLASH.csa.out_group_dim) == (8, 1024)
    assert V4_FLASH.csa.window_size == 128

    assert (V4_PRO.hidden_size, V4_PRO.num_layers) == (7168, 61)
    assert V4_PRO.csa.top_k == 1024
    assert (V4_PRO.csa.num_heads, V4_PRO.csa.q_lora_rank) == (128, 1536)
    assert (V4_PRO.csa.num_out_groups, V4_PRO.csa.out_group_dim) == (16, 1024)
    assert V4_PRO.csa.rope_dim == 64 and V4_PRO.hca.rope_dim == 64


def test_grouped_output_projection_is_cheaper_than_a_dense_one():
    """The reason the two-stage projection exists, as arithmetic.

    A single ``[c * n_h, d]`` matrix would dominate the layer.
    """
    cfg = V4_FLASH.csa
    dense = cfg.head_dim * cfg.num_heads * cfg.hidden_size
    g, dg = cfg.num_out_groups, cfg.out_group_dim
    grouped = g * (cfg.head_dim * cfg.num_heads // g) * dg + g * dg * cfg.hidden_size
    assert grouped < dense / 1.9


def test_layer_config_validation():
    with pytest.raises(ValueError, match="rope_block_pos"):
        csa_cfg(rope_block_pos="middle")
    with pytest.raises(ValueError, match="divisible"):
        csa_cfg(num_heads=4, num_out_groups=3)
    with pytest.raises(ValueError, match="must be even"):
        csa_cfg(rope_dim=7)
    with pytest.raises(ValueError, match="exceeds head_dim"):
        csa_cfg(rope_dim=32, head_dim=16)
    with pytest.raises(IndexError):
        V4_FLASH.layer_kind(999)


def test_build_attention_stack_matches_the_config():
    cfg = DSV4Config(
        num_layers=6, hidden_size=64, csa=csa_cfg(), hca=hca_cfg(),
        num_prefix_layers=2, prefix_kind=LayerKind.SWA,
    )
    stack = build_attention_stack(cfg)
    assert len(stack) == 6
    assert [type(m).__name__ for m in stack] == [
        "SWAAttention", "SWAAttention",
        "CSAAttention", "HCAAttention", "CSAAttention", "HCAAttention",
    ]


def test_sequence_must_divide_the_compression_rate():
    cfg = hca_cfg(compress_rate=8)
    layer = _layer(HCAAttention, cfg)
    with pytest.raises(ValueError, match="multiple of compress_rate"):
        layer(torch.randn(1, 20, cfg.hidden_size, dtype=D))


def test_backend_resolution():
    x = torch.zeros(2, 2)
    assert ops.resolve_backend("torch", x) == "torch"
    assert ops.resolve_backend("auto", x) == "torch"  # CPU tensors
    with pytest.raises(ValueError, match="needs CUDA"):
        ops.resolve_backend("tilelang", x)
    with pytest.raises(ValueError, match="unknown backend"):
        ops.resolve_backend("cuda", x)


def test_indexer_kl_loss_is_zero_on_a_perfect_indexer():
    """The auxiliary objective bottoms out when the indexer already agrees."""
    torch.manual_seed(4)
    B, N, NB, m = 1, 16, 4, 4
    I = torch.randn(B, N, NB, dtype=D, requires_grad=True)

    t = torch.arange(N)
    s = torch.arange(NB)
    allowed = ((s.unsqueeze(0) + 1) * m <= t.unsqueeze(1)).unsqueeze(0)
    target = torch.softmax(I.detach().masked_fill(~allowed, float("-inf")), dim=-1)
    target = torch.nan_to_num(target)

    loss = ops.indexer_kl_loss(I, target, m)
    assert loss.abs() < 1e-9

    # And it is positive, differentiable, and finite when they disagree.
    other = torch.rand(B, N, NB, dtype=D)
    loss2 = ops.indexer_kl_loss(I, other, m)
    assert loss2 > 0 and torch.isfinite(loss2)
    loss2.backward()
    assert torch.isfinite(I.grad).all()
