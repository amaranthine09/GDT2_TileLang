"""Decoding must agree with the parallel forward, whatever the step size.

The serving path is where a state bug is hardest to notice: prefill looks fine,
then generation drifts after a few hundred tokens. These tests pin the
invariant that actually matters -- decoding ``n`` tokens at once, one at a time,
or as a single parallel forward must all produce the same thing.
"""

from __future__ import annotations

import pytest
import torch

from gdn2 import GatedDeltaNet2, GatedDeltaNet2Config, gdn2_decode, recurrent_gdn2
from tests.test_gdn2 import make_inputs, rel_err

CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA, reason="requires CUDA + TileLang")


@pytest.mark.parametrize("n", [1, 3, 8, 16])
def test_decode_matches_recurrence(n):
    """gdn2_decode is the recurrence, so it must equal the recurrence."""
    q, k, v, g, b, w = make_inputs(2, n, 3, 32, 24)
    s0 = torch.randn(2, 3, 32, 24) * 0.3
    scale = 32**-0.5

    o_ref, s_ref = recurrent_gdn2(q, k, v, g, b, w, scale, s0, output_final_state=True)
    state = s0.clone()
    o = gdn2_decode(q, k, v, g, b, w, state, scale)

    assert rel_err(o, o_ref) < 1e-5
    assert rel_err(state, s_ref) < 1e-5


def test_multi_token_decode_equals_repeated_single_steps():
    """Amortising the state traffic must not change the answer.

    This is the invariant the multi-step kernel exists to preserve: holding the
    state in registers across n tokens has to give exactly what n separate
    launches would.
    """
    q, k, v, g, b, w = make_inputs(2, 12, 3, 32, 24)
    s0 = torch.randn(2, 3, 32, 24) * 0.3
    scale = 32**-0.5

    state_multi = s0.clone()
    o_multi = gdn2_decode(q, k, v, g, b, w, state_multi, scale)

    state_one = s0.clone()
    o_one = torch.cat(
        [gdn2_decode(*(t[:, i : i + 1] for t in (q, k, v, g, b, w)), state_one, scale)
         for i in range(12)],
        dim=1,
    )

    assert rel_err(o_multi, o_one) < 1e-5
    assert rel_err(state_multi, state_one) < 1e-5


@pytest.mark.parametrize("draft", [1, 2, 4, 5])
def test_layer_decode_matches_parallel_forward(draft):
    """Prefill then generate in ``draft``-sized chunks == one parallel forward.

    Covers the short convolution's decode window as well as the recurrent
    state; a draft size that does not divide the sequence exercises the ragged
    tail.
    """
    torch.manual_seed(0)
    cfg = GatedDeltaNet2Config(hidden_size=64, num_heads=2, head_dim_k=16, head_dim_v=16)
    layer = GatedDeltaNet2(cfg).eval()
    x = torch.randn(2, 24, 64)

    with torch.no_grad():
        full, _ = layer(x)
        cache = layer.init_cache(2)
        chunks = [layer.decode(x[:, i : i + draft], cache)[0] for i in range(0, 24, draft)]
    assert rel_err(torch.cat(chunks, dim=1), full) < 1e-4


def test_prefill_then_decode_matches_full_forward():
    """The realistic serving shape: a prompt, then token-by-token generation."""
    torch.manual_seed(0)
    cfg = GatedDeltaNet2Config(hidden_size=64, num_heads=2, head_dim_k=16, head_dim_v=16)
    layer = GatedDeltaNet2(cfg).eval()
    x = torch.randn(1, 20, 64)

    with torch.no_grad():
        full, _ = layer(x)
        prompt, cache = layer(x[:, :12], use_cache=True)
        gen = [layer.step(x[:, t], cache)[0] for t in range(12, 20)]
    stitched = torch.cat([prompt, torch.stack(gen, dim=1)], dim=1)
    assert rel_err(stitched, full) < 1e-4


@cuda_only
@pytest.mark.cuda
@pytest.mark.parametrize("n", [1, 4, 16])
def test_decode_kernel_matches_reference(n):
    q, k, v, g, b, w = make_inputs(2, n, 4, 64, 64, device="cuda")
    s0 = torch.randn(2, 4, 64, 64, device="cuda") * 0.2
    scale = 64**-0.5

    o_ref, s_ref = recurrent_gdn2(q, k, v, g, b, w, scale, s0, output_final_state=True)
    state = s0.clone()
    o = gdn2_decode(q.bfloat16(), k.bfloat16(), v.bfloat16(), g,
                    b.bfloat16(), w.bfloat16(), state, scale)

    assert rel_err(o.float(), o_ref) < 3e-2
    assert rel_err(state, s_ref) < 3e-2


# --------------------------------------------------------------------------
# Prefill
# --------------------------------------------------------------------------


@pytest.mark.parametrize("S", [64, 128, 100, 7])
def test_prefill_matches_recurrence(S):
    """Includes lengths with a ragged tail, which route through decode."""
    from gdn2 import gdn2_prefill

    q, k, v, g, b, w = make_inputs(2, S, 3, 32, 24)
    s0 = torch.randn(2, 3, 32, 24) * 0.3
    scale = 32**-0.5

    o_ref, s_ref = recurrent_gdn2(q, k, v, g, b, w, scale, s0, output_final_state=True)
    o, state = gdn2_prefill(q, k, v, g, b, w, s0.clone(), scale)

    assert o.shape == (2, S, 3, 24)
    assert rel_err(o, o_ref) < 2e-5
    assert rel_err(state, s_ref) < 2e-5


def test_segmented_prefill_equals_one_pass():
    """A long prompt fed in segments must equal prefilling it all at once."""
    from gdn2 import gdn2_prefill

    q, k, v, g, b, w = make_inputs(2, 192, 3, 32, 24)
    scale = 32**-0.5

    o_all, s_all = gdn2_prefill(q, k, v, g, b, w, None, scale)

    outs, state = [], None
    for lo, hi in ((0, 64), (64, 160), (160, 192)):
        o, state = gdn2_prefill(*(t[:, lo:hi] for t in (q, k, v, g, b, w)), state, scale)
        outs.append(o)

    assert rel_err(torch.cat(outs, dim=1), o_all) < 2e-5
    assert rel_err(state, s_all) < 2e-5


def test_layer_prefill_matches_forward():
    """prefill() and forward() differ only in plumbing, not in maths."""
    torch.manual_seed(0)
    cfg = GatedDeltaNet2Config(hidden_size=64, num_heads=2, head_dim_k=16, head_dim_v=16)
    layer = GatedDeltaNet2(cfg).eval()
    x = torch.randn(2, 70, 64)  # ragged: 70 is not a multiple of 64

    with torch.no_grad():
        full, _ = layer(x)
        pre, _ = layer.prefill(x)
    assert rel_err(pre, full) < 1e-4


def test_serving_loop_matches_forward():
    """The whole serving path end to end: prefill, then generate."""
    torch.manual_seed(0)
    cfg = GatedDeltaNet2Config(hidden_size=64, num_heads=2, head_dim_k=16, head_dim_v=16)
    layer = GatedDeltaNet2(cfg).eval()
    x = torch.randn(1, 30, 64)

    with torch.no_grad():
        full, _ = layer(x)
        pre, cache = layer.prefill(x[:, :18])
        draft, cache = layer.decode(x[:, 18:22], cache)
        gen = torch.stack([layer.step(x[:, t], cache)[0] for t in range(22, 30)], dim=1)
    assert rel_err(torch.cat([pre, draft, gen], dim=1), full) < 1e-4


@cuda_only
@pytest.mark.cuda
@pytest.mark.parametrize("S", [512, 1000])
def test_prefill_kernel_matches_reference(S):
    """Fusing the scan into the output must not change the answer.

    S=1000 has a ragged tail, so it also covers the hand-off to the decode
    kernel at the end of a prompt.
    """
    from gdn2 import gdn2_prefill

    q, k, v, g, b, w = make_inputs(2, S, 4, 64, 64, device="cuda")
    s0 = torch.randn(2, 4, 64, 64, device="cuda") * 0.2
    scale = 64**-0.5

    o_ref, s_ref = recurrent_gdn2(q, k, v, g, b, w, scale, s0, output_final_state=True)
    o, state = gdn2_prefill(q.bfloat16(), k.bfloat16(), v.bfloat16(), g,
                            b.bfloat16(), w.bfloat16(), s0.clone(), scale)

    assert rel_err(o.float(), o_ref) < 3e-2
    assert rel_err(state, s_ref) < 3e-2


@cuda_only
@pytest.mark.cuda
def test_prefill_does_not_allocate_chunk_states():
    """The point of the fused kernel: no [B, NC, H, DK, DV] array.

    Compares peak memory against the training forward, which stores it.
    """
    from gdn2 import chunk_gdn2_fwd, gdn2_prefill

    B, S, H, DK, DV = 2, 2048, 8, 128, 128
    q, k, v, g, b, w = make_inputs(B, S, H, DK, DV, device="cuda")
    args = [x.bfloat16() for x in (q, k, v)] + [g] + [x.bfloat16() for x in (b, w)]
    scale = DK**-0.5

    torch.cuda.reset_peak_memory_stats()
    chunk_gdn2_fwd(*args, scale=scale)
    peak_train = torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    gdn2_prefill(*args, scale=scale)
    peak_prefill = torch.cuda.max_memory_allocated()

    h_bytes = B * (S // 64) * H * DK * DV * 2  # bf16 per-chunk states
    assert peak_prefill < peak_train - h_bytes * 0.5, (
        f"expected prefill to save ~{h_bytes / 2**20:.0f} MiB, "
        f"got {(peak_train - peak_prefill) / 2**20:.0f} MiB"
    )
