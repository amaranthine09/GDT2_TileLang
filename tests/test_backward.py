"""The chunkwise backward must match autograd through the chunkwise forward.

``gdn2.backward.chunk_gdn2_bwd_torch`` is the specification the TileLang
backward kernels implement, staged the same way. Checking it against autograd
here validates the derivation -- the inverse-of-triangular term, the reverse
state scan, the shared ``pairwise_bwd``, and every path that touches the decay
-- on CPU, before any kernel runs.
"""

from __future__ import annotations

import pytest
import torch

from gdn2.reference import chunk_gdn2_bwd_torch
from gdn2.reference import chunk_gdn2_torch
from tests.test_gdn2 import make_inputs, rel_err

NAMES = ["dq", "dk", "dv", "dg", "db", "dw", "dS0"]


def autograd_reference(inputs, s0, do, dht, BS, BC, scale):
    ts = [t.clone().requires_grad_(True) for t in inputs]
    s0t = None if s0 is None else s0.clone().requires_grad_(True)
    o, sf = chunk_gdn2_torch(*ts, scale, s0t, dht is not None, chunk_size=BS, sub_chunk_size=BC)
    loss = (o * do).sum()
    if dht is not None:
        loss = loss + (sf * dht).sum()
    loss.backward()
    return [t.grad for t in ts] + [None if s0t is None else s0t.grad]


@pytest.mark.parametrize("BS,BC", [(64, 16), (64, 32), (32, 16)])
def test_backward_matches_autograd(BS, BC):
    q, k, v, g, b, w = make_inputs(2, 2 * BS, 3, 32, 24)
    scale = 32**-0.5
    s0 = torch.randn(2, 3, 32, 24) * 0.3
    do = torch.randn(2, 2 * BS, 3, 24)
    dht = torch.randn(2, 3, 32, 24) * 0.5

    ref = autograd_reference((q, k, v, g, b, w), s0, do, dht, BS, BC, scale)
    got = chunk_gdn2_bwd_torch(q, k, v, g, b, w, scale, s0, do, dht, BS, BC)

    for name, a, r in zip(NAMES, got, ref):
        assert rel_err(a, r) < 1e-4, f"{name}: {rel_err(a, r):.2e}"


def test_backward_without_state_grads():
    """No initial state and no final-state gradient is the common training case."""
    q, k, v, g, b, w = make_inputs(2, 128, 2, 32, 32)
    scale = 32**-0.5
    do = torch.randn(2, 128, 2, 32)

    ref = autograd_reference((q, k, v, g, b, w), None, do, None, 64, 16, scale)
    got = chunk_gdn2_bwd_torch(q, k, v, g, b, w, scale, None, do, None, 64, 16)

    for name, a, r in zip(NAMES[:-1], got[:-1], ref[:-1]):
        assert rel_err(a, r) < 1e-4, f"{name}: {rel_err(a, r):.2e}"


def test_backward_survives_fast_decay():
    """No stage may reconstruct 1/gamma, in the backward any more than forward."""
    q, k, v, g, b, w = make_inputs(1, 128, 2, 32, 32, decay_scale=25.0)
    scale = 32**-0.5
    do = torch.randn(1, 128, 2, 32)

    ref = autograd_reference((q, k, v, g, b, w), None, do, None, 64, 16, scale)
    got = chunk_gdn2_bwd_torch(q, k, v, g, b, w, scale, None, do, None, 64, 16)

    for name, a, r in zip(NAMES[:-1], got[:-1], ref[:-1]):
        assert torch.isfinite(a).all(), f"{name} produced non-finite gradients"
        assert rel_err(a, r) < 1e-4, f"{name}: {rel_err(a, r):.2e}"
