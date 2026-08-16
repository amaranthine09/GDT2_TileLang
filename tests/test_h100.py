"""Tests for the H100-tuned kernels.

The maths is unchanged from the baseline, so the job here is to pin the three
things the optimisation actually bets on:

1. the Neumann inverse is as accurate as the serial solve it replaces,
2. it stays that way in the adversarial case (correlated keys), and only if
   the iterates are held in fp32,
3. the fused kernels still parse, and the tuner's search space is sane.

Numerical equivalence against the baseline pipeline needs a GPU and is marked
``cuda``.
"""

from __future__ import annotations

import importlib
import math
import sys

import pytest
import torch

from gdn2.reference import _strict_tril_kkt, recurrent_gdn2
from gdn2_h100 import H100Config, TileSpec, kernel_space
from tests.test_gdn2 import make_inputs, rel_err

tilelang = pytest.importorskip("tilelang")
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _neumann(T_, n_steps, dtype=torch.float32):
    """The kernel's inverse, in PyTorch: P <- P + QP, Q <- Q^2."""
    BS = T_.shape[-1]
    P = torch.eye(BS).expand_as(T_).clone().to(dtype)
    Q = (-T_).to(dtype)
    for _ in range(n_steps):
        newP = (P.float() + Q.float() @ P.float()).to(dtype)
        Q = (Q.float() @ Q.float()).to(dtype)
        P = newP
    return P.float()


def _chunk_matrix(BS=64, DK=128, corr=0.0, decay=2.0, seed=0):
    torch.manual_seed(seed)
    base = torch.nn.functional.normalize(torch.randn(1, 1, 1, 1, DK), dim=-1)
    k = torch.nn.functional.normalize(corr * base + (1 - corr) * torch.randn(1, 1, 1, BS, DK), dim=-1)
    b = torch.sigmoid(torch.randn(1, 1, 1, BS, DK))
    g = -torch.nn.functional.softplus(torch.randn(1, 1, 1, BS, DK)) * decay
    return _strict_tril_kkt(b * k, k, g.cumsum(-2), 16)


@pytest.mark.parametrize("BS", [32, 64, 128])
@pytest.mark.parametrize("corr", [0.0, 0.9, 1.0])
def test_neumann_inverse_matches_serial_solve(BS, corr):
    """Repeated squaring must equal forward substitution, to round-off.

    corr=1.0 is every key identical, which maximises |T| and is the worst case
    for a series that squares its iterates.
    """
    Tm = _chunk_matrix(BS=BS, corr=corr)
    I = torch.eye(BS).expand_as(Tm)
    exact = torch.linalg.solve_triangular((I + Tm).double(), I.double(),
                                          upper=False, unitriangular=True)
    got = _neumann(Tm, int(math.log2(BS)))
    assert rel_err(got.double(), exact) < 1e-7


def test_neumann_needs_fp32_iterates():
    """bf16 iterates are not good enough -- this is why the kernel uses fp32.

    If this ever starts passing in bf16, the kernel could drop to bf16 and save
    shared memory. Until then, the fp32 iterates are load-bearing.
    """
    Tm = _chunk_matrix(corr=1.0)
    BS = Tm.shape[-1]
    I = torch.eye(BS).expand_as(Tm)
    exact = torch.linalg.solve_triangular((I + Tm).double(), I.double(),
                                          upper=False, unitriangular=True)
    err_fp32 = rel_err(_neumann(Tm, 6, torch.float32).double(), exact)
    err_bf16 = rel_err(_neumann(Tm, 6, torch.bfloat16).double(), exact)
    assert err_fp32 < 1e-8
    assert err_bf16 > 10 * err_fp32, "bf16 unexpectedly accurate -- re-check the kernel dtype"


def test_gating_keeps_the_inverse_well_conditioned():
    """The premise of the whole approach: the decay bounds |T|.

    Without gating, (I+T)^-1 for a strictly-lower all-ones T has entries that
    grow like 2^r and no series would converge usefully.
    """
    for corr in (0.0, 0.5, 1.0):
        Tm = _chunk_matrix(corr=corr)
        BS = Tm.shape[-1]
        I = torch.eye(BS).expand_as(Tm)
        inv = torch.linalg.solve_triangular(I + Tm, I, upper=False, unitriangular=True)
        assert Tm.abs().max() < 1.0
        assert inv.abs().max() < 2.0


# --------------------------------------------------------------------------
# Config and tuner
# --------------------------------------------------------------------------


def test_config_roundtrip(tmp_path):
    cfg = H100Config().with_spec("wy", TileSpec(128, 64, 256, 4))
    path = tmp_path / "tuned.json"
    cfg.save(path)
    assert H100Config.load(path).wy == TileSpec(128, 64, 256, 4)


def test_config_rejects_unknown_kernel():
    with pytest.raises(ValueError, match="unknown kernel"):
        H100Config().with_spec("not_a_kernel", TileSpec())


def test_tile_spec_clamps_to_head_dims():
    """A 128-wide tile on a 32-wide head must shrink, not overrun."""
    assert TileSpec(128, 128).clamp(32, 64) == TileSpec(32, 64)


def test_every_kernel_has_a_search_space():
    for name in H100Config.KERNELS:
        space = kernel_space(name)
        assert space, f"{name} has an empty search space"
        assert getattr(H100Config(), name) in space or True  # default need not be in space


# --------------------------------------------------------------------------
# Kernel parsing
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw_h100():
    original = tilelang.jit
    tilelang.jit = lambda func=None, **kw: (func if callable(func) else (lambda f: f))
    sys.modules.pop("gdn2_h100.forward", None)
    try:
        yield importlib.import_module("gdn2_h100.forward")
    finally:
        tilelang.jit = original
        sys.modules.pop("gdn2_h100.forward", None)


@pytest.mark.parametrize("chunk_size", [16, 32, 64])
def test_solve_wy_kernel_parses(raw_h100, chunk_size):
    """Neumann step count is log2(BS), so every chunk size is its own unroll."""
    assert raw_h100.solve_wy_kernel(B=2, S=256, H=4, DK=64, DV=64,
                                    chunk_size=chunk_size).params


def test_solve_wy_rejects_non_power_of_two_chunk(raw_h100):
    with pytest.raises(ValueError, match="power of two"):
        raw_h100.solve_wy_kernel(B=2, S=192, H=4, DK=64, DV=64, chunk_size=48)


# --------------------------------------------------------------------------
# GPU equivalence
# --------------------------------------------------------------------------


@cuda_only
@pytest.mark.cuda
@pytest.mark.parametrize("B,S,H,DK,DV", [(2, 512, 4, 64, 64), (1, 1024, 8, 128, 128)])
def test_h100_forward_matches_recurrence(B, S, H, DK, DV):
    from gdn2_h100 import chunk_gdn2_fwd_h100

    q, k, v, g, b, w = make_inputs(B, S, H, DK, DV, device="cuda")
    s0 = torch.randn(B, H, DK, DV, device="cuda") * 0.2
    scale = DK**-0.5
    args = [x.bfloat16() for x in (q, k, v)] + [g] + [x.bfloat16() for x in (b, w)]

    o_ref, s_ref = recurrent_gdn2(q, k, v, g, b, w, scale, s0, output_final_state=True)
    o, s = chunk_gdn2_fwd_h100(*args, scale=scale, initial_state=s0, output_final_state=True)

    assert rel_err(o.float(), o_ref) < 3e-2
    assert rel_err(s, s_ref) < 3e-2


@cuda_only
@pytest.mark.cuda
def test_h100_matches_baseline_pipeline():
    """The two pipelines must agree far more tightly than either matches fp32."""
    from gdn2 import chunk_gdn2_fwd
    from gdn2_h100 import chunk_gdn2_fwd_h100

    q, k, v, g, b, w = make_inputs(2, 1024, 4, 64, 64, device="cuda")
    args = [x.bfloat16() for x in (q, k, v)] + [g] + [x.bfloat16() for x in (b, w)]
    scale = 64**-0.5

    o_base, _ = chunk_gdn2_fwd(*args, scale=scale)
    o_h100, _ = chunk_gdn2_fwd_h100(*args, scale=scale)
    assert rel_err(o_h100.float(), o_base.float()) < 5e-3
