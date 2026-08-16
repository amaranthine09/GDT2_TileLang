"""Parse the TileLang kernels into TVM PrimFuncs without needing a GPU.

Full kernel validation needs CUDA, but TVMScript parsing catches a useful class
of mistakes on its own -- undefined names, bad buffer slices, dtype and arity
errors, unsupported intrinsics -- and it runs anywhere. Keep this green on
laptops so the CUDA suite only has to find genuinely numerical problems.
"""

from __future__ import annotations

import importlib
import sys

import pytest

tilelang = pytest.importorskip("tilelang")


def _strip_jit():
    original = tilelang.jit
    tilelang.jit = lambda func=None, **kw: (func if callable(func) else (lambda f: f))
    return original


@pytest.fixture(scope="module")
def raw_kernels():
    """``gdn2.forward`` with ``@tilelang.jit`` neutered.

    The decorator defers compilation, so calling a decorated factory would try
    to reach a GPU. Stripping it leaves the plain factory, which still runs the
    entire TVMScript parse and hands back a ``PrimFunc``.
    """
    original = _strip_jit()
    sys.modules.pop("gdn2.forward", None)
    try:
        yield importlib.import_module("gdn2.forward")
    finally:
        tilelang.jit = original
        sys.modules.pop("gdn2.forward", None)


DIMS = dict(B=2, S=256, H=4, DK=64, DV=64)


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("cumsum_kernel", dict(B=2, S=256, H=4, DK=64)),
        ("intra_diag_kernel", dict(B=2, S=256, H=4, DK=64, scale=0.125)),
        ("inter_blocks_kernel", dict(B=2, S=256, H=4, DK=64, scale=0.125)),
        ("solve_kernel", dict(B=2, S=256, H=4)),
        ("wy_kernel", DIMS),
        ("state_kernel", DIMS),
        ("output_kernel", dict(**DIMS, scale=0.125)),
    ],
)
def test_kernel_parses(raw_kernels, name, kwargs):
    prim_func = getattr(raw_kernels, name)(**kwargs)
    assert prim_func.params, f"{name} produced a PrimFunc with no parameters"


@pytest.mark.parametrize("sub_chunk_size", [8, 16, 32])
def test_inter_blocks_parses_for_each_sub_chunk(raw_kernels, sub_chunk_size):
    """The grid is sized by BS // BC, so every ratio must parse."""
    prim_func = raw_kernels.inter_blocks_kernel(
        B=1, S=128, H=2, DK=64, scale=0.125, chunk_size=64, sub_chunk_size=sub_chunk_size
    )
    assert prim_func.params


def test_inter_blocks_rejects_ragged_sub_chunk(raw_kernels):
    with pytest.raises(ValueError, match="divisible"):
        raw_kernels.inter_blocks_kernel(B=1, S=128, H=2, DK=64, scale=0.125, chunk_size=64, sub_chunk_size=24)


@pytest.mark.parametrize("chunk_size", [16, 32, 64, 128])
def test_log_step_scan_schedule_matches_cumsum(chunk_size):
    """Mirror of ``cumsum_kernel``'s scan, checked against ``torch.cumsum``.

    The kernel replaces ``T.cumsum(dim=0)`` with an explicit Hillis-Steele
    sweep, so the offset schedule and the ``t >= off`` predicate are ours to get
    right. This runs the identical recurrence in PyTorch.
    """
    import torch

    x = torch.randn(chunk_size, 7)
    cur = x.clone()
    for step in range(chunk_size.bit_length() - 1):
        off = 1 << step
        prev = cur[(torch.arange(chunk_size) - off).clamp_min(0)]
        mask = (torch.arange(chunk_size) >= off).float().unsqueeze(-1)
        cur = cur + prev * mask
    torch.testing.assert_close(cur, x.cumsum(0), rtol=1e-5, atol=1e-5)


def test_cumsum_kernel_rejects_non_power_of_two_chunk(raw_kernels):
    with pytest.raises(ValueError, match="power of two"):
        raw_kernels.cumsum_kernel(B=1, S=192, H=2, DK=64, chunk_size=48)


def test_inter_blocks_rejects_single_sub_chunk(raw_kernels):
    """BC == BS leaves no off-diagonal work and would launch an empty grid."""
    with pytest.raises(ValueError, match="at least 2 sub-chunks"):
        raw_kernels.inter_blocks_kernel(B=1, S=128, H=2, DK=64, scale=0.125, chunk_size=64, sub_chunk_size=64)


# --------------------------------------------------------------------------
# Inference kernels
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw_kernels_inf():
    """``gdn2.inference`` with ``@tilelang.jit`` neutered."""
    original = _strip_jit()
    sys.modules.pop("gdn2.inference", None)
    try:
        yield importlib.import_module("gdn2.inference")
    finally:
        tilelang.jit = original
        sys.modules.pop("gdn2.inference", None)


def test_decode_step_kernel_parses(raw_kernels_inf):
    assert raw_kernels_inf.decode_step_kernel(B=2, H=4, DK=64, DV=64, scale=0.125).params


@pytest.mark.parametrize("n_steps", [1, 4, 8, 16])
def test_decode_multi_kernel_parses(raw_kernels_inf, n_steps):
    """n_steps is compile-time, so each draft length is its own kernel."""
    assert raw_kernels_inf.decode_multi_kernel(
        B=2, H=4, DK=64, DV=64, n_steps=n_steps, scale=0.125
    ).params


# --------------------------------------------------------------------------
# Backward kernels
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw_kernels_bwd():
    """``gdn2.backward`` with ``@tilelang.jit`` neutered."""
    original = _strip_jit()
    sys.modules.pop("gdn2.backward", None)
    try:
        yield importlib.import_module("gdn2.backward")
    finally:
        tilelang.jit = original
        sys.modules.pop("gdn2.backward", None)


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("dstate_kernel", dict(**DIMS, scale=0.125)),
        ("dwy_a_kernel", DIMS),
        ("daqk_kernel", dict(B=2, S=256, H=4, DV=64)),
        ("dwy_b_kernel", DIMS),
        ("dintra_diag_kernel", dict(B=2, S=256, H=4, DK=64, scale=0.125)),
        ("dinter_row_kernel", dict(B=2, S=256, H=4, DK=64, scale=0.125)),
        ("dinter_col_kernel", dict(B=2, S=256, H=4, DK=64, scale=0.125)),
        ("dassemble_kernel", dict(**DIMS, scale=0.125)),
    ],
)
def test_backward_kernel_parses(raw_kernels_bwd, name, kwargs):
    prim_func = getattr(raw_kernels_bwd, name)(**kwargs)
    assert prim_func.params, f"{name} produced a PrimFunc with no parameters"


@pytest.mark.parametrize("sub_chunk_size", [8, 16, 32])
def test_backward_inter_kernels_parse_for_each_sub_chunk(raw_kernels_bwd, sub_chunk_size):
    for name in ("dinter_row_kernel", "dinter_col_kernel"):
        prim_func = getattr(raw_kernels_bwd, name)(
            B=1, S=128, H=2, DK=64, scale=0.125, chunk_size=64, sub_chunk_size=sub_chunk_size
        )
        assert prim_func.params
