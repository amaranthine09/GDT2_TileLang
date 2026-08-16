"""Parse the DSV4 TileLang kernels into TVM PrimFuncs without a GPU.

Numerical validation of the kernels needs CUDA. TVMScript parsing does not, and
it catches a useful class of mistakes on its own: undefined names, buffer slices
that do not match their destination, dtype and arity errors, unsupported
intrinsics, and the shape-dependent guards each kernel raises for itself.

Keep this green on laptops so a CUDA run only has to find genuinely numerical
problems. It is not a substitute for one -- see ``README.md`` on what remains
unverified.
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


def _raw(module: str):
    """Import ``module`` with ``@tilelang.jit`` neutered.

    The decorator defers compilation, so calling a decorated factory would try
    to reach a GPU. Stripping it leaves the plain factory, which still runs the
    entire TVMScript parse and hands back a ``PrimFunc``.
    """
    original = _strip_jit()
    sys.modules.pop(module, None)
    try:
        yield importlib.import_module(module)
    finally:
        tilelang.jit = original
        sys.modules.pop(module, None)


@pytest.fixture(scope="module")
def compress():
    yield from _raw("dsv4.compress")


@pytest.fixture(scope="module")
def indexer():
    yield from _raw("dsv4.indexer")


@pytest.fixture(scope="module")
def core_attn():
    yield from _raw("dsv4.core_attn")


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

HCA_DIMS = dict(B=2, N=256, c=128, m=32)
CSA_DIMS = dict(B=2, N=256, c=128, m=4, block_I=8)


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("hca_compress_kernel", HCA_DIMS),
        ("hca_compress_bwd_kernel", HCA_DIMS),
        ("csa_compress_kernel", CSA_DIMS),
        ("csa_compress_bwd_kernel", CSA_DIMS),
    ],
)
def test_compress_kernels_parse(compress, name, kwargs):
    assert getattr(compress, name)(**kwargs).params


@pytest.mark.parametrize("m", [1, 2, 4, 8, 128])
def test_csa_compress_parses_for_each_rate(compress, m):
    assert compress.csa_compress_kernel(B=1, N=256, c=64, m=m, block_I=4).params


def test_compress_rejects_ragged_sequence(compress):
    with pytest.raises(ValueError, match="multiple of compress_rate"):
        compress.hca_compress_kernel(B=1, N=100, c=64, m=32)


def test_compress_rejects_ragged_channels(compress):
    with pytest.raises(ValueError, match="multiple of block_D"):
        compress.hca_compress_kernel(B=1, N=256, c=100, m=32, block_D=64)


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

IDX_DIMS = dict(B=2, N=256, NB=64, nhI=8, cI=64, block_S=32)


@pytest.mark.parametrize(
    "name",
    ["lightning_index_kernel", "lightning_index_bwd_q_kernel", "lightning_index_bwd_k_kernel"],
)
def test_indexer_kernels_parse(indexer, name):
    assert getattr(indexer, name)(**IDX_DIMS).params


def test_indexer_kernels_take_a_runtime_offset(indexer):
    """``t_off`` must be a runtime scalar, not a compile-time constant.

    Chunking the query axis produces a different offset for every chunk. If it
    were baked into the kernel signature, a long sequence would recompile once
    per chunk.
    """
    pf = indexer.lightning_index_kernel(**IDX_DIMS)
    assert len(pf.params) == 5, "expected (qI, wI, KI, t_off, I)"


@pytest.mark.parametrize("block_S", [16, 32, 64])
def test_indexer_parses_for_each_key_tile(indexer, block_S):
    kw = dict(IDX_DIMS, block_S=block_S)
    assert indexer.lightning_index_kernel(**kw).params


# ---------------------------------------------------------------------------
# Core attention
# ---------------------------------------------------------------------------

ATTN = dict(B=2, N=256, H=16, c=64, NB=64, scale=0.125, block_H=8, block_KV=16)


def test_sparse_attn_kernels_parse(core_attn):
    kw = dict(ATTN, k=16, window=32)
    assert core_attn.sparse_attn_fwd_kernel(**kw).params
    assert core_attn.sparse_attn_bwd_q_kernel(**kw).params


def test_dense_attn_kernels_parse(core_attn):
    kw = dict(ATTN, compress_rate=4, window=32, block_T=2)
    assert core_attn.dense_attn_fwd_kernel(**kw).params
    assert core_attn.dense_attn_bwd_q_kernel(**kw).params
    assert core_attn.dense_attn_bwd_kv_kernel(
        **dict(ATTN, compress_rate=4, block_T=8)
    ).params


@pytest.mark.parametrize("window", [0, 16, 32, 128])
def test_attn_parses_with_and_without_the_window(core_attn, window):
    assert core_attn.sparse_attn_fwd_kernel(**dict(ATTN, k=16, window=window)).params
    assert core_attn.dense_attn_fwd_kernel(
        **dict(ATTN, compress_rate=4, window=window, block_T=2)
    ).params


@pytest.mark.parametrize("block_H", [4, 8, 16])
def test_attn_parses_for_each_head_tile(core_attn, block_H):
    kw = dict(ATTN, k=16, window=32, block_H=block_H)
    assert core_attn.sparse_attn_fwd_kernel(**kw).params


def test_attn_rejects_a_head_count_it_cannot_tile(core_attn):
    with pytest.raises(ValueError, match="multiple of block_H"):
        core_attn.sparse_attn_fwd_kernel(**dict(ATTN, H=12, k=16, block_H=8))


def test_kv_backward_is_not_jit_allocated(core_attn):
    """``dense_attn_bwd_kv_kernel`` accumulates atomically, so it takes its
    destination as a parameter rather than returning one.

    A JIT-allocated output would arrive as uninitialised memory, and adding into
    that produces garbage that still has the right shape and dtype.
    """
    pf = core_attn.dense_attn_bwd_kv_kernel(**dict(ATTN, compress_rate=4, block_T=8))
    assert len(pf.params) == 6, "expected (q, kv_comp, do, lse, delta, dkv_comp)"


def test_csa_backward_is_not_jit_allocated(compress):
    """Stream *b*'s last span is written by no block, so it cannot be allocated.

    Entry ``i`` reads stream *b* rows ``[(i-1)m, im)``. The final ``m`` rows
    would belong to entry ``NB``, which does not exist, so nothing ever assigns
    them. A JIT-allocated output would hand back uninitialised memory there --
    and ``dbias_b`` sums over every row, so the garbage would land in a
    parameter gradient while every shape and dtype stayed correct.
    """
    pf = compress.csa_compress_bwd_kernel(**CSA_DIMS)
    assert len(pf.params) == 12, "expected 8 inputs + 4 caller-provided gradients"


def test_every_documented_kernel_exists(compress, indexer, core_attn):
    """``__all__`` and the module docstrings must not drift from the code."""
    for module in (compress, indexer, core_attn):
        for name in module.__all__:
            assert hasattr(module, name), f"{module.__name__}.{name} is missing"
