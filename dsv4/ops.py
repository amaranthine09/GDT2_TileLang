"""Differentiable ops: autograd wiring over the kernels, with a CPU fallback.

Each stage of the pipeline is exposed as a single differentiable function that
dispatches on ``backend``:

``"tilelang"``
    The kernels in :mod:`dsv4.compress`, :mod:`dsv4.indexer` and
    :mod:`dsv4.core_attn`, wrapped in :class:`torch.autograd.Function` so the
    hand-written backward is what autograd calls.
``"torch"``
    The oracle in :mod:`dsv4.reference`, used directly. It is written in
    ordinary differentiable PyTorch, so it needs no ``Function`` at all --
    autograd differentiates it, and that is exactly what makes it a valid check
    on the hand-written backward.
``"auto"`` (default)
    ``"tilelang"`` when the tensors are on CUDA and TileLang imported,
    ``"torch"`` otherwise. A whole layer therefore runs on a laptop, which is
    what lets the maths be tested without a GPU.

The top-k selection is not differentiable in either backend: it is a hard
gather, and no gradient flows through the choice of *which* entries were
selected. Gradients reach the indexer only through whatever auxiliary loss
trains it -- see :func:`indexer_kl_loss`.
"""

from __future__ import annotations

import torch

from . import reference as R
from .config import DEFAULT_KERNEL_CONFIG, KernelConfig

__all__ = [
    "resolve_backend",
    "csa_compress",
    "hca_compress",
    "lightning_index",
    "topk_select",
    "index_and_select",
    "sparse_attention",
    "dense_attention",
    "indexer_kl_loss",
]

_TILELANG_OK: bool | None = None


def _tilelang_available() -> bool:
    """Whether the TileLang kernels can be imported at all."""
    global _TILELANG_OK
    if _TILELANG_OK is None:
        try:  # pragma: no cover - depends on the install
            from . import compress, core_attn, indexer  # noqa: F401

            _TILELANG_OK = True
        except Exception:
            _TILELANG_OK = False
    return _TILELANG_OK


def resolve_backend(backend: str, *tensors: torch.Tensor) -> str:
    """Resolve ``"auto"`` against the tensors' device and the install.

    Args:
        backend: ``"auto"``, ``"tilelang"`` or ``"torch"``.
        *tensors: the tensors the op will run on.

    Returns:
        ``"tilelang"`` or ``"torch"``.

    Raises:
        ValueError: for an unknown name, or for ``"tilelang"`` on CPU tensors.
    """
    if backend == "torch":
        return "torch"
    on_cuda = all(t.is_cuda for t in tensors if t is not None)
    if backend == "tilelang":
        if not on_cuda:
            raise ValueError("backend='tilelang' needs CUDA tensors")
        return "tilelang"
    if backend != "auto":
        raise ValueError(f"unknown backend {backend!r}; expected auto/tilelang/torch")
    return "tilelang" if on_cuda and _tilelang_available() else "torch"


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


class _HCACompress(torch.autograd.Function):
    """HCA pooling with the hand-written backward."""

    @staticmethod
    def forward(ctx, C, Z, bias, config):
        from .compress import hca_compress_fwd

        CComp = hca_compress_fwd(C, Z, bias, config)
        ctx.save_for_backward(C, Z, bias, CComp)
        ctx.config = config
        return CComp

    @staticmethod
    def backward(ctx, dCComp):
        from .compress import hca_compress_bwd

        C, Z, bias, CComp = ctx.saved_tensors
        dC, dZ, dbias = hca_compress_bwd(C, Z, bias, CComp, dCComp, ctx.config)
        return dC, dZ, dbias, None


class _CSACompress(torch.autograd.Function):
    """CSA overlapped pooling with the hand-written backward."""

    @staticmethod
    def forward(ctx, Ca, Cb, Za, Zb, bias_a, bias_b, config):
        from .compress import csa_compress_fwd

        CComp = csa_compress_fwd(Ca, Cb, Za, Zb, bias_a, bias_b, config)
        ctx.save_for_backward(Ca, Cb, Za, Zb, bias_a, bias_b, CComp)
        ctx.config = config
        return CComp

    @staticmethod
    def backward(ctx, dCComp):
        from .compress import csa_compress_bwd

        Ca, Cb, Za, Zb, bias_a, bias_b, CComp = ctx.saved_tensors
        grads = csa_compress_bwd(
            Ca, Cb, Za, Zb, bias_a, bias_b, CComp, dCComp, ctx.config
        )
        return (*grads, None)


def hca_compress(C, Z, bias, backend="auto", config: KernelConfig = DEFAULT_KERNEL_CONFIG):
    """Differentiable HCA pooling (eq. 22/23).

    Args:
        C, Z: ``[B, N, c]``.
        bias: ``[m', c]``.
        backend: see :func:`resolve_backend`.
        config: tiling configuration.

    Returns:
        ``[B, N // m', c]``.
    """
    if resolve_backend(backend, C, Z) == "torch":
        return R.hca_compress(C, Z, bias)[0]
    return _HCACompress.apply(C, Z, bias, config)


def csa_compress(
    Ca, Cb, Za, Zb, bias_a, bias_b,
    backend="auto", config: KernelConfig = DEFAULT_KERNEL_CONFIG,
):
    """Differentiable CSA overlapped pooling (eq. 11/12).

    Returns:
        ``[B, N // m, c]``.
    """
    if resolve_backend(backend, Ca, Cb, Za, Zb) == "torch":
        return R.csa_compress(Ca, Cb, Za, Zb, bias_a, bias_b)[0]
    return _CSACompress.apply(Ca, Cb, Za, Zb, bias_a, bias_b, config)


# ---------------------------------------------------------------------------
# Indexer and selection
# ---------------------------------------------------------------------------


class _LightningIndex(torch.autograd.Function):
    """Index scores with the hand-written split backward."""

    @staticmethod
    def forward(ctx, qI, wI, KI, compress_rate, t_offset, config):
        from .indexer import lightning_index_fwd

        I = lightning_index_fwd(qI, wI, KI, compress_rate, t_offset, config)
        ctx.save_for_backward(qI, wI, KI)
        ctx.compress_rate = compress_rate
        ctx.t_offset = t_offset
        ctx.config = config
        return I

    @staticmethod
    def backward(ctx, dI):
        from .indexer import lightning_index_bwd

        qI, wI, KI = ctx.saved_tensors
        dqI, dwI, dKI = lightning_index_bwd(
            dI, qI, wI, KI, ctx.compress_rate, ctx.t_offset, ctx.config
        )
        return dqI, dwI, dKI, None, None, None


def lightning_index(
    qI, wI, KI, compress_rate, t_offset: int = 0,
    backend="auto", config: KernelConfig = DEFAULT_KERNEL_CONFIG,
):
    """Differentiable lightning-indexer scores (eq. 16).

    Returns:
        ``[B, N, NB]``; masked positions are strongly negative, not ``-inf``,
        so downstream arithmetic cannot produce NaN.
    """
    if resolve_backend(backend, qI, wI, KI) == "torch":
        return R.lightning_index(qI, wI, KI, compress_rate, t_offset=t_offset)
    return _LightningIndex.apply(qI, wI, KI, compress_rate, t_offset, config)


def topk_select(I: torch.Tensor, k: int, config: KernelConfig = DEFAULT_KERNEL_CONFIG):
    """Top-k selection over index scores (eq. 17). Not differentiable.

    Args:
        I: ``[B, N, NB]`` scores.
        k: entries to keep.
        config: accepted for signature symmetry; unused.

    Returns:
        ``[B, N, k]`` int32 indices, ``-1`` padded, sorted ascending.

    Note:
        This is :func:`torch.topk`, deliberately. Unlike the attention and
        pooling stages, selection has a well-optimised library implementation
        already, and it is not on the critical path: with ``NB`` entries and
        ``k`` kept, selection is O(NB) per row against the O(k * c * n_h) the
        attention then spends on the result. A fused score-and-select kernel
        that never writes the score row would save the round trip -- that is
        the remaining optimisation here, and it is not implemented.
        :func:`index_and_select` chunks the query axis instead, which captures
        the memory win without the data-dependent control flow a radix select
        would need.
    """
    with torch.no_grad():
        return R.topk_select(I.detach(), k)


def index_and_select(
    qI, wI, KI, compress_rate, k,
    chunk: int = 4096, backend="auto", config: KernelConfig = DEFAULT_KERNEL_CONFIG,
):
    """Score and select in query chunks, never materialising the full matrix.

    The score matrix is ``[B, N, NB]``. At the context lengths V4 targets that
    is far larger than the model -- for a 1M-token sequence with ``m = 4`` it is
    250 billion entries per batch element. But only the top ``k`` of each row
    survives, so nothing needs to see more than one row at a time.

    This walks the query axis in chunks, scoring and immediately collapsing each
    chunk to its ``[chunk, k]`` selection. Peak memory is ``chunk * NB`` instead
    of ``N * NB`` -- at the published settings a reduction of roughly ``NB / k``,
    or 500x.

    Args:
        qI, wI, KI, compress_rate: as in :func:`lightning_index`.
        k: entries to keep per query.
        chunk: query tokens scored at once.
        backend, config: as elsewhere.

    Returns:
        ``[B, N, k]`` int32 indices.

    Note:
        Selection is not differentiable, so this runs under ``no_grad``. Train
        the indexer through :func:`indexer_kl_loss`, which needs the scores for
        one chunk at a time and nothing more.
    """
    N = qI.shape[1]
    out = []
    with torch.no_grad():
        for t0 in range(0, N, chunk):
            t1 = min(t0 + chunk, N)
            # t_offset, not a post-hoc mask: the scorer has to know where the
            # chunk sits, because a mask applied afterwards can only remove
            # candidates the local boundary already wrongly removed.
            I = lightning_index(
                qI[:, t0:t1], wI[:, t0:t1], KI, compress_rate, t_offset=t0,
                backend=backend, config=config,
            )
            out.append(R.topk_select(I, k))
    return torch.cat(out, dim=1)


def indexer_kl_loss(
    I: torch.Tensor,
    target_p: torch.Tensor,
    compress_rate: int,
) -> torch.Tensor:
    """Auxiliary loss that trains the indexer to imitate the core attention.

    Top-k selection is a hard gather, so no gradient reaches the indexer through
    it. The report says the indexer is warmed up before sparsity is switched on
    but does not write the objective down; the DeepSeek Sparse Attention recipe
    V4 cites trains it to match the distribution the *dense* core attention
    produces, summed over heads. That is what this implements, so it is a
    documented reconstruction rather than a quotation.

    Args:
        I: ``[B, N, NB]`` index scores, masked with a large negative.
        target_p: ``[B, N, NB]`` target distribution -- the head-summed,
            renormalised core attention probabilities over compressed entries.
        compress_rate: ``m``, for the causal mask.

    Returns:
        Scalar KL divergence, averaged over query tokens.
    """
    B, N, NB = I.shape
    t = torch.arange(N, device=I.device)
    s = torch.arange(NB, device=I.device)
    allowed = ((s.unsqueeze(0) + 1) * compress_rate <= t.unsqueeze(1)).unsqueeze(0)

    # A large finite negative, not -inf: a row with no legal candidate would
    # otherwise be a softmax over all -inf, which is NaN -- and the NaN survives
    # a `where` on the way back, poisoning the gradient rather than the value.
    # R._f, not .float(): an fp64 caller (the gradient tests) must stay in fp64,
    # or a perfectly-matched indexer bottoms out at fp32 roundoff rather than 0.
    logp = torch.log_softmax(R._f(I).masked_fill(~allowed, R.NO_CANDIDATE), dim=-1)
    tgt = R._f(target_p).masked_fill(~allowed, 0.0)
    tgt = tgt / tgt.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    # Rows with no legal candidate contribute nothing. xlogy rather than
    # `tgt * tgt.clamp_min(eps).log()`: the clamp floors the entropy term for
    # every small probability, so a perfectly-matched indexer would bottom out
    # at ~1e-8 instead of at 0 and the loss would never look converged.
    live = allowed.any(dim=-1)
    kl = (torch.xlogy(tgt, tgt) - tgt * logp).sum(dim=-1)
    return (kl * live).sum() / live.sum().clamp_min(1)


# ---------------------------------------------------------------------------
# Core attention
# ---------------------------------------------------------------------------


class _SparseAttention(torch.autograd.Function):
    """CSA core attention with the hand-written backward."""

    @staticmethod
    def forward(ctx, q, kv_comp, kv_win, sink, idx, scale, window, config):
        from .core_attn import sparse_attn_fwd

        o, lse = sparse_attn_fwd(q, kv_comp, kv_win, sink, idx, scale, window, config)
        ctx.save_for_backward(q, kv_comp, kv_win, sink, idx, o, lse)
        ctx.scale, ctx.window, ctx.config = scale, window, config
        return o

    @staticmethod
    def backward(ctx, do):
        from .core_attn import sparse_attn_bwd

        q, kv_comp, kv_win, sink, idx, o, lse = ctx.saved_tensors
        dq, dkv, dwin, dsink = sparse_attn_bwd(
            do.contiguous(), q, kv_comp, kv_win, sink, idx,
            ctx.scale, ctx.window, o, lse, ctx.config,
        )
        return (
            dq.to(q.dtype), dkv.to(kv_comp.dtype), dwin.to(kv_win.dtype), dsink,
            None, None, None, None,
        )


class _DenseAttention(torch.autograd.Function):
    """HCA core attention with the hand-written backward."""

    @staticmethod
    def forward(ctx, q, kv_comp, kv_win, sink, scale, compress_rate, window, config):
        from .core_attn import dense_attn_fwd

        o, lse = dense_attn_fwd(
            q, kv_comp, kv_win, sink, scale, compress_rate, window, config
        )
        ctx.save_for_backward(q, kv_comp, kv_win, sink, o, lse)
        ctx.scale, ctx.compress_rate = scale, compress_rate
        ctx.window, ctx.config = window, config
        return o

    @staticmethod
    def backward(ctx, do):
        from .core_attn import dense_attn_bwd

        q, kv_comp, kv_win, sink, o, lse = ctx.saved_tensors
        dq, dkv, dwin, dsink = dense_attn_bwd(
            do.contiguous(), q, kv_comp, kv_win, sink,
            ctx.scale, ctx.compress_rate, ctx.window, o, lse, ctx.config,
        )
        return (
            dq.to(q.dtype), dkv.to(kv_comp.dtype), dwin.to(kv_win.dtype), dsink,
            None, None, None, None,
        )


def sparse_attention(
    q, kv_comp, kv_win, sink, idx, compress_rate, window, scale,
    backend="auto", config: KernelConfig = DEFAULT_KERNEL_CONFIG,
):
    """Differentiable CSA core attention (eq. 19, sink of eq. 27).

    Args:
        q: ``[B, N, H, c]``, normed and RoPE'd.
        kv_comp: ``[B, NB, c]``, normed and RoPE'd.
        kv_win: ``[B, N, c]`` window entries, or ``None``.
        sink: ``[H]``.
        idx: ``[B, N, k]`` selection.
        compress_rate, window, scale: layer parameters.
        backend, config: as elsewhere.

    Returns:
        ``[B, N, H, c]`` before the ``-t`` un-rotation.
    """
    if resolve_backend(backend, q, kv_comp) == "torch":
        return R.core_attention(
            q, kv_comp, kv_win, sink, idx, compress_rate, window, scale
        )
    return _SparseAttention.apply(q, kv_comp, kv_win, sink, idx, scale, window, config)


def dense_attention(
    q, kv_comp, kv_win, sink, compress_rate, window, scale,
    backend="auto", config: KernelConfig = DEFAULT_KERNEL_CONFIG,
):
    """Differentiable HCA core attention (eq. 26, sink of eq. 27).

    Returns:
        ``[B, N, H, c]`` before the ``-t`` un-rotation.
    """
    if resolve_backend(backend, q, kv_comp) == "torch":
        return R.core_attention(
            q, kv_comp, kv_win, sink, None, compress_rate, window, scale
        )
    return _DenseAttention.apply(
        q, kv_comp, kv_win, sink, scale, compress_rate, window, config
    )
