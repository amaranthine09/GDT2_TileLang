"""DeepSeek-V4 attention layers as ``nn.Module``.

Three layer types, matching the report's stack:

:class:`CSAAttention`
    Compressed Sparse Attention -- overlapped pooling, lightning indexer,
    top-k, shared-KV MQA.
:class:`HCAAttention`
    Heavily Compressed Attention -- aggressive non-overlapped pooling, dense
    over the compressed stream.
:class:`SWAAttention`
    Pure sliding window, for the first layers of V4-Flash.

:func:`build_attention_stack` lays them out in the published interleaving.

Order of operations
-------------------
Per the report, RMSNorm and partial RoPE are both applied "just before the core
attention operation". Norm comes first here -- normalising after rotating would
rescale the rotated lanes against the unrotated ones and break the property the
rotation exists to provide. RoPE covers only the trailing ``rope_dim``
channels of each query and each KV entry.

The compressed entries are position-tagged *once*, when they are built, not per
query. That is what makes them cacheable: a compressed entry is written to the
KV cache already rotated, which is also why the report can describe storing its
RoPE lanes and its remaining lanes at different precisions.

Because each entry is used as key *and* value, the raw attention output is a
weighted sum of rotated vectors and carries absolute position. It is rotated
back by the query's own position at the end, which converts every entry's
contribution into a function of its distance from the query.

Reconstructed details
---------------------
Beyond the three noted in :mod:`dsv4.reference`, one more choice is not fixed by
the report: whether the *indexer* gets RoPE. The report specifies partial RoPE
for the core attention and is silent about the indexer. DeepSeek Sparse
Attention, which the report cites for the indexer design, does apply it, and an
indexer with no position information would be a poor long-context selector --
so ``indexer_rope`` defaults to ``True``. Set it to ``False`` to match a
checkpoint that disagrees.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import ops
from . import reference as R
from .config import (
    DEFAULT_KERNEL_CONFIG,
    CSAConfig,
    DSV4Config,
    HCAConfig,
    KernelConfig,
    LayerKind,
)

__all__ = [
    "CSAAttention",
    "HCAAttention",
    "SWAAttention",
    "build_attention_stack",
]


def _linear(d_in: int, d_out: int) -> nn.Parameter:
    """A bias-free projection matrix, fan-in scaled."""
    w = torch.empty(d_in, d_out)
    nn.init.normal_(w, std=d_in**-0.5)
    return nn.Parameter(w)


class _BaseAttention(nn.Module):
    """Everything CSA, HCA and SWA share: queries, window, sink, output.

    Subclasses supply :meth:`compressed_entries`, which produces the compressed
    KV stream (or ``None`` for the pure sliding-window layer), and
    :meth:`core_attention`, which runs the variant's attention over it.
    """

    def __init__(self, config, kernel: KernelConfig = DEFAULT_KERNEL_CONFIG, backend: str = "auto"):
        super().__init__()
        self.cfg = config
        self.kernel = kernel
        self.backend = backend

        d, c, H = config.hidden_size, config.head_dim, config.num_heads
        dc, g, dg = config.q_lora_rank, config.num_out_groups, config.out_group_dim

        # Queries, low-rank. The latent is shared with the indexer, so it is
        # returned by _queries rather than kept private.
        self.w_dq = _linear(d, dc)
        self.w_uq = _linear(dc, c * H)
        self.q_norm = nn.Parameter(torch.ones(c))
        self.kv_norm = nn.Parameter(torch.ones(c))

        self.w_win_kv = None if config.window_shares_kv_proj else _linear(d, c)

        # Sink logits start at 0: one unit of unnormalised mass, so a head
        # begins by holding back about 1/(1 + n_attended) of its attention.
        self.sink = nn.Parameter(torch.zeros(H))

        self.w_group = nn.Parameter(
            torch.empty(g, c * H // g, dg).normal_(std=(c * H // g) ** -0.5)
        )
        self.w_out = nn.Parameter(torch.empty(g * dg, d).normal_(std=(g * dg) ** -0.5))

    # -- pieces ---------------------------------------------------------

    def _positions(self, x: torch.Tensor, position_ids: torch.Tensor | None) -> torch.Tensor:
        if position_ids is not None:
            return position_ids
        return torch.arange(x.shape[1], device=x.device)

    def _queries(self, x: torch.Tensor, pos: torch.Tensor):
        """``h -> (q, c_q)`` with ``q`` normed and rotated, ``[B, N, H, c]``."""
        cfg = self.cfg
        B, N, _ = x.shape
        cq = x @ self.w_dq
        q = (cq @ self.w_uq).view(B, N, cfg.num_heads, cfg.head_dim)
        q = R.rms_norm(q, self.q_norm, cfg.norm_eps)
        cos, sin = R.rope_cos_sin(pos, cfg.rope_dim, cfg.rope_theta, dtype=q.dtype)
        q = R.apply_partial_rope(
            q, cos.view(1, N, 1, -1), sin.view(1, N, 1, -1), cfg.rope_interleaved
        )
        return q, cq

    def _tag(self, kv: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Norm and rotate a KV stream at the given positions, ``[B, L, c]``."""
        cfg = self.cfg
        kv = R.rms_norm(kv, self.kv_norm, cfg.norm_eps)
        cos, sin = R.rope_cos_sin(pos, cfg.rope_dim, cfg.rope_theta, dtype=kv.dtype)
        return R.apply_partial_rope(
            kv, cos.unsqueeze(0), sin.unsqueeze(0), cfg.rope_interleaved
        )

    def _window_entries(self, x: torch.Tensor, pos: torch.Tensor, fallback: torch.Tensor):
        """Uncompressed KV entries for the sliding-window branch."""
        if self.cfg.window_size <= 0:
            return None
        raw = fallback if self.w_win_kv is None else x @ self.w_win_kv
        return self._tag(raw, pos)

    def _finish(self, o: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Un-rotate by the query position, then the grouped output projection."""
        cfg = self.cfg
        o = R.unrope_output(o, pos, cfg.rope_dim, cfg.rope_theta, cfg.rope_interleaved)
        return R.grouped_out_proj(o, self.w_group, self.w_out)

    # -- interface ------------------------------------------------------

    def compressed_entries(self, x):  # pragma: no cover - abstract
        raise NotImplementedError

    def core_attention(self, q, kv, win, x, cq, pos):  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Run the layer.

        Args:
            x: ``[B, N, d]`` hidden states. ``N`` must be a multiple of the
                layer's compression rate.
            position_ids: ``[N]`` positions; defaults to ``arange(N)``.

        Returns:
            ``[B, N, d]``.
        """
        pos = self._positions(x, position_ids)
        q, cq = self._queries(x, pos)
        kv, raw_stream = self.compressed_entries(x)
        win = self._window_entries(x, pos, raw_stream)
        o = self.core_attention(q, kv, win, x, cq, pos)
        return self._finish(o, pos)


class SWAAttention(_BaseAttention):
    """Pure sliding-window attention -- no compressed branch at all.

    V4-Flash uses this for its first two layers, where the model is still
    working at the token level and a compressed summary would have nothing to
    summarise.
    """

    def __init__(self, config, kernel=DEFAULT_KERNEL_CONFIG, backend="auto"):
        if config.window_shares_kv_proj:
            raise ValueError("a sliding-window layer has no compression stream to share")
        super().__init__(config, kernel, backend)

    def compressed_entries(self, x):
        return None, None

    def core_attention(self, q, kv, win, x, cq, pos):
        return R.sliding_window_attention(
            q, win, self.sink, self.cfg.window_size, self.cfg.scale
        )


class HCAAttention(_BaseAttention):
    """Heavily Compressed Attention (section 2.3.2).

    Pools every ``m'`` tokens into one entry with a per-channel softmax and
    attends to all of them. No indexer and no selection -- the compression
    ratio alone is what makes long context affordable.
    """

    def __init__(self, config: HCAConfig, kernel=DEFAULT_KERNEL_CONFIG, backend="auto"):
        super().__init__(config, kernel, backend)
        d, c, m = config.hidden_size, config.head_dim, config.compress_rate
        self.w_kv = _linear(d, c)
        self.w_z = _linear(d, c)
        # Zero bias is a uniform pool at init: every token in the span
        # contributes equally until the model learns otherwise.
        self.bias = nn.Parameter(torch.zeros(m, c))

    def compressed_entries(self, x):
        cfg = self.cfg
        C, Z = x @ self.w_kv, x @ self.w_z
        comp = ops.hca_compress(C, Z, self.bias, self.backend, self.kernel)
        pos = R.block_positions(
            comp.shape[1], cfg.compress_rate, cfg.rope_block_pos, comp.device
        )
        return self._tag(comp, pos), C

    def core_attention(self, q, kv, win, x, cq, pos):
        cfg = self.cfg
        return ops.dense_attention(
            q, kv, win, self.sink, cfg.compress_rate, cfg.window_size, cfg.scale,
            self.backend, self.kernel,
        )


class CSAAttention(_BaseAttention):
    """Compressed Sparse Attention (section 2.3.1).

    Pools every ``m`` tokens into one entry using the overlapped two-stream
    scheme, scores the entries with a lightning indexer, and attends to the top
    ``k``. The indexer keys are built by running the *same* pooling on a second
    pair of streams at the indexer's own width.

    Warning:
        **The indexer parameters receive no gradient from this layer's output.**
        Top-k selection is a hard gather: the loss can see *which* entries were
        chosen only through a discrete argmax, so nothing flows back into the
        scores that ranked them. After ``loss.backward()`` the eight indexer
        tensors -- ``w_aki``, ``w_bki``, ``w_azi``, ``w_bzi``, ``bias_ai``,
        ``bias_bi``, ``w_iuq``, ``w_w`` -- will have ``grad`` of ``None``.

        That is the architecture, not a bug, and it is why the report warms the
        indexer up as a separate stage. Train it by adding
        :func:`dsv4.ops.indexer_kl_loss` to your objective, using
        :meth:`index_scores` to get the scores. An optimiser built over
        ``layer.parameters()`` without that term will leave the indexer at its
        initialisation forever, and the model will still train -- just selecting
        at random.
    """

    def __init__(self, config: CSAConfig, kernel=DEFAULT_KERNEL_CONFIG, backend="auto"):
        super().__init__(config, kernel, backend)
        d, c, m = config.hidden_size, config.head_dim, config.compress_rate
        cI, nhI, dc = config.indexer_dim, config.indexer_heads, config.q_lora_rank

        # Two compression streams for the KV entries (eq. 9/10).
        self.w_akv, self.w_bkv = _linear(d, c), _linear(d, c)
        self.w_az, self.w_bz = _linear(d, c), _linear(d, c)
        self.bias_a = nn.Parameter(torch.zeros(m, c))
        self.bias_b = nn.Parameter(torch.zeros(m, c))

        # The same pooling again at indexer width, for the indexer keys.
        self.w_aki, self.w_bki = _linear(d, cI), _linear(d, cI)
        self.w_azi, self.w_bzi = _linear(d, cI), _linear(d, cI)
        self.bias_ai = nn.Parameter(torch.zeros(m, cI))
        self.bias_bi = nn.Parameter(torch.zeros(m, cI))

        # Indexer queries share the query latent c_q (eq. 13/14).
        self.w_iuq = _linear(dc, cI * nhI)
        self.w_w = _linear(d, nhI)

    def compressed_entries(self, x):
        cfg = self.cfg
        Ca, Cb = x @ self.w_akv, x @ self.w_bkv
        Za, Zb = x @ self.w_az, x @ self.w_bz
        comp = ops.csa_compress(
            Ca, Cb, Za, Zb, self.bias_a, self.bias_b, self.backend, self.kernel
        )
        pos = R.block_positions(
            comp.shape[1], cfg.compress_rate, cfg.rope_block_pos, comp.device
        )
        return self._tag(comp, pos), Ca

    def _indexer_keys(self, x: torch.Tensor) -> torch.Tensor:
        """Compressed indexer keys ``K^IComp``, ``[B, NB, cI]``."""
        Ca, Cb = x @ self.w_aki, x @ self.w_bki
        Za, Zb = x @ self.w_azi, x @ self.w_bzi
        return ops.csa_compress(
            Ca, Cb, Za, Zb, self.bias_ai, self.bias_bi, self.backend, self.kernel
        )

    def index_scores(self, x: torch.Tensor, cq: torch.Tensor, pos: torch.Tensor):
        """Lightning-indexer scores, exposed so the auxiliary loss can reach them.

        Returns:
            ``[B, N, NB]`` scores. Only useful at lengths where the full matrix
            fits; :meth:`forward` uses the chunked path instead.
        """
        cfg = self.cfg
        qI, wI, KI = self._indexer_inputs(x, cq, pos)
        return ops.lightning_index(
            qI, wI, KI, cfg.compress_rate, 0, self.backend, self.kernel
        )

    def _indexer_inputs(self, x, cq, pos):
        cfg = self.cfg
        B, N, _ = x.shape
        qI = (cq @ self.w_iuq).view(B, N, cfg.indexer_heads, cfg.indexer_dim)
        wI = x @ self.w_w
        KI = self._indexer_keys(x)

        if cfg.indexer_rope:
            rd = min(cfg.rope_dim, cfg.indexer_dim)
            kpos = R.block_positions(
                KI.shape[1], cfg.compress_rate, cfg.rope_block_pos, KI.device
            )
            cos, sin = R.rope_cos_sin(pos, rd, cfg.rope_theta, dtype=qI.dtype)
            qI = R.apply_partial_rope(
                qI, cos.view(1, N, 1, -1), sin.view(1, N, 1, -1), cfg.rope_interleaved
            )
            kcos, ksin = R.rope_cos_sin(kpos, rd, cfg.rope_theta, dtype=KI.dtype)
            KI = R.apply_partial_rope(
                KI, kcos.unsqueeze(0), ksin.unsqueeze(0), cfg.rope_interleaved
            )
        return qI, wI, KI

    def core_attention(self, q, kv, win, x, cq, pos):
        cfg = self.cfg
        qI, wI, KI = self._indexer_inputs(x, cq, pos)
        idx = ops.index_and_select(
            qI, wI, KI, cfg.compress_rate, cfg.top_k,
            backend=self.backend, config=self.kernel,
        )
        return ops.sparse_attention(
            q, kv, win, self.sink, idx, cfg.compress_rate, cfg.window_size,
            cfg.scale, self.backend, self.kernel,
        )


def build_attention_stack(
    config: DSV4Config, backend: str = "auto"
) -> nn.ModuleList:
    """Instantiate every attention layer of a stack in the published order.

    Args:
        config: the stack configuration; :meth:`DSV4Config.layer_kind` decides
            what sits at each depth.
        backend: passed through to every layer.

    Returns:
        A :class:`torch.nn.ModuleList` of ``config.num_layers`` layers.
    """
    kinds = {
        LayerKind.CSA: (CSAAttention, "csa"),
        LayerKind.HCA: (HCAAttention, "hca"),
        LayerKind.SWA: (SWAAttention, "hca"),
    }
    layers = []
    for i in range(config.num_layers):
        kind = config.layer_kind(i)
        cls, attr = kinds[kind]
        layers.append(cls(getattr(config, attr), config.kernel, backend))
    return nn.ModuleList(layers)
