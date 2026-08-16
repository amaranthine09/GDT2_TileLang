"""The Gated DeltaNet-2 attention layer -- the entry point for this package.

Two things live here, and nothing else:

:func:`gdn2_attention`
    The functional op. Takes ``q, k, v`` and the three gates, returns the
    output and (optionally) the final recurrent state. This is where the
    autograd wiring lives: forward calls :mod:`gdn2.forward`, backward calls
    :mod:`gdn2.backward`.

:class:`GatedDeltaNet2`
    The ``nn.Module`` you drop into a transformer in place of self-attention.
    It owns the projections, the short causal convolution, the gate
    parameterisation and the output norm, then calls ``gdn2_attention``.

Use the module if you are building a model; use the function if you already
have projections and gates of your own.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .backward import chunk_gdn2_bwd
from .config import DEFAULT_KERNEL_CONFIG, GDN2Config, GatedDeltaNet2Config
from .forward import _pad_to_chunk, chunk_gdn2_fwd
from .inference import gdn2_decode, gdn2_prefill
from .reference import chunk_gdn2_bwd_torch, chunk_gdn2_torch

__all__ = ["gdn2_attention", "GatedDeltaNet2", "GDN2Cache", "resolve_backend"]


# ---------------------------------------------------------------------------
# Functional op
# ---------------------------------------------------------------------------

_BACKENDS = ("auto", "tilelang", "torch")


def resolve_backend(backend: str, on_cuda: bool, needs_grad: bool) -> str:
    """Turn ``"auto"`` into a concrete backend, or validate an explicit one.

    Args:
        backend: requested backend.
        on_cuda: whether the inputs live on a CUDA device.
        needs_grad: whether a backward pass will be required.

    Returns:
        Either ``"tilelang"`` or ``"torch"``.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {_BACKENDS}")
    if backend == "auto":
        return "tilelang" if on_cuda else "torch"
    if backend != "torch" and not on_cuda:
        raise ValueError(f"backend {backend!r} requires a CUDA device")
    return backend


class _GDN2Function(torch.autograd.Function):
    """Fused TileLang forward, chunkwise-PyTorch backward."""

    @staticmethod
    def forward(ctx, q, k, v, g, b, w, scale, initial_state, output_final_state, config):
        with torch.no_grad():
            o, final_state = chunk_gdn2_fwd(
                q, k, v, g, b, w, scale, initial_state, output_final_state, config
            )
        ctx.save_for_backward(q, k, v, g, b, w, initial_state)
        ctx.scale = scale
        ctx.config = config
        ctx.output_final_state = output_final_state
        return o, final_state

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, do, dfinal_state):
        q, k, v, g, b, w, initial_state = ctx.saved_tensors
        cfg = ctx.config
        needs = ctx.needs_input_grad
        dht = dfinal_state if ctx.output_final_state else None

        if cfg.backward == "tilelang":
            grads = chunk_gdn2_bwd(q, k, v, g, b, w, ctx.scale, initial_state,
                                   do.contiguous(), dht, cfg)
        else:
            grads = chunk_gdn2_bwd_torch(
                q.float(), k.float(), v.float(), g.float(), b.float(), w.float(),
                ctx.scale, initial_state, do.float(), None if dht is None else dht.float(),
                chunk_size=cfg.chunk_size, sub_chunk_size=cfg.sub_chunk_size,
            )
        inputs = (q, k, v, g, b, w)
        out = [gr.to(t.dtype) if needs[i] else None for i, (t, gr) in enumerate(zip(inputs, grads))]
        ds0 = grads[6].to(initial_state.dtype) if (initial_state is not None and needs[7]) else None
        return (*out, None, ds0, None, None)


def gdn2_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    config: GDN2Config = DEFAULT_KERNEL_CONFIG,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Gated DeltaNet-2 attention over a full sequence.

    Dispatches to the backend chosen by ``config.backend``; see the module
    docstring for what ``"auto"`` resolves to.

    Args:
        q, k: ``[B, S, H, DK]`` query / key. The layer L2-normalises these.
        v: ``[B, S, H, DV]`` value.
        g: ``[B, S, H, DK]`` log decay, ``<= 0`` (natural log).
        b: ``[B, S, H, DK]`` channel-wise erase gate.
        w: ``[B, S, H, DV]`` channel-wise write gate.
        scale: query scale; defaults to ``DK ** -0.5``.
        initial_state: ``[B, H, DK, DV]`` incoming state, or ``None``.
        output_final_state: also return the state after the last token.
        config: tiling and backend configuration.
        cu_seqlens: ``[N + 1]`` packed-sequence offsets. Not implemented yet;
            passing it raises.

    Returns:
        ``(o, final_state)`` with ``o`` of shape ``[B, S, H, DV]``.
    """
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    needs_grad = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (q, k, v, g, b, w, initial_state)
    )
    backend = resolve_backend(config.backend, q.device.type == "cuda", needs_grad)

    if cu_seqlens is not None:
        raise NotImplementedError(
            "packed variable-length batches are not implemented yet; pad the batch instead"
        )

    S = q.shape[1]
    BS = config.chunk_size
    S_pad = ((S + BS - 1) // BS) * BS
    if S_pad != S:
        # Zero padding is inert: g = 0 keeps the decay at 1 while b = w = 0
        # makes both the erase and the write vanish, so the state is unchanged.
        q, k, v, g, b, w = (_pad_to_chunk(x, S_pad) for x in (q, k, v, g, b, w))

    if backend == "torch":
        o, final_state = chunk_gdn2_torch(
            q, k, v, g, b, w, scale, initial_state, output_final_state,
            chunk_size=BS, sub_chunk_size=config.sub_chunk_size,
        )
    else:
        o, final_state = _GDN2Function.apply(
            q, k, v, g, b, w, scale, initial_state, output_final_state, config
        )
    return o[:, :S], final_state


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


@dataclass
class GDN2Cache:
    """Per-layer decoding cache.

    Attributes:
        recurrent_state: ``[B, H_v, DK, DV]`` carried state.
        conv_state: ``[B, C, conv_size - 1]`` tail of the short-conv input, or
            ``None`` when the layer has no short convolution.
    """

    recurrent_state: torch.Tensor
    conv_state: torch.Tensor | None = None


class _ShortConv(nn.Module):
    """Causal depthwise conv1d over the channel axis, with a decoding cache."""

    def __init__(self, channels: int, kernel_size: int = 4, activation: bool = True):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.conv = nn.Conv1d(channels, channels, kernel_size, groups=channels, padding=kernel_size - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, S, C]`` -> ``[B, S, C]``."""
        y = self.conv(x.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        return F.silu(y) if self.activation else y

    def step(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One token. ``x``: ``[B, C]``, ``state``: ``[B, C, kernel_size - 1]``."""
        y, state = self.decode(x.unsqueeze(1), state)
        return y[:, 0], state

    def decode(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``n`` tokens at once. ``x``: ``[B, n, C]``.

        Prepends the cached ``kernel_size - 1`` tail so the first token sees the
        same window it would have during a full forward pass.
        """
        buf = torch.cat([state, x.transpose(1, 2)], dim=-1)  # [B, C, K-1+n]
        y = F.conv1d(buf, self.conv.weight, self.conv.bias, groups=self.channels)
        y = y.transpose(1, 2)
        return (F.silu(y) if self.activation else y), buf[..., -(self.kernel_size - 1):]


class _GatedRMSNorm(nn.Module):
    """RMSNorm with a SiLU output gate, applied per value head."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight * F.silu(gate.float())).to(gate.dtype)


class GatedDeltaNet2(nn.Module):
    """Gated DeltaNet-2 token mixer.

    Decoupled gating is the whole point of GDN-2, so the two gates get their own
    projections: ``b`` lives on the key axis (which coordinates of the old read
    to erase) and ``w`` on the value axis (which coordinates of the incoming
    value to write). Tying them would collapse this layer to KDA.

    Args:
        config: see :class:`GatedDeltaNet2Config`.
    """

    def __init__(self, config: GatedDeltaNet2Config):
        super().__init__()
        self.cfg = config
        H = config.num_heads
        HK = config.num_kv_heads or H
        if H % HK != 0:
            raise ValueError(f"num_heads {H} must be a multiple of num_kv_heads {HK}")
        DK = config.head_dim_k
        DV = config.head_dim_v or config.head_dim_k

        self.num_heads, self.num_kv_heads = H, HK
        self.head_dim_k, self.head_dim_v = DK, DV
        self.head_repeat = H // HK

        d_qk, d_v = HK * DK, H * DV
        self.q_proj = nn.Linear(config.hidden_size, d_qk, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, d_qk, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, d_v, bias=False)
        # Erase gate on the key axis, write gate on the value axis.
        self.b_proj = nn.Linear(config.hidden_size, d_qk, bias=False)
        self.w_proj = nn.Linear(config.hidden_size, d_v, bias=False)
        # Channel-wise decay: g = -exp(A_log) * softplus(f_proj(x) + dt_bias)
        self.f_proj = nn.Linear(config.hidden_size, d_qk, bias=False)
        self.o_gate_proj = nn.Linear(config.hidden_size, d_v, bias=False)
        self.o_proj = nn.Linear(d_v, config.hidden_size, bias=False)

        if config.conv_size > 0:
            self.q_conv = _ShortConv(d_qk, config.conv_size)
            self.k_conv = _ShortConv(d_qk, config.conv_size)
            self.v_conv = _ShortConv(d_v, config.conv_size)
        else:
            self.q_conv = self.k_conv = self.v_conv = None

        # g = -exp(A_log) * softplus(f_proj(x) + dt_bias): the rate is per key
        # head and broadcast over channels, the bias is per channel.
        lo, hi = config.a_init_range
        self.A_log = nn.Parameter(torch.empty(HK).uniform_(lo, hi).log_())
        dt_lo, dt_hi = config.dt_init_range
        dt = torch.empty(HK, DK).uniform_(dt_lo, dt_hi)
        # Invert softplus so the initial step size is exactly `dt`.
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        self.o_norm = _GatedRMSNorm(DV, config.norm_eps)
        self.scale = DK**-0.5
        self.erase_scale = config.erase_gate_scale

    def _gates(self, x: torch.Tensor, lead: tuple[int, ...]) -> torch.Tensor:
        """Channel-wise log decay from the raw ``f`` projection."""
        f = self.f_proj(x).view(*lead, self.num_kv_heads, self.head_dim_k)
        return -self.A_log.exp().unsqueeze(-1) * F.softplus(f.float() + self.dt_bias)

    def _expand(self, x: torch.Tensor) -> torch.Tensor:
        """Broadcast key-head tensors up to the number of value heads."""
        return x if self.head_repeat == 1 else x.repeat_interleave(self.head_repeat, dim=-2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: GDN2Cache | None = None,
        use_cache: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, GDN2Cache | None]:
        """Run the mixer over a sequence.

        Args:
            hidden_states: ``[B, S, hidden_size]``.
            cache: state carried in from a previous call, or ``None``.
            use_cache: return an updated cache for incremental decoding.
            cu_seqlens: ``[N + 1]`` packed-sequence offsets for variable-length
                batches. Not implemented yet; passing it raises.

        Returns:
            ``(output, cache)`` with ``output`` of shape
            ``[B, S, hidden_size]``.
        """
        B, S, _ = hidden_states.shape
        H, HK = self.num_heads, self.num_kv_heads
        DK, DV = self.head_dim_k, self.head_dim_v

        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        conv_state = None
        if self.q_conv is not None:
            if use_cache:
                width = self.cfg.conv_size - 1
                tail = torch.cat([q, k, v], dim=-1).transpose(1, 2)
                conv_state = F.pad(tail, (max(0, width - S), 0))[..., -width:]
            q, k, v = self.q_conv(q), self.k_conv(k), self.v_conv(v)

        q = F.normalize(q.view(B, S, HK, DK), p=2, dim=-1)
        k = F.normalize(k.view(B, S, HK, DK), p=2, dim=-1)
        v = v.view(B, S, H, DV)

        g = self._gates(hidden_states, (B, S))
        b = torch.sigmoid(self.b_proj(hidden_states).view(B, S, HK, DK)) * self.erase_scale
        w = torch.sigmoid(self.w_proj(hidden_states).view(B, S, H, DV))

        q, k, b, g = (self._expand(t) for t in (q, k, b, g))

        o, final_state = gdn2_attention(
            q.contiguous(), k.contiguous(), v.contiguous(),
            g.contiguous(), b.contiguous(), w.contiguous(),
            scale=self.scale,
            initial_state=cache.recurrent_state if cache is not None else None,
            output_final_state=use_cache,
            config=self.cfg.kernel,
            cu_seqlens=cu_seqlens,
        )

        gate = self.o_gate_proj(hidden_states).view(B, S, H, DV)
        o = self.o_norm(o, gate).reshape(B, S, H * DV)
        out = self.o_proj(o)

        new_cache = GDN2Cache(final_state, conv_state) if use_cache else None
        return out, new_cache

    @torch.no_grad()
    def prefill(
        self, hidden_states: torch.Tensor, cache: GDN2Cache | None = None
    ) -> tuple[torch.Tensor, GDN2Cache]:
        """Prefill a prompt and return the cache to decode from.

        Same maths as :meth:`forward`, different plumbing: no autograd graph,
        and the state scan is fused into the output so the per-chunk states are
        never materialised -- that array is ``[B, NC, H, DK, DV]``, hundreds of
        MB on a long prompt, and serving has no backward to spend it on. A
        ragged tail goes through the decode kernel instead of padding the last
        chunk.

        Pass the returned cache back in to prefill a long prompt in segments.

        Args:
            hidden_states: ``[B, S, hidden_size]``.
            cache: cache to continue from, or ``None`` to start fresh.

        Returns:
            ``(output, cache)`` with ``output`` of shape
            ``[B, S, hidden_size]``.
        """
        B, S, _ = hidden_states.shape
        H, HK = self.num_heads, self.num_kv_heads
        DK, DV = self.head_dim_k, self.head_dim_v

        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        if cache is None:
            cache = self.init_cache(B, device=hidden_states.device)
        if self.q_conv is not None:
            d_qk = HK * DK
            cs = cache.conv_state
            q, cq = self.q_conv.decode(q, cs[:, :d_qk])
            k, ck = self.k_conv.decode(k, cs[:, d_qk : 2 * d_qk])
            v, cv = self.v_conv.decode(v, cs[:, 2 * d_qk :])
            cache.conv_state = torch.cat([cq, ck, cv], dim=1)

        q = F.normalize(q.view(B, S, HK, DK), p=2, dim=-1)
        k = F.normalize(k.view(B, S, HK, DK), p=2, dim=-1)
        v = v.view(B, S, H, DV)
        g = self._gates(hidden_states, (B, S))
        b = torch.sigmoid(self.b_proj(hidden_states).view(B, S, HK, DK)) * self.erase_scale
        w = torch.sigmoid(self.w_proj(hidden_states).view(B, S, H, DV))
        q, k, b, g = (self._expand(t) for t in (q, k, b, g))

        o, state = gdn2_prefill(
            q.contiguous(), k.contiguous(), v.contiguous(),
            g.contiguous(), b.contiguous(), w.contiguous(),
            cache.recurrent_state, scale=self.scale, config=self.cfg.kernel,
        )
        cache.recurrent_state = state

        gate = self.o_gate_proj(hidden_states).view(B, S, H, DV)
        o = self.o_norm(o, gate).reshape(B, S, H * DV)
        return self.o_proj(o), cache

    @torch.no_grad()
    def decode(self, hidden_states: torch.Tensor, cache: GDN2Cache) -> tuple[torch.Tensor, GDN2Cache]:
        """Decode ``n`` tokens, updating ``cache`` in place.

        Applies the recurrence directly rather than the chunkwise form, and
        holds the recurrent state in registers across all ``n`` steps, so the
        state is read and written once per call instead of once per token. Pass
        several tokens at a time whenever you have them -- speculative decoding
        drafts, or a short prefill tail -- since the state traffic is what
        decoding actually costs.

        Args:
            hidden_states: ``[B, n, hidden_size]``.
            cache: cache from a previous :meth:`forward`, :meth:`decode` or
                :meth:`init_cache`.

        Returns:
            ``(output, cache)`` with ``output`` of shape
            ``[B, n, hidden_size]``.
        """
        B, n, _ = hidden_states.shape
        H, HK = self.num_heads, self.num_kv_heads
        DK, DV = self.head_dim_k, self.head_dim_v

        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        if self.q_conv is not None:
            d_qk = HK * DK
            cs = cache.conv_state
            q, cq = self.q_conv.decode(q, cs[:, :d_qk])
            k, ck = self.k_conv.decode(k, cs[:, d_qk : 2 * d_qk])
            v, cv = self.v_conv.decode(v, cs[:, 2 * d_qk :])
            cache.conv_state = torch.cat([cq, ck, cv], dim=1)

        q = F.normalize(q.view(B, n, HK, DK), p=2, dim=-1)
        k = F.normalize(k.view(B, n, HK, DK), p=2, dim=-1)
        v = v.view(B, n, H, DV)
        g = self._gates(hidden_states, (B, n))
        b = torch.sigmoid(self.b_proj(hidden_states).view(B, n, HK, DK)) * self.erase_scale
        w = torch.sigmoid(self.w_proj(hidden_states).view(B, n, H, DV))
        q, k, b, g = (self._expand(t) for t in (q, k, b, g))

        o = gdn2_decode(
            q.contiguous(), k.contiguous(), v.contiguous(),
            g.contiguous(), b.contiguous(), w.contiguous(),
            cache.recurrent_state, scale=self.scale,
        )

        gate = self.o_gate_proj(hidden_states).view(B, n, H, DV)
        o = self.o_norm(o, gate).reshape(B, n, H * DV)
        return self.o_proj(o), cache

    @torch.no_grad()
    def step(self, hidden_states: torch.Tensor, cache: GDN2Cache) -> tuple[torch.Tensor, GDN2Cache]:
        """Decode a single token. Thin wrapper over :meth:`decode`.

        Args:
            hidden_states: ``[B, hidden_size]`` for one position.
            cache: cache from a previous call.

        Returns:
            ``(output, cache)`` with ``output`` of shape ``[B, hidden_size]``.
        """
        out, cache = self.decode(hidden_states.unsqueeze(1), cache)
        return out[:, 0], cache

    def init_cache(self, batch_size: int, device=None, dtype=torch.float32) -> GDN2Cache:
        """Allocate an empty cache for decoding from scratch."""
        state = torch.zeros(
            batch_size, self.num_heads, self.head_dim_k, self.head_dim_v,
            device=device, dtype=dtype,
        )
        conv_state = None
        if self.q_conv is not None:
            channels = 2 * self.num_kv_heads * self.head_dim_k + self.num_heads * self.head_dim_v
            conv_state = torch.zeros(batch_size, channels, self.cfg.conv_size - 1, device=device, dtype=dtype)
        return GDN2Cache(state, conv_state)
