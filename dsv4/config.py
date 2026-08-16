"""Configuration objects for DeepSeek-V4 hybrid attention.

Three levels, deliberately separate:

``KernelConfig``
    How the kernels are *executed* -- tile sizes, thread counts, dtypes.
    Nothing here changes the maths.
``CSAConfig`` / ``HCAConfig``
    What one attention *layer* is. These are the numbers from the technical
    report's model-setup section.
``DSV4Config``
    A whole stack: layer count, hidden size, and which layer type sits where.

The published configurations are available as :data:`V4_FLASH` and
:data:`V4_PRO`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch

__all__ = [
    "KernelConfig",
    "CSAConfig",
    "HCAConfig",
    "DSV4Config",
    "LayerKind",
    "DEFAULT_KERNEL_CONFIG",
    "V4_FLASH",
    "V4_PRO",
]


class LayerKind:
    """Which attention variant a layer uses.

    ``SWA`` is the pure sliding-window layer V4-Flash puts at the bottom of the
    stack: no compressed branch at all, just the local window.
    """

    CSA = "csa"
    HCA = "hca"
    SWA = "swa"

    ALL = (CSA, HCA, SWA)


@dataclass(frozen=True)
class KernelConfig:
    """Tiling and dtype knobs for the kernel pipeline.

    Attributes:
        block_T: query-token tile for the compression and indexer kernels, and
            for the HCA key-side backward's query sweep.
        block_T_attn: query-token tile for the HCA core attention itself. Much
            smaller than ``block_T`` because that kernel's output accumulator
            is ``[block_T_attn * block_H, head_dim]`` in registers, and
            ``head_dim`` is 512. The CSA core attention has no such knob --
            per-token top-k means a token tile shares nothing, so it is always
            one block per token.
        block_H: attention heads per block. The core attention kernels are MQA,
            so a whole head tile shares one loaded KV tile; this is the main
            arithmetic-intensity knob, and it trades directly against register
            pressure -- see ``block_T_attn``.
        block_KV: KV-entry tile for the core attention loops.
        block_D: channel tile for the reductions over the head dim ``c``.
        block_S: compressed-block tile for the indexer's key axis.
        threads: threads per block for the GEMM-heavy kernels.
        num_stages: software pipeline depth.
        accum_dtype: accumulator dtype. fp32 throughout; the softmax
            denominators and the compression reductions are not safe in fp16.
        input_dtype: dtype of the activations handed to the kernels.
        kv_rope_dtype: the report stores the RoPE lanes of the KV cache in bf16
            and the rest in fp8. Split storage is a cache-layout decision, so it
            lives here rather than in the model config.
        kv_nope_dtype: dtype for the non-RoPE lanes of the KV cache.
        indexer_dtype: the report runs indexer logits in fp4. Anything narrower
            than the tensor-core minimum falls back to ``input_dtype``.
    """

    block_T: int = 64
    block_T_attn: int = 2
    block_H: int = 16
    block_KV: int = 64
    block_D: int = 64
    block_S: int = 128
    threads: int = 128
    num_stages: int = 2
    accum_dtype: torch.dtype = torch.float32
    input_dtype: torch.dtype = torch.bfloat16
    kv_rope_dtype: torch.dtype = torch.bfloat16
    kv_nope_dtype: torch.dtype = torch.bfloat16
    indexer_dtype: torch.dtype = torch.bfloat16


DEFAULT_KERNEL_CONFIG = KernelConfig()


@dataclass(frozen=True)
class _CoreAttnConfig:
    """Fields CSA and HCA share -- everything except the compression stage.

    Attributes:
        hidden_size: model width ``d``.
        head_dim: head dimension ``c``. Also the compressed-entry width, since
            each entry serves as both key and value (shared-KV MQA).
        num_heads: number of query heads ``n_h``. There is exactly one KV
            "head": all query heads read the same compressed stream.
        q_lora_rank: query compression dim ``d_c``. Queries are produced
            low-rank, ``h W^DQ W^UQ``, and the latent is shared with the
            indexer.
        rope_dim: how many trailing channels carry RoPE. 64 in every published
            configuration.
        window_size: sliding-window branch width ``n_win``. Uncompressed KV
            entries for the most recent tokens, attended alongside the
            compressed ones under a single softmax.
        num_out_groups: ``g``, the number of groups the output projection
            splits ``n_h`` into.
        out_group_dim: ``d_g``, the width each group projects down to before
            the groups are concatenated and projected to ``d``.
        norm_eps: epsilon for the query / KV RMSNorm applied before core
            attention.
        softmax_scale: logit scale. ``None`` means ``head_dim ** -0.5``.
        rope_theta: RoPE base.
        rope_interleaved: ``False`` pairs channel ``i`` with ``i + rope_dim/2``
            (the "rotate_half" convention used by the HuggingFace DeepSeek
            models); ``True`` pairs adjacent channels. Both are self-consistent
            -- this only has to match the checkpoint you are loading.
        window_shares_kv_proj: reuse the compression-stream projection for the
            sliding-window entries instead of giving the window its own
            ``W^winKV``. See the note in :mod:`dsv4.reference`.
        rope_block_pos: which token position stands for a pooled entry when it
            is rotated -- ``"last"``, ``"first"`` or ``"center"`` of its span.
            The compressed and window branches share a softmax, so this has to
            be on the token scale either way; the choice only shifts every
            compressed entry by a constant. Not fixed by the report.
    """

    hidden_size: int
    head_dim: int = 512
    num_heads: int = 64
    q_lora_rank: int = 1024
    rope_dim: int = 64
    window_size: int = 128
    num_out_groups: int = 8
    out_group_dim: int = 1024
    norm_eps: float = 1e-6
    softmax_scale: float | None = None
    rope_theta: float = 10000.0
    rope_interleaved: bool = False
    window_shares_kv_proj: bool = False
    rope_block_pos: str = "last"

    @property
    def scale(self) -> float:
        """The logit scale actually used."""
        return self.head_dim**-0.5 if self.softmax_scale is None else self.softmax_scale

    def _validate_core(self) -> None:
        if self.rope_block_pos not in ("last", "first", "center"):
            raise ValueError(
                f"rope_block_pos must be last/first/center, got {self.rope_block_pos!r}"
            )
        if self.rope_dim > self.head_dim:
            raise ValueError(f"rope_dim {self.rope_dim} exceeds head_dim {self.head_dim}")
        if self.rope_dim % 2:
            raise ValueError(f"rope_dim {self.rope_dim} must be even")
        if self.num_heads % self.num_out_groups:
            raise ValueError(
                f"num_heads {self.num_heads} must be divisible by "
                f"num_out_groups {self.num_out_groups}"
            )


@dataclass(frozen=True)
class CSAConfig(_CoreAttnConfig):
    """Compressed Sparse Attention.

    Compresses every ``compress_rate`` tokens into one KV entry using the
    overlapped two-stream scheme, then keeps only the ``top_k`` entries a
    lightning indexer scores highest.

    Attributes:
        compress_rate: ``m``. Each compressed entry pools ``2m`` source rows
            (``m`` from stream *a*, ``m`` from stream *b*), but consecutive
            entries reuse rows, so the sequence still shrinks by exactly ``m``.
        top_k: ``k``, compressed entries kept per query.
        indexer_heads: ``n_h^I``.
        indexer_dim: ``c^I``.
        indexer_rope: apply partial RoPE to the indexer's queries and keys. The
            report specifies RoPE for the core attention and says nothing about
            the indexer; DeepSeek Sparse Attention, which it cites for the
            indexer design, does apply it, and a position-blind indexer would
            select poorly at long range. A reconstruction -- see
            :mod:`dsv4.attention`.
    """

    compress_rate: int = 4
    top_k: int = 512
    indexer_heads: int = 64
    indexer_dim: int = 128
    indexer_rope: bool = True

    kind = LayerKind.CSA

    def __post_init__(self) -> None:
        self._validate_core()
        if self.compress_rate < 1:
            raise ValueError(f"compress_rate must be >= 1, got {self.compress_rate}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")


@dataclass(frozen=True)
class HCAConfig(_CoreAttnConfig):
    """Heavily Compressed Attention.

    Same shared-KV MQA as :class:`CSAConfig`, but the compression is much more
    aggressive (``m' >> m``), non-overlapped, and there is no sparse selection
    -- every compressed entry is attended.

    Attributes:
        compress_rate: ``m'``.
    """

    compress_rate: int = 128

    kind = LayerKind.HCA

    def __post_init__(self) -> None:
        self._validate_core()
        if self.compress_rate < 1:
            raise ValueError(f"compress_rate must be >= 1, got {self.compress_rate}")


@dataclass
class DSV4Config:
    """A whole DeepSeek-V4 attention stack.

    Attributes:
        num_layers: transformer depth.
        hidden_size: model width ``d``.
        csa: the CSA layer configuration.
        hca: the HCA layer configuration.
        num_prefix_layers: how many layers at the bottom of the stack use
            ``prefix_kind`` instead of the interleaved pattern. 2 in both
            published configurations.
        prefix_kind: what those bottom layers are. V4-Flash uses pure sliding
            window, V4-Pro uses HCA.
        interleave_start: which kind the interleaved region starts with.
        kernel: tiling configuration passed through to the kernels.
    """

    num_layers: int
    hidden_size: int
    csa: CSAConfig
    hca: HCAConfig
    num_prefix_layers: int = 2
    prefix_kind: str = LayerKind.SWA
    interleave_start: str = LayerKind.CSA
    kernel: KernelConfig = field(default_factory=KernelConfig)

    def __post_init__(self) -> None:
        for name in ("prefix_kind", "interleave_start"):
            value = getattr(self, name)
            if value not in LayerKind.ALL:
                raise ValueError(f"{name} must be one of {LayerKind.ALL}, got {value!r}")
        # The layer configs carry their own hidden_size so they can be used
        # standalone; keep them consistent with the stack.
        self.csa = replace(self.csa, hidden_size=self.hidden_size)
        self.hca = replace(self.hca, hidden_size=self.hidden_size)

    def layer_kind(self, layer_idx: int) -> str:
        """Which attention variant layer ``layer_idx`` uses.

        The bottom ``num_prefix_layers`` are ``prefix_kind``; above that CSA and
        HCA alternate, starting from ``interleave_start``.
        """
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range for {self.num_layers} layers")
        if layer_idx < self.num_prefix_layers:
            return self.prefix_kind
        offset = layer_idx - self.num_prefix_layers
        first, second = (
            (LayerKind.CSA, LayerKind.HCA)
            if self.interleave_start == LayerKind.CSA
            else (LayerKind.HCA, LayerKind.CSA)
        )
        return first if offset % 2 == 0 else second

    def layer_config(self, layer_idx: int) -> _CoreAttnConfig:
        """The configuration object for layer ``layer_idx``.

        A pure sliding-window layer is expressed as the HCA config with its
        compressed branch switched off, so downstream code has one shape to
        handle.
        """
        kind = self.layer_kind(layer_idx)
        if kind == LayerKind.CSA:
            return self.csa
        if kind == LayerKind.HCA:
            return self.hca
        return self.hca


# The two published configurations, from the model-setup section of the report.
V4_FLASH = DSV4Config(
    num_layers=43,
    hidden_size=4096,
    csa=CSAConfig(
        hidden_size=4096,
        head_dim=512,
        num_heads=64,
        q_lora_rank=1024,
        num_out_groups=8,
        out_group_dim=1024,
        window_size=128,
        compress_rate=4,
        top_k=512,
        indexer_heads=64,
        indexer_dim=128,
    ),
    hca=HCAConfig(
        hidden_size=4096,
        head_dim=512,
        num_heads=64,
        q_lora_rank=1024,
        num_out_groups=8,
        out_group_dim=1024,
        window_size=128,
        compress_rate=128,
    ),
    num_prefix_layers=2,
    prefix_kind=LayerKind.SWA,
)

V4_PRO = DSV4Config(
    num_layers=61,
    hidden_size=7168,
    csa=CSAConfig(
        hidden_size=7168,
        head_dim=512,
        num_heads=128,
        q_lora_rank=1536,
        num_out_groups=16,
        out_group_dim=1024,
        window_size=128,
        compress_rate=4,
        top_k=1024,
        indexer_heads=64,
        indexer_dim=128,
    ),
    hca=HCAConfig(
        hidden_size=7168,
        head_dim=512,
        num_heads=128,
        q_lora_rank=1536,
        num_out_groups=16,
        out_group_dim=1024,
        window_size=128,
        compress_rate=128,
    ),
    num_prefix_layers=2,
    prefix_kind=LayerKind.HCA,
)
