"""Configuration objects for Gated DeltaNet-2.

Two levels, deliberately separate:

``GDN2Config``
    How the kernels are *executed* -- tile sizes, thread counts, dtypes,
    backend. Nothing here changes the maths.
``GatedDeltaNet2Config``
    What the *model* is -- widths, head counts, gate parameterisation. This is
    the one you put in a model config file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = ["GDN2Config", "GatedDeltaNet2Config"]


@dataclass(frozen=True)
class GDN2Config:
    """Tiling and dtype knobs for the kernel pipeline.

    Attributes:
        chunk_size: chunk length ``BS``. 64 matches the tensor-core tile shapes
            the kernels are written around.
        sub_chunk_size: ``BC``, the granularity of the overflow-safe decay
            rescaling. Smaller is safer and slower; ``BS % BC`` must be 0.
        block_DK, block_DV: channel tiles for the GEMM-heavy kernels.
        threads: threads per block for the GEMM-heavy kernels.
        num_stages: software pipeline depth.
        state_dtype: dtype of the carried state. fp32 is recommended -- the
            state accumulates over the whole sequence.
        backend: ``"auto"``, ``"tilelang"``, ``"fla"`` or ``"torch"``. See the
            module docstring for what ``"auto"`` resolves to.
    """

    chunk_size: int = 64
    sub_chunk_size: int = 16
    block_DK: int = 64
    block_DV: int = 64
    threads: int = 128
    num_stages: int = 2
    state_dtype: torch.dtype = torch.float32
    backend: str = "auto"
    backward: str = "tilelang"



DEFAULT_KERNEL_CONFIG = GDN2Config()


@dataclass
class GatedDeltaNet2Config:
    """Configuration for :class:`GatedDeltaNet2`.

    Attributes:
        hidden_size: model width.
        num_heads: number of *value* heads.
        num_kv_heads: number of *key* heads; ``num_heads`` must be a multiple of
            it. Defaults to ``num_heads``.
        head_dim_k: key/query head dim.
        head_dim_v: value head dim. Defaults to ``head_dim_k``.
        conv_size: short-convolution kernel width; 0 disables it.
        norm_eps: epsilon for the output RMSNorm.
        erase_gate_scale: upper bound of the erase gate ``b``. The paper puts
            ``b`` in ``[0, 1]``, which is the default; FLA documents ``[0, 2]``,
            matching DeltaNet's usual ``beta`` range. Values above 2 make the
            erase step non-contractive.
        a_init_range: initialisation range for ``exp(A_log)``, the per-head
            decay rate multiplier.
        dt_init_range: initialisation range for the softplus bias, controlling
            the initial expected decay per step.
        kernel: tiling configuration passed through to the kernels.
    """

    hidden_size: int
    num_heads: int
    num_kv_heads: int | None = None
    head_dim_k: int = 128
    head_dim_v: int | None = None
    conv_size: int = 4
    norm_eps: float = 1e-5
    erase_gate_scale: float = 1.0
    a_init_range: tuple[float, float] = (1.0, 16.0)
    dt_init_range: tuple[float, float] = (0.001, 0.1)
    kernel: GDN2Config = field(default_factory=GDN2Config)
