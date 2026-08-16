"""DeepSeek-V4 hybrid attention in TileLang.

Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA) from
*DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*
(arXiv:2606.19348), as TileLang kernels with hand-written backward passes, a
differentiable PyTorch oracle, and ``nn.Module`` layers.

    from dsv4 import CSAAttention, V4_FLASH

    layer = CSAAttention(V4_FLASH.csa).cuda().bfloat16()
    out = layer(hidden_states)          # [B, N, d]

Layers run on CPU through the PyTorch reference and on CUDA through the
kernels; ``backend="auto"`` picks between them. See ``README.md`` for what is
verified and what is not.
"""

from .attention import CSAAttention, HCAAttention, SWAAttention, build_attention_stack
from .config import (
    DEFAULT_KERNEL_CONFIG,
    V4_FLASH,
    V4_PRO,
    CSAConfig,
    DSV4Config,
    HCAConfig,
    KernelConfig,
    LayerKind,
)
from .ops import (
    csa_compress,
    dense_attention,
    hca_compress,
    index_and_select,
    indexer_kl_loss,
    lightning_index,
    sparse_attention,
    topk_select,
)

__all__ = [
    "CSAAttention",
    "HCAAttention",
    "SWAAttention",
    "build_attention_stack",
    "CSAConfig",
    "HCAConfig",
    "DSV4Config",
    "KernelConfig",
    "LayerKind",
    "DEFAULT_KERNEL_CONFIG",
    "V4_FLASH",
    "V4_PRO",
    "csa_compress",
    "hca_compress",
    "lightning_index",
    "topk_select",
    "index_and_select",
    "indexer_kl_loss",
    "sparse_attention",
    "dense_attention",
]
