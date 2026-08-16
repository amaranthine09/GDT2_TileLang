"""Gated DeltaNet-2 (GDN-2) linear attention, chunkwise, in TileLang.

    from gdn2 import GatedDeltaNet2, GatedDeltaNet2Config
    layer = GatedDeltaNet2(GatedDeltaNet2Config(hidden_size=2048, num_heads=16))

Module layout::

    config.py      what to build (model) and how to run it (kernel tiling)
    attention.py   gdn2_attention() + the GatedDeltaNet2 module   <- start here
    forward.py     forward TileLang kernels + their driver
    backward.py    backward TileLang kernels + their driver
    reference.py   PyTorch oracles: the recurrence, the chunkwise form,
                   the staged backward. Also the CPU fallback.
"""

from .attention import GDN2Cache, GatedDeltaNet2, gdn2_attention, resolve_backend
from .backward import chunk_gdn2_bwd
from .config import GDN2Config, GatedDeltaNet2Config
from .forward import chunk_gdn2_fwd, gdn2_decode_step
from .reference import chunk_gdn2_bwd_torch, chunk_gdn2_torch, recurrent_gdn2

__all__ = [
    # model
    "GatedDeltaNet2",
    "GatedDeltaNet2Config",
    "GDN2Cache",
    # functional op
    "gdn2_attention",
    "gdn2_decode_step",
    "GDN2Config",
    # kernel drivers
    "chunk_gdn2_fwd",
    "chunk_gdn2_bwd",
    # references / oracles
    "recurrent_gdn2",
    "chunk_gdn2_torch",
    "chunk_gdn2_bwd_torch",
    "resolve_backend",
]
