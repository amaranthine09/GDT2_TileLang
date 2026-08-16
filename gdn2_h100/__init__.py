"""H100-tuned GDN-2 kernels.

A separate package from :mod:`gdn2` on purpose: the baseline stays verified and
untouched, and the two can be benchmarked against each other.

    from gdn2_h100 import H100Config, chunk_gdn2_fwd_h100
    cfg = H100Config.load("tuned_h100.json")   # after running the tuner

What differs from the baseline, and why, is in ``gdn2_h100/README.md``. The
maths is identical -- both are validated against the same oracles in
:mod:`gdn2.reference`.
"""

from .config import H100_PASS_CONFIGS, SM90_ARCH, H100Config, TileSpec
from .forward import chunk_gdn2_fwd_h100, neumann_wy_stages, solve_wy_kernel
from .tuning import benchmark_config, kernel_space, tune

__all__ = [
    "H100Config",
    "TileSpec",
    "H100_PASS_CONFIGS",
    "SM90_ARCH",
    "chunk_gdn2_fwd_h100",
    "neumann_wy_stages",
    "solve_wy_kernel",
    "tune",
    "kernel_space",
    "benchmark_config",
]
