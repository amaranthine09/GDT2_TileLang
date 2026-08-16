"""Per-kernel tiling and H100 compiler settings.

The baseline ``gdn2.GDN2Config`` applies one ``block_DK`` / ``block_DV`` /
``threads`` to all eighteen kernels. That is convenient and wrong: a kernel
reducing over ``DV`` into a ``[BS, DK]`` accumulator wants a different shape
from one doing a ``[BS, BS] x [BS, DV]`` product. :class:`H100Config` gives
each kernel its own :class:`TileSpec`, which is also the unit the autotuner
searches.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import tilelang
import torch

__all__ = ["TileSpec", "H100Config", "H100_PASS_CONFIGS", "SM90_ARCH"]

SM90_ARCH = "sm_90a"

# Hopper features TileLang enables by default; listed so that a regression in
# the toolchain is visible rather than silent. The DISABLE_* keys are all left
# False on purpose -- WGMMA, TMA and warp specialisation are the reason to
# target sm_90a at all.
H100_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    # cp.async / TMA pipelining behind T.Pipelined(num_stages=...)
    tilelang.PassConfigKey.TL_ENABLE_ASYNC_COPY: True,
    tilelang.PassConfigKey.TIR_USE_ASYNC_COPY: True,
    # Hopper tensor cores and bulk-async copies
    tilelang.PassConfigKey.TL_DISABLE_WGMMA: False,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
    # Producer/consumer warp groups
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    # Our kernels hold many short-lived tiles; letting them share smem raises
    # occupancy, which is what most of these kernels are actually limited by.
    tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    tilelang.PassConfigKey.TIR_MERGE_STATIC_SMEM: True,
}


@dataclass(frozen=True)
class TileSpec:
    """Launch shape for one kernel.

    Attributes:
        block_DK: key-dim tile.
        block_DV: value-dim tile.
        threads: threads per block. 128 is one Hopper warpgroup; 256 lets the
            compiler split producer and consumer warp groups.
        num_stages: software pipeline depth. Deeper hides more HBM latency and
            costs shared memory, so it interacts with the tile sizes.
    """

    block_DK: int = 64
    block_DV: int = 64
    threads: int = 128
    num_stages: int = 2

    def clamp(self, DK: int, DV: int) -> TileSpec:
        """Shrink tiles that exceed the actual head dims."""
        return replace(self, block_DK=min(self.block_DK, DK), block_DV=min(self.block_DV, DV))


def _spec(**kw) -> TileSpec:
    return field(default_factory=lambda: TileSpec(**kw))


@dataclass(frozen=True)
class H100Config:
    """Tiling for the whole pipeline, one :class:`TileSpec` per kernel.

    The defaults are reasoned, not measured -- see ``gdn2_h100/README.md`` for
    what each one is betting on. Run ``python -m gdn2_h100.tuning`` on the
    target GPU to replace them with searched values.

    Attributes:
        chunk_size: ``BS``. 64 matches the baseline; 128 halves the number of
            sequential scan steps and gives WGMMA longer m-tiles, at the cost
            of a 4x larger ``[BS, BS]`` intermediate. Worth tuning.
        sub_chunk_size: ``BC`` for the overflow-safe decay rescaling.
        fuse_solve_into_wy: build ``Ai`` in registers inside the WY kernel
            instead of writing it to HBM and reading it back.
        neumann_inverse: invert ``I + Akk`` by repeated squaring on tensor
            cores rather than by serial forward substitution.
        state_dtype: dtype of the carried state.
        use_swizzle: reorder block indices for L2 locality on the big
            chunk-parallel kernels.
    """

    chunk_size: int = 64
    sub_chunk_size: int = 16
    fuse_solve_into_wy: bool = True
    neumann_inverse: bool = True
    state_dtype: torch.dtype = torch.float32
    use_swizzle: bool = True

    # Reductions over DV into [BS, DK]: want a wide DV sweep, deep pipeline.
    intra_diag: TileSpec = _spec(block_DK=64, block_DV=64, threads=64, num_stages=1)
    inter_blocks: TileSpec = _spec(block_DK=32, block_DV=64, threads=128, num_stages=3)
    wy: TileSpec = _spec(block_DK=64, block_DV=64, threads=256, num_stages=3)
    state: TileSpec = _spec(block_DK=64, block_DV=64, threads=128, num_stages=2)
    output: TileSpec = _spec(block_DK=64, block_DV=64, threads=256, num_stages=3)
    prefill: TileSpec = _spec(block_DK=64, block_DV=64, threads=128, num_stages=2)
    decode: TileSpec = _spec(block_DK=64, block_DV=64, threads=128, num_stages=1)
    # Backward stages
    dstate: TileSpec = _spec(block_DK=64, block_DV=64, threads=128, num_stages=2)
    dwy_a: TileSpec = _spec(block_DK=64, block_DV=64, threads=256, num_stages=3)
    daqk: TileSpec = _spec(block_DK=64, block_DV=64, threads=128, num_stages=3)
    dwy_b: TileSpec = _spec(block_DK=64, block_DV=64, threads=256, num_stages=3)
    dintra_diag: TileSpec = _spec(block_DK=64, block_DV=64, threads=64, num_stages=1)
    dinter_row: TileSpec = _spec(block_DK=32, block_DV=64, threads=128, num_stages=3)
    dinter_col: TileSpec = _spec(block_DK=32, block_DV=64, threads=128, num_stages=3)
    dassemble: TileSpec = _spec(block_DK=64, block_DV=64, threads=128, num_stages=2)

    KERNELS = (
        "intra_diag", "inter_blocks", "wy", "state", "output", "prefill", "decode",
        "dstate", "dwy_a", "daqk", "dwy_b", "dintra_diag", "dinter_row", "dinter_col",
        "dassemble",
    )

    def with_spec(self, kernel: str, spec: TileSpec) -> H100Config:
        """Return a copy with one kernel's tiling replaced."""
        if kernel not in self.KERNELS:
            raise ValueError(f"unknown kernel {kernel!r}; expected one of {self.KERNELS}")
        return replace(self, **{kernel: spec})

    def save(self, path: str | Path) -> None:
        """Write the tuned tiling to JSON."""
        blob = {
            "chunk_size": self.chunk_size,
            "sub_chunk_size": self.sub_chunk_size,
            "fuse_solve_into_wy": self.fuse_solve_into_wy,
            "neumann_inverse": self.neumann_inverse,
            "use_swizzle": self.use_swizzle,
            "kernels": {k: asdict(getattr(self, k)) for k in self.KERNELS},
        }
        Path(path).write_text(json.dumps(blob, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> H100Config:
        """Read a tiling written by :meth:`save`."""
        blob = json.loads(Path(path).read_text())
        cfg = cls(
            chunk_size=blob["chunk_size"],
            sub_chunk_size=blob["sub_chunk_size"],
            fuse_solve_into_wy=blob["fuse_solve_into_wy"],
            neumann_inverse=blob["neumann_inverse"],
            use_swizzle=blob["use_swizzle"],
        )
        for name, spec in blob["kernels"].items():
            cfg = cfg.with_spec(name, TileSpec(**spec))
        return cfg
