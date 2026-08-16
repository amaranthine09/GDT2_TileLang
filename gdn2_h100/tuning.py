"""Autotuning for every kernel in the pipeline.

    python -m gdn2_h100.tuning --save tuned_h100.json
    python -m gdn2_h100.tuning -B 4 -S 8192 -H 16 --dk 128 --dv 128 --quick

Why not ``@tilelang.autotuner.autotune`` on each factory: that tunes one kernel
against synthetic inputs in isolation, and picks the config that minimises
*that* kernel. Our pipeline is fifteen kernels whose costs interact -- a bigger
``block_DV`` in the WY kernel changes how much L2 the state scan finds warm,
and the chunk size changes every stage at once. So this searches per kernel but
*measures end to end*, which is the number that matters, and it searches the
two whole-pipeline knobs (``chunk_size``, ``sub_chunk_size``) as an outer loop.

Coordinate descent, not exhaustive: the full product of fifteen kernels' spaces
is astronomically large, while one pass over each kernel holding the rest fixed
is minutes and captures nearly all of the win. Two passes by default, because
the second pass sees the first pass's improvements.

Everything is cached to JSON. Tune once per (GPU, shape) and load it:

    from gdn2_h100 import H100Config
    cfg = H100Config.load("tuned_h100.json")
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import H100Config, TileSpec

__all__ = ["kernel_space", "tune", "benchmark_config"]

# Per-kernel search spaces. Kept deliberately small and shape-aware: 128 threads
# is one Hopper warpgroup and 256 lets the compiler split producer/consumer, so
# there is little point going wider. num_stages beyond 4 exhausts shared memory
# at these tile sizes.
_GEMM_HEAVY = dict(
    block_DK=(32, 64, 128),
    block_DV=(32, 64, 128),
    threads=(128, 256),
    num_stages=(1, 2, 3, 4),
)
_STATE_BOUND = dict(  # sequential over chunks: tiles matter more than depth
    block_DK=(64, 128),
    block_DV=(32, 64, 128),
    threads=(128, 256),
    num_stages=(1, 2, 3),
)
_TOKEN_PARALLEL = dict(  # one block per token; tiny blocks, no pipelining win
    block_DK=(64,),
    block_DV=(64,),
    threads=(32, 64, 128),
    num_stages=(1, 2),
)

SPACES = {
    "intra_diag": _TOKEN_PARALLEL,
    "dintra_diag": _TOKEN_PARALLEL,
    "inter_blocks": dict(_GEMM_HEAVY, block_DK=(16, 32, 64)),
    "dinter_row": dict(_GEMM_HEAVY, block_DK=(16, 32, 64)),
    "dinter_col": dict(_GEMM_HEAVY, block_DK=(16, 32, 64)),
    "wy": _GEMM_HEAVY,
    "dwy_a": _GEMM_HEAVY,
    "dwy_b": _GEMM_HEAVY,
    "daqk": _GEMM_HEAVY,
    "output": _GEMM_HEAVY,
    "dassemble": _GEMM_HEAVY,
    "state": _STATE_BOUND,
    "dstate": _STATE_BOUND,
    "prefill": _STATE_BOUND,
    "decode": _STATE_BOUND,
}


def kernel_space(kernel: str) -> list[TileSpec]:
    """Every :class:`TileSpec` considered for one kernel."""
    space = SPACES[kernel]
    keys = ("block_DK", "block_DV", "threads", "num_stages")
    return [TileSpec(**dict(zip(keys, combo))) for combo in itertools.product(*(space[k] for k in keys))]


def _inputs(B, S, H, DK, DV, dtype, device="cuda"):
    q = F.normalize(torch.randn(B, S, H, DK, device=device), dim=-1).to(dtype)
    k = F.normalize(torch.randn(B, S, H, DK, device=device), dim=-1).to(dtype)
    v = torch.randn(B, S, H, DV, device=device, dtype=dtype)
    g = -F.softplus(torch.randn(B, S, H, DK, device=device)) * 4.0
    b = torch.sigmoid(torch.randn(B, S, H, DK, device=device)).to(dtype)
    w = torch.sigmoid(torch.randn(B, S, H, DV, device=device)).to(dtype)
    return q, k, v, g, b, w


def benchmark_config(cfg: H100Config, shape, dtype=torch.bfloat16, train=True,
                     warmup=3, rep=10) -> float:
    """Median milliseconds for one forward (plus backward if ``train``).

    Returns ``inf`` if the config fails to compile or run -- an out-of-shared-
    memory tile is a normal search outcome, not an error.
    """
    from gdn2.backward import chunk_gdn2_bwd

    from .forward import chunk_gdn2_fwd_h100

    B, S, H, DK, DV = shape
    q, k, v, g, b, w = _inputs(B, S, H, DK, DV, dtype)
    scale = DK**-0.5
    do = torch.randn(B, S, H, DV, device="cuda", dtype=dtype)

    def run():
        o, _ = chunk_gdn2_fwd_h100(q, k, v, g, b, w, scale, None, False, cfg)
        if train:
            chunk_gdn2_bwd(q, k, v, g, b, w, scale, None, do, None, _as_baseline(cfg))
        return o

    try:
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()
        times = []
        for _ in range(rep):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
        times.sort()
        return times[len(times) // 2]
    except Exception:
        return float("inf")


def _as_baseline(cfg: H100Config):
    """Adapt an :class:`H100Config` to the baseline's single-spec config.

    The backward still lives in ``gdn2.backward``, which takes one global
    tiling. Feed it the ``dwy_a`` spec, which is the shape most of its kernels
    share.
    """
    from gdn2 import GDN2Config

    return GDN2Config(
        chunk_size=cfg.chunk_size, sub_chunk_size=cfg.sub_chunk_size,
        block_DK=cfg.dwy_a.block_DK, block_DV=cfg.dwy_a.block_DV,
        threads=cfg.dwy_a.threads, num_stages=cfg.dwy_a.num_stages,
        state_dtype=cfg.state_dtype,
    )


def tune(shape, dtype=torch.bfloat16, train=True, passes=2, chunk_sizes=(64,),
         sub_chunk_sizes=(16,), kernels=None, verbose=True) -> tuple[H100Config, float]:
    """Coordinate-descent search over the per-kernel tilings.

    Args:
        shape: ``(B, S, H, DK, DV)`` to tune for. Tune at your serving or
            training shape -- the winner is shape-specific.
        dtype: input dtype.
        train: include a backward in the measured step.
        passes: sweeps over the kernel list. The second pass sees the first's
            improvements; a third rarely moves anything.
        chunk_sizes, sub_chunk_sizes: outer loop over the whole-pipeline knobs.
        kernels: restrict to these kernels (default: all).
        verbose: print each improvement as it is found.

    Returns:
        ``(best_config, best_ms)``.
    """
    kernels = list(kernels or H100Config.KERNELS)
    best_cfg, best_ms = None, float("inf")

    for cs, bcs in itertools.product(chunk_sizes, sub_chunk_sizes):
        if cs % bcs or cs // bcs < 2:
            continue
        cfg = H100Config(chunk_size=cs, sub_chunk_size=bcs)
        ms = benchmark_config(cfg, shape, dtype, train)
        if verbose:
            print(f"[chunk {cs}/{bcs}] baseline {ms:.3f} ms")
        if ms == float("inf"):
            continue

        for p in range(passes):
            for name in kernels:
                cur = getattr(cfg, name)
                for spec in kernel_space(name):
                    if spec == cur:
                        continue
                    trial = cfg.with_spec(name, spec)
                    t = benchmark_config(trial, shape, dtype, train)
                    if t < ms - 1e-4:
                        ms, cfg, cur = t, trial, spec
                        if verbose:
                            print(f"  pass{p} {name:<13} {spec} -> {ms:.3f} ms")
        if ms < best_ms:
            best_cfg, best_ms = cfg, ms

    return best_cfg, best_ms


def main():
    p = argparse.ArgumentParser(description="Autotune the GDN-2 H100 pipeline.")
    p.add_argument("-B", type=int, default=2)
    p.add_argument("-S", type=int, default=4096)
    p.add_argument("-H", type=int, default=16)
    p.add_argument("--dk", type=int, default=128)
    p.add_argument("--dv", type=int, default=128)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--inference", action="store_true", help="tune the forward only")
    p.add_argument("--chunk", type=int, nargs="+", default=[64, 128])
    p.add_argument("--sub-chunk", type=int, nargs="+", default=[16, 32])
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--quick", action="store_true", help="one pass, chunk 64 only")
    p.add_argument("--save", default="tuned_h100.json")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("autotuning needs a CUDA device")

    chunks = [64] if args.quick else args.chunk
    subs = [16] if args.quick else args.sub_chunk
    shape = (args.B, args.S, args.H, args.dk, args.dv)
    print(f"tuning {torch.cuda.get_device_name()} for B={args.B} S={args.S} "
          f"H={args.H} DK={args.dk} DV={args.dv} "
          f"({'forward only' if args.inference else 'forward+backward'})")

    cfg, ms = tune(shape, getattr(torch, args.dtype), train=not args.inference,
                   passes=1 if args.quick else args.passes,
                   chunk_sizes=chunks, sub_chunk_sizes=subs)
    if cfg is None:
        raise SystemExit("no config compiled successfully")
    cfg.save(args.save)
    print(f"\nbest {ms:.3f} ms -> {Path(args.save).resolve()}")
    for name in H100Config.KERNELS:
        print(f"  {name:<13} {getattr(cfg, name)}")


if __name__ == "__main__":
    main()
