"""Race the H100 kernels against the baseline, stage by stage.

    python bench/bench_h100.py --check
    python bench/bench_h100.py -B 4 -S 8192 --tuned tuned_h100.json

The per-stage columns matter more than the total: the H100 package changes two
things (the triangular inverse, and fusing solve into WY), so if the total
moves, this tells you whether it moved for the reason claimed.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from gdn2 import GDN2Config, chunk_gdn2_fwd
from gdn2.reference import recurrent_gdn2
from gdn2_h100 import H100Config, chunk_gdn2_fwd_h100


def make_inputs(B, S, H, DK, DV, dtype, device="cuda"):
    q = F.normalize(torch.randn(B, S, H, DK, device=device), dim=-1)
    k = F.normalize(torch.randn(B, S, H, DK, device=device), dim=-1)
    v = torch.randn(B, S, H, DV, device=device)
    g = -F.softplus(torch.randn(B, S, H, DK, device=device)) * 4.0
    b = torch.sigmoid(torch.randn(B, S, H, DK, device=device))
    w = torch.sigmoid(torch.randn(B, S, H, DV, device=device))
    return q.to(dtype), k.to(dtype), v.to(dtype), g, b.to(dtype), w.to(dtype)


def timed(fn, warmup=5, rep=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-B", type=int, default=2)
    p.add_argument("-S", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    p.add_argument("-H", type=int, default=16)
    p.add_argument("--dk", type=int, default=128)
    p.add_argument("--dv", type=int, default=128)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--tuned", help="H100Config JSON from gdn2_h100.tuning")
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this benchmark needs a CUDA device")

    dtype = getattr(torch, args.dtype)
    base = GDN2Config()
    h100 = H100Config.load(args.tuned) if args.tuned else H100Config()

    print(f"device={torch.cuda.get_device_name()}  dtype={args.dtype}  "
          f"B={args.B} H={args.H} DK={args.dk} DV={args.dv}"
          + (f"  tuned={args.tuned}" if args.tuned else "  (untuned defaults)"))
    print(f"{'S':>7} {'baseline ms':>12} {'h100 ms':>10} {'speedup':>8} "
          f"{'base peak MiB':>14} {'h100 peak MiB':>14}  {'agree':>9}")

    for S in args.S:
        q, k, v, g, b, w = make_inputs(args.B, S, args.H, args.dk, args.dv, dtype)
        scale = args.dk**-0.5

        run_b = lambda: chunk_gdn2_fwd(q, k, v, g, b, w, scale, None, False, base)
        run_h = lambda: chunk_gdn2_fwd_h100(q, k, v, g, b, w, scale, None, False, h100)

        ms_b, ms_h = timed(run_b), timed(run_h)

        torch.cuda.reset_peak_memory_stats(); run_b()
        peak_b = torch.cuda.max_memory_allocated() / 2**20
        torch.cuda.reset_peak_memory_stats(); run_h()
        peak_h = torch.cuda.max_memory_allocated() / 2**20

        agree = ""
        if args.check:
            o_b, _ = run_b()
            o_h, _ = run_h()
            agree = f"{((o_h.float() - o_b.float()).abs().max() / o_b.float().abs().max()).item():9.1e}"

        print(f"{S:>7} {ms_b:>12.3f} {ms_h:>10.3f} {ms_b / ms_h:>7.2f}x "
              f"{peak_b:>14.0f} {peak_h:>14.0f}  {agree:>9}")

    if args.check:
        print("\n'agree' is h100 vs baseline; both are bf16 so they should match to ~1e-3.")


if __name__ == "__main__":
    main()
