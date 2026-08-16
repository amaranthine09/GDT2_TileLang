"""Benchmark and validate the GDN-2 TileLang kernels on a GPU.

    python bench/bench_gdn2.py                 # default sweep
    python bench/bench_gdn2.py --check         # also diff against the reference
    python bench/bench_gdn2.py -B 4 -S 8192 -H 16 --dk 128 --dv 128

Reports the wall time of our TileLang forward against two baselines:

- ``fla``: flash-linear-attention's Triton ``chunk_gdn2``. This is the real
  kernel-vs-kernel comparison, and the number that matters. Shown only when
  ``flash-linear-attention`` is installed.
- ``torch``: the PyTorch chunkwise form. Not a fair kernel baseline -- it loops
  over chunks in Python -- but it is the fallback used off-CUDA.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from gdn2 import fla_backend
from gdn2 import GDN2Config, chunk_gdn2_fwd
from gdn2.reference import chunk_gdn2_torch, recurrent_gdn2


def make_inputs(B, S, H, DK, DV, dtype, device="cuda"):
    q = F.normalize(torch.randn(B, S, H, DK, device=device), dim=-1)
    k = F.normalize(torch.randn(B, S, H, DK, device=device), dim=-1)
    v = torch.randn(B, S, H, DV, device=device)
    g = -F.softplus(torch.randn(B, S, H, DK, device=device)) * 4.0
    b = torch.sigmoid(torch.randn(B, S, H, DK, device=device))
    w = torch.sigmoid(torch.randn(B, S, H, DV, device=device))
    return q.to(dtype), k.to(dtype), v.to(dtype), g, b.to(dtype), w.to(dtype)


def timed(fn, *args, warmup=5, rep=20, **kwargs):
    """Median wall time in milliseconds."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    times = []
    for _ in range(rep):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


def flops(B, S, H, DK, DV, chunk_size):
    """Dominant GEMM FLOPs in the chunkwise forward.

    Four ``[BS, BS] x [BS, D]``-class products per chunk (the two intra-chunk
    matrices, the two WY applications) plus two ``[BS, DK] x [DK, DV]``
    cross-chunk products.
    """
    nc = S // chunk_size
    per_chunk = 2 * chunk_size * chunk_size * (2 * DK + 2 * DV) + 2 * 2 * chunk_size * DK * DV
    return B * H * nc * per_chunk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-B", type=int, default=2)
    p.add_argument("-S", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    p.add_argument("-H", type=int, default=8)
    p.add_argument("--dk", type=int, default=128)
    p.add_argument("--dv", type=int, default=128)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--sub-chunk", type=int, default=16)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--check", action="store_true", help="diff against the fp32 recurrence")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this benchmark needs a CUDA device")

    dtype = getattr(torch, args.dtype)
    cfg = GDN2Config(chunk_size=args.chunk, sub_chunk_size=args.sub_chunk)
    scale = args.dk**-0.5
    has_fla = fla_backend.is_available()
    print(f"device={torch.cuda.get_device_name()}  dtype={args.dtype}  "
          f"B={args.B} H={args.H} DK={args.dk} DV={args.dv} chunk={args.chunk}/{args.sub_chunk}")
    if not has_fla:
        print("note: flash-linear-attention not installed, skipping the Triton baseline")
    print(f"{'S':>7} {'tilelang ms':>12} {'TFLOP/s':>9} {'fla ms':>9} {'vs fla':>7} "
          f"{'torch ms':>10}  {'max rel err':>11}")

    for S in args.S:
        q, k, v, g, b, w = make_inputs(args.B, S, args.H, args.dk, args.dv, dtype)

        def run_tl():
            return chunk_gdn2_fwd(q, k, v, g, b, w, scale, None, False, cfg)

        def run_torch():
            return chunk_gdn2_torch(q, k, v, g, b, w, scale, chunk_size=args.chunk,
                                    sub_chunk_size=args.sub_chunk)

        ms_tl = timed(run_tl)
        ms_pt = timed(run_torch, warmup=2, rep=5)
        tfs = flops(args.B, S, args.H, args.dk, args.dv, args.chunk) / (ms_tl * 1e-3) / 1e12

        ms_fla, ratio = float("nan"), ""
        if has_fla:
            ms_fla = timed(lambda: fla_backend.chunk_gdn2_fla(q, k, v, g, b, w, scale))
            ratio = f"{ms_fla / ms_tl:.2f}x"

        err = ""
        if args.check:
            o_tl, _ = run_tl()
            # The literal recurrence is O(S) Python steps, so only check a prefix.
            n = min(S, 512)
            o_ref, _ = recurrent_gdn2(q[:, :n], k[:, :n], v[:, :n], g[:, :n], b[:, :n], w[:, :n], scale)
            rel = ((o_tl[:, :n].float() - o_ref).abs().max() / o_ref.abs().max()).item()
            err = f"{rel:11.2e}"

        fla_cell = f"{ms_fla:9.3f}" if has_fla else f"{'-':>9}"
        print(f"{S:>7} {ms_tl:>12.3f} {tfs:>9.1f} {fla_cell} {ratio:>7} {ms_pt:>10.3f}  {err:>11}")


if __name__ == "__main__":
    main()
