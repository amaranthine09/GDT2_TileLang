"""Benchmark the GDN-2 decode path on a GPU.

    python bench/bench_decode.py
    python bench/bench_decode.py -B 32 --dk 128 --dv 128 --check

Decoding is memory bound: each step touches the whole ``[DK, DV]`` state and
does only rank-1 work with it, so what matters is bytes moved, not FLOPs. The
table reports achieved state bandwidth -- compare it against your card's HBM
peak, not against a TFLOP/s number.

``n=1`` is ordinary autoregressive decoding. Larger ``n`` is what speculative
decoding gives you, and shows what holding the state in registers across steps
is worth: the ``per-token`` column should fall roughly as 1/n until the kernel
stops being state-bound.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from gdn2 import gdn2_decode, recurrent_gdn2


def make_inputs(B, n, H, DK, DV, dtype, device="cuda"):
    q = F.normalize(torch.randn(B, n, H, DK, device=device), dim=-1)
    k = F.normalize(torch.randn(B, n, H, DK, device=device), dim=-1)
    v = torch.randn(B, n, H, DV, device=device)
    g = -F.softplus(torch.randn(B, n, H, DK, device=device)) * 4.0
    b = torch.sigmoid(torch.randn(B, n, H, DK, device=device))
    w = torch.sigmoid(torch.randn(B, n, H, DV, device=device))
    return q.to(dtype), k.to(dtype), v.to(dtype), g, b.to(dtype), w.to(dtype)


def timed(fn, warmup=10, rep=50):
    """Median wall time in milliseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(rep):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-B", type=int, default=8, help="batch (concurrent sequences)")
    p.add_argument("-n", type=int, nargs="+", default=[1, 2, 4, 8, 16], help="tokens per launch")
    p.add_argument("-H", type=int, default=16)
    p.add_argument("--dk", type=int, default=128)
    p.add_argument("--dv", type=int, default=128)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--check", action="store_true", help="diff against the fp32 recurrence")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this benchmark needs a CUDA device")

    dtype = getattr(torch, args.dtype)
    scale = args.dk**-0.5
    # One read + one write of the state per launch, regardless of n.
    state_bytes = 2 * args.B * args.H * args.dk * args.dv * 4

    print(f"device={torch.cuda.get_device_name()}  dtype={args.dtype}  "
          f"B={args.B} H={args.H} DK={args.dk} DV={args.dv}")
    print(f"state = {state_bytes / 2 / 2**20:.1f} MiB fp32")
    print(f"{'n':>4} {'ms':>9} {'per-token us':>13} {'state GB/s':>11}  {'max rel err':>11}")

    for n in args.n:
        q, k, v, g, b, w = make_inputs(args.B, n, args.H, args.dk, args.dv, dtype)
        state = torch.zeros(args.B, args.H, args.dk, args.dv, device="cuda")

        ms = timed(lambda: gdn2_decode(q, k, v, g, b, w, state, scale))
        gbs = state_bytes / (ms * 1e-3) / 1e9

        err = ""
        if args.check:
            st = torch.zeros(args.B, args.H, args.dk, args.dv, device="cuda")
            o = gdn2_decode(q, k, v, g, b, w, st, scale)
            o_ref, _ = recurrent_gdn2(q, k, v, g, b, w, scale)
            err = f"{((o.float() - o_ref).abs().max() / o_ref.abs().max()).item():11.2e}"

        print(f"{n:>4} {ms:>9.4f} {ms * 1e3 / n:>13.2f} {gbs:>11.1f}  {err:>11}")


if __name__ == "__main__":
    main()
