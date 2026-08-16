# GDN-2 on H100

A separate package from `gdn2/`, so the verified baseline stays untouched and
the two can be raced against each other. **The maths is identical** — both are
validated against the same oracles in `gdn2/reference.py`.

```bash
python -m gdn2_h100.tuning --save tuned_h100.json   # once per GPU + shape
```
```python
from gdn2_h100 import H100Config, chunk_gdn2_fwd_h100
cfg = H100Config.load("tuned_h100.json")
o, state = chunk_gdn2_fwd_h100(q, k, v, g, b, w, scale, config=cfg)
```

## Honest status

None of this has run on an H100. I have no GPU here, so everything below is
either **proven** (a numerical property I measured in PyTorch, pinned by a
test) or **reasoned** (a structural argument about memory traffic or
dependency chains). Nothing is *measured on hardware*. The tuner exists
precisely because the remaining questions can only be answered by running it.

Treat the claimed wins as hypotheses with stated mechanisms, and let
`bench/bench_h100.py` settle them.

## What changed

### 1. The triangular inverse moved to tensor cores — proven safe

The baseline `solve_kernel` runs `BS - 2 = 62` **dependent** forward-
substitution steps per chunk. Each is an elementwise multiply over `[BS, BS]`,
a 64-way reduction, and a barrier. It does no tensor-core work at all, and 62
serialised shared-memory round trips is the single worst dependency chain in
the pipeline.

`Akk` is strictly lower triangular, hence nilpotent, so the Neumann series
terminates *exactly*:

```
(I + T)^-1 = sum_{j=0}^{BS-1} (-T)^j
```

and repeated squaring evaluates it in `log2(BS)` steps:

```
P_1 = I,  Q_1 = -T
P_2m = P_m + Q_m P_m,    Q_2m = Q_m^2
```

**6 steps of 2 GEMMs, instead of 62 dependent reductions.**

This is only usable if it is numerically safe, which is not obvious — squaring
can amplify. Measured against a float64 solve, on the worst case (every key
identical, which maximises `|T|`):

| iterate dtype | relative error |
| --- | --- |
| bf16 | 4.9e-04 |
| fp16 | 5.9e-05 |
| **fp32** | **2.3e-09** |
| serial substitution (fp32) | 1.5e-09 |

So fp32 iterates are exactly as accurate as the method they replace, and
**bf16 is not good enough** — it loses four orders of magnitude, and `Ai` feeds
the state scan where error compounds across chunks. The kernel therefore holds
`P` and `Q` in fp32 (TF32 GEMMs) and rounds to bf16 once at the end, which is
precisely what the baseline does when it stores `Ai`.

The reason this works at all: the decay gates keep `|T| < 0.2` and
`‖(I+T)^-1‖ = 1.0` even at full key correlation. Without gating, a strictly
lower all-ones `T` would give inverse entries growing like `2^r` and no series
would help. `test_gating_keeps_the_inverse_well_conditioned` pins that premise.

### 2. `Ai` never reaches HBM — reasoned

It is produced by the solve and consumed only by WY, so `solve_wy_kernel`
fuses them and keeps it in shared memory. That removes a `[B, S, H, BS]` write
and read: at `B=2, S=8192, H=16` roughly **270 MB of traffic per forward**.
Seven forward kernels become six.

### 3. Per-kernel tiling — reasoned

The baseline applies one `block_DK`/`block_DV`/`threads` to all eighteen
kernels. A kernel reducing over `DV` into a `[BS, DK]` accumulator wants a
different shape from one doing `[BS, BS] x [BS, DV]`. `H100Config` gives each
kernel its own `TileSpec`, which is also the unit the tuner searches.

### 4. Hopper features made explicit — reasoned

`H100_PASS_CONFIGS` pins WGMMA, TMA lowering, warp specialisation and
`cp.async` **on**, and enables aggressive shared-memory merging (these kernels
hold many short-lived tiles; sharing them raises occupancy, which is what most
of them are limited by). TileLang enables most of these by default — listing
them means a toolchain regression shows up as a diff rather than a silent
slowdown.

`threads=256` on the GEMM-heavy kernels gives the compiler two warp groups to
split into producer and consumer; `128` is a single warpgroup. Which wins is a
tuner question.

## Autotuning

`python -m gdn2_h100.tuning` searches **per kernel but measures end to end**,
which is the number that matters — a bigger `block_DV` in WY changes how much
L2 the state scan finds warm.

Coordinate descent, not exhaustive: the full product is ~10^24 configs, one
pass over each kernel holding the rest fixed is **804** and captures nearly all
of the win. Two passes by default, since the second sees the first's
improvements. `chunk_size` and `sub_chunk_size` are an outer loop because they
change every stage at once. Results cache to JSON; tune once per (GPU, shape).

Configs that fail to compile — an out-of-shared-memory tile — score `inf` and
are skipped. That is a normal search outcome, not an error.

## Considered and *not* done

Being explicit about what was rejected, and why, so nobody re-derives it:

- **Blocked triangular inverse** (invert `BC x BC` diagonal blocks, merge with
  small GEMMs). Asymptotically cheaper than Neumann — ~10 GEMMs of `16^3`
  versus 12 of `64^3`. Not done because TileLang's eager builder turns every
  Python `for` into a TIR loop, so a dict of per-block accumulators cannot be
  indexed; it needs manual unrolling per `NSUB`. Neumann gets most of the win
  with none of that. **This is the next thing to try if the solve still shows
  up in a profile.**
- **`chunk_size = 128`.** Halves the sequential scan steps and gives WGMMA
  longer m-tiles, but the `[BS, BS]` Neumann buffers grow 4x — at fp32 that is
  128 KB for `P` and `Q` alone, which will not fit alongside the tiles. Left in
  the tuner's outer loop so it gets measured, but expect it to need the blocked
  inverse first.
- **Two-level (Blelloch) scan over chunks.** The state scan is sequential over
  `NC` with only `B*H*(DV/block_DV)` blocks — at `B=1, H=16, DV=128` that is 32
  blocks on 132 SMs, ~24% occupancy. A scan-then-propagate pass would fix it.
  This is the **largest remaining win for small batches** and is real work: the
  chunk transition is an affine map on the state, so the combining operator is
  well-defined but not cheap to store.
- **Persistent kernels / `T.use_swizzle`.** Cheap to add, but with no way to
  measure L2 behaviour, adding them blind is as likely to hurt as help. The
  config flag exists (`use_swizzle`); wiring it is a one-liner once there are
  numbers.
- **fp8 for the intra-chunk matrices.** H100 has the tensor cores for it, but
  `Aqk`/`Akk` feed a matrix inverse and the fp32-vs-bf16 measurement above
  shows how little headroom there is. Not without a lot of error analysis.

## What to run first

```bash
python -m pytest tests/test_h100.py -v -m cuda   # correctness before speed
python bench/bench_h100.py --check               # baseline vs H100, per stage
python -m gdn2_h100.tuning --quick               # sanity-check the tuner
python -m gdn2_h100.tuning --save tuned_h100.json
```

If `bench_h100.py` shows the fused solve+WY kernel *slower* than the baseline
pair, the likely cause is shared-memory pressure from the fp32 `P`/`Q` buffers
capping occupancy — check with `--ptxas-options=-v` and try `threads=128`.
