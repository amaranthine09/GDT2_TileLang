# Gated DeltaNet-2 in TileLang

Chunkwise GDN-2 linear attention: TileLang kernels, a PyTorch reference, and an
`nn.Module` token mixer.

GDN-2 ([arXiv:2605.22791](https://arxiv.org/abs/2605.22791)) splits the delta
rule's single scalar gate into a channel-wise **erase** gate on the key axis and
a channel-wise **write** gate on the value axis. Per head, per token:

```
S_t = (I - k_t (b_t * k_t)^T) diag(a_t) S_{t-1} + k_t (w_t * v_t)^T
o_t = S_t^T q_t
```

| symbol | shape | meaning |
| --- | --- | --- |
| `S_t` | `[DK, DV]` | recurrent state |
| `q_t, k_t` | `[DK]` | query / key, L2-normalised per head |
| `v_t` | `[DV]` | value |
| `a_t = exp(g_t)` | `[DK]` | channel-wise decay, `g_t <= 0` |
| `b_t` | `[DK]` | channel-wise **erase** gate |
| `w_t` | `[DV]` | channel-wise **write** gate |

Tying `b_t = w_t = beta_t` recovers KDA; additionally collapsing `a_t` to a
scalar recovers Gated DeltaNet. Both degeneracies are covered by tests.

## Install and run

```bash
pip install tilelang torch
```

Optional, and only for the benchmark's comparison column:

```bash
pip install flash-linear-attention
```

```bash
python -m pytest tests/ -v
```

The maths, gradient, and layer tests run on CPU. Kernel tests are marked `cuda`
and skip without a GPU:

```bash
python -m pytest tests/ -v -m cuda
```

```bash
python bench/bench_gdn2.py --check
```

## Use

```python
import torch
from gdn2 import GatedDeltaNet2, GatedDeltaNet2Config

layer = GatedDeltaNet2(GatedDeltaNet2Config(
    hidden_size=2048, num_heads=16, num_kv_heads=8,
    head_dim_k=128, head_dim_v=128,
)).cuda().bfloat16()

x = torch.randn(2, 4096, 2048, device="cuda", dtype=torch.bfloat16)
out, _ = layer(x)                       # [2, 4096, 2048]
```

### Inference

Serving is a pair — `prefill` then `decode` — and neither goes through
`forward()`, which is the training path and builds an autograd graph.

```python
out, cache = layer.prefill(prompt)              # prompt -> cache
for _ in range(n_new):
    tok, cache = layer.step(tok_hidden, cache)  # one token at a time
```

**Prefill** runs the same chunkwise maths as training with different plumbing.
The training forward stores the per-chunk states `h`, shape
`[B, NC, H, DK, DV]` — hundreds of MB on a long prompt — so its output kernel
can run parallel over chunks and so the backward can reuse them. Serving needs
neither, so `prefill_scan_kernel` fuses the scan with the output and never
materialises `h`. The trade is chunk parallelism for that allocation; with a
real serving batch there is plenty left in `B × H × DV-tiles`. A ragged tail
routes through the decode kernel rather than padding, so a 1000-token prompt
does not pay for 1024. Call it repeatedly, passing the cache back, to prefill
in segments.

**Decode** is memory bound rather than compute bound. Each step touches the
whole `[DK, DV]` state — 64 KiB per head at 128×128 fp32 — and does only rank-1
work with it, so state traffic *is* the cost.

When you have several tokens at once — a speculative-decoding draft, a short
prefill tail — pass them together. `decode_multi_kernel` keeps the state in
registers across all `n` steps, so it is read and written **once per call
instead of once per token**:

```python
out, cache = layer.decode(draft_hidden, cache)  # [B, n, hidden]
```

The recurrence is still sequential in `n`; what is saved is the memory traffic,
which at `n = 8` turns 16 state round trips into 2.

The functional op, if you already have projections and gates:

```python
from gdn2 import gdn2_attention
o, final_state = gdn2_attention(q, k, v, g, b, w, output_final_state=True)
```

All tensors are `[B, S, H, D]`. `g` is fp32 natural-log decay (`<= 0`). Sequence
lengths that aren't a multiple of the chunk size are zero-padded internally;
padding is inert because `g = 0` holds the decay at 1 while `b = w = 0` makes
both the erase and the write vanish.

### Backends

Forward **and backward** are TileLang. There is no third-party kernel
dependency — flash-linear-attention is a benchmark target, not a runtime one.

| backend | forward | backward |
| --- | --- | --- |
| `tilelang` | 7 chunkwise + 3 serving kernels | 8 kernels |
| `torch` | PyTorch | staged PyTorch backward |

`auto` (the default) is `tilelang` on CUDA, `torch` elsewhere. The backward can
be pinned separately for debugging — `GDN2Config(backward="torch")` runs the
exact staged PyTorch backward against the TileLang forward, which is how you
localise a bad gradient to a kernel.

## Chunkwise algorithm

Substituting `S_t = diag(gamma_t) S_hat_t` with `gamma_t = exp(cumsum(g))`
absorbs the decay and leaves a plain delta rule, which unrolls over a chunk into
a WY representation. Per chunk, with `S` the state entering it:

```
Tm  = strict_tril(E K_bar^T)      E_r     = b_r * k_r * gamma_r
Ai  = (I + Tm)^{-1}               K_bar_s = k_s / gamma_s
W   = Ai E                        U       = Ai (w * v)
Un  = U - W S
O   = Q_gamma S + tril(Aqk) Un    Q_gamma_r = scale * q_r * gamma_r
S'  = diag(gamma_C) S + K_tail^T Un
```

Only the state scan over chunks is sequential; everything else is batched GEMMs.

### Kernels

| # | kernel | grid | role |
| --- | --- | --- | --- |
| 1 | `cumsum_kernel` | chunk × BH | `G = log2(e) · cumsum(g)` within each chunk |
| 2 | `intra_diag_kernel` | token × H | diagonal sub-blocks of `Aqk`, `Akk` |
| 3 | `inter_blocks_kernel` | sub-block × chunk × BH | off-diagonal sub-blocks |
| 4 | `solve_kernel` | chunk × BH | `Ai = (I + Akk)^{-1}` |
| 5 | `wy_kernel` | chunk × BH | `W`, `U`, `KG` |
| 6 | `state_kernel` | DV-tile × BH | sequential scan → per-chunk states, `Un` |
| 7 | `output_kernel` | DV-tile × chunk × BH | `o` |

Decoding does not use these. `gdn2/inference.py` has `decode_step_kernel`
(one token) and `decode_multi_kernel` (`n` tokens, state resident in registers),
both applying the recurrence directly — with a handful of tokens there is no
intra-chunk structure worth the WY machinery.

Kernels 2 and 3 write disjoint column ranges of the same two matrices and must
run in that order. All decay arithmetic is in log2 space so the kernels can use
`exp2`.

### Overflow safety

The WY form needs `k_s / gamma_s`, and `1 / gamma_s` overflows fp32 once a
channel's cumulative decay within a chunk passes ~88 nats. Real gates reach
this: with `g = -exp(A_log) · softplus(·)` a fast channel can lose `e^-4` per
token, i.e. `e^-256` across a 64-token chunk.

So no block of `Aqk` or `Akk` is ever formed from `exp(G) · exp(-G)` directly.
Each is rescaled by the cumulative decay at the **first row of its row
sub-block**, which puts both exponents at or below zero:

```
row[r, c] = f_r[c] * exp2(G_r[c] - ref[c])     r at or after the reference row
col[s, c] = k_s[c] * exp2(ref[c] - G_s[c])     s strictly before it
```

Underflow to zero is the right answer there, since the true product is itself
vanishing. Diagonal sub-blocks have no valid reference row, so kernel 2 computes
them token-parallel from the exact pairwise difference `exp2(G_r - G_s)`, which
is non-positive for `r >= s` by construction.

`test_fast_forgetting_channels_do_not_overflow` exercises this at a decay scale
that loses `e^-1900` per chunk.

## Status

Verified on CPU (80 tests):

- the chunkwise WY form reproduces the literal recurrence to ~1e-6 relative,
  across chunk/sub-chunk sizes, with and without an initial state;
- **the seven-kernel decomposition** reproduces it too —
  `tests/test_pipeline_simulation.py` transcribes each kernel stage into
  PyTorch (log2 decay, per-sub-block reference rows, the exact pairwise
  diagonal, the substitution schedule, the state scan) and checks the result
  against the recurrence, so indexing and masking mistakes in the kernel
  *design* surface without a GPU;
- gradients match those of the literal recurrence;
- the backward matches autograd through the chunkwise form to ~1e-7 on every
  gradient, including the decay;
- reduction to KDA when the two gates are tied; zero gates leave the state to
  pure decay;
- the serving path agrees with the parallel forward end to end: prefill (whole
  and segmented, with a ragged tail), an `n`-token draft, and single steps all
  stitch back to one `forward()`;
- all 18 kernels parse to valid TVM PrimFuncs, for each `BS`/`BC` ratio and
  each decode length.

**Not yet run on a GPU.** The algorithm is validated and every kernel parses,
but no numerical result has come out of compiled TileLang code — the machine
this was written on has no CUDA device. What remains unverified is the TileLang
lowering itself: tile/layout inference, shared-memory budgets, the
synchronisation the compiler inserts around the serial substitution loop and
the state scan, and `T.gemm` on the small `[BC, BC]` tiles. Before trusting the
kernels, run:

```bash
python -m pytest tests/ -v -m cuda
python bench/bench_gdn2.py --check
```

Two library-level hazards were found and worked around during development, both
worth re-checking against your tilelang version:

- `from __future__ import annotations` breaks `T.prim_func`: TVMScript evaluates
  `T.Tensor(...)` parameter annotations at definition time, and postponed
  annotations turn them into unresolvable strings. `gdn2/kernels.py` therefore
  deliberately omits that import.
- `T.cumsum(dim=0)` on a non-square 2-D tile is out of bounds in tilelang 0.1.9:
  `CumSum2D<Axis=0>` addresses `src[col * W + gRow]` while iterating `col` up to
  `W`. Kernel 1 uses an explicit log-step scan instead.

## Limitations and next steps

- **No fused TileLang backward.** The TileLang side is forward only; its
  backward re-runs the chunkwise PyTorch form under `enable_grad`. Gradients are
  exact (same algebra, fp32) and it is all batched GEMMs with one Python loop
  over chunks, but training step time is dominated by it. Use the `fla` backend
  for training until a fused TileLang backward exists — that is the main piece
  of work remaining here.
- **`solve_kernel` is the serial stage**: `BS - 2` substitution steps per chunk.
  The blocked variant — invert `BC × BC` diagonal blocks, then merge with small
  GEMMs — is the first optimisation to try. It was not written that way here
  because tilelang's eager builder turns every Python `for` into a TIR loop, so
  a dict of per-block accumulators cannot be indexed; the blocked form needs
  manual unrolling.
- **No varlen in the TileLang path.** Use the `fla` backend for packed
  sequences. Adding it here means chunk-boundary handling in kernels 1, 2, 3
  and 6.
- **Grouped heads expand by materialisation.** When `num_heads > num_kv_heads`
  the layer `repeat_interleave`s `q`, `k`, `b` and `g` up to the value-head
  count, so the kernels always see one head count. Costs memory bandwidth at
  high group ratios.
- **Tile sizes are not autotuned.** `GDN2Config` exposes `block_DK`, `block_DV`,
  `threads` and `num_stages`; wrapping the factories in
  `tilelang.autotuner.autotune` is straightforward once a GPU is available.
- `h` (per-chunk states) is `[B, NC, H, DK, DV]` in the input dtype. At long
  sequence lengths this is the dominant allocation, and is the cost of making
  kernel 7 fully parallel.

## Layout

Six modules, each with one job:

| file | what's in it |
| --- | --- |
| `gdn2/config.py` | `GatedDeltaNet2Config` (what the model is) and `GDN2Config` (how the kernels run) |
| `gdn2/attention.py` | **start here** — `gdn2_attention()` and the `GatedDeltaNet2` module |
| `gdn2/forward.py` | 7 chunkwise TileLang kernels (training + prefill) + `chunk_gdn2_fwd` |
| `gdn2/inference.py` | serving: `gdn2_prefill` (fused scan) and `gdn2_decode` (1 or `n` tokens) |
| `gdn2/backward.py` | 8 backward TileLang kernels + `chunk_gdn2_bwd` |
| `gdn2/reference.py` | PyTorch oracles: the recurrence, the chunkwise form, the staged backward |

Dependencies run one way: `config` → `forward` → {`backward`, `inference`} →
`attention`, with `reference` standing alone.

| test | covers |
| --- | --- |
| `tests/test_gdn2.py` | maths, gate degeneracies, the module, backend selection |
| `tests/test_backward.py` | the backward derivation, against autograd |
| `tests/test_pipeline_simulation.py` | the forward kernel decomposition, in PyTorch |
| `tests/test_inference.py` | prefill and decode agree with the parallel forward |
| `tests/test_kernel_parse.py` | all 18 kernels parse, without a GPU |
| `bench/bench_gdn2.py` | training/prefill benchmark |
| `bench/bench_decode.py` | decode benchmark: state bandwidth vs tokens per launch |
