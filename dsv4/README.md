# DeepSeek-V4 hybrid attention in TileLang

Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA) from
[DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348)
(arXiv:2606.19348, sections 2.3.1–2.3.3), as TileLang kernels with hand-written
backward passes, a differentiable PyTorch oracle, and `nn.Module` layers.

## The architecture

V4 stops treating the past as a flat list of tokens. Every layer pools its KV
cache into compressed entries and attends over those, interleaving two variants
through the stack.

| | CSA | HCA |
| --- | --- | --- |
| compression rate | `m = 4` | `m' = 128` |
| pooling | overlapped, two streams (eq. 11/12) | non-overlapped, one stream (eq. 22/23) |
| selection | lightning indexer + top-k (eq. 16/17) | none, dense over the compressed stream |
| top-k | 512 (Flash) / 1024 (Pro) | — |

Both share the same core attention: **shared-KV MQA** — one compressed stream
read by every query head, each entry acting as *both* key and value — plus a
128-token sliding window of uncompressed entries under the same softmax, a
learnable per-head attention sink in the denominator (eq. 27), per-head RMSNorm,
and partial RoPE on the trailing 64 channels.

Pooling is a softmax over the span taken **per channel**: with `C, Z` of shape
`[B, N, c]` it is `c` independent length-`m` softmaxes per block, not one
softmax across channels.

```
h --> C, Z        projections
  --> CComp       softmax-weighted pooling
  --> KIComp      same pooling, indexer width      (CSA only)
  --> I           lightning index scores           (CSA only)
  --> idx         top-k selection                  (CSA only)
  --> o           shared-KV MQA + window + sink
  --> o_hat       grouped output projection
```

## Kernels

Twelve, all with a backward.

| module | forward | backward |
| --- | --- | --- |
| `compress.py` | `hca_compress_kernel`, `csa_compress_kernel` | `hca_compress_bwd_kernel`, `csa_compress_bwd_kernel` |
| `indexer.py` | `lightning_index_kernel` | `lightning_index_bwd_q_kernel`, `lightning_index_bwd_k_kernel` |
| `core_attn.py` | `sparse_attn_fwd_kernel`, `dense_attn_fwd_kernel` | `sparse_attn_bwd_q_kernel`, `dense_attn_bwd_q_kernel`, `dense_attn_bwd_kv_kernel` |

Design notes live in each module's docstring. The load-bearing ones:

**The backward is parallelised differently for the two variants,** because the
reuse is in different places. CSA goes query-parallel: all `n_h` heads share the
one gathered KV tile, so the head axis is free. Inverting the top-k and going
key-parallel instead would make every (entry, query) pair re-read `q` and `do`
for all heads — about **128× more traffic** at the published settings. The price
is that `dkv_comp` becomes a scatter and needs atomics, but the atomic volume
equals the forward's gather volume, so it costs roughly one extra pass. HCA has
no gather, so its key direction has arithmetic intensity `block_KV` and is
solidly compute-bound; it gets a key-parallel kernel and needs no atomics on the
stream.

**Each entry is key *and* value,** so its gradient is the sum of a value path
and a key path. Dropping the key path leaves a gradient that still descends —
the model trains, just worse. `test_shared_key_value_gradient_has_both_paths`
pins it.

**Masked logits use `-1e30`, not `-inf`,** and are explicitly zeroed after the
exponential. A fully-masked tile would otherwise give `max = -inf` and
`exp(-inf - -inf) = 1`, turning masked slots into weight-1 entries.

**Both softmax Jacobians collapse.** For pooling, `dZ_j = S_j * dO * (C_j -
CComp)` — the forward's own output *is* the second term, so no extra reduction
is needed. For attention, the sink changes the probabilities but not the *form*
of the logit gradient, so a flash kernel needs no structural change; it just
picks up `dsink = -exp(z' - lse) * delta`.

**RoPE angles are built in fp64.** `inv_freq` is not exactly representable in
fp32 and the angle scales with position, so at `rope_dim = 64` and position 1e6
an all-fp32 pipeline is off by ~0.016 on a cosine. Only the cos/sin are cast
down — they are bounded, so that is free.

Three things are left to PyTorch on purpose, because a kernel could not beat a
bandwidth-bound library reduction: the bias gradients (a strided sum over a
`dZ` the backward already wrote), `delta = rowsum(do * o)`, and `dsink`.

## Use

```python
import torch
from dsv4 import CSAAttention, HCAAttention, V4_FLASH

layer = CSAAttention(V4_FLASH.csa).cuda().bfloat16()
out = layer(hidden_states)                 # [B, N, d]
```

`build_attention_stack(V4_FLASH)` lays out the whole published stack: two
sliding-window layers, then CSA and HCA alternating.

Layers run on CPU through the PyTorch reference and on CUDA through the kernels.
`backend="auto"` (the default) picks between them, which is what lets the maths
be tested on a laptop.

### Training the indexer

**The indexer parameters get no gradient from the layer's output.** Top-k is a
hard gather, so nothing flows back into the scores that did the ranking. After
`loss.backward()`, `w_aki`, `w_bki`, `w_azi`, `w_bzi`, `bias_ai`, `bias_bi`,
`w_iuq` and `w_w` will all have `grad is None`.

That is the architecture, not a bug — it is why the report warms the indexer up
as a separate stage. But it fails silently: an optimiser over
`layer.parameters()` leaves the indexer at its initialisation, the model still
trains, and it is simply selecting compressed entries at random. Add the
auxiliary loss:

```python
q, cq = layer._queries(x, positions)
scores = layer.index_scores(x, cq, positions)
loss = main_loss + alpha * indexer_kl_loss(scores, target_p, m)
```

## What is verified, and what is not

I have no GPU, so this splits cleanly and you should read the split before
trusting anything.

**Verified, and run in CI on CPU** (`python -m pytest tests/ -q`, 152 passing):

- Every hand-written backward formula in `reference.py` checked against autograd
  in **fp64**, to `1e-11` — compression (both variants), the indexer, and core
  attention with and without the window and the sparse gather. The reference
  promotes to fp64 whenever its inputs are, so this is not capped at fp32
  roundoff.
- Strict causality of all three layer types, compared **exactly**: perturbing
  token `j` leaves every output before `j` bit-identical.
- The chunked selection path agrees exactly with a single full-sequence pass.
- CSA with `k >= NB` collapses exactly onto HCA's core attention.
- Sink, padding, and empty-row edge cases: a query with no candidates outputs
  exactly zero with `lse = z'`, not NaN.
- The RoPE relative-position property and the output un-rotation.
- All twelve kernels parse into TVM PrimFuncs, across a sweep of tile sizes,
  compression rates and window sizes, with the shape guards firing as intended.

**Not verified — needs a GPU:**

- **No kernel has ever been executed.** Parsing is not compiling and compiling
  is not running. Numerical agreement between the kernels and the fp64 oracle is
  the one thing that matters most here and is exactly what is missing.
- Tile sizes are reasoned from register pressure (`M * 512` fp32 for the output
  accumulator, so `M = 32` at 256 threads is 64 registers a thread) and
  arithmetic intensity, not measured. Treat `KernelConfig` defaults as a
  starting point for autotuning, not as tuned values.
- The atomic-throughput argument behind CSA's query-parallel backward is an
  analysis of traffic volume, not a benchmark.
- fp8/fp4 storage and the mixed-precision KV cache the report describes are
  configuration fields only; the kernels run bf16 in and fp32 accumulate.

The fastest way to close the gap: run `tests/test_dsv4_reference.py` on a CUDA
box with the layers forced to `backend="tilelang"` and compare against
`backend="torch"` on the same inputs. Any disagreement beyond bf16 tolerance is
a kernel bug, and the oracle is already trustworthy to fp64.

## Details the report does not fix

Each is a named parameter with the default written out, so a checkpoint that
disagrees can be matched without touching kernels.

1. **Output un-rotation position.** The report says RoPE is applied "with
   position `-i`" to the outputs, having just used `i` for both the head index
   and the block index. The stated purpose — making each entry's contribution
   depend on the query-to-entry *distance* — forces the query position.
2. **Position tag of a compressed entry** (`rope_block_pos`, default `"last"`).
   A pooled entry spans many tokens but needs one position. The compressed and
   window branches share a softmax, so it has to be on the token scale.
3. **Sliding-window projection** (`window_shares_kv_proj`, default `False`).
   "We additionally produce `n_win` uncompressed KV entries" reads as a
   projection of its own.
4. **Indexer RoPE** (`indexer_rope`, default `True`). The report specifies
   partial RoPE for the core attention and is silent on the indexer. DSA, which
   it cites for the indexer design, applies it, and a position-blind indexer
   would select poorly at long range.
5. **Indexer loss.** The report says the indexer is warmed up but not against
   what. `indexer_kl_loss` implements the DSA recipe — match the head-summed
   dense attention distribution — as a documented reconstruction.

## Not implemented

- **A fused score-and-select kernel.** Selection currently goes through
  `torch.topk`, which is well optimised and off the critical path (O(NB) per row
  against the O(k·c·n_h) the attention then spends). `index_and_select` chunks
  the query axis so peak memory is `chunk × NB` rather than `N × NB` — a ~500×
  reduction at the published settings — which captures the memory win without
  the data-dependent control flow a radix select would need. Fusing the two
  would still save the round trip.
- **Fused RMSNorm+RoPE.** These run as PyTorch elementwise ops. Correct, and a
  real but modest bandwidth win if fused into the attention epilogue.
- **Inference.** Prefill and decode paths, and the compressed KV cache itself.
  The kernels are the training/prefill maths; a decode path would reuse the
  compressed entries rather than rebuild them.
