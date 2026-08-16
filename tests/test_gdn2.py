"""Correctness tests for Gated DeltaNet-2.

The maths tests (chunkwise form vs. the literal recurrence, gradients, gate
degeneracies, the layer itself) run anywhere, CPU included. The kernel tests
are marked ``cuda`` and skip without a GPU.

    pytest tests/test_gdn2.py -v
    pytest tests/test_gdn2.py -v -m cuda      # on a GPU box
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from gdn2 import GatedDeltaNet2, GatedDeltaNet2Config, chunk_gdn2_torch, recurrent_gdn2
from gdn2 import GDN2Config

CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA, reason="requires CUDA + TileLang")


def make_inputs(B, S, H, DK, DV, decay_scale=4.0, device="cpu", seed=0):
    """Random GDN-2 inputs with the same shapes and ranges the layer produces."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    Ke = dict(generator=gen, dtype=torch.float32)
    q = F.normalize(torch.randn(B, S, H, DK, **Ke), dim=-1).to(device)
    k = F.normalize(torch.randn(B, S, H, DK, **Ke), dim=-1).to(device)
    v = torch.randn(B, S, H, DV, **Ke).to(device)
    # g <= 0; decay_scale controls how aggressively channels forget.
    g = (-F.softplus(torch.randn(B, S, H, DK, **Ke)) * decay_scale).to(device)
    b = torch.sigmoid(torch.randn(B, S, H, DK, **Ke)).to(device)
    w = torch.sigmoid(torch.randn(B, S, H, DV, **Ke)).to(device)
    return q, k, v, g, b, w


def rel_err(a: torch.Tensor, ref: torch.Tensor) -> float:
    return ((a - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()


# --------------------------------------------------------------------------
# The chunkwise WY form must reproduce the definition.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("S,BS,BC", [(128, 64, 16), (256, 64, 32), (192, 32, 16), (64, 64, 8)])
def test_chunk_matches_recurrent(S, BS, BC):
    q, k, v, g, b, w = make_inputs(2, S, 3, 32, 48)
    o_ref, s_ref = recurrent_gdn2(q, k, v, g, b, w, output_final_state=True)
    o, s = chunk_gdn2_torch(q, k, v, g, b, w, output_final_state=True, chunk_size=BS, sub_chunk_size=BC)
    assert rel_err(o, o_ref) < 2e-5
    assert rel_err(s, s_ref) < 2e-5


def test_chunk_matches_recurrent_with_initial_state():
    q, k, v, g, b, w = make_inputs(2, 128, 3, 32, 48)
    s0 = torch.randn(2, 3, 32, 48) * 0.5
    o_ref, s_ref = recurrent_gdn2(q, k, v, g, b, w, initial_state=s0, output_final_state=True)
    o, s = chunk_gdn2_torch(q, k, v, g, b, w, initial_state=s0, output_final_state=True)
    assert rel_err(o, o_ref) < 2e-5
    assert rel_err(s, s_ref) < 2e-5


@pytest.mark.parametrize("decay_scale", [0.0, 1.0, 8.0, 30.0])
def test_fast_forgetting_channels_do_not_overflow(decay_scale):
    """The 1/gamma factorisation must survive channels that decay hard.

    At decay_scale=30 a channel can lose ~e^-1900 across one 64-token chunk,
    which overflows any implementation that materialises exp(-cumsum(g)).
    """
    q, k, v, g, b, w = make_inputs(1, 128, 2, 32, 32, decay_scale=decay_scale)
    o_ref, _ = recurrent_gdn2(q, k, v, g, b, w)
    o, _ = chunk_gdn2_torch(q, k, v, g, b, w)
    assert torch.isfinite(o).all()
    assert rel_err(o, o_ref) < 1e-4


# --------------------------------------------------------------------------
# Gate degeneracies: GDN-2 must contain KDA and Gated DeltaNet.
# --------------------------------------------------------------------------


def test_reduces_to_kda_when_gates_are_tied():
    """Tying b and w to one scalar per token is exactly KDA's delta gate."""
    q, k, v, g, _, _ = make_inputs(2, 128, 3, 32, 32)
    beta = torch.rand(2, 128, 3, 1)
    b = beta.expand(-1, -1, -1, 32).contiguous()
    w = beta.expand(-1, -1, -1, 32).contiguous()

    o, _ = chunk_gdn2_torch(q, k, v, g, b, w)

    # KDA reference: S = (I - beta k k^T) diag(a) S + beta k v^T
    S = torch.zeros(2, 3, 32, 32)
    o_kda = torch.empty_like(o)
    for t in range(128):
        S = S * g[:, t].exp().unsqueeze(-1)
        kt, bt = k[:, t], beta[:, t, :, 0]
        S = S - bt[..., None, None] * kt.unsqueeze(-1) * torch.einsum("bhkd,bhk->bhd", S, kt).unsqueeze(-2)
        S = S + bt[..., None, None] * kt.unsqueeze(-1) * v[:, t].unsqueeze(-2)
        o_kda[:, t] = torch.einsum("bhkd,bhk->bhd", S, q[:, t] * 32**-0.5)
    assert rel_err(o, o_kda) < 2e-5


def test_zero_gates_leave_state_untouched():
    """b = w = 0 means neither erase nor write: pure decay of the prior state."""
    q, k, v, g, _, _ = make_inputs(1, 128, 2, 32, 32)
    b = torch.zeros(1, 128, 2, 32)
    w = torch.zeros(1, 128, 2, 32)
    s0 = torch.randn(1, 2, 32, 32)
    _, s = chunk_gdn2_torch(q, k, v, g, b, w, initial_state=s0, output_final_state=True)
    expected = s0 * g.sum(dim=1).exp().unsqueeze(-1)
    assert rel_err(s, expected) < 2e-5


def test_erase_and_write_gates_are_independent():
    """Changing only the write gate must not change the erase behaviour."""
    q, k, v, g, b, w = make_inputs(1, 128, 2, 32, 32)
    o1, _ = chunk_gdn2_torch(q, k, v, g, b, w)
    o2, _ = chunk_gdn2_torch(q, k, v, g, b, w * 0.5)
    assert not torch.allclose(o1, o2)


# --------------------------------------------------------------------------
# Gradients
# --------------------------------------------------------------------------


def test_gradients_match_recurrent():
    q, k, v, g, b, w = make_inputs(2, 128, 2, 16, 24)
    names = ["q", "k", "v", "g", "b", "w"]

    def grads(fn):
        ts = [t.clone().requires_grad_(True) for t in (q, k, v, g, b, w)]
        o, _ = fn(*ts)
        o.square().mean().backward()
        return [t.grad for t in ts]

    ref = grads(recurrent_gdn2)
    got = grads(chunk_gdn2_torch)
    for name, a, r in zip(names, got, ref):
        assert rel_err(a, r) < 1e-4, f"grad mismatch for {name}: {rel_err(a, r):.2e}"


# --------------------------------------------------------------------------
# Layer
# --------------------------------------------------------------------------


def test_layer_forward_shape_and_backward():
    cfg = GatedDeltaNet2Config(hidden_size=128, num_heads=4, head_dim_k=32, head_dim_v=32)
    layer = GatedDeltaNet2(cfg)
    x = torch.randn(2, 64, 128, requires_grad=True)
    out, cache = layer(x, use_cache=True)
    assert out.shape == (2, 64, 128)
    assert cache.recurrent_state.shape == (2, 4, 32, 32)
    out.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, p in layer.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"


def test_layer_grouped_heads():
    cfg = GatedDeltaNet2Config(hidden_size=128, num_heads=8, num_kv_heads=2, head_dim_k=16, head_dim_v=32)
    layer = GatedDeltaNet2(cfg)
    out, _ = layer(torch.randn(2, 64, 128))
    assert out.shape == (2, 64, 128)


def test_decode_step_matches_forward():
    """Token-by-token decoding must track the parallel forward exactly."""
    torch.manual_seed(0)
    cfg = GatedDeltaNet2Config(hidden_size=64, num_heads=2, head_dim_k=16, head_dim_v=16)
    layer = GatedDeltaNet2(cfg).eval()
    x = torch.randn(2, 40, 64)

    with torch.no_grad():
        full, _ = layer(x)
        cache = layer.init_cache(2)
        outs = [layer.step(x[:, t], cache)[0] for t in range(40)]
    stepped = torch.stack(outs, dim=1)
    assert rel_err(stepped, full) < 1e-4


# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------


def test_auto_backend_policy():
    """Our kernels on CUDA, PyTorch elsewhere. No third-party kernel path."""
    from gdn2 import resolve_backend

    assert resolve_backend("auto", on_cuda=False, needs_grad=False) == "torch"
    assert resolve_backend("auto", on_cuda=False, needs_grad=True) == "torch"
    assert resolve_backend("auto", on_cuda=True, needs_grad=False) == "tilelang"
    assert resolve_backend("auto", on_cuda=True, needs_grad=True) == "tilelang"


def test_explicit_backend_validation():
    from gdn2 import resolve_backend

    assert resolve_backend("torch", on_cuda=False, needs_grad=True) == "torch"
    with pytest.raises(ValueError, match="requires a CUDA device"):
        resolve_backend("tilelang", on_cuda=False, needs_grad=False)
    with pytest.raises(ValueError, match="unknown backend"):
        resolve_backend("fla", on_cuda=True, needs_grad=False)


def test_varlen_is_rejected_not_ignored():
    """Packed sequences are unimplemented; that must be loud, not silent."""
    from gdn2 import gdn2_attention
    from gdn2 import GDN2Config

    q, k, v, g, b, w = make_inputs(1, 128, 2, 16, 16)
    cu = torch.tensor([0, 64, 128])
    with pytest.raises(NotImplementedError, match="variable-length"):
        gdn2_attention(q, k, v, g, b, w, config=GDN2Config(backend="torch"), cu_seqlens=cu)


# --------------------------------------------------------------------------
# TileLang kernels (GPU only)
# --------------------------------------------------------------------------


@cuda_only
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("B,S,H,DK,DV", [(2, 512, 4, 64, 64), (1, 1024, 8, 128, 128), (2, 320, 2, 64, 128)])
def test_kernel_matches_reference(dtype, B, S, H, DK, DV):
    from gdn2 import chunk_gdn2_fwd

    q, k, v, g, b, w = make_inputs(B, S, H, DK, DV, device="cuda")
    s0 = torch.randn(B, H, DK, DV, device="cuda") * 0.2

    o_ref, s_ref = recurrent_gdn2(q, k, v, g, b, w, initial_state=s0, output_final_state=True)
    o, s = chunk_gdn2_fwd(
        q.to(dtype), k.to(dtype), v.to(dtype), g, b.to(dtype), w.to(dtype),
        scale=DK**-0.5, initial_state=s0, output_final_state=True,
    )
    # bf16 GEMM operands: a few 1e-2 of relative error is expected.
    tol = 3e-2 if dtype is torch.bfloat16 else 1e-2
    assert rel_err(o.float(), o_ref) < tol
    assert rel_err(s.float(), s_ref) < tol


@cuda_only
@pytest.mark.cuda
def test_kernel_decode_step_matches_reference():
    from gdn2 import gdn2_decode_step

    q, k, v, g, b, w = make_inputs(2, 8, 4, 64, 64, device="cuda")
    s0 = torch.randn(2, 4, 64, 64, device="cuda") * 0.2

    o_ref, _ = recurrent_gdn2(q, k, v, g, b, w, initial_state=s0)

    state = s0.clone()
    outs = []
    for t in range(8):
        outs.append(
            gdn2_decode_step(
                q[:, t].bfloat16(), k[:, t].bfloat16(), v[:, t].bfloat16(),
                g[:, t], b[:, t].bfloat16(), w[:, t].bfloat16(), state,
            ).float()
        )
    assert rel_err(torch.stack(outs, 1), o_ref) < 3e-2


@cuda_only
@pytest.mark.cuda
def test_kernel_autograd_end_to_end():
    cfg = GatedDeltaNet2Config(hidden_size=256, num_heads=4, head_dim_k=64, head_dim_v=64,
                               kernel=GDN2Config(chunk_size=64, sub_chunk_size=16))
    layer = GatedDeltaNet2(cfg).cuda()
    x = torch.randn(2, 256, 256, device="cuda", requires_grad=True)
    out, _ = layer(x)
    out.square().mean().backward()
    assert torch.isfinite(x.grad).all()
