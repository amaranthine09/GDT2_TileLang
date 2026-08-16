"""Simulate the seven-kernel pipeline stage by stage, in PyTorch.

``gdn2.reference.chunk_gdn2_torch`` proves the chunkwise *maths*. This proves
the *decomposition*: that splitting ``Aqk``/``Akk`` across two kernels with
different numerics, inverting in a third, and scanning the state in a fourth
still reconstructs the recurrence. Everything here mirrors what each kernel
does -- log2-space decay, per-sub-block reference rows, the exact pairwise
diagonal, the strictly-lower/inclusive-diagonal split -- so an indexing or
masking mistake in the kernel design shows up here, on CPU, without a GPU.

It does not execute any TileLang code; see ``tests/test_kernel_parse.py`` for
that, and the ``cuda``-marked tests for the real thing.
"""

from __future__ import annotations

import math

import pytest
import torch

from gdn2.reference import recurrent_gdn2
from tests.test_gdn2 import make_inputs, rel_err

LOG2E = math.log2(math.e)


def simulate_pipeline(q, k, v, g, b, w, scale, initial_state, BS=64, BC=16):
    """Transcription of kernels 1-7 for one ``[B, S, H, D]`` batch."""
    B, S, H, DK = q.shape
    DV = v.shape[-1]
    NC, NSUB = S // BS, BS // BC
    dev = q.device

    # --- kernel 1: chunk-local cumsum, rescaled to log2 space ----------------
    G = (g.view(B, NC, BS, H, DK).cumsum(dim=2) * LOG2E).view(B, S, H, DK)

    def blk(x):
        return x.view(B, NC, BS, H, x.shape[-1]).permute(0, 3, 1, 2, 4)

    qb, kb, vb, Gb, bb_, wb = (blk(x) for x in (q, k, v, G, b, w))  # [B,H,NC,BS,D]

    Aqk = torch.zeros(B, H, NC, BS, BS, device=dev)
    Akk = torch.zeros(B, H, NC, BS, BS, device=dev)

    # --- kernel 2: diagonal sub-blocks, exact pairwise decay -----------------
    # One CUDA block per token; j walks that token's own sub-chunk only.
    for p in range(BS):
        t_sub = (p // BC) * BC
        for j in range(t_sub, p + 1):
            decay = torch.exp2(Gb[..., p, :] - Gb[..., j, :])
            Aqk[..., p, j] = (qb[..., p, :] * kb[..., j, :] * decay).sum(-1) * scale
            if j < p:  # Akk is strictly lower
                Akk[..., p, j] = (bb_[..., p, :] * kb[..., p, :] * kb[..., j, :] * decay).sum(-1)

    # --- kernel 3: off-diagonal sub-blocks, factorised through a reference ---
    for i in range(1, NSUB):
        rs = slice(i * BC, (i + 1) * BC)
        ref = Gb[..., i * BC : i * BC + 1, :]  # decay at the sub-block's first row
        row_decay = torch.exp2(Gb[..., rs, :] - ref)
        row_q = qb[..., rs, :] * row_decay
        row_e = bb_[..., rs, :] * kb[..., rs, :] * row_decay
        for j in range(i):
            cs = slice(j * BC, (j + 1) * BC)
            col = kb[..., cs, :] * torch.exp2(ref - Gb[..., cs, :])
            Aqk[..., rs, cs] = (row_q @ col.transpose(-1, -2)) * scale
            Akk[..., rs, cs] = row_e @ col.transpose(-1, -2)

    # --- kernel 4: forward substitution for (I + Akk)^-1 ---------------------
    # Rows 0 and 1 of the inverse minus I are just -Akk; substitute from row 2.
    Ai = torch.where(
        torch.arange(BS, device=dev)[:, None] > torch.arange(BS, device=dev)[None, :],
        -Akk,
        torch.zeros_like(Akk),
    )
    for r in range(2, BS):
        row = -Akk[..., r, :].clone()
        row[..., r:] = 0
        Ai[..., r, :] = row + torch.einsum("...c,...cd->...d", row, Ai)
    Ai = Ai + torch.eye(BS, device=dev)

    # --- kernel 5: WY factors ------------------------------------------------
    gamma = torch.exp2(Gb)
    W = Ai @ (bb_ * kb * gamma)
    U = Ai @ (wb * vb)
    KG = kb * torch.exp2(Gb[..., -1:, :] - Gb)

    # --- kernel 6: sequential scan over chunks -------------------------------
    S_st = torch.zeros(B, H, DK, DV, device=dev) if initial_state is None else initial_state.clone()
    h = torch.empty(B, H, NC, DK, DV, device=dev)
    Un = torch.empty(B, H, NC, BS, DV, device=dev)
    for n in range(NC):
        h[:, :, n] = S_st  # state entering the chunk
        Un[:, :, n] = U[:, :, n] - W[:, :, n] @ S_st
        S_st = gamma[:, :, n, -1].unsqueeze(-1) * S_st + KG[:, :, n].transpose(-1, -2) @ Un[:, :, n]

    # --- kernel 7: combine cross-chunk and intra-chunk ------------------------
    tri = torch.ones(BS, BS, dtype=torch.bool, device=dev).tril()
    o = (qb * gamma * scale) @ h + Aqk.masked_fill(~tri, 0.0) @ Un
    return o.permute(0, 2, 3, 1, 4).reshape(B, S, H, DV), S_st


@pytest.mark.parametrize("BS,BC", [(64, 16), (64, 32), (32, 16)])
def test_pipeline_reconstructs_the_recurrence(BS, BC):
    q, k, v, g, b, w = make_inputs(1, 2 * BS, 2, 32, 32)
    scale = 32**-0.5
    o_ref, s_ref = recurrent_gdn2(q, k, v, g, b, w, scale, output_final_state=True)
    o, s = simulate_pipeline(q, k, v, g, b, w, scale, None, BS=BS, BC=BC)
    assert rel_err(o, o_ref) < 1e-4
    assert rel_err(s, s_ref) < 1e-4


def test_pipeline_with_initial_state():
    q, k, v, g, b, w = make_inputs(2, 128, 2, 32, 48)
    scale = 32**-0.5
    s0 = torch.randn(2, 2, 32, 48) * 0.3
    o_ref, s_ref = recurrent_gdn2(q, k, v, g, b, w, scale, initial_state=s0, output_final_state=True)
    o, s = simulate_pipeline(q, k, v, g, b, w, scale, s0)
    assert rel_err(o, o_ref) < 1e-4
    assert rel_err(s, s_ref) < 1e-4


def test_pipeline_survives_fast_decay():
    """The two-kernel Aqk/Akk split must hold up when 1/gamma would overflow."""
    q, k, v, g, b, w = make_inputs(1, 128, 2, 32, 32, decay_scale=25.0)
    scale = 32**-0.5
    o_ref, _ = recurrent_gdn2(q, k, v, g, b, w, scale)
    o, _ = simulate_pipeline(q, k, v, g, b, w, scale, None)
    assert torch.isfinite(o).all()
    assert rel_err(o, o_ref) < 1e-4
