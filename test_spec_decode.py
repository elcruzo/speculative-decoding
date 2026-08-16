"""Speculative decoding tests — EAGLE-3 default; Leviathan / Medusa named variants."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from spec_decode import (
    BigramLM,
    Eagle3Draft,
    LayerFeatures,
    MedusaDraftModel,
    MedusaHead,
    MultiLayerCausalLM,
    TinyCausalLM,
    draft_tree,
    eagle3_decode,
    eagle3_draft_tree,
    greedy_decode,
    leviathan_decode,
    medusa_draft_logits,
    residual_after_reject,
    tree_verify,
)


def test_leviathan_greedy_accepted_is_prefix_of_target():
    torch.manual_seed(0)
    v = 8
    target = BigramLM(v)
    draft = BigramLM(v)
    prefix = torch.tensor([[0, 1, 2]])
    tgt = greedy_decode(target, prefix, n=8)
    out = leviathan_decode(draft, target, prefix, gamma=4, temperature=0.0)
    assert out.tokens == tgt[: len(out.tokens)]
    assert len(out.tokens) >= 1


def test_leviathan_rejection_sampling_lossless_tv():
    """First generated token ~ target p(x | prefix), even with a mismatched draft."""
    torch.manual_seed(1)
    v = 5
    target = BigramLM(v)
    draft = BigramLM(v)
    with torch.no_grad():
        target.table.zero_()
        target.table[0].copy_(torch.tensor([0.0, 2.2, 0.4, -0.3, 0.1]))
        draft.table.zero_()
        draft.table[0].copy_(torch.tensor([1.5, 0.2, 1.0, 0.8, 0.3]))
        target.table[1:].copy_(torch.randn(v - 1, v) * 0.3)
        draft.table[1:].copy_(torch.randn(v - 1, v) * 0.3)

    p = F.softmax(target.table[0], dim=0).detach().cpu().numpy()
    n = 3500
    counts = np.zeros(v, dtype=np.int64)
    g = torch.Generator().manual_seed(7)
    prefix = torch.tensor([[0]])
    for _ in range(n):
        r = leviathan_decode(draft, target, prefix, gamma=3, temperature=1.0, generator=g)
        counts[r.tokens[0]] += 1
    emp = counts / n
    tv = 0.5 * np.abs(emp - p).sum()
    expected = n * p
    chi2 = float(np.sum((counts - expected) ** 2 / np.maximum(expected, 1.0)))
    assert tv < 0.07, f"TV {tv} emp={emp} p={p}"
    assert chi2 < 22.0, f"chi2 {chi2} (df=4)"


def test_tree_verify_accepts_valid_path():
    torch.manual_seed(2)
    v = 8
    draft = BigramLM(v)
    target = BigramLM(v)
    prefix = torch.tensor([[0, 1]])
    nodes = draft_tree(draft, prefix, branches=(2, 2), temperature=0.0)
    assert len(nodes) == 2 + 4
    path = tree_verify(target, prefix, nodes, temperature=0.0)
    assert len(path) >= 1
    roots = {i for i, n in enumerate(nodes) if n.parent < 0}
    child_of: dict[int, set[int]] = {i: set() for i in range(len(nodes))}
    for i, n in enumerate(nodes):
        if n.parent >= 0:
            child_of[n.parent].add(i)
    cur_ids = roots
    for depth, tok in enumerate(path):
        match = [i for i in cur_ids if nodes[i].token == tok]
        if not match:
            assert depth == len(path) - 1
            break
        cur_ids = child_of[match[0]]


def test_residual_updated_per_rejected_sibling():
    """AGENTS pitfall: p ← (p−q)_+ after **each** reject, not one residual at the end."""
    p0 = torch.tensor([0.40, 0.30, 0.20, 0.10])
    q_a = torch.tensor([0.50, 0.20, 0.20, 0.10])
    q_b = torch.tensor([0.10, 0.60, 0.20, 0.10])
    p1 = residual_after_reject(p0, q_a)
    # (p0 - q_a)_+ = [0, 0.10, 0, 0] → [0, 1, 0, 0]
    assert torch.allclose(p1, torch.tensor([0.0, 1.0, 0.0, 0.0]), atol=1e-5)
    p2 = residual_after_reject(p1, q_b)
    assert torch.allclose(p2, torch.tensor([0.0, 1.0, 0.0, 0.0]), atol=1e-5)
    one_shot = residual_after_reject(p0, q_a + q_b)
    assert not torch.allclose(one_shot, p2, atol=1e-5)


def test_mean_accepted_gt_one_when_draft_matches_target():
    torch.manual_seed(3)
    v = 8
    target = BigramLM(v)
    draft = BigramLM(v)
    draft.load_state_dict(target.state_dict())
    prefix = torch.tensor([[0, 1]])
    g = torch.Generator().manual_seed(0)
    n_gen = []
    n_acc = []
    for _ in range(40):
        r = leviathan_decode(draft, target, prefix, gamma=4, temperature=0.7, generator=g)
        n_gen.append(len(r.tokens))
        n_acc.append(r.n_draft_accepted)
    assert float(np.mean(n_gen)) > 1.0
    assert float(np.mean(n_acc)) > 1.0


def test_medusa_head_shape_and_leviathan_decode():
    torch.manual_seed(4)
    m = TinyCausalLM(8, d_model=16, n_heads=4)
    head = MedusaHead(16, 8, n_heads=2)
    prefix = torch.tensor([[1, 2, 3]])
    logits_list = medusa_draft_logits(m, head, prefix)
    assert len(logits_list) == 2
    assert logits_list[0].shape == (1, 8)
    draft = MedusaDraftModel(m, head)
    out = leviathan_decode(draft, m, prefix, gamma=2, temperature=0.0)
    tgt = greedy_decode(m, prefix, n=len(out.tokens))
    assert out.tokens == tgt[: len(out.tokens)]


def test_eagle3_fusion_uses_all_three_layers():
    """Changing mid-layer features must change fused g (not last-hidden-only)."""
    torch.manual_seed(5)
    d, v = 16, 8
    draft = Eagle3Draft(d, v, n_heads=4)
    low = torch.randn(1, 3, d)
    mid = torch.randn(1, 3, d)
    high = torch.randn(1, 3, d)
    g0 = draft.fuse_features(LayerFeatures(low=low, mid=mid, high=high))
    mid2 = mid + 2.0
    g1 = draft.fuse_features(LayerFeatures(low=low, mid=mid2, high=high))
    assert not torch.allclose(g0, g1), "fusion ignored mid layer"
    with torch.no_grad():
        draft.fuse.weight[:, d : 2 * d].zero_()
        draft.fuse.bias.zero_()
    g2 = draft.fuse_features(LayerFeatures(low=low, mid=mid, high=high))
    g3 = draft.fuse_features(LayerFeatures(low=low, mid=mid2, high=high))
    assert torch.allclose(g2, g3, atol=1e-5)


def test_eagle3_draft_tree_and_decode_path():
    torch.manual_seed(6)
    v, d = 8, 16
    target = MultiLayerCausalLM(v, d_model=d, n_heads=4)
    draft = Eagle3Draft(d, v, n_heads=4)
    prefix = torch.tensor([[0, 1, 2]])
    nodes = eagle3_draft_tree(target, draft, prefix, branches=(2, 2), temperature=0.0)
    assert len(nodes) == 2 + 4
    path = eagle3_decode(target, draft, prefix, branches=(2, 2), temperature=0.0)
    assert len(path) >= 1
    roots = [i for i, n in enumerate(nodes) if n.parent < 0]
    child_of = {i: [] for i in range(len(nodes))}
    for i, n in enumerate(nodes):
        if n.parent >= 0:
            child_of[n.parent].append(i)
    cur = roots
    for depth, tok in enumerate(path):
        match = [i for i in cur if nodes[i].token == tok]
        if not match:
            assert depth == len(path) - 1
            break
        cur = child_of[match[0]]


def test_eagle3_rejects_wrong_target_type():
    draft = Eagle3Draft(16, 8, n_heads=4)
    prefix = torch.tensor([[0, 1]])
    try:
        eagle3_draft_tree(TinyCausalLM(8, 16, 4), draft, prefix)  # type: ignore[arg-type]
        assert False, "expected TypeError"
    except TypeError:
        pass


def test_multilayer_forward_layers_distinct():
    torch.manual_seed(7)
    m = MultiLayerCausalLM(6, d_model=12, n_heads=3)
    x = torch.tensor([[0, 1, 2, 3]])
    logits, feats = m.forward_layers(x)
    assert logits.shape == (1, 4, 6)
    assert feats.low.shape == feats.mid.shape == feats.high.shape == (1, 4, 12)
    assert not torch.allclose(feats.low, feats.mid)
    assert not torch.allclose(feats.mid, feats.high)
