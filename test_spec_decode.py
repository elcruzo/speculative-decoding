"""Speculative decoding tests — fail if the paper algorithm is wrong."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from spec_decode import (
    BigramLM,
    FeatureDraftHead,
    FeatureDraftModel,
    TinyCausalLM,
    draft_tree,
    feature_draft_logits,
    greedy_decode,
    speculative_decode,
    tree_verify,
)


def test_greedy_accepted_is_prefix_of_target():
    torch.manual_seed(0)
    v = 8
    target = BigramLM(v)
    draft = BigramLM(v)
    prefix = torch.tensor([[0, 1, 2]])
    tgt = greedy_decode(target, prefix, n=8)
    out = speculative_decode(draft, target, prefix, gamma=4, temperature=0.0)
    assert out.tokens == tgt[: len(out.tokens)]
    assert len(out.tokens) >= 1


def test_rejection_sampling_lossless_tv():
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
        # Keep other rows defined so γ>1 still runs.
        target.table[1:].copy_(torch.randn(v - 1, v) * 0.3)
        draft.table[1:].copy_(torch.randn(v - 1, v) * 0.3)

    p = F.softmax(target.table[0], dim=0).detach().cpu().numpy()
    n = 3500
    counts = np.zeros(v, dtype=np.int64)
    g = torch.Generator().manual_seed(7)
    prefix = torch.tensor([[0]])
    for _ in range(n):
        r = speculative_decode(draft, target, prefix, gamma=3, temperature=1.0, generator=g)
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
    # Walk: each draft-accepted token (until a residual that is not a child) follows an edge.
    cur_ids = roots
    for depth, tok in enumerate(path):
        match = [i for i in cur_ids if nodes[i].token == tok]
        if not match:
            # Residual correction — must be the last token of the path.
            assert depth == len(path) - 1
            break
        cur_ids = child_of[match[0]]


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
        r = speculative_decode(draft, target, prefix, gamma=4, temperature=0.7, generator=g)
        n_gen.append(len(r.tokens))
        n_acc.append(r.n_draft_accepted)
    assert float(np.mean(n_gen)) > 1.0
    assert float(np.mean(n_acc)) > 1.0


def test_feature_draft_head_shape_and_decode():
    torch.manual_seed(4)
    m = TinyCausalLM(8, d_model=16, n_heads=4)
    head = FeatureDraftHead(16, 8)
    prefix = torch.tensor([[1, 2, 3]])
    logits = feature_draft_logits(m, head, prefix)
    assert logits.shape == (1, 8)
    draft = FeatureDraftModel(m, head)
    out = speculative_decode(draft, m, prefix, gamma=2, temperature=0.0)
    tgt = greedy_decode(m, prefix, n=len(out.tokens))
    assert out.tokens == tgt[: len(out.tokens)]
