"""CPU demo: draft-verify, tree verify, feature head."""

from __future__ import annotations

import torch

from spec_decode import (
    BigramLM,
    FeatureDraftHead,
    FeatureDraftModel,
    TinyCausalLM,
    draft_tree,
    greedy_decode,
    speculative_decode,
    tree_verify,
)

if __name__ == "__main__":
    torch.manual_seed(0)
    target = BigramLM(8)
    draft = BigramLM(8)
    prefix = torch.tensor([[0, 1]])
    tgt = greedy_decode(target, prefix, n=6)
    spec = speculative_decode(draft, target, prefix, gamma=4, temperature=0.0)
    print("target greedy", tgt)
    print("spec greedy  ", spec.tokens, "draft_accepted", spec.n_draft_accepted)

    nodes = draft_tree(draft, prefix, branches=(2, 2), temperature=0.0)
    path = tree_verify(target, prefix, nodes, temperature=0.0)
    print("tree path    ", path, "nodes", len(nodes))

    lm = TinyCausalLM(8, d_model=16, n_heads=4)
    head = FeatureDraftHead(16, 8)
    fd = speculative_decode(FeatureDraftModel(lm, head), lm, prefix, gamma=3, temperature=0.0)
    print("feature-head ", fd.tokens)
