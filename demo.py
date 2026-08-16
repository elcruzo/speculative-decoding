"""CPU demo: EAGLE-3 default, Leviathan chain, Medusa head."""

from __future__ import annotations

import torch

from spec_decode import (
    BigramLM,
    Eagle3Draft,
    MedusaDraftModel,
    MedusaHead,
    MultiLayerCausalLM,
    TinyCausalLM,
    eagle3_decode,
    greedy_decode,
    leviathan_decode,
)


if __name__ == "__main__":
    torch.manual_seed(0)

    # Default: EAGLE-3 multi-layer fusion + tree verify
    target = MultiLayerCausalLM(8, d_model=16, n_heads=4)
    draft = Eagle3Draft(16, 8, n_heads=4)
    prefix = torch.tensor([[0, 1, 2]])
    path = eagle3_decode(target, draft, prefix, branches=(2, 2), temperature=0.0)
    print("eagle3 path   ", path)

    # Named variant: Leviathan chain draft-verify
    tgt_bi = BigramLM(8)
    dr_bi = BigramLM(8)
    pref = torch.tensor([[0, 1]])
    tgt = greedy_decode(tgt_bi, pref, n=6)
    spec = leviathan_decode(dr_bi, tgt_bi, pref, gamma=4, temperature=0.0)
    print("target greedy ", tgt)
    print("leviathan     ", spec.tokens, "draft_accepted", spec.n_draft_accepted)

    # Named variant: Medusa head + Leviathan verify
    lm = TinyCausalLM(8, d_model=16, n_heads=4)
    head = MedusaHead(16, 8, n_heads=2)
    fd = leviathan_decode(MedusaDraftModel(lm, head), lm, pref, gamma=3, temperature=0.0)
    print("medusa+leviat ", fd.tokens)
