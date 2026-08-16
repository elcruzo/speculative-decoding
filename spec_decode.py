"""Speculative decoding: Leviathan et al. 2023 (lossless draft-then-verify) + tree verify.

Classic: draft proposes γ tokens; target scores them in one forward; accept the
prefix until the first rejection; on reject, sample from the residual (p − q)_+.
If all γ are accepted, sample one bonus token from the target.

Tree: draft expands a small top-k tree; one target forward with a tree attention
mask; accept a single root-to-leaf path by the same residual rule.

Feature draft head: linear map from the target's last hidden state → next-token
logits (Medusa-like / EAGLE-3-shaped, no separate draft weights beyond the head).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


class BigramLM(nn.Module):
    """Tiny torch LM: logits[t] = table[token[t]]. Enough for lossless distribution tests."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = vocab_size
        self.table = nn.Parameter(torch.randn(vocab_size, vocab_size) * 0.5)

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits, _ = self.forward_with_hidden(tokens, attn_mask)
        return logits

    def forward_with_hidden(
        self, tokens: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.table[tokens]
        return hidden, hidden


class TinyCausalLM(nn.Module):
    """One-layer causal transformer — used for tree-mask verify and the feature head."""

    def __init__(self, vocab_size: int, d_model: int = 32, n_heads: int = 4, max_len: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits, _ = self.forward_with_hidden(tokens, attn_mask)
        return logits

    def forward_with_hidden(
        self, tokens: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t = tokens.shape
        h = self.emb(tokens) + self.pos(torch.arange(t, device=tokens.device))
        hd = self.d_model // self.n_heads
        q = self.wq(h).view(b, t, self.n_heads, hd).transpose(1, 2)
        k = self.wk(h).view(b, t, self.n_heads, hd).transpose(1, 2)
        v = self.wv(h).view(b, t, self.n_heads, hd).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (hd ** 0.5)
        if attn_mask is None:
            allowed = torch.tril(torch.ones(t, t, dtype=torch.bool, device=tokens.device))
        else:
            allowed = attn_mask.bool()
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1) @ v
        h = h + self.wo(attn.transpose(1, 2).reshape(b, t, self.d_model))
        h = self.ln1(h)
        h = self.ln2(h + self.ff(h))
        return self.head(h), h


class FeatureDraftHead(nn.Module):
    """Linear draft head: hidden → next-token logits. No separate draft model."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)


class FeatureDraftModel(nn.Module):
    """Wrap a target LM + feature head so it looks like a draft model."""

    def __init__(self, target: nn.Module, head: FeatureDraftHead):
        super().__init__()
        self.target = target
        self.head = head

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        _logits, hidden = self.target.forward_with_hidden(tokens, attn_mask)
        return self.head(hidden)


def _softmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        # Greedy: one-hot at argmax (used as p/q for the accept test).
        idx = logits.argmax(dim=-1)
        p = torch.zeros_like(logits)
        p.scatter_(-1, idx.unsqueeze(-1), 1.0)
        return p
    return torch.softmax(logits / temperature, dim=-1)


def _sample(probs: torch.Tensor, temperature: float, generator: torch.Generator | None) -> int:
    if temperature <= 0:
        return int(probs.argmax(dim=-1).item())
    return int(torch.multinomial(probs, 1, generator=generator).item())


@dataclass
class SpecResult:
    tokens: list[int]
    n_draft_accepted: int
    rejected: bool


@torch.no_grad()
def speculative_decode(
    draft: nn.Module,
    target: nn.Module,
    prefix: torch.Tensor,
    gamma: int,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> SpecResult:
    """Leviathan et al. lossless draft-then-verify. `prefix` is (1, T)."""
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    device = prefix.device
    ctx = prefix
    draft_tokens: list[int] = []
    draft_probs: list[torch.Tensor] = []
    for _ in range(gamma):
        q_logits = draft(ctx)[:, -1, :]
        q_prob = _softmax(q_logits, temperature)[0]
        tok = _sample(q_prob, temperature, generator)
        draft_tokens.append(tok)
        draft_probs.append(q_prob)
        ctx = torch.cat([ctx, torch.tensor([[tok]], device=device)], dim=1)

    # One target forward over prefix + all draft tokens.
    seq = torch.cat([prefix, torch.tensor([draft_tokens], device=device)], dim=1)
    t_logits = target(seq)
    t0 = prefix.shape[1] - 1
    accepted: list[int] = []
    for i in range(gamma):
        p = _softmax(t_logits[:, t0 + i, :], temperature)[0]
        q = draft_probs[i]
        x = draft_tokens[i]
        if temperature <= 0:
            accept = x == int(p.argmax().item())
        else:
            ratio = float((p[x] / q[x].clamp(min=1e-12)).item())
            u = float(torch.rand((), generator=generator).item())
            accept = u < min(1.0, ratio)
        if accept:
            accepted.append(x)
            continue
        residual = (p - q).clamp(min=0)
        z = residual.sum()
        if float(z) <= 0:
            residual = p
        else:
            residual = residual / z
        y = _sample(residual, temperature if temperature > 0 else 0.0, generator)
        if temperature <= 0:
            y = int(p.argmax().item())
        accepted.append(y)
        return SpecResult(tokens=accepted, n_draft_accepted=i, rejected=True)

    bonus_p = _softmax(t_logits[:, t0 + gamma, :], temperature)[0]
    accepted.append(_sample(bonus_p, temperature, generator))
    return SpecResult(tokens=accepted, n_draft_accepted=gamma, rejected=False)


@torch.no_grad()
def greedy_decode(model: nn.Module, prefix: torch.Tensor, n: int) -> list[int]:
    ctx = prefix
    out: list[int] = []
    for _ in range(n):
        logits = model(ctx)[:, -1, :]
        tok = int(logits.argmax(dim=-1).item())
        out.append(tok)
        ctx = torch.cat([ctx, torch.tensor([[tok]], device=prefix.device)], dim=1)
    return out


@dataclass
class TreeNode:
    token: int
    parent: int
    q_prob: torch.Tensor
    children: list[int] = field(default_factory=list)


@torch.no_grad()
def draft_tree(
    draft: nn.Module,
    prefix: torch.Tensor,
    branches: tuple[int, ...] = (2, 2),
    temperature: float = 1.0,
) -> list[TreeNode]:
    """Expand a small tree: at depth d take top-`branches[d]` tokens from the draft."""
    nodes: list[TreeNode] = []
    # Virtual root = last prefix token; children are first draft tokens.
    # We store only draft nodes; parent -1 means the prefix.
    frontier = [(-1, prefix)]
    for depth, k in enumerate(branches):
        nxt: list[tuple[int, torch.Tensor]] = []
        for parent_idx, ctx in frontier:
            logits = draft(ctx)[:, -1, :]
            probs = _softmax(logits, temperature)[0]
            if temperature <= 0:
                topk = torch.topk(probs, k=min(k, probs.numel()))
                choices = [(int(i), probs) for i in topk.indices.tolist()]
            else:
                topk = torch.topk(probs, k=min(k, probs.numel()))
                choices = [(int(i), probs) for i in topk.indices.tolist()]
            for tok, q_prob in choices:
                idx = len(nodes)
                nodes.append(TreeNode(token=tok, parent=parent_idx, q_prob=q_prob))
                if parent_idx >= 0:
                    nodes[parent_idx].children.append(idx)
                child_ctx = torch.cat([ctx, torch.tensor([[tok]], device=prefix.device)], dim=1)
                nxt.append((idx, child_ctx))
        frontier = nxt
    return nodes


def tree_attention_mask(prefix_len: int, nodes: list[TreeNode]) -> torch.Tensor:
    """(T, T) bool mask: prompt is causal; each draft node attends to prompt + ancestors."""
    t = prefix_len + len(nodes)
    mask = torch.zeros(t, t, dtype=torch.bool)
    for i in range(prefix_len):
        mask[i, : i + 1] = True
    # ancestor chain for each node
    ancestors: list[list[int]] = []
    for i, node in enumerate(nodes):
        chain = []
        p = node.parent
        while p >= 0:
            chain.append(p)
            p = nodes[p].parent
        chain.reverse()
        ancestors.append(chain)
        pos = prefix_len + i
        mask[pos, :prefix_len] = True
        for a in chain:
            mask[pos, prefix_len + a] = True
        mask[pos, pos] = True
    return mask


@torch.no_grad()
def tree_verify(
    target: nn.Module,
    prefix: torch.Tensor,
    nodes: list[TreeNode],
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> list[int]:
    """One target forward with a tree mask; accept a single path (paper residual rule)."""
    if not nodes:
        return []
    device = prefix.device
    prefix_len = prefix.shape[1]
    tokens = torch.cat([prefix, torch.tensor([[n.token for n in nodes]], device=device)], dim=1)
    mask = tree_attention_mask(prefix_len, nodes).to(device)
    logits = target(tokens, attn_mask=mask)

    def parent_logit_row(node_idx: int) -> torch.Tensor:
        parent = nodes[node_idx].parent
        if parent < 0:
            pos = prefix_len - 1
        else:
            pos = prefix_len + parent
        return _softmax(logits[:, pos, :], temperature)[0]

    # Roots = nodes whose parent is -1
    roots = [i for i, n in enumerate(nodes) if n.parent < 0]
    path: list[int] = []
    candidates = roots
    while candidates:
        # Score each sibling against the same parent p; pick by speculative accept in sibling order (higher q first).
        scored = []
        for idx in candidates:
            p = parent_logit_row(idx)
            q = nodes[idx].q_prob.to(device)
            x = nodes[idx].token
            scored.append((idx, p, q, x))
        scored.sort(key=lambda t: float(t[2][t[3]].item()), reverse=True)
        chosen = None
        p_res = scored[0][1].clone()
        for idx, _p, q, x in scored:
            if temperature <= 0:
                ok = x == int(p_res.argmax().item())
            else:
                ratio = float((p_res[x] / q[x].clamp(min=1e-12)).item())
                u = float(torch.rand((), generator=generator).item())
                ok = u < min(1.0, ratio)
            if ok:
                chosen = idx
                path.append(x)
                break
            p_res = (p_res - q).clamp(min=0)
            z = p_res.sum()
            if float(z) > 0:
                p_res = p_res / z
        if chosen is None:
            y = int(p_res.argmax().item()) if temperature <= 0 else _sample(p_res, temperature, generator)
            path.append(y)
            return path
        candidates = nodes[chosen].children
    return path


@torch.no_grad()
def feature_draft_logits(target: nn.Module, head: FeatureDraftHead, prefix: torch.Tensor) -> torch.Tensor:
    _logits, hidden = target.forward_with_hidden(prefix)
    return head(hidden[:, -1, :])
