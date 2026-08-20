"""Speculative decoding — EAGLE-3 default + named Leviathan / Medusa variants.

Default (EAGLE-3, Li et al. 2025 arXiv:2503.01840):
  Target exposes low / mid / high hidden states. Fuse g = FC(concat(l,m,h)).
  Draft: FC(concat(g_or_a, e_token)) → one decoder layer → a → LM head (direct
  token prediction). Expand a draft tree; verify with one target forward + tree
  attention; accept a root-to-leaf path with Leviathan residual (p−q)_+ updated
  after **each** rejected sibling.

Named variants (wrong draft type raises ``TypeError``):
  - ``leviathan_decode`` — classic chain draft-then-verify (Leviathan 2023).
  - ``MedusaHead`` — parallel linear heads on the **last** hidden only (Cai et al.).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Toy LMs
# ---------------------------------------------------------------------------


class BigramLM(nn.Module):
    """Tiny torch LM: logits[t] = table[token[t]]. Enough for lossless TV tests."""

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


class _DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.d_model = d_model
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = h.shape
        hd = self.d_model // self.n_heads
        q = self.wq(h).view(b, t, self.n_heads, hd).transpose(1, 2)
        k = self.wk(h).view(b, t, self.n_heads, hd).transpose(1, 2)
        v = self.wv(h).view(b, t, self.n_heads, hd).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (hd**0.5)
        if attn_mask is None:
            allowed = torch.tril(torch.ones(t, t, dtype=torch.bool, device=h.device))
        else:
            allowed = attn_mask.bool()
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1) @ v
        h = self.ln1(h + self.wo(attn.transpose(1, 2).reshape(b, t, self.d_model)))
        return self.ln2(h + self.ff(h))


@dataclass
class LayerFeatures:
    """Low / mid / high hidden sequences from a multi-layer target (EAGLE-3 §3.1)."""

    low: torch.Tensor
    mid: torch.Tensor
    high: torch.Tensor


class MultiLayerCausalLM(nn.Module):
    """Three-block causal LM so EAGLE-3 can fuse distinct layer depths."""

    def __init__(self, vocab_size: int, d_model: int = 32, n_heads: int = 4, max_len: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([_DecoderBlock(d_model, n_heads) for _ in range(3)])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits, _feats = self.forward_layers(tokens, attn_mask)
        return logits

    def forward_with_hidden(
        self, tokens: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, feats = self.forward_layers(tokens, attn_mask)
        return logits, feats.high

    def forward_layers(
        self, tokens: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, LayerFeatures]:
        b, t = tokens.shape
        h = self.emb(tokens) + self.pos(torch.arange(t, device=tokens.device))
        h0 = self.blocks[0](h, attn_mask)
        h1 = self.blocks[1](h0, attn_mask)
        h2 = self.blocks[2](h1, attn_mask)
        return self.head(h2), LayerFeatures(low=h0, mid=h1, high=h2)


class TinyCausalLM(nn.Module):
    """Single decoder block + LM head (Medusa / Leviathan host)."""

    def __init__(self, vocab_size: int, d_model: int = 32, n_heads: int = 4, max_len: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.block = _DecoderBlock(d_model, n_heads)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits, _ = self.forward_with_hidden(tokens, attn_mask)
        return logits

    def forward_with_hidden(
        self, tokens: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t = tokens.shape
        h = self.emb(tokens) + self.pos(torch.arange(t, device=tokens.device))
        h = self.block(h, attn_mask)
        return self.head(h), h


# ---------------------------------------------------------------------------
# EAGLE-3 draft (default)
# ---------------------------------------------------------------------------


class Eagle3Draft(nn.Module):
    """EAGLE-3 draft: multi-layer fusion + token-emb mix + one decoder + direct logits.

    Paper §3.1:
      g = FC(concat(l, m, h))                         # 3k → k
      x = FC(concat(g_or_a, e_token))                  # 2k → k
      a = DecoderLayer(x);  logits = LMHead(a)         # direct token prediction

    Weights start at PyTorch default init (untrained). Demos exercise fusion + tree
    verify on that random draft.
    """

    def __init__(self, d_model: int, vocab_size: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.fuse = nn.Linear(3 * d_model, d_model)
        self.in_proj = nn.Linear(2 * d_model, d_model)
        self.decoder = _DecoderBlock(d_model, n_heads)
        self.emb = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def fuse_features(self, feats: LayerFeatures) -> torch.Tensor:
        return self.fuse(torch.cat([feats.low, feats.mid, feats.high], dim=-1))

    def forward_draft(
        self,
        features: torch.Tensor,
        tokens: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """features/tokens: (B, T, …). Returns (a, logits) both (B, T, …)."""
        e = self.emb(tokens)
        x = self.in_proj(torch.cat([features, e], dim=-1))
        a = self.decoder(x, attn_mask)
        return a, self.lm_head(a)


# ---------------------------------------------------------------------------
# Named variant: Medusa head (last-hidden only)
# ---------------------------------------------------------------------------


class MedusaHead(nn.Module):
    """Medusa-style parallel heads: last hidden → next-k token logits (no layer fusion)."""

    def __init__(self, d_model: int, vocab_size: int, n_heads: int = 1):
        super().__init__()
        if n_heads < 1:
            raise ValueError("n_heads must be >= 1")
        self.n_heads = n_heads
        self.heads = nn.ModuleList([nn.Linear(d_model, vocab_size) for _ in range(n_heads)])

    def forward(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        """hidden: (B, d) or (B, T, d) — applies each head to the last position."""
        if hidden.dim() == 3:
            h = hidden[:, -1, :]
        else:
            h = hidden
        return [head(h) for head in self.heads]


class MedusaDraftModel(nn.Module):
    """Wrap target + first Medusa head so it looks like a chain draft model for Leviathan."""

    def __init__(self, target: nn.Module, head: MedusaHead):
        super().__init__()
        self.target = target
        self.head = head

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        _logits, hidden = self.target.forward_with_hidden(tokens, attn_mask)
        return self.head(hidden)[0].unsqueeze(1).expand(-1, tokens.size(1), -1)


# ---------------------------------------------------------------------------
# Sampling primitives
# ---------------------------------------------------------------------------


def _softmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        idx = logits.argmax(dim=-1)
        p = torch.zeros_like(logits)
        p.scatter_(-1, idx.unsqueeze(-1), 1.0)
        return p
    return torch.softmax(logits / temperature, dim=-1)


def _sample(probs: torch.Tensor, temperature: float, generator: torch.Generator | None) -> int:
    if temperature <= 0:
        return int(probs.argmax(dim=-1).item())
    return int(torch.multinomial(probs, 1, generator=generator).item())


def residual_after_reject(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Leviathan residual: normalize((p − q)_+). Named path when mass is zero → renormalize p."""
    r = (p - q).clamp(min=0)
    z = r.sum()
    if float(z) <= 0:
        # Zero residual mass: paper normalise is undefined; recover with target p (named, not silent).
        return p / p.sum().clamp(min=1e-12)
    return r / z


@dataclass
class SpecResult:
    tokens: list[int]
    n_draft_accepted: int
    rejected: bool


# ---------------------------------------------------------------------------
# Named variant: Leviathan chain draft-verify
# ---------------------------------------------------------------------------


@torch.no_grad()
def leviathan_decode(
    draft: nn.Module,
    target: nn.Module,
    prefix: torch.Tensor,
    gamma: int,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> SpecResult:
    """Leviathan et al. lossless draft-then-verify. ``prefix`` is (1, T)."""
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
        residual = residual_after_reject(p, q)
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


# ---------------------------------------------------------------------------
# Tree draft + verify (shared by EAGLE-3 default)
# ---------------------------------------------------------------------------


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
    """Expand a small tree: at depth d take top-``branches[d]`` tokens from the draft."""
    nodes: list[TreeNode] = []
    frontier = [(-1, prefix)]
    for depth, k in enumerate(branches):
        nxt: list[tuple[int, torch.Tensor]] = []
        for parent_idx, ctx in frontier:
            logits = draft(ctx)[:, -1, :]
            probs = _softmax(logits, temperature)[0]
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
    for i, node in enumerate(nodes):
        chain: list[int] = []
        p = node.parent
        while p >= 0:
            chain.append(p)
            p = nodes[p].parent
        chain.reverse()
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
    """One target forward with a tree mask; accept a single path (residual per rejected sibling)."""
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

    roots = [i for i, n in enumerate(nodes) if n.parent < 0]
    path: list[int] = []
    candidates = roots
    while candidates:
        scored = []
        for idx in candidates:
            p = parent_logit_row(idx)
            q = nodes[idx].q_prob.to(device)
            x = nodes[idx].token
            scored.append((idx, p, q, x))
        scored.sort(key=lambda t: float(t[2][t[3]].item()), reverse=True)
        chosen = None
        # Fresh p at this depth; update residual after **each** rejected sibling.
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
            p_res = residual_after_reject(p_res, q)
        if chosen is None:
            y = int(p_res.argmax().item()) if temperature <= 0 else _sample(p_res, temperature, generator)
            path.append(y)
            return path
        candidates = nodes[chosen].children
    return path


# ---------------------------------------------------------------------------
# EAGLE-3 default: multi-layer fusion draft tree + tree verify
# ---------------------------------------------------------------------------


@torch.no_grad()
def eagle3_draft_tree(
    target: MultiLayerCausalLM,
    draft: Eagle3Draft,
    prefix: torch.Tensor,
    branches: tuple[int, ...] = (2, 2),
    temperature: float = 1.0,
) -> list[TreeNode]:
    """Build a draft tree with EAGLE-3 fused features; self-predicted ``a`` replaces ``g`` on draft nodes."""
    if not isinstance(target, MultiLayerCausalLM):
        raise TypeError("eagle3_draft_tree requires MultiLayerCausalLM (multi-layer features)")
    if not isinstance(draft, Eagle3Draft):
        raise TypeError("eagle3_draft_tree requires Eagle3Draft")

    device = prefix.device
    _logits, feats = target.forward_layers(prefix)
    g_prefix = draft.fuse_features(feats)  # (1, T, d)

    nodes: list[TreeNode] = []
    # frontier: (parent_idx, token_seq, feature_seq) — features are g on prefix, a on draft
    frontier: list[tuple[int, torch.Tensor, torch.Tensor]] = [(-1, prefix, g_prefix)]

    for depth, k in enumerate(branches):
        nxt: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for parent_idx, tok_seq, feat_seq in frontier:
            _a, logits = draft.forward_draft(feat_seq, tok_seq)
            probs = _softmax(logits[:, -1, :], temperature)[0]
            topk = torch.topk(probs, k=min(k, probs.numel()))
            for tok_i in topk.indices.tolist():
                tok = int(tok_i)
                idx = len(nodes)
                nodes.append(TreeNode(token=tok, parent=parent_idx, q_prob=probs))
                if parent_idx >= 0:
                    nodes[parent_idx].children.append(idx)
                # Next step: append token; feature for new position = draft output a at last pos
                # (paper: a replaces g for unverified positions).
                child_tok = torch.cat([tok_seq, torch.tensor([[tok]], device=device)], dim=1)
                a_last = _a[:, -1:, :]
                # Extend feature sequence: keep prior features, append a as stand-in for g_new
                child_feat = torch.cat([feat_seq, a_last], dim=1)
                nxt.append((idx, child_tok, child_feat))
        frontier = nxt
    return nodes


@torch.no_grad()
def eagle3_decode(
    target: MultiLayerCausalLM,
    draft: Eagle3Draft,
    prefix: torch.Tensor,
    branches: tuple[int, ...] = (2, 2),
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Default path: EAGLE-3 multi-layer fusion draft tree + tree verify."""
    nodes = eagle3_draft_tree(target, draft, prefix, branches=branches, temperature=temperature)
    return tree_verify(target, prefix, nodes, temperature=temperature, generator=generator)


@torch.no_grad()
def medusa_draft_logits(target: nn.Module, head: MedusaHead, prefix: torch.Tensor) -> list[torch.Tensor]:
    _logits, hidden = target.forward_with_hidden(prefix)
    return head(hidden)
