# Speculative decoding

**Default:** EAGLE-3 multi-layer hidden fusion + tree verify (Li et al. 2025).

**Named variants:** Leviathan chain draft-verify; Medusa last-hidden heads.

## Papers

- Li et al., *EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test* (2025) ([arXiv:2503.01840](https://arxiv.org/abs/2503.01840)). Abandons feature regression for **direct token prediction**. Fuses **low / mid / high** target hiddens: $g=\mathrm{FC}(\mathrm{concat}(l,m,h))$. Draft mixes $g$ (or self-predicted $a$) with the token embedding, runs **one decoder layer**, then an LM head. Compatible with EAGLE-2 **dynamic draft trees** and tree-attention verify.
- Leviathan, Kalman & Matias, *Fast Inference from Transformers via Speculative Decoding* (ICML 2023). Draft proposes $\gamma$ tokens; target verifies in one forward; accept until first rejection; on reject sample from $(p-q)_+$. **Lossless** vs target-only sampling.
- Cai et al., *Medusa* (2024): parallel heads on the **last** hidden predict future tokens (no multi-layer fusion).

## Default algorithm (EAGLE-3)

1. Target forward on the prefix → layer features $l,m,h$ → fused $g=\mathrm{FC}(\mathrm{concat}(l,m,h))$.
2. Draft tree: at each node, $x=\mathrm{FC}(\mathrm{concat}(g\text{ or }a,\,e_{\mathrm{tok}}))$, $a=\mathrm{Decoder}(x)$, sample top-$k$ from $\mathrm{LMHead}(a)$. Unverified positions reuse draft $a$ in place of target $g$.
3. One target forward over $\mathrm{concat}(\mathrm{prefix},\,\mathrm{node\ tokens})$ with a **tree attention mask**.
4. Walk root→leaf; among siblings apply Leviathan accept with residual updated after **each** rejected sibling: $p\leftarrow\mathrm{normalize}((p-q)_+)$.

## Named variants

| API | Role |
|---|---|
| `eagle3_decode` / `Eagle3Draft` | Default — fusion + tree |
| `leviathan_decode` | Classic $\gamma$-chain draft-verify |
| `MedusaHead` / `MedusaDraftModel` | Last-hidden parallel heads → Leviathan verify |

No silent fallback between these paths: wrong types raise `TypeError`.

## Papers on disk

- [`papers/leviathan-speculative-decoding-2023.pdf`](papers/leviathan-speculative-decoding-2023.pdf) — Leviathan et al. (2023) ([arXiv:2211.17192](https://arxiv.org/abs/2211.17192))
- [`papers/li-eagle-2024.pdf`](papers/li-eagle-2024.pdf) — Li et al. EAGLE (2024) ([arXiv:2401.15077](https://arxiv.org/abs/2401.15077))
- [`papers/li-eagle-3-2025.pdf`](papers/li-eagle-3-2025.pdf) — Li et al. EAGLE-3 (2025) ([arXiv:2503.01840](https://arxiv.org/abs/2503.01840))
- [`papers/cai-medusa-2024.pdf`](papers/cai-medusa-2024.pdf) — Cai et al. Medusa (2024) ([arXiv:2401.10774](https://arxiv.org/abs/2401.10774))

## Compared to EAGLE-3 / Leviathan

**What you learn here:**
- EAGLE-3 multi-layer hidden fusion $g=\mathrm{FC}(\mathrm{concat}(l,m,h))$ + tree verify
- Leviathan accept/reject with residual $p\leftarrow(p-q)_+$ after each rejected sibling
- Named Medusa last-hidden heads — no silent path switch

| | This repo | EAGLE-3 (Li et al. 2025) |
|---|---|---|
| Target | Tiny MultiLayerCausalLM (d=16) | LLaMA / DeepSeek-scale |
| Draft | `Eagle3Draft` one decoder layer | Trained draft on ShareGPT/UltraChat |
| Verify | Tree mask + Leviathan walk | Same accept math; CUDA/SGLang stack |

### Numbers (2026-08-16, Darwin 25.5.0 arm64 / Apple M5)

| Metric | This repo | Baseline | Source |
|---|---|---|---|
| EAGLE-3 accepted path | `[0]` (toy untrained) | τ ≈ 4.0–7.5 | arXiv:2503.01840; `main.py` |
| Leviathan `draft_accepted` | 0 (BigramLM γ=4) | speedup up to 6.5× | same paper; `main.py` |
| Target greedy length | 6 tokens | — | `main.py` |

```bash
python main.py
```

## Run

```bash
python main.py
python -m pytest test_spec_decode.py -q
```
