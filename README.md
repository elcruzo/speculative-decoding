# Speculative decoding

Lossless draft-then-verify (Leviathan et al.) plus a small tree-verify path and a Medusa-like feature draft head.

## Papers

- Leviathan, Kalman & Matias, *Fast Inference from Transformers via Speculative Decoding* (ICML 2023). Draft model proposes γ tokens; target verifies in **one** forward; accept until the first rejection; on reject, sample from the residual `(p − q)_+`. This is **lossless**: the output distribution equals target-only sampling.
- Li et al., *EAGLE-3* (2025): the draft is not a second LM. A lightweight head reads **fused hidden features from multiple target layers** and predicts the next tokens. Training uses a feature-regression + draft-token loss. We **document** that design; the code implements the no-separate-model core: a linear `hidden → vocab` draft head (Medusa-shaped). Multi-layer fusion is the EAGLE-3 increment on top of that head.

## Classic algorithm (implemented)

1. Draft samples `x_1..x_γ ~ q(· | prefix, x_<i)`.
2. Target computes `p_i = p(· | prefix, x_<i)` for `i = 1..γ+1` in one forward.
3. For each i: accept `x_i` with probability `min(1, p_i(x_i)/q_i(x_i))`.
4. On first reject: sample `y ~ normalize((p_i − q_i)_+)` and stop.
5. If all γ accepted: sample one bonus token from `p_{γ+1}`.

Greedy (`temperature=0`) is the same rule with one-hot `p, q`: accept while draft argmax equals target argmax; the residual is the target argmax.

## Tree verify

Draft expands a top-2 then top-2 tree. The target sees `concat(prefix, node_tokens)` with a **tree attention mask** (each node attends to the prompt and its ancestors only). Acceptance walks from the root, applying the same residual rule among siblings, and returns one path.

## Run

```bash
python demo.py
python -m pytest test_spec_decode.py -q
```
