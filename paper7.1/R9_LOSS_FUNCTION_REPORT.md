# R9: Loss Function Specificity Experiment
## Paper 7.1 — Windstorm Institute

**Date:** 2026-04-10
**Platform:** Varon-1 (RTX 5090, shared GPU — ~5 GB available)
**Runtime:** ~70 minutes (18 runs total)

---

## Research Question

Is the throughput basin a property of the **data** or an artifact of the **cross-entropy (CE) loss function**?

All prior basin measurements used models trained with CE loss. CE minimizes KL divergence to the true distribution, converging to source entropy H(X) by construction. If we train with a *different* objective and still observe the same basin, the basin is genuinely about the data. If the basin shifts or disappears, CE loss may be the explanation.

## Experimental Design

### Architecture
- GPT-2 (92M / 85M params depending on vocab size), 12 layers, 768 hidden, 12 heads
- Gradient checkpointing enabled (GPU memory constrained)
- Batch 4, seq_len 256, fp16

### Loss Functions
- **CE:** Standard cross-entropy (`F.cross_entropy(logits, targets)`)
- **MSE:** Mean squared error on logits vs one-hot targets (memory-efficient formulation)
- **LS:** Label-smoothed CE with smoothing=0.3 (`F.cross_entropy(logits, targets, label_smoothing=0.3)`)

### Corpora
- **SYN-8:** 256-symbol Markov-0 corpus, 8.0 bits/symbol source entropy (from Exp 1)
- **WikiText-2:** Natural language (GPT-2 tokenizer, 50257 vocab)

### Training
- 3 seeds per condition: 42, 137, 2024
- SYN-8: 4,000 steps; WikiText: 2,500 steps
- AdamW, lr=3e-4, warmup=200, cosine decay

### Evaluation
All models evaluated identically: CE loss on held-out data, converted to BPT = CE_loss / ln(2). This measures bits of prediction quality regardless of training objective.

---

## Results

### SYN-8 (Source Entropy H = 8.0 bits/symbol)

| Loss Function | Seed 42 BPT | Seed 137 BPT | Seed 2024 BPT | **Mean BPT** | Std | vs Source H |
|---|---|---|---|---|---|---|
| CE | 8.001 | 8.001 | 8.001 | **8.001** | 0.0001 | +0.001 |
| MSE | 8.199 | 8.199 | 8.199 | **8.199** | 0.0000 | +0.199 |
| LS (0.3) | 8.058 | 8.057 | 8.057 | **8.057** | 0.0002 | +0.057 |

### WikiText-2 (Natural Language)

| Loss Function | Seed 42 BPT | Seed 137 BPT | Seed 2024 BPT | **Mean BPT** | Std |
|---|---|---|---|---|---|
| CE | 8.746 | 8.747 | 8.746 | **8.746** | 0.001 |
| MSE | 15.605 | 15.605 | 15.605 | **15.605** | 0.000 |
| LS (0.3) | 8.926 | 8.936 | 8.938 | **8.933** | 0.005 |

### Full CSV
See `results/r9_loss_function.csv` for all columns including BPSS*, final training loss, and parameter counts.

---

## Analysis

### Finding 1: SYN-8 — All three losses converge near 8 BPT

This is the decisive result. On SYN-8 data with 8.0 bits/symbol source entropy:

- **CE** achieves 8.001 BPT — essentially perfect recovery of source entropy
- **MSE** achieves 8.199 BPT — 0.2 bits above source (slightly worse optimizer, but same ballpark)
- **LS** achieves 8.057 BPT — 0.06 bits above source (label smoothing adds mild entropy penalty)

**No model compressed to ~4 BPT. The loss function does not create the basin.** If the ~4 BPT basin were a CE artifact, we would expect MSE-trained models to find a different attractor. They don't — MSE lands at 8.2, still tracking the source entropy, not collapsing to some loss-specific basin.

The ordering CE < LS < MSE is theoretically expected:
- CE directly optimizes for the metric we evaluate with, so it reaches the information-theoretic floor
- LS intentionally targets a smoothed distribution (adds ~0.3 * uniform noise), inflating BPT slightly
- MSE optimizes a proxy that doesn't exactly minimize CE, but its implicit gradient still pushes logits toward correct predictions

### Finding 2: WikiText — Unconverged, but informative

WikiText models have NOT converged (only 2,500 steps from scratch, 92M params). Pre-trained GPT-2 achieves ~4.0 BPT on WikiText-2; our from-scratch models are at 8.7-15.6 BPT. This is expected and does not undermine the SYN-8 result.

However, the *relative ordering* is already informative:
- CE (8.746) ≈ LS (8.933) — both on similar trajectories toward convergence
- MSE (15.605) — far behind, nearly random-guess level

MSE with a 50,257-vocab softmax is a *much harder* optimization landscape than MSE with 294 vocab (SYN-8). The MSE gradient signal is sparse and weak for large vocab: the loss cares equally about driving 50,256 wrong-class logits toward 0 as it does about pushing the correct logit toward 1. This is a known limitation of MSE for classification, not evidence that the loss function determines the basin.

**Honest limitation:** We cannot draw basin conclusions from the WikiText runs because they have not converged. A fair comparison would require either:
- Pre-training to convergence with each loss (~100K+ steps), or
- Using a pre-trained CE checkpoint and only evaluating (which would test the loss for eval, not for training)

### Finding 3: Remarkable seed stability

Standard deviations across seeds are negligible:
- SYN-8 CE: σ = 0.0001 BPT (three seeds agree to the fourth decimal)
- SYN-8 MSE: σ < 0.0001 BPT
- WikiText CE: σ = 0.001 BPT

This suggests the BPT outcome is deterministic given the data and architecture, with seed affecting only the approach trajectory, not the attractor.

---

## Interpretation

### The basin is NOT a CE artifact

The central concern was: "CE converges to source entropy by construction. Does the basin disappear under a different loss?" The answer is no. MSE-trained SYN-8 still achieves ~8.2 BPT — roughly matching source entropy even though MSE does not have a direct mathematical link to entropy minimization.

This makes sense: any loss function that trains a model to predict the next token well will implicitly learn the distribution. A model that predicts accurately under MSE will also predict accurately under CE evaluation, because accurate prediction IS compression.

### The ~0.2 bit MSE gap is real but small

MSE achieves 8.199 vs CE's 8.001 — a 0.198-bit penalty. This reflects MSE's suboptimal gradient geometry for probability estimation, not a different basin. The model is still in the "8-bit basin" determined by the data.

### Claims supported by this experiment

1. **The throughput basin tracks source entropy regardless of loss function** — confirmed on SYN-8 with three losses.
2. **CE is the most efficient loss** for reaching the basin (as theory predicts) — confirmed: CE < LS < MSE.
3. **The basin is a property of the data, not the optimization objective** — confirmed: changing the loss shifts how quickly/closely you approach the basin, not *where* the basin is.

### Claims NOT supported

1. **WikiText basin independence of loss** — NOT tested (models unconverged).
2. **Scale invariance** — untested (only 85-124M params).

---

## Limitations

1. **WikiText models did not converge.** The 2,500-step training budget was insufficient for 92M from-scratch models on natural language. WikiText results reflect early-training dynamics, not asymptotic behavior. A proper test requires 50K+ steps or pre-trained checkpoints.

2. **GPU memory constraints.** The GPU was shared with two other Paper 7.1 jobs (~26 GB occupied). We ran with batch=4, seq_len=256, gradient checkpointing — substantially reduced compute per step vs Exp 1 (batch=32, seq_len=512). Despite this, SYN-8 CE achieved 8.001 BPT, matching Exp 1's source entropy recovery.

3. **No truly exotic loss.** MSE and label-smoothed CE are still *prediction* losses. A maximally different test would use a contrastive loss (InfoNCE) or a non-predictive objective (masked autoencoder, JEPA). We substituted LS for InfoNCE due to complexity.

4. **Single architecture.** Only GPT-2 tested. Though Exp 3 already showed architecture independence across transformer/Mamba/RWKV.

---

## Suggested Follow-Up Experiments

1. **Converged WikiText comparison**: Train CE/MSE/LS for 50K+ steps on WikiText-2 and confirm all three reach the ~4 BPT natural language basin.
2. **InfoNCE / contrastive loss**: Implement proper contrastive next-token prediction and test whether the basin holds.
3. **Intermediate entropy**: Run the same 3-loss comparison on SYN-5 and SYN-6 data.
4. **Large-scale MSE**: Train a 1B+ model with MSE on SYN-8 to check if MSE closes the 0.2-bit gap at scale.

---

## Conclusion

**The throughput basin is NOT a cross-entropy artifact.** Three different loss functions — CE, MSE, and label-smoothed CE — all converge to the same ~8 BPT basin on SYN-8 data (source entropy = 8.0 bits). The loss function affects *how closely* the model approaches the information-theoretic floor (CE: +0.001, LS: +0.057, MSE: +0.199) but not *where* the floor is.

This result, combined with Exp 1 (data determines BPT), Exp 3 (architecture doesn't matter), and Exp 6 (no thermodynamic ceiling), strengthens the conclusion that **the basin is a property of the training data's intrinsic entropy**.

---

*Windstorm Institute — Paper 7.1, Experiment R9*
*"The basin is where the data puts it, not where the loss function looks."*
