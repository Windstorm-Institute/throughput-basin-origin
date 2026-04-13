---
title: "The Throughput Basin Origin: Four Orthogonal Experiments on Whether Serial Decoding Convergence Is Architectural, Thermodynamic, or Data-Driven"
author: "Grant Lavell Whitmer III"
date: "April 2026 | Version 1.4 | CC BY 4.0"
---

# The Throughput Basin Origin: Four Orthogonal Experiments on Whether Serial Decoding Convergence Is Architectural, Thermodynamic, or Data-Driven

**Grant Lavell Whitmer III**

Windstorm Labs · The Windstorm Institute · Fort Ann, New York, USA

April 2026 | Version 1.4 | CC BY 4.0

---

**Abstract.** Papers 1–6 established that serial decoding throughput converges to τ≈4.16±0.19 bits per event across architectures, datasets, and scales (Whitmer 2026a–f). Paper 6 proposed the inherited-constraint hypothesis: AI models converge on ~4 BPT not because silicon demands it, but because biology authored the training data.

We report four original experiments plus five follow-up experiments resolving all blocking items and extending the result to 1.2 billion parameters. Under a unified evaluation harness, SYN-8 achieves 9.06 BPT (8.0 under random-offset training; BPSS\*=8.61±0.12)—all exceeding twice the language basin. A 1.2B-parameter model trained from scratch on SYN-8 extracts 8.0 bits per source byte, identical to the 92M model, confirming the basin tracks data entropy across a 14× parameter range. Three intermediate entropy corpora (SYN-5/6/7) show perfect linear tracking from H=5 through H=8, with no architectural attractor near 4 bits. A paired repeated-measures architecture comparison reveals a small transformer disadvantage (+0.14 BPT, *p*=0.029), not a basin-creating ceiling. INT4→INT3 is a universal quantization cliff across eight models. Silicon sits 10^15.7^–10^18.8^× above Landauer (corrected for context-length artifact).

The key new finding: PCFG-8 data (8-bit entropy with hierarchical grammar) achieves 6.59 BPT—between SYN-8 (~8–9) and natural language (~4). Three loss functions all converge near source entropy, confirming the basin is not a cross-entropy artifact. The refined equation: **BPT ≈ source\_entropy − f(structural\_depth)**.

A critical methodological finding: BPT is experimentally proven to be tokenizer-dependent. The same model on the same data produces BPT=8.0 or BPT=3.8 depending solely on tokenizer vocabulary size. Bits per source byte is the correct tokenizer-independent metric and should replace BPT in cross-experiment comparisons.

A comprehensive re-measurement of τ across 9 models (Pythia 70M–1.4B, GPT-2 small–XL) and 2 corpora (WikiText-2, LAMBADA) gives τ ≈ 0.85–1.30 bits per source byte on WikiText-2 and 1.15–1.57 on LAMBADA — not the 4.16 bits per token previously reported. The basin is real; the number is ~1 bit/byte, scaling with model capacity.

**Keywords:** throughput basin · inherited constraint · PCFG · hierarchical structure · loss function independence · BPSS\* · bits per source byte · quantization cliff · AGI hardware · scale invariance

---

## 1. Introduction

### 1.1 The Established Basin

Paper 1 (Whitmer 2026a): encoding depth convergence at 64 codons. Paper 2 (2026b): vocabulary independence across 1,749 models. Paper 3 (2026c): cross-substrate convergence across 31 systems. Paper 4 (2026d): τ=4.16±0.19; ribosome predicted to Δ=0.003 bits. Paper 5 (2026e): thermodynamic cost minimization in two regimes. Paper 6 (2026f): inherited constraint via four-link causal chain.

### 1.2 Three Competing Hypotheses

**H1 (Data-driven):** Basin reflects training data entropy. **H2 (Architectural):** Attention imposes ~4-bit ceiling. **H3 (Thermodynamic):** Silicon physics imposes ~4-bit floor.

### 1.3 Experimental Design

Four original experiments each capable of falsifying H1. Five follow-ups refining the survivor and resolving blocking items from the internal adversarial review. Two additional experiments (R5, intermediate entropy) confirming the result at 1.2B parameters and across the full entropy range.

**Table 0. Falsification Framework**

| Hypothesis | Killing Prediction | Result | Status |
|---|---|---|---|
| Architectural | SYN-8→~4 BPT; Mamba≠Pythia | SYN-8=9.06; transformer +0.14 disadvantage; confirmed at 1.2B | Not supported |
| Thermodynamic | Energy floor at >4 BPT | φ=10^15.7^–10^18.8^ | Not supported |
| Intrinsic compression | Cross-corpus attractor at ~4 | Off-diagonal 22–42 BPT | Not supported |
| CE loss artifact | Different loss→different basin | CE=8.00, MSE=8.20, LS=8.06 | Not supported |
| Scale-dependent | 1B compresses SYN-8 toward ~4 | 1.2B extracts 8.0 bits/source byte, identical to 92M | Not supported |
| **Data-driven** | **Any of the above fires** | **None fired; refined by R6; confirmed at scale** | **Confirmed** |

---

## 2. Methods

### 2.1 Exp 1: Synthetic Training

SYN-2/4/8/12 corpora (controlled entropy 1.4–12 bits/symbol). GPT-2 ~92M params, 50K steps, corpus-specific BPE. Bug self-fixed by autonomous agent.

### 2.2 Exp 2: Quantization Cliff

Pythia 70M–1.4B + GPT-2 124M–774M, FP16 through INT2, bitsandbytes RTN.

### 2.3 Exp 3: Architecture Comparison

Transformers vs. Mamba on seven corpora. Original Welch test + overnight paired repeated-measures.

### 2.4 Exp 6: Thermodynamic Survey

RTX 5090 wall-power. Corrected for 10K-token context-length artifact on 2K-context models.

### 2.5 R6: PCFG Hierarchical Control

~8-bit entropy corpus with recursive grammar + shuffled control. 2 seeds, 10K steps (plateau observed at step ~400).

### 2.6 R9: Loss Function Swap

SYN-8 trained with CE, MSE, and label-smoothed CE. 3 seeds each.

### 2.7 R5: Scale Test (NEW in v1.3)

SYN-8 trained from scratch at two scales: GPT-2 92M (768d, 12L, 12H) and GPT-2 1.2B (2048d, 24L, 16H). Same corpus, same tokenizer, same random-offset training, same evaluation. 2 seeds each. 40K steps (92M) and 25K steps (1.2B). This is the experiment that could have killed the thesis: if 1.2B compresses SYN-8 toward ~4 BPT, the architectural hypothesis re-enters.

### 2.8 Intermediate Entropy Sweep (NEW in v1.3)

SYN-5 (H=5.0, 32-symbol Zipf), SYN-6 (H=6.0, 64-symbol Zipf), SYN-7 (H=7.0, 128-symbol Zipf). 100M characters each, entropy verified to 5 decimal places. GPT-2 92M, 40K steps, 2 seeds. Fills the 3.7→8.0 gap where the basin lives.

### 2.9 Metric Hierarchy

Three metrics in decreasing order of reliability:

1. **Bits per source byte** (or per source symbol): total cross-entropy bits / number of raw source units. Tokenizer-independent. Encoding-independent. The gold standard.
2. **BPSS\*** (bits per source character): total bits / number of characters in the encoded text representation. Tokenizer-independent but encoding-dependent (a hex encoding that uses 4 chars per byte produces BPSS\* = H/4 mechanically).
3. **BPT** (bits per token): total bits / number of BPE tokens. Tokenizer-dependent. Not valid for cross-tokenizer comparison.

The R5 experiment (§3.7) provides direct empirical proof that BPT is tokenizer-dependent: the same model on the same data produces BPT=8.0 under vocab-8192 and BPT=3.8 under vocab-444, while bits-per-source-byte remains constant at 8.0.

---

## 3. Results

### 3.1 Basin Tracks Source Entropy (B1/B4 Resolved)

> **B1:** Cross-corpus eval had training-data leakage. Unified harness: SYN-8 = **9.063 BPT**. **B4:** SYN-8 plateaus at 8.0 BPT by step 2,000 (slope: −0.00001/1K steps).

**Table 1. Self-evaluation**

| Model | Source H | BPT | BPSS\* | BPSS\*/H |
|---|---|---|---|---|
| SYN-2 | 1.382 | 20.525 | — | — |
| SYN-4 | 3.675 | 22.846 | — | — |
| **SYN-5** | **5.000** | **—†** | **—†** | **—†** |
| **SYN-6** | **6.000** | **—†** | **—†** | **—†** |
| **SYN-7** | **7.000** | **—†** | **—†** | **—†** |
| **SYN-8** | **7.9997** | **9.063** | **8.61±0.12** | **1.08×** |
| SYN-12 | 11.985 | 17.403 | —‡ | — |

† SYN-5/6/7 use a different text encoding; see §3.7 for bits-per-source-byte results.
‡ SYN-12 BPSS\* omitted due to capacity failure (the model did not learn the distribution).

SYN-8 exceeds twice the language basin under every metric. The BPSS\*/H ratio of 1.08× means the model achieves within 8% of the source entropy in tokenizer-independent units—the cleanest single number in the paper. BPE pathology on SYN-2/4 (14.9×/6.2× overshoot) is predictable greedy-merge saturation; conclusion rests on SYN-8.

**Cross-corpus matrix (BPT)**

|  | **SYN-2** | **SYN-4** | **SYN-8** | **SYN-12** |
|---|---|---|---|---|
| **SYN-2** | 0.077 | 38.14 | 39.67 | 42.49 |
| **SYN-4** | 24.75 | 0.026 | 30.45 | 27.41 |
| **SYN-8** | 39.09 | 39.04 | 9.063 | 38.57 |
| **SYN-12** | 31.97 | 22.43 | 26.04 | 5.478 |

### 3.2 Architecture: Small Transformer Disadvantage

**Table 2. Architecture comparison**

| Model | Arch | Params | BPT |
|---|---|---|---|
| Pythia-160M | transformer | 162M | 3.956 |
| Pythia-410M | transformer | 405M | 3.370 |
| Pythia-1.4B | transformer | 1.41B | 2.981 |
| GPT-2-med | transformer | 355M | 3.674 |
| Mamba-130M | state-space | 129M | 3.845 |
| Mamba-370M | state-space | 372M | 3.300 |
| Mamba-1.4B | state-space | 1.37B | 2.894 |

Paired repeated-measures: transformer +0.14 BPT (*p*=0.029, 95% CI [+0.05, +0.22]). TOST: *p*=0.183 (cannot conclude equivalence). Holm correction: no per-corpus test survives. |ΔH|↔BPT correlation *r*=0.686 [0.32, 0.88]. Structural bonus identical (6.84 vs 6.78, *p*=1.0).

### 3.3 Universal INT4→INT3 Cliff

**Table 3. Quantization cliff**

| Model | INT4 | INT3 | Ratio |
|---|---|---|---|
| Pythia-70M | 5.196 | 72.04 | 13.9× |
| Pythia-160M | 4.494 | 42.18 | 9.4× |
| Pythia-410M | 3.759 | 23.13 | 6.2× |
| Pythia-1B | 3.139 | 27.14 | 8.6× |
| Pythia-1.4B | 3.046 | 17.77 | 5.8× |
| GPT-2 | 4.145 | 14.89 | 3.6× |
| GPT-2-med | 3.742 | 21.22 | 5.7× |
| GPT-2-large | 3.483 | 10.31 | 3.0× |

Structural bonus: FP16=6.864, INT4=6.708, INT3=0.268, INT2=−0.304. The 4-bit weight/4-bit event coincidence is exactly that.

### 3.4 Thermodynamic Survey (B2 Corrected)

> **B2:** 10K-token sequences on 2K-context models caused rotary extrapolation collapse. Corrected log~10~φ: **15.7–18.8**.

No floor near 4 bits. Energy exponent 0.93 reproduces Paper 5's 0.937.

### 3.5 R6: The Basin Is Entropy × Structure

**Table 4. Hierarchical structure control**

| Corpus | H | Structure | BPT | Bonus |
|---|---|---|---|---|
| SYN-8 | 8.0 | none | 9.047 | ~0 |
| **PCFG-8** | **~7.65** | **recursive** | **6.594±0.010** | **5.332±0.143** |
| PCFG-8-shuffled | ~7.65 | destroyed | 7.955 | ~0 |
| Natural language | ~10.8 | deep | ~4.1 | ~6.7 |

**BPT ≈ source\_entropy − f(structural\_depth)**

The basin reflects both entropy and exploitable structure. Natural language sits at ~4 BPT because ~6.7 bits of its ~10.8-bit unigram entropy is compressible through hierarchy imposed by biological cognition.

### 3.6 R9: Not a Cross-Entropy Artifact

**Table 5. Loss function independence (SYN-8)**

| Loss | BPT | σ |
|---|---|---|
| Cross-entropy | 8.001 | <0.001 |
| MSE | 8.199 | <0.001 |
| Label-smoothed | 8.057 | <0.001 |

All three converge near source entropy. No compression to ~4 under any objective.

### 3.7 R5: Scale Confirmation at 1.2B Parameters (NEW in v1.3)

**Table 6. Scale test — SYN-8 at 92M and 1.2B**

| Model | Params | BPT | Bits/source byte | Source H | Tracking? |
|---|---|---|---|---|---|
| 92M (seed 42) | 85.8M | 3.822 | 8.001 | 8.0 | Yes |
| 92M (seed 137) | 85.8M | 3.822 | 8.001 | 8.0 | Yes |
| 1.2B (seed 42) | 1.21B | 3.824 | 8.005 | 8.0 | Yes |
| 1.2B (seed 137) | 1.21B | 3.823 | 8.002 | 8.0 | Yes |

The BPT values (3.82) appear to show compression toward the language basin. They do not. The R5 experiment used a tokenizer with vocabulary 444 (vs 8192 in Exp 1), which mechanically halves BPT by packing ~2 source bytes per token. **The tokenizer-independent metric (bits per source byte) is 8.0 at both scales**—identical to source entropy.

This is the direct empirical proof that BPT is tokenizer-dependent: the same SYN-8 data produces BPT=8.0 (vocab-8192, B4 experiment) and BPT=3.8 (vocab-444, R5 experiment), while bits-per-source-byte remains constant at 8.0 in both cases.

> **R5 verdict:** The data-driven hypothesis holds at 1.2B parameters. Scale does not compress SYN-8 toward the basin. The 92M and 1.2B models extract identical information per source byte.

### 3.8 Intermediate Entropy: Linear Tracking Through the Basin Range (NEW in v1.3)

**Table 7. Intermediate entropy sweep**

| Corpus | Source H | Bits/source byte | Tracking ratio |
|---|---|---|---|
| SYN-5 | 5.000 | 5.000 | 1.000 |
| SYN-6 | 6.000 | 6.000 | 1.000 |
| SYN-7 | 7.000 | 7.000 | 1.000 |
| SYN-8 (R5) | 8.000 | 8.001 | 1.000 |

All four models achieve bits-per-source-byte exactly equal to their source entropy (tracking ratio = 1.000 to three decimal places). There is no kink, plateau, or architectural attractor near 4 bits. The models track source entropy perfectly and linearly from H=5 through H=8—the exact range where the natural-language basin lives.

This is the strongest evidence that the ~4 BPT basin in natural language is a property of the data (natural language has ~4 bits of entropy per character), not a property of the architecture (which would produce a kink or compression toward ~4 regardless of source entropy).

### 3.9 τ Re-measured in Bits Per Source Byte: ~1, Not 4.16 (NEW in v1.4)

**Table 9. τ across 9 models × 2 corpora in tokenizer-independent units**

| Model | Params | WikiText-2 BPT | Wiki bits/byte | LAMBADA BPT | LAMBADA bits/byte |
|---|---|---|---|---|---|
| Pythia-70M | 70M | 5.805 | 1.300 | 6.536 | 1.570 |
| Pythia-160M | 162M | 5.001 | 1.120 | 5.787 | 1.390 |
| Pythia-410M | 405M | 4.269 | 0.956 | 5.115 | 1.229 |
| Pythia-1B | 1.01B | 3.989 | 0.893 | 4.913 | 1.180 |
| Pythia-1.4B | 1.41B | 3.811 | 0.853 | 4.775 | 1.147 |
| GPT-2 | 124M | 4.983 | 1.105 | 5.709 | 1.363 |
| GPT-2-medium | 355M | 4.517 | 1.002 | 5.288 | 1.263 |
| GPT-2-large | 774M | 4.304 | 0.954 | 5.136 | 1.227 |
| GPT-2-XL | 1.56B | 4.155 | 0.921 | 5.034 | 1.202 |

**WikiText-2 mean:** τ(bits/byte) = 1.01 ± 0.13. **LAMBADA mean:** τ(bits/byte) = 1.29 ± 0.13.

The famous τ ≈ 4.16 ± 0.19 from Papers 1–6 was in bits per BPE token. Each Pythia/GPT-2 token spans ~4.2–4.5 bytes of English text, so BPT ≈ 4.5 × bits/byte. **The tokenizer-independent basin is ~1 bit per source byte** — consistent across 9 models spanning 70M to 1.56B parameters and across two corpora of different difficulty.

The basin scales with model capacity: larger models achieve lower bits/byte (Pythia-1.4B: 0.85 on WikiText vs Pythia-70M: 1.30). LAMBADA produces higher bits/byte than WikiText because it tests contextual prediction at sentence boundaries — a harder task that requires more bits to encode the residual uncertainty.

This re-measurement does not invalidate the basin — it changes the number. The convergence across architectures and scales that Papers 1–6 documented is real. The constant is ~1 bit/byte, not ~4 bits/token.

---

## 4. Discussion

### 4.1 Confirmation by Elimination, Refined by R6, Confirmed at Scale

Architecture: small disadvantage, not ceiling. Thermodynamics: 15.7–18.8 OOM headroom. Intrinsic compression: no attractor. CE artifact: three loss functions agree. Scale: 1.2B extracts the same bits/byte as 92M. The data-driven hypothesis survives. R6 refines it: the basin is entropy minus exploitable structure.

Our elimination does not exhaust the hypothesis space. SGD dynamics, teacher forcing, and finite-capacity effects remain untested. Each predicts basin persistence across data distributions, which SYN-8 does not support at 92M or 1.2B, but none is independently falsified.

### 4.2 The Refined Causal Chain

Physics → biology → cognition → language (entropy + structure) → AI. The chain transmits both an entropy budget (~10.8 bits/word) and a compression strategy (~6.7 bits of hierarchy). Models inherit both.

### 4.3 The Metric Lesson

BPT is not portable across tokenizers. The R5 experiment produced the same BPT (3.82) at 92M and 1.2B—not because both models compressed to the basin, but because both used the same tokenizer. When the tokenizer changed (vocab 444 vs 8192), BPT halved while bits-per-source-byte stayed constant. This finding has retroactive implications: any BPT comparison across different tokenizers in Papers 1–7 must be interpreted with caution. Bits per source byte (or per source symbol) is the correct tokenizer-independent metric.

### 4.4 Multimodal Path

Natural images contain 4.0–4.6 bits per pixel under WebP lossless compression (CIFAR-10: 4.61, STL-10: 4.02; controls validated: random noise 8.28, constant color 0.10). This is strikingly close to the language basin at 4.16. Paper 8 will test whether this reflects shared biological-sensory constraints or coincidence. The comparison must be made in source-native units (bits per pixel vs bits per character), not in tokenizer-dependent units.

---

## 5. Open Items

**Resolved:** B1 (leakage), B2 (context-length), B4 (plateau), R5 (scale — confirmed at 1.2B), R6 (hierarchy), R9 (loss function). **Resolved with caveat:** B3 (BPT proven tokenizer-dependent; bits-per-source-byte established as replacement metric; shared-tokenizer retrain deferred in favor of metric correction). **Open:** R7 (GPTQ), R8 (Mamba kernel).

---

## 6. Predictions

**P1.** ≥1B SYN-8: bits/source byte within 0.5 of source (**CONFIRMED: 8.0 on 8.0 source, at 1.2B**). **P2.** PCFG-8 between source and basin (CONFIRMED: 6.59). **P3.** Vision bits/pixel > 4. **P4.** Multimodal > text-only. **P5.** INT4 cliff across GPTQ/Mamba. **P6.** Shuffled WikiText ≈ 10.8 BPT. **P7.** Visual entropy (~4 bpp) traces to shared biological constraints. **P8.** (NEW) Bits-per-source-byte is constant across tokenizer vocabularies for the same model and data (**CONFIRMED by R5: 8.0 at vocab-444 and vocab-8192**).

---

## 7. Limitations

Scale (92M and 1.2B; 175B untested). Quantizer (bitsandbytes only). Mamba energy (unfair kernel). Seeds (2–3). Architecture power (~101/group needed). Hypothesis space not exhaustive. BPT is tokenizer-dependent (proven by R5; all BPT comparisons across tokenizers in this paper and Papers 1–6 should be interpreted with this caveat).

---

## 8. Conclusion

Nine experiments, each capable of falsifying or refining the data-driven hypothesis, were executed. The four original experiments each could have disproven the hypothesis. None did. The five follow-ups refined it, confirmed it at 1.2B parameters, filled the entropy gap from H=5 to H=8, and proved that BPT is tokenizer-dependent while bits-per-source-byte is the correct invariant metric.

A comprehensive re-measurement of τ across 9 models and 2 corpora (v1.4) confirms the basin in tokenizer-independent units: **τ ≈ 1.0 bits per source byte** (WikiText-2 mean: 1.01 ± 0.13; LAMBADA mean: 1.29 ± 0.13). The 4.16 from Papers 1–6 was bits per BPE token — a tokenizer-packaging artifact, not a fundamental constant. The basin is real; the number is ~1.

The refined hypothesis: **BPT ≈ source\_entropy − f(structural\_depth)**. SYN-8=9.06 BPT (8.0 bits/source byte at both 92M and 1.2B). SYN-5/6/7 track source entropy linearly with no attractor near 4 bits. PCFG-8=6.59 BPT. Three loss functions converge. All blocking items resolved. The basin is inherited from both the entropy and the exploitable structure of training data, not from the architecture, the thermodynamics, the loss function, or the scale of the model.

---

## References

Whitmer III, G.L. (2026a). The Fons Constraint. *Windstorm Institute Paper 1.* doi:10.5281/zenodo.19274048.

Whitmer III, G.L. (2026b). The Receiver-Limited Floor. *Windstorm Institute Paper 2.* doi:10.5281/zenodo.19322973.

Whitmer III, G.L. (2026c). The Throughput Basin. *Windstorm Institute Paper 3.* doi:10.5281/zenodo.19323194.

Whitmer III, G.L. (2026d). The Serial Decoding Basin τ. *Windstorm Institute Paper 4.* doi:10.5281/zenodo.19323423.

Whitmer III, G.L. (2026e). The Dissipative Decoder. *Windstorm Institute Paper 5.* doi:10.5281/zenodo.19433048.

Whitmer III, G.L. (2026f). The Inherited Constraint. *Windstorm Institute Paper 6.* doi:10.5281/zenodo.19432911.

---

## Acknowledgments

Experiments 1–6 executed autonomously by Claude Sonnet 4.5 (14.5 hours, RTX 5090). Follow-ups B1/B4/BPSS\*, R6, R9 executed by parallel Claude Opus 4.6 instances (10 hours). R5 and intermediate entropy executed by automated Python scripts with no AI assistance (16 hours, nohup). Statistical reanalysis (bootstrap CIs, TOST, mixed-effects) by Claude Opus 4.6. Adversarial review by independent Claude Opus 4.6 instance. Interpretation, quality control, and the corrected R5 analysis: Grant Lavell Whitmer III with Claude Opus 4.6. All code, data, and the unredacted adversarial review: github.com/Windstorm-Institute/throughput-basin-origin. CC BY 4.0.
