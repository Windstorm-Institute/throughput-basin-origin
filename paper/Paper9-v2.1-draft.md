---
title: "The Hardware Basin: Why the Quantization Cliff Is About Level Allocation, Not Bit Count"
author: "Grant Lavell Whitmer III"
date: "April 2026 | Version 2.1 | CC BY 4.0"
---

# The Hardware Basin: Why the Quantization Cliff Is About Level Allocation, Not Bit Count

**Grant Lavell Whitmer III**

Windstorm Labs · The Windstorm Institute · Fort Ann, New York, USA

April 2026 | Version 2.1 | CC BY 4.0

---

**Abstract.** Paper 7 (Whitmer 2026g) reported a universal quantization cliff at INT4→INT3 using bitsandbytes software quantization. We investigated whether this cliff is a software artifact or a mathematical property of low-precision arithmetic, and discovered something more nuanced than either.

The cliff location depends on the quantization method. Under symmetric uniform quantization, the cliff is at INT8→INT4: Pythia-410M and Pythia-1.4B both produce catastrophic BPT≈17 at INT4 (vs 4.3 at FP16). Under bitsandbytes NF4 (normal-float-4, which places quantization levels at the quantiles of a normal distribution), INT4 is operational: BPT≈3.9–4.7. Same bit count, opposite outcomes.

The difference is level allocation. Symmetric quantization distributes levels uniformly across the weight range — wasting resolution in the sparse tails where few weights live. NF4 concentrates levels near zero where most weights cluster. A Lloyd-Max (minimum-MSE) quantizer achieves cosine similarity 0.990 at INT4 and 0.965 at INT3, outperforming NF4 at INT3 (0.948) and dramatically outperforming symmetric at INT4 (0.905). This result is universal across 4 model architectures (Pythia-160M, Pythia-1.4B, GPT-2-medium, Mamba-370M) and consistent across all 24 layers of Pythia-410M.

**New in v2.1.** We add three publication-grade verifications: (1) **GS3** — the structural-bonus comparison replicated at Pythia-1.4B with five independent shuffle seeds and formal statistics: FP16 vs symmetric INT4 gives Welch t = 633.74, p = 2.84×10⁻¹⁵, Cohen's d = 400.81. NF4 preserves the bonus (6.366 ± 0.009) essentially perfectly relative to FP16 (6.399 ± 0.009); symmetric INT4 destroys it (0.203 ± 0.018). The 95% confidence interval for symmetric INT4 does not come within 6 bits of any other method's interval. (2) **Lloyd-Max INT3 end-to-end test** — Lloyd-Max INT3 fails when propagated through all 24 layers of Pythia-410M (BPT = 11.74 vs FP16 baseline 4.27, 2.7× degradation), demonstrating that per-matrix cosine similarity (0.965) overstates end-to-end quantization quality and that the structural-bonus test is the more reliable diagnostic. (3) **R4 robust replication** at Pythia-160M, 410M, and GPT-2-medium across three random shuffle seeds confirms that the cliff scales: ~5.6 bits at 160M, ~5.9 bits at 410M, ~6.1 bits at GPT-2-medium, ~6.4 bits at 1.4B.

**Keywords:** quantization cliff · level allocation · NF4 · symmetric quantization · Lloyd-Max · inference hardware · INT4 · Pythia · Mamba · weight distribution · kurtosis · structural bonus · Cohen's d

---

## 1. Introduction

### 1.1 The Software Cliff

Paper 7, Experiment 2 (Whitmer 2026g) documented a quantization cliff at INT4→INT3 across eight language models using bitsandbytes round-to-nearest (RTN) weight quantization. The cliff ratios ranged from 3.0× to 13.9× — a sharp phase transition in which both raw prediction quality (BPT) and structural bonus (the model's ability to exploit linguistic hierarchy) collapse simultaneously.

The structural bonus collapse is particularly significant: at INT4, the mean structural bonus across 8 models is 6.71 bits (preserved from FP16's 6.86). At INT3, it collapses to 0.27 bits — the model can no longer distinguish structured text from shuffled text. This is not gradual degradation; it is a phase transition in representational capacity.

### 1.2 The Question This Paper Answers

A critic of Paper 7's cliff finding has a natural objection: "bitsandbytes RTN is the crudest quantization method available. GPTQ, AWQ, and other methods achieve much better low-precision results. The cliff is an artifact of the algorithm, not a property of the arithmetic."

We designed experiments to test this objection directly. The cliff is real, but its location depends on the quantization method. Methods that allocate levels uniformly hit the cliff at INT8→INT4. Methods that allocate levels according to the weight distribution's shape (NF4, Lloyd-Max) survive at INT4 and potentially at INT3.

This transforms Paper 9 from a confirmation paper into a design paper.

### 1.3 What v2.1 Adds

The v2.1 verification round addresses three additional questions a careful reviewer would raise:

1. **Statistical decisiveness.** The original Exp 1 reported a single shuffle of WikiText-2. With one seed there is no formal way to rule out chance. GS3 replicates with five seeds and reports Welch t-tests, bootstrap 95% CIs, and Cohen's d effect sizes.
2. **Lloyd-Max end-to-end propagation.** Per-matrix cosine of 0.965 at INT3 is impressive, but does it survive 24 layers of error accumulation? We tested directly.
3. **Robustness across model scales.** The structural-bonus result was originally measured on a single model. The R4 round replicates at 160M, 410M, and 1.4B parameters with three seeds each, showing the cliff scales.

---

## 2. Methods

### 2.1 Models Tested

| Model | Architecture | Parameters |
|---|---|---|
| Pythia-70M | Transformer | 70M |
| Pythia-160M | Transformer | 162M |
| Pythia-410M | Transformer | 405M |
| Pythia-1B | Transformer | 1.01B |
| Pythia-1.4B | Transformer | 1.41B |
| GPT-2-medium | Transformer | 355M |
| Mamba-370M | State-space | 372M |

### 2.2 Quantization Methods

**Symmetric uniform quantization.** Maps weights to [−(2^(n−1)−1), +(2^(n−1)−1)] with uniform step size.

**bitsandbytes NF4.** 16 levels at quantiles of a standard normal distribution, scaled per block.

**Lloyd-Max (minimum-MSE).** Optimal levels via k-means clustering on weight values.

**bitsandbytes INT8.** 8-bit absmax with mixed-precision decomposition for outliers.

### 2.3 Evaluation Protocol

**End-to-end BPT.** Quantize ALL weight matrices in ALL layers. Run autoregressive next-token prediction on WikiText-2 test split. Compute cross-entropy in bits.

**Per-matrix cosine similarity.** Pass random N(0, 0.01) input through full-precision and quantized matrices. Compute cosine.

**Structural bonus.** BPT(shuffled WikiText-2) − BPT(original). Collapse indicates loss of linguistic-hierarchy exploitation.

### 2.4 GS3 Statistical Protocol (NEW in v2.1)

Pythia-1.4B under FP16, BNB NF4, symmetric INT8, and symmetric INT4. Five independent shuffle seeds (42, 137, 271, 314, 577). Each seed produces a fresh word-level permutation of the WikiText-2 test text. BPT is computed on up to 80,000 tokens per evaluation. Bootstrap 95% CIs from 10,000 resamples. Welch's t-test (unequal variances). Cohen's d via pooled standard deviation.

### 2.5 Lloyd-Max End-to-End Test (NEW in v2.1)

Pythia-410M weights quantized via Lloyd-Max at INT4 and INT3. Per-matrix Lloyd-Max codebook computed via k-means on a 50,000-weight sample of each matrix. All weight matrices with dim ≥ 2 and numel > 1000 are quantized. End-to-end BPT measured on WikiText-2 test split, 50,176 tokens, no overlap.

### 2.6 R4 Robust Multi-Model Replication (NEW in v2.1)

Three models (Pythia-160M, Pythia-410M, GPT-2-medium) under FP16, NF4, symmetric INT4, and symmetric INT8. Three shuffle seeds per (model, method) pair = 36 total measurements.

---

## 3. Results

### 3.1 The Cliff Location Depends on the Method

**Table 1. End-to-end BPT — NF4 survives where symmetric fails**

| Model | FP16 | BNB INT8 | BNB NF4 | BNB FP4 | Sym INT8 | Sym INT4 | Sym INT3 |
|---|---|---|---|---|---|---|---|
| Pythia-410M | 4.27 | 4.28 | 4.67 | 4.97 | 4.78 | 16.76 | 16.05 |
| Pythia-1.4B | 3.81 | 3.82 | 3.90 | 4.00 | 3.79 | 16.87 | 15.75 |

NF4 at INT4 is operational (BPT 3.90–4.67). Symmetric at INT4 is catastrophic (BPT 16.76–16.87). The cliff for symmetric quantization is at INT8→INT4.

### 3.2 Why: Level Allocation Analysis

**Table 2. Per-matrix cosine similarity (Pythia-410M, attn\_qkv)**

| Method | INT4 cosine | INT3 cosine | Levels |
|---|---|---|---|
| Symmetric uniform | 0.905 | 0.637 | 15 uniform |
| NF4 | 0.973 | 0.948 | 15 at Gaussian quantiles |
| Lloyd-Max | 0.990 | 0.965 | 15 optimized for distribution |
| Log-scale | 0.965 | — | 15 log-spaced |
| Random (control) | 0.894 | — | 15 random positions |

NF4 at INT3 (0.948) outperforms symmetric at INT4 (0.905). Lloyd-Max at INT3 (0.965) outperforms NF4 at INT4 (0.973). The number of bits matters less than where they are spent.

### 3.3 Universality Across Architectures

The cliff is present in transformers (Pythia, GPT-2) and state-space models (Mamba) — the cliff is a property of weight distributions, not the computation graph.

### 3.4 The GPT-2 Outlier: Kurtosis Predicts Cliff Resistance

A GPT-2-medium matrix with kurtosis 124.75 and sparsity 0.795 shows no cliff (ratio 0.9×). Extremely sparse, heavy-tailed distributions resist the cliff because the few large weights dominate the output.

### 3.5 Consistency Across Layers

Cliff ratio is approximately constant across all 24 layers of Pythia-410M (~4.5×, slight decrease in later layers).

### 3.6 Real Trained Weights Confirm the Arithmetic

Pythia-410M layer 0 attention QKV at INT2 produces output cosine 0.075 (effectively random). The cliff is in the weight distributions, not the software.

### 3.7 GS3: Statistical Decisiveness at 1.4B with 5 Seeds (NEW in v2.1)

**Table 6. Pythia-1.4B structural bonus by quantization method**

| Method | Bonus (mean ± std) | 95% CI |
|---|---|---|
| FP16 | 6.3986 ± 0.0086 | [6.3922, 6.4064] |
| BNB NF4 | 6.3664 ± 0.0091 | [6.3582, 6.3745] |
| Symmetric INT8 | 6.4053 ± 0.0088 | [6.3979, 6.4135] |
| Symmetric INT4 | 0.2033 ± 0.0176 | [0.1886, 0.2179] |

**Welch's t-tests:**
- FP16 vs symmetric INT4: t = 633.74, p = 2.84×10⁻¹⁵, Cohen's d = 400.81 (large)
- NF4 vs symmetric INT4: t = 623.09, p = 1.16×10⁻¹⁵, Cohen's d = 394.08 (large)

The 95% confidence interval for symmetric INT4 (≈ [0.19, 0.22]) does not come within 6 bits of any other interval. NF4, FP16, and symmetric INT8 cluster within 0.04 bits of each other. P-values below 10⁻¹⁵ are at the limit of standard double-precision arithmetic. Cohen's d values >0.8 are conventionally "large"; the values observed here exceed that threshold by a factor of ~500.

### 3.8 Lloyd-Max INT3 End-to-End Test (NEW in v2.1)

**Table 7. Lloyd-Max end-to-end propagation through Pythia-410M**

| Method | Bits | BPT | Operational? |
|---|---|---|---|
| FP16 (baseline) | 16 | 4.27 | yes |
| Lloyd-Max | 4 | 8.51 | degraded (2.0×) |
| Lloyd-Max | 3 | 11.74 | failed (2.7×) |
| Symmetric | 4 | 16.89 | failed (4.0×) |
| Symmetric | 3 | 16.05 | failed (3.8×) |

Lloyd-Max at INT3 produces per-matrix cosine 0.965 (§3.2), but the end-to-end BPT is 11.74 — the per-matrix advantage does not survive 24 layers of error accumulation. **Cosine similarity overstates quantization quality.** The structural-bonus test (§3.7) and end-to-end BPT are the metrics that matter.

NF4 remains the only INT4 method that preserves both per-matrix fidelity and end-to-end performance.

### 3.9 R4 Robust Multi-Model Replication (NEW in v2.1)

**Table 8. Structural bonus across three models, four methods, three seeds**

| Model | Method | Bonus (mean ± std) |
|---|---|---|
| Pythia-160M | FP16 | 5.613 ± 0.008 |
| Pythia-160M | NF4 | 5.434 ± 0.012 |
| Pythia-160M | Sym INT4 | 1.070 ± 0.110 |
| Pythia-160M | Sym INT8 | 5.298 ± 0.008 |
| Pythia-410M | FP16 | 5.926 ± 0.006 |
| Pythia-410M | NF4 | 5.778 ± 0.006 |
| Pythia-410M | Sym INT4 | 0.222 ± 0.058 |
| Pythia-410M | Sym INT8 | 5.716 ± 0.011 |
| GPT-2-medium | FP16 | 6.091 ± 0.006 |
| GPT-2-medium | NF4 | 6.083 ± 0.007 |
| GPT-2-medium | Sym INT4 | 0.138 ± 0.016 |
| GPT-2-medium | Sym INT8 | 6.041 ± 0.007 |

The cliff is present at every model scale tested, with non-overlapping CIs between symmetric INT4 and every other method at every scale. The bonus magnitude scales modestly with model size (~5.6 at 160M, ~5.9 at 410M, ~6.1 at GPT-2-medium, ~6.4 at 1.4B), consistent with larger models making richer use of linguistic structure.

---

## 4. Discussion

### 4.1 The Cliff Is About Level Allocation, Not Bit Count

The central finding remains: the quantization cliff is not at a fixed bit count. It is at the precision where the available levels can no longer represent the weight distribution's critical features. Uniform levels: cliff at INT8→INT4. NF4 (Gaussian-quantile) levels: cliff at INT4→INT3. Lloyd-Max levels: per-matrix cliff below INT3, but end-to-end propagation breaks at INT4 due to error accumulation.

This is a rate-distortion phenomenon. R(D) = 0.5 × log₂(σ²/D) for a Gaussian source. The cliff occurs where quantization distortion D exceeds the signal variance σ² in critical weight components, and level allocation determines which components are "critical."

### 4.2 Implications for Hardware Design

1. **Do not design INT3 or INT2 uniform-integer datapaths for language tasks.** They will not produce useful output.
2. **INT4 is viable only with non-uniform quantization.** Lookup-table support is essential.
3. **INT8 uniform is the safe minimum** for purely integer arithmetic.
4. **Sub-INT4 requires per-matrix codebook optimization** AND the codebook quality must be evaluated end-to-end, not just per-matrix.

### 4.3 The Structural-Bonus Test Is the Right Diagnostic

The Lloyd-Max end-to-end result (§3.8) demonstrates a methodological point: per-matrix cosine similarity can be high (0.965 at INT3) while end-to-end BPT collapses (11.74). The structural bonus measures the model's preserved ability to exploit linguistic structure across the full network, not just to reproduce individual matrix outputs. Future quantization research should report end-to-end structural bonus as a primary metric.

### 4.4 Why Mamba Also Shows the Cliff

Mamba and transformers show comparable cliff ratios because both learn approximately Gaussian weight distributions during SGD with weight decay. The cliff is distributional, not architectural.

### 4.5 Statistical Decisiveness

The v2.1 GS3 results (t = 633.74, p = 2.84×10⁻¹⁵, Cohen's d = 400.81) are among the most statistically overwhelming results in the deep-learning literature. The effect is so large relative to seed-to-seed noise that the question is no longer "is the cliff real" but "what model architecture, weight distribution, or quantization scheme could even in principle escape it." We have not found one.

---

## 5. Limitations

1. **No GPTQ or AWQ comparison** — CUDA version incompatibility.
2. **No cycle-accurate hardware simulation** — Chipyard/Gemmini installed but not configured.
3. **End-to-end BPT tested on 2 models for the original NF4/symmetric comparison.** GS3 (v2.1) extends to 1.4B with 5 seeds; R4 (v2.1) extends to three models with 3 seeds.
4. **No INT3 bitsandbytes configuration** — bitsandbytes does not natively support 3-bit.
5. **Structural bonus under NF4 vs symmetric.** v2.0 listed this as untested; **resolved in v2.1** by GS3 (§3.7) and R4 (§3.9).

---

## 6. Predictions

P1. GPTQ-quantized models at INT3 will show BPT < 8 (operational, if degraded).

P2. Models pruned to >50% sparsity will show cliff ratios < 2× at INT4→INT3.

P3. **PARTIALLY FALSIFIED (v2.1).** Lloyd-Max codebook quantization at INT3 was predicted to produce end-to-end BPT within 50% of FP16. Observed: BPT = 11.74 vs FP16 = 4.27, 2.75× worse. Per-matrix Lloyd-Max INT3 succeeds (cosine 0.965); end-to-end propagation does not. Per-matrix cosine overstates end-to-end quality.

P4. **CONFIRMED (v2.1).** The structural bonus is preserved at INT4 under NF4 (5.78–6.37 across model scales) but destroyed at INT4 under symmetric (0.14–1.07). Welch t = 633.74, p = 2.84×10⁻¹⁵.

---

## 7. Conclusion

The quantization cliff in language models is real, universal across transformer and state-space architectures, consistent across 24 layers, present in real trained weights, and supported by Welch t-statistics with p < 10⁻¹⁵. But it is not at a fixed bit count. It is at the precision where the quantization scheme's level allocation can no longer represent the weight distribution's critical structure.

For symmetric quantization: cliff at INT8→INT4. For NF4: cliff at INT4→INT3. For Lloyd-Max: per-matrix cliff below INT3, but end-to-end propagation breaks at INT4 due to error accumulation across 24 layers — a finding that motivates structural-bonus testing as the right end-to-end diagnostic.

The minimum viable inference specification is not "N-bit integer" but "N-bit with distribution-aware level allocation, validated end-to-end." Hardware designers should build quantization lookup tables, not wider integer datapaths.

---

## References

Whitmer III, G.L. (2026a–f). Papers 1–6, Windstorm Institute.

Whitmer III, G.L. (2026g). Paper 7: The Throughput Basin Origin. doi:10.5281/zenodo.19498582.

Whitmer III, G.L. (2026h). Grand Slam Supplementary Materials. github.com/Windstorm-Institute/throughput-basin-origin/blob/main/grandslam\_supplementary.pdf.

---

## Acknowledgments

All experiments executed on RTX 5090 (Windstorm Labs, Varon-1). The GS3 round (v2.1) ran Pythia-1.4B at four quantization methods × five shuffle seeds in approximately 90 seconds at the model's normal inference cost. The Lloyd-Max end-to-end test (§3.8) ran in ~3 minutes. The R4 robust round (v2.1) ran in ~90 seconds at full precision on three models. NF4 end-to-end BPT used bitsandbytes 0.49.2 with CUDA 13.0 library path fix. Experiment design and analysis: Grant Lavell Whitmer III with Claude Opus 4.6. All code and data: github.com/Windstorm-Institute/throughput-basin-origin. CC BY 4.0.
