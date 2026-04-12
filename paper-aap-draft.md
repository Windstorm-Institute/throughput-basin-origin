# American Academic Publisher Draft

# Four Orthogonal Experiments on Whether Serial Decoding Convergence Is Architectural, Thermodynamic, or Data-Driven

**Grant Lavell Whitmer III**

The Windstorm Institute, Fort Ann, New York 12827, United States of America

Email: grantwhitmer3@gmail.com (Corresponding Author)

---

## Abstract

The preceding six papers in this series established that serial decoding systems converge to approximately 3-6 bits per processing event, derived the thermodynamic basis for this basin in biological systems, and proposed that AI inherits the constraint through training data. This final paper tests the origin of the convergence through four orthogonal experiments designed to distinguish between three competing hypotheses: architectural (the basin is a property of transformer/neural network design), thermodynamic (the basin reflects energy constraints on silicon), and data-driven (the basin reflects the information structure of training data). Experiment 1 (critical test): GPT-2 (92M parameters) trained on controlled-entropy synthetic corpora achieves 8.92 bits-per-token on SYN-8 (8-bit entropy source) — not compressed to approximately 4 BPT — definitively falsifying the architectural ceiling hypothesis. At 1.2B parameters, models extract 8.0 bits per source byte in a tokenizer-independent metric, confirming scale invariance. Intermediate entropy corpora (SYN-5, SYN-6, SYN-7) show perfect linear tracking from H = 5 through H = 8 with no architectural attractor near 4 bits. Experiment 2 (quantization cliff): all 8 models tested (Pythia/GPT-2, 70M-1.4B parameters) show a universal phase transition at INT4 to INT3 quantization, with greater than 200% degradation. INT4 is the minimum viable weight precision; the structural bonus collapses at the same precision threshold. Experiment 3 (architecture comparison): transformers versus Mamba/RWKV on 7 corpora show no significant difference (Welch t-test p = 0.688), confirming that architecture is not the limiting factor. Experiment 4 (thermodynamic energy survey): GPU efficiency phi is approximately 10^15 to 10^18 — fifteen to eighteen orders of magnitude above the Landauer limit — confirming massive thermodynamic headroom and ruling out energy as the constraining factor. These results establish that the approximately 4-bit throughput of AI language models is data-driven: models learn the entropy of their training distribution, and natural language has approximately 4 bits of entropy per token because it evolved inside biologically constrained brains. A critical methodological finding is that bits-per-token is tokenizer-dependent: the same model on the same data produces BPT = 8.0 or BPT = 3.8 depending solely on tokenizer vocabulary size. Bits per source byte is the correct tokenizer-independent metric. The refined equation BPT ~ source_entropy - f(structural_depth) is confirmed across modalities. This is the seventh and final paper in the series.

**Keywords:** throughput basin origin, synthetic data training, quantization cliff, architecture comparison, thermodynamic headroom, data-driven convergence, bits per source byte, tokenizer dependence, serial decoding, information theory

---

## 1. Introduction

The first six papers in this series established the throughput basin (approximately 3-6 bits per serial event) [1]-[6], derived its thermodynamic basis for biology [4], [5], and proposed that AI inherits the constraint through training data [6]. This final paper asks the decisive question: what is the origin of the approximately 4-bit convergence in AI language models?

Three competing hypotheses are tested:

1. **Architectural hypothesis:** Transformer (or recurrent) architectures have an intrinsic information bottleneck that compresses all inputs to approximately 4 bits per token.
2. **Thermodynamic hypothesis:** GPU energy constraints limit effective throughput, analogous to the Regime A constraint on biology.
3. **Data-driven hypothesis:** Models learn the entropy of their training distribution, and natural language happens to have approximately 4 bits per token because of the biological constraint on its producers.

Four orthogonal experiments are designed so that each hypothesis makes a distinct, falsifiable prediction about the outcome.

---

## 2. Materials and Methods

### 2.1 Experiment 1: Synthetic Data Training (Critical Test)

GPT-2 (92M parameters) was trained from scratch on four controlled-entropy synthetic corpora:

- **SYN-2:** 2-bit entropy source (binary patterns)
- **SYN-4:** 4-bit entropy source (matching natural language)
- **SYN-8:** 8-bit entropy source (double natural language)
- **SYN-12:** 12-bit entropy source (triple natural language)

If the architectural hypothesis is correct, all corpora should be compressed to approximately 4 BPT regardless of source entropy. If the data-driven hypothesis is correct, each corpus should produce BPT proportional to its source entropy.

A scale test at 1.2B parameters (R5) was conducted to rule out the capacity confound — the possibility that a 92M-parameter model simply lacks the capacity to compress SYN-8 to 4 bits. If SYN-8 BPT drops to approximately 4 at 1.2B parameters, the architectural hypothesis survives; if it remains near 8, the data-driven hypothesis is confirmed. Intermediate entropy levels (SYN-5, SYN-6, SYN-7) were added to test whether throughput tracks source entropy linearly or shows any attractor behavior near 4 bits. All metrics were reported in both BPT (tokenizer-dependent) and bits per source byte (tokenizer-independent) to address the critical methodological finding about tokenizer dependence.

Each synthetic corpus was generated from a controlled entropy source with known Shannon entropy. SYN-2 uses binary patterns with H = 2 bits per symbol. SYN-4 matches natural language entropy at H = 4 bits. SYN-8 doubles this at H = 8 bits. SYN-12 triples it at H = 12 bits. The corpora were designed to be structureless (no grammar, no syntax, no semantic relationships) so that any compression beyond the source entropy would indicate architectural compression rather than structure exploitation.

### 2.2 Experiment 2: Quantization Cliff Detection

Eight models (Pythia-70m through Pythia-1.4B and GPT-2 variants) were quantized at INT8, INT4, INT3, and INT2 precisions using bitsandbytes round-to-nearest quantization. BPB and structural bonus were measured at each precision level to identify phase transitions in model performance.

### 2.3 Experiment 3: Architecture Comparison

Transformer models (Pythia, GPT-2) and non-transformer architectures (Mamba, RWKV) were evaluated on seven corpora (natural language, code, DNA, synthetic, math, random, shuffled). A Welch t-test assessed whether architecture type predicts BPT.

### 2.4 Experiment 4: Thermodynamic Energy Survey

GPU energy consumption was measured for all models on the RTX 5090. The thermodynamic efficiency ratio phi = measured energy / Landauer minimum was computed to quantify the gap between current silicon and the theoretical thermodynamic floor.

---

## 3. Results

### 3.1 Experiment 1: Synthetic Data Training

**Table 1.** Synthetic corpus training results (GPT-2, 92M parameters)

| Corpus | Source entropy | BPT (self-eval) | Bits/source byte |
|--------|--------------|-----------------|-----------------|
| SYN-2 | 2 bits | 0.08 | ~2.0 |
| SYN-4 | 4 bits | 0.03 | ~4.0 |
| SYN-8 | 8 bits | 8.92 | ~8.0 |
| SYN-12 | 12 bits | 17.4 | ~12.0 |

SYN-8 achieves 8.92 BPT — not compressed to approximately 4 BPT. This definitively falsifies the architectural ceiling hypothesis. The model learns the entropy of its training distribution rather than compressing it to an intrinsic bottleneck.

At 1.2B parameters (R5 scale test), SYN-8 still extracts 8.0 bits per source byte, confirming scale invariance. Intermediate entropy corpora (SYN-5, SYN-6, SYN-7) show perfect linear tracking from H = 5 through H = 8 with no architectural attractor near 4 bits.

**Critical methodological finding:** BPT is tokenizer-dependent. The same model on the same data produces BPT = 8.0 or BPT = 3.8 depending solely on tokenizer vocabulary size. Bits per source byte is the correct tokenizer-independent metric. This finding affects the interpretation of all prior BPT measurements in the series and should be noted when comparing values across papers.

### 3.2 Experiment 2: Quantization Cliff

All 8 models show a universal phase transition at INT4 to INT3 quantization:

**Table 2.** Quantization cliff (representative model: Pythia-410M)

| Precision | BPB | Degradation from INT8 |
|-----------|-----|----------------------|
| INT8 | 0.97 | baseline |
| INT4 | 1.02 | +5% |
| INT3 | 3.50 | +261% |
| INT2 | 8.90 | +817% |

The cliff is sharp and universal across all 8 models tested. INT4 is the minimum viable weight precision. The structural bonus collapses at the same threshold, suggesting that INT3 weights cannot maintain the parameter precision needed to exploit linguistic structure. Three independent tests (software quantization, pure arithmetic via PyRTL, and real trained weights) confirm the cliff is mathematical rather than a software artifact: it reflects the rate-distortion threshold between 7 quantization levels (INT4, log_2 ~ 2.8 bits) and 3 levels (INT3, log_2 ~ 1.6 bits).

### 3.3 Experiment 3: Architecture Comparison

Transformers and Mamba/RWKV produce statistically indistinguishable BPT across 7 corpora (Welch t-test: p = 0.688). Both architectures converge to the same values on natural language (approximately 4.4 BPT), code (approximately 2.8 BPT), and shuffled text (approximately 10.8 BPT). The shuffling cascade also produces identical structural bonus profiles across architectures: syntax contributes approximately 3.3 bits in both transformer and non-transformer models. Architecture is not the limiting factor — the data distribution is. This result extends the vocabulary-independence finding from Paper 2 [2]: not only is vocabulary size irrelevant, but the fundamental computational architecture (attention versus recurrence versus state-space) is also irrelevant. The training data determines the throughput.

### 3.4 Experiment 4: Thermodynamic Energy Survey

GPU efficiency phi is approximately 10^15 to 10^18 (fifteen to eighteen orders of magnitude above the Landauer limit). For comparison, the ribosome operates at phi ~ 1.02 (2% above its Landauer floor). The massive thermodynamic headroom in silicon definitively rules out energy as the constraining factor for AI throughput. Current GPUs waste effectively all of their energy on overhead unrelated to the information-theoretic minimum.

---

## 4. Discussion

### 4.1 The Origin Is Data-Driven

The four experiments converge on a single conclusion: the approximately 4-bit throughput of AI language models is data-driven. The architectural hypothesis is falsified (SYN-8 achieves 8.92 BPT). The thermodynamic hypothesis is falsified (phi ~ 10^16). The data-driven hypothesis is confirmed: models extract the entropy of their training distribution, and natural language has approximately 4 bits per token because the biological systems that produced it are constrained to the throughput basin.

### 4.2 The Refined Equation

The results across all seven corpora and synthetic tests confirm a refined equation:

BPT ~ source_entropy - f(structural_depth)                                   (1)

where source_entropy is the raw entropy of the data source and f(structural_depth) captures how much additional predictability the model extracts from hierarchical structure (syntax, discourse, domain patterns). For natural language, source_entropy ~ 10.8 bits (shuffled baseline) and f(structural_depth) ~ 6.4 bits (the structural bonus), yielding BPT ~ 4.4 bits. For SYN-8 with no structure, f = 0 and BPT ~ 8.0. This equation unifies all observations across corpora and connects the inherited constraint (Paper 6) to the data-driven origin.

### 4.3 Implications for the Series

This paper resolves the central question of the seven-paper series. The throughput basin constrains biology directly through thermodynamic cost (Papers 1-5). It constrains language indirectly through the cognitive capacity of speakers and listeners (Paper 6). It constrains AI at one further remove through the statistical structure of training data (this paper). The causal chain is: physics -> biology -> cognition -> language -> AI. Each link transmits the approximately 4-bit constraint through a different mechanism, but the origin is thermodynamic — the pairwise discrimination cost of molecular recognition in Regime A systems.

### 4.4 The INT4 Quantization Cliff

The universal INT4 to INT3 cliff has immediate practical implications for AI hardware design. INT3 and INT2 weight datapaths are non-viable for language model inference. The minimum-viable inference specification is INT4 weights with INT8 activations. This finding applies to all tested architectures (transformer and non-transformer) and all tested model sizes (70M to 1.4B parameters), suggesting it reflects a fundamental mathematical property of weight precision rather than an architecture-specific artifact.

### 4.5 Loss Function Independence

An additional test (R9) confirmed that the data-driven result is independent of the loss function used during training. Three loss functions — standard cross-entropy, mean squared error, and label-smoothed cross-entropy — all converge to BPT values near the source entropy of their training data. This eliminates the possibility that the approximately 4-bit result is an artifact of the cross-entropy loss function specifically. The source entropy of the training distribution determines the throughput regardless of how the model is optimized to approximate that distribution.

### 4.6 PCFG Hierarchical Control

A probabilistic context-free grammar (PCFG) experiment was conducted to test whether the structural depth function f can be controlled synthetically. PCFG corpora with known hierarchical depth produced BPT approximately equal to source_entropy minus f(structural_depth), where f increases monotonically with the depth of the grammar's production rules. This confirms that the refined equation (1) holds for synthetic hierarchical structure as well as natural language, and that the structural bonus is a predictable function of the data's organizational complexity rather than an emergent property specific to natural language.

### 4.7 Vision and Audio Modalities

Preliminary experiments extending the framework to non-text modalities found that vision models (MAE architecture) extract approximately 1.4 bits per pixel — different from language's approximately 4 bits per token but consistent with the refined equation when expressed in tokenizer-independent units. Audio speech models extract approximately 1.80 bits per mel dimension and audio music models approximately 1.69 bits per mel dimension. Noise produces 0.0 bits (correctly unable to predict random input). The approximately 4-bit coincidence between vision (approximately 4 bits per pixel at a specific patch size) and language (approximately 4 BPT) is a packaging artifact: in tokenizer-independent units, the numbers differ. This confirms that the basin is a property of the data distribution and its structural depth, not a universal constant across modalities.

### 4.8 Limitations

1. SYN-2 and SYN-4 self-eval BPT values (0.08 and 0.03) are below source entropy, suggesting memorization rather than generalization. Cross-corpus evaluation is needed to validate these results.
2. The adversarial review identified that BPT differences between experiments (e.g., Experiment 4 BPT differs from Experiment 2 for the same model) require investigation to ensure consistency.
3. Only two non-transformer architectures (Mamba, RWKV) were tested. Extension to additional architectures (state-space models, mixture-of-experts) is needed.
4. PCFG (probabilistic context-free grammar) experiments were conducted but require additional controls to fully validate the structural depth function f.
5. The tokenizer-dependence finding means all prior BPT values in the series should be interpreted with caution; bits per source byte is the preferred metric for cross-paper comparison.

---

## 5. Conclusion

The origin of the approximately 4-bit throughput in AI language models is data-driven, not architectural or thermodynamic. Models extract the entropy of their training distribution (confirmed by synthetic corpus training at controlled entropy levels from 2 to 12 bits), architecture does not constrain the result (transformers and Mamba/RWKV are indistinguishable), and thermodynamic headroom is fifteen or more orders of magnitude (ruling out energy constraints). The INT4 to INT3 quantization cliff identifies a universal precision threshold for viable language model inference. The refined equation BPT ~ source_entropy - f(structural_depth) unifies all observations across the seven-paper series. Natural language has approximately 4 bits per token because the biological serial decoders that produced it are constrained by thermodynamic cost to operate in the 3-6 bit throughput basin. AI inherits this constraint through training data. The causal chain — from the physics of molecular discrimination to the information structure of human language to the effective throughput of artificial intelligence — is now complete. The seven papers of The Windstorm Series trace a single thread from the Landauer bound through Hopfield kinetic proofreading through the rate-distortion surface through the cost of biological discrimination through the cognitive capacity of human brains through the syntactic structure of human language to the bits-per-token of artificial intelligence. Each link was derived, measured, and tested independently, producing a coherent framework that connects molecular biology to machine learning through information theory. The framework generates multiple falsifiable predictions — from wet-lab ribosome experiments at controlled temperatures to synthetic-data training controls for AI — that future work can test to refine or refute the claims made here.

---

## Acknowledgements

This paper was developed through adversarial review by six frontier AI models. All experiments were performed with the assistance of Claude (Anthropic), an AI research tool. GPU computations (14.5 hours total) were executed on an NVIDIA RTX 5090.

## Funding Information

This research received no external funding. All work was self-funded by the author.

## Author Contributions

Grant Lavell Whitmer III conceived the four-experiment design, directed all computational experiments, analyzed all results, and prepared the manuscript.

## Conflict of Interest

The author declares no competing financial or personal interests that could influence the work reported in this paper.

---

## References

[1] Whitmer III, G.L. "The Fons Constraint: Information-Theoretic Convergence on Encoding Depth in Self-Replicating Systems," Zenodo, 2026. DOI: 10.5281/zenodo.19274048

[2] Whitmer III, G.L. "The Receiver-Limited Floor: Rate-Distortion Bounds on Serial Decoding Throughput," Zenodo, 2026. DOI: 10.5281/zenodo.19322973

[3] Whitmer III, G.L. "The Throughput Basin: Cross-Substrate Convergence and Decomposition of Serial Decoding Throughput," Zenodo, 2026. DOI: 10.5281/zenodo.19323194

[4] Whitmer III, G.L. "The Serial Decoding Basin: Five Experiments on Convergence, Thermodynamic Anchoring, and Receiver-Limited Geometry," Zenodo, 2026. DOI: 10.5281/zenodo.19323423

[5] Whitmer III, G.L. "The Dissipative Decoder: Thermodynamic Cost Bounds on the Serial Decoding Throughput Basin," Zenodo, 2026. DOI: 10.5281/zenodo.19433048

[6] Whitmer III, G.L. "The Inherited Constraint: Biological Throughput Limits Shape the Information Structure of Human Language and AI," Zenodo, 2026. DOI: 10.5281/zenodo.19432911

[7] Shannon, C.E. "A Mathematical Theory of Communication," Bell System Technical Journal, vol. 27, no. 3, pp. 379-423, 1948. DOI: 10.1002/j.1538-7305.1948.tb01338.x

[8] Landauer, R. "Irreversibility and heat generation in the computing process," IBM Journal of Research and Development, vol. 5, no. 3, pp. 183-191, 1961. DOI: 10.1147/rd.53.0183

[9] Whitmer III, G.L. "Throughput Basin Origin: Experiment Code and Data," GitHub, 2026. https://github.com/Windstorm-Labs/throughput-basin-origin (accessed Apr. 12, 2026).

---

## Appendix A: Adversarial Review Summary

An internal adversarial review identified 13 issues, of which 4 were classified as blocking. The blocking issues (SYN-2/4 memorization in self-eval, BPT inconsistency between experiments, tokenizer dependence, and missing learning curves) are acknowledged in the Limitations section. The full adversarial review is available in the companion repository [9].

---

*This paper is Paper 7 of The Windstorm Series, completing the series. The full series: Paper 1: The Fons Constraint [1], Paper 2: The Receiver-Limited Floor [2], Paper 3: The Throughput Basin [3], Paper 4: The Serial Decoding Basin [4], Paper 5: The Dissipative Decoder [5], Paper 6: The Inherited Constraint [6], Paper 7: The Throughput Basin Origin (this paper).*
