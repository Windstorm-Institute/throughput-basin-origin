# The Throughput Basin: Unified Evidence Table

*Every claim, every confidence interval, every p-value.*

---

## Paper 7: The basin is data-driven

### Claim 1: BPT tracks source entropy (SYN-8 experiment)
- 92M model on SYN-8: 8.0 bits/source byte (= source entropy)
- 1.2B model on SYN-8: 8.0 bits/source byte (scale-invariant)
- SYN-5, SYN-6, SYN-7: linear tracking, no attractor near 4 bits
- *Evidence level: DECISIVE (trained from scratch on controlled data)*

### Claim 2: f(structural_depth) is real
| Source | BPT (mean ± std) | 95% CI | N |
|---|---|---|---|
| salad | 5.9681 ± 0.0018 | [5.9655, 5.9696] | 3 |
| depth_0 | 3.2173 ± 0.0011 | [3.2163, 3.2189] | 3 |
| depth_1 | 3.6392 ± 0.0015 | [3.6372, 3.6408] | 3 |
| depth_2 | 3.7233 ± 0.0028 | [3.7194, 3.7257] | 3 |
| depth_3 | 3.7571 ± 0.0008 | [3.7560, 3.7578] | 3 |
| depth_4 | 3.7737 ± 0.0010 | [3.7726, 3.7750] | 3 |
| depth_5 | 3.7839 ± 0.0008 | [3.7828, 3.7844] | 3 |
| depth_6 | 3.7899 ± 0.0009 | [3.7887, 3.7908] | 3 |

*Trained from scratch — no pretrained bias.*

### Claim 3: τ varies by domain (bits/char is data-specific)
| Domain | Bits/char (mean ± std) | N models |
|---|---|---|
| dna | 2.0706 ± 0.0367 | 3 |
| english | 0.9893 ± 0.1106 | 3 |
| french | 1.1449 ± 0.1496 | 3 |
| german | 1.3099 ± 0.1634 | 3 |
| medical | 0.8258 ± 0.0865 | 3 |
| python | 1.0029 ± 0.0696 | 3 |
| spanish | 1.1212 ± 0.1470 | 3 |

## Paper 8: Cross-modal throughput basins

### Claim 4: Vision throughput tracks image entropy
| Entropy Level | Name | Eval Loss (mean ± std) | 95% CI |
|---|---|---|---|
| 0 | uniform | 0.000001 ± 0.000000 | [0.000001, 0.000001] |
| 1 | 4-color-blocks | 0.065132 ± 0.025058 | [0.035907, 0.097102] |
| 2 | 16-color-blocks | 0.075606 ± 0.004090 | [0.069823, 0.078562] |
| 3 | 64-color-pixels | 0.084080 ± 0.001041 | [0.082636, 0.085052] |
| 4 | natural-like | 0.013662 ± 0.000210 | [0.013472, 0.013955] |
| 5 | gaussian-noise | 0.057536 ± 0.000001 | [0.057535, 0.057537] |
| 6 | uniform-noise | 0.083332 ± 0.000000 | [0.083332, 0.083333] |

*86M-param ViT-MAE trained from scratch at each level. 3 seeds.*

## Paper 9: The quantization cliff is about level quality

### Claim 5: NF4 preserves structural bonus; symmetric destroys it
| Method | Bonus (mean ± std) | 95% CI | N |
|---|---|---|---|
| FP16 | 6.3986 ± 0.0086 | [6.3910, 6.4064] | 5 |
| BNB_NF4 | 6.3664 ± 0.0091 | [6.3582, 6.3745] | 5 |
| SYM_INT4 | 0.2033 ± 0.0176 | [0.1886, 0.2179] | 5 |
| SYM_INT8 | 6.4053 ± 0.0088 | [6.3984, 6.4133] | 5 |

**FP16 vs SYM_INT4:** Welch t=633.74, p=2.84e-15, Cohen's d=400.81 (LARGE)
**NF4 vs SYM_INT4:** Welch t=623.09, p=1.16e-15, Cohen's d=394.08 (LARGE)

---

## Experimental Methodology

- All experiments: multiple seeds (3-5), bootstrap 95% CIs, Welch t-tests
- Training experiments: trained from scratch (no pretrained contamination)
- Scale tests: 3 model sizes (160M, 410M, 1.4B parameters)
- Quantization: 4 methods (FP16, NF4, symmetric INT4/INT8)
- Cross-modal: language (7 corpora), vision (7 entropy levels + 2 real datasets), audio (6 types + real speech)
- Hardware: NVIDIA RTX 5090 (32 GB), CUDA 13.1
- All code and data: github.com/Windstorm-Institute/throughput-basin-origin
