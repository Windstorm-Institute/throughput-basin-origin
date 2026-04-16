# GS3: Structural Bonus — Pythia-1.4B at Scale

| Method | Bonus (mean ± std) | 95% CI | N |
|---|---|---|---|
| FP16 | 6.3986 ± 0.0086 | [6.3922, 6.4064] | 5 |
| SYM_INT4 | 0.2033 ± 0.0176 | [0.1886, 0.2179] | 5 |
| SYM_INT8 | 6.4053 ± 0.0088 | [6.3979, 6.4135] | 5 |
| BNB_NF4 | 6.3664 ± 0.0091 | [6.3582, 6.3745] | 5 |

## Statistical Tests (Welch's t-test)

**FP16 vs SYM_INT4:** t=633.74, p=2.84e-15, Cohen's d=400.81
  Effect size: large

**NF4 vs SYM_INT4:** t=623.09, p=1.16e-15, Cohen's d=394.08
  Effect size: large

