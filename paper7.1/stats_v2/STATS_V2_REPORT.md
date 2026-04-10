# Paper 7.1 — Statistics v2 Report

## Summary

Paper 7's headline statistical claims rest almost entirely on point estimates
without uncertainty quantification, on a Welch t-test (p=0.688) that has no
power to either reject or establish equivalence, and on per-corpus tests that
treat repeated measures as independent. Stats v2 re-analyses every reported
quantity from the existing CSVs (no models re-run, no GPU touched) with
bootstrap confidence intervals, a pre-specified TOST equivalence test,
mixed-effects / paired analysis of the 7-corpus battery, a power analysis,
and Holm/BH multiple-comparison corrections. The good news: the INT4 cliff
and structural-bonus claims survive intact, and one new finding emerges
(the paired-by-corpus test reveals a small but statistically detectable
transformer−serial gap that the naive Welch on n=4 vs 3 missed). The bad
news: Paper 7's "no architecture-specific bias" framing is unsupported in
either direction by the existing data and must be rewritten.

---

## 1. Bootstrap confidence intervals
File: `bootstrap_cis.csv` (10,000 percentile resamples).

Key results:
- WikiText transformer mean BPT = 3.495, 95% CI [3.15, 3.82] (n=4 models)
- WikiText serial mean BPT       = 3.346, 95% CI [2.89, 3.84] (n=3 models)
- WikiText diff (T−S)            = 0.149, 95% CI [−0.41, 0.70] — straddles zero
- Paired-by-corpus diff (T−S)    = 0.140, 95% CI [+0.05, +0.22] — **excludes zero**
- Mean structural bonus (FP16, n=8) = 6.864, 95% CI [6.75, 6.97]
- Mean cliff ratio int3/int4 (n=8)  = 7.01,  95% CI [4.89, 9.50]
- Mean log10(φ) (exp-6, n=31)       = 17.04, 95% CI [16.81, 17.27]

All SYN-* self-eval and 4×4 cross-corpus cells are flagged
**single-seed, CI not estimable** rather than fabricating uncertainty.

Interpretation: the *naive* WikiText-only comparison is genuinely
inconclusive (CI ±0.55 BPT), but pooling across corpora reveals a
small consistent transformer-disadvantage of ~0.14 BPT — see §3.

## 2. TOST equivalence test
File: `tost_results.md`.

With pre-specified margin ±0.5 BPT (defended as ½ pooled SD), TOST returns
p = 0.183 → **cannot conclude equivalence**. The minimum equivalence
margin this design could have detected at 80% power is roughly ±2.6 BPT,
which is scientifically meaningless. **The data are insufficient to
support either an equivalence or a difference claim from the WikiText
slice alone.**

## 3. Mixed-effects / paired-by-corpus analysis
File: `mixed_effects.txt`.

The 7-corpus battery is a repeated-measures design (each model on every
corpus). MixedLM `bpt ~ arch_bin + (1|model) + (1|corpus)` runs but the
boundary-of-parameter-space convergence warning means the variance
components are not trustworthy on n=49 with small group sizes. The
robust paired-by-corpus fallback (one-sample t on the 7 paired corpus
differences):
- mean(transformer − serial) = +0.140 BPT
- t(6) = 2.83, **p = 0.029**

This is a *new* finding that the underpowered Welch in
`exp3_statistics.csv` (p = 0.688) missed. After accounting for repeated
measures across corpora, transformers in this panel are slightly *worse*
on average (higher BPT) than the size-matched serial models. The effect
is small (~0.14 BPT, ~4% relative on WikiText) but consistent across
corpora — every single corpus had transformer ≥ serial in mean BPT.

Note: this is still only 4 vs 3 models. The paired test is more powerful
than the Welch because it cancels per-corpus difficulty, not because it
invents data. It should be reported as suggestive, not definitive.

## 4. Power analysis
Files: `power_curves.png`, `power_curves.csv`.

- Current design (n=4 vs 3): MDE at 80% power ≈ Cohen's d = **2.05**
- Observed d = 0.397
- Sample size needed to detect d = 0.397 at 80% power: **≈ 101 models per group**

Paper 7's "p=0.688 → no difference" claim is therefore exactly the kind
of underpowered null that absence-of-evidence-evidence-of-absence
guidance warns against.

## 5. Holm-Bonferroni / BH on the 7 per-corpus tests
File: `holm_corrected_corpus_tests.csv`.

None of the 7 raw per-corpus Welch tests survives at α=0.05 even before
correction (smallest raw p = 0.262, csv corpus). After Holm all
adjusted p = 1.0; after BH FDR all adjusted p ≥ 0.74. Per-corpus
testing offers no support for arch differences; the *paired* test in §3
does, by exploiting the corpus-as-block structure rather than testing
each corpus in isolation.

## 6. Cliff ratio CIs
File: `cliff_ratio_cis.csv`.

Per-model cliff ratios (int3/int4 BPT) range 2.96 (gpt2-large) to 13.87
(pythia-70m); all are reported as point estimates with the explicit
"single measurement, no within-cell uncertainty" flag. The
across-models population mean cliff ratio is **7.01, 95% CI
[4.89, 9.50]**, computed by bootstrapping the 8-model population.
Crucially, every model's cliff ratio is well above 1.0; the 8/8
universality claim survives.

## 7. Correlation replay
File: `correlation_cis.csv` (cross-corpus, n=16 cells, bootstrap r CIs).

| Pair | r | p | 95% CI on r |
|---|---|---|---|
| train_H vs BPT     | −0.100 | 0.713 | [−0.62, +0.39] |
| target_H vs BPT    | +0.115 | 0.673 | [−0.42, +0.57] |
| Δ_H vs BPT         | +0.152 | 0.575 | [−0.27, +0.54] |
| **|Δ_H| vs BPT**   | **+0.686** | **0.003** | **[+0.32, +0.88]** |

The first three correlations the deep-analysis report quoted as evidence
of "no relationship" have CIs that span the entire interpretable range —
they are non-findings, not null findings. The *unsigned* entropy gap
|Δ_H|, however, has a CI that excludes zero by a wide margin and is the
correlation that should anchor the discussion: train↔target entropy
*mismatch in either direction* is a strong predictor of cross-corpus BPT
inflation, which is the actual story the cross-corpus matrix tells.

---

## What we can and cannot say

| Claim | Original support | Support after stats v2 | Verdict |
|---|---|---|---|
| Transformer ≈ serial (architecture-agnostic) | Welch p=0.688 on n=4 vs 3, WikiText only | TOST p=0.183 (cannot conclude eq.); paired-by-corpus shows T is **worse** by 0.14 BPT, p=0.029 | **WEAKENED — must be rewritten** |
| INT4 cliff is universal (8/8) | 8/8 models qualitatively | 8/8 models, mean cliff ratio = 7.01 [4.89, 9.50]; every per-model ratio ≫ 1 | **HOLD / strengthened** |
| Structural bonus ≈ 6.8 across panel | Point estimate | Mean = 6.86 [6.75, 6.97] | **HOLD / strengthened** |
| log10(φ) ≈ 17 | Point estimate | Mean = 17.04 [16.81, 17.27] | **HOLD / strengthened** |
| Per-corpus T vs S differences | 7 raw Welch tests | All raw p ≥ 0.26; all Holm p = 1.0; all BH p ≥ 0.74 | HOLD as null |
| SYN-8 self-eval = 8.92 BPT | Single seed | Single seed, no CI possible | **NEEDS REPLICATION (multi-seed)** |
| All 4×4 cross-corpus BPTs | Single seed each | Single seed, no CI possible | **NEEDS REPLICATION** |
| "train_H, target_H, Δ_H uncorrelated with BPT" | r close to 0 | r CIs span [-0.6, +0.6]; non-findings, not nulls | DOWNGRADED |
| **|Δ_H| predicts cross-corpus BPT** | not reported | r=0.686, 95% CI [+0.32, +0.88], p=0.003 | **NEW FINDING — promote to main text** |

## What additional data would resolve the remaining uncertainty

1. **Multiple seeds for the synthetic-corpus self-eval and 4×4 cross-corpus
   matrix** (Exp 1). At minimum 5 seeds per cell would let us put real CIs
   on the SYN-8 = 8.92 BPT and the diagonal/off-diagonal comparison.
2. **More architectures in Exp 3.** Honestly comparing transformer to
   serial requires roughly 100 models per group to detect the observed
   effect at d=0.4. A practical compromise: pre-register a one-sided
   hypothesis (transformer worse than serial by some margin) on a panel
   of ~10–15 each, motivated by the §3 paired result.
3. **A second seed per model on the 7-corpus battery** would let MixedLM
   actually estimate model-level variance instead of hitting the
   parameter boundary.

## Required edits to Paper 7 §5.2 (adversarial review section)

The current §5.2 should be updated as follows:

> **Architecture comparison.** The original Welch t-test (t=0.43, p=0.688)
> on the WikiText slice was severely underpowered: with n=4 transformers
> and n=3 serial models the minimum detectable Cohen's d at 80% power is
> ~2.0, far above the observed d=0.40. A pre-specified TOST equivalence
> test at margin ±0.5 BPT also fails (p=0.18), so the data are likewise
> insufficient to *establish* equivalence. We therefore retract the
> framing "no architecture-specific bias was observed" and replace it
> with **"no architecture-specific bias is detectable from this dataset,
> and a small bias of either sign is fully consistent with the observed
> data."** A paired-by-corpus reanalysis across the 7-corpus battery
> (which respects the repeated-measures structure that the per-corpus
> Welch tests violated) finds transformers ~0.14 BPT *worse* than
> size-matched serial models (t(6)=2.83, p=0.029). This is suggestive
> but should be confirmed at larger n before being treated as a primary
> finding.

All point estimates elsewhere in Paper 7 should be reported with the
bootstrap CIs in `bootstrap_cis.csv` and `cliff_ratio_cis.csv`. The
non-finding correlations from the deep-analysis report should either be
removed or accompanied by their (very wide) bootstrap CIs; the
|Δ_H|–BPT correlation (r=0.69 [0.32, 0.88]) should be promoted from
supplementary to main text.
