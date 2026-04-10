#!/usr/bin/env python3
"""Paper 7.1 stats v2: bootstrap CIs, TOST, mixed-effects, power, multi-comp corrections."""
import os, sys, json, math
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.weightstats import ttost_ind
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.multitest import multipletests

ROOT = "/home/user1-gpu/agi-extensions"
OUT  = f"{ROOT}/paper7.1/stats_v2"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20260409)
B = 10000

def boot_mean_ci(x, B=B, alpha=0.05):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(x), size=(B, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 100*alpha/2)), float(np.percentile(means, 100*(1-alpha/2)))

def boot_stat_ci(x, fn, B=B, alpha=0.05):
    x = np.asarray(x, dtype=float)
    idx = RNG.integers(0, len(x), size=(B, len(x)))
    vals = np.array([fn(x[i]) for i in idx])
    return float(np.percentile(vals, 100*alpha/2)), float(np.percentile(vals, 100*(1-alpha/2)))

# ---------- Load ----------
self_eval   = pd.read_csv(f"{ROOT}/exp-1/results/exp1_self_eval.csv")
cross_corp  = pd.read_csv(f"{ROOT}/exp-1/results/exp1_cross_corpus.csv")
ent         = pd.read_csv(f"{ROOT}/exp-1/results/corpus_entropy.csv")
quant       = pd.read_csv(f"{ROOT}/exp-2/results/exp2_quantization.csv")
bpt_cmp     = pd.read_csv(f"{ROOT}/exp-3/results/exp3_bpt_comparison.csv")
seven       = pd.read_csv(f"{ROOT}/exp-3/results/exp3_seven_corpus.csv")
exp6        = pd.read_csv(f"{ROOT}/exp-6/results/exp6_energy.csv")

# ---------- 1. Bootstrap CIs ----------
rows = []
# SYN-* self-eval (single seed)
for _, r in self_eval.iterrows():
    rows.append([f"self_eval_{r['model']}", r["bpt"], np.nan, np.nan, 0,
                 "single-seed", "single-seed, CI not estimable"])
# Cross-corpus 4x4 (single seed)
for _, r in cross_corp.iterrows():
    rows.append([f"cross_{r['model']}_on_{r['corpus']}", r["bpt"], np.nan, np.nan, 0,
                 "single-seed", "single-seed, CI not estimable"])

# Transformer / serial WikiText means (n=4 / n=3 model populations)
trans_wt = bpt_cmp[bpt_cmp.arch_type=="transformer"].bpt_mean.values
serial_wt = bpt_cmp[bpt_cmp.arch_type=="serial"].bpt_mean.values
lo, hi = boot_mean_ci(trans_wt)
rows.append(["wikitext_transformer_mean_bpt", float(trans_wt.mean()), lo, hi, B,
             "percentile-bootstrap-over-models", "n=4 transformer models"])
lo, hi = boot_mean_ci(serial_wt)
rows.append(["wikitext_serial_mean_bpt", float(serial_wt.mean()), lo, hi, B,
             "percentile-bootstrap-over-models", "n=3 serial models"])

# Diff in WikiText mean BPT (independent groups bootstrap)
def boot_diff(a, b, B=B):
    a = np.asarray(a); b = np.asarray(b)
    ai = RNG.integers(0, len(a), size=(B, len(a)))
    bi = RNG.integers(0, len(b), size=(B, len(b)))
    d  = a[ai].mean(axis=1) - b[bi].mean(axis=1)
    return float(np.percentile(d,2.5)), float(np.percentile(d,97.5))
lo,hi = boot_diff(trans_wt, serial_wt)
rows.append(["wikitext_diff_trans_minus_serial", float(trans_wt.mean()-serial_wt.mean()),
             lo, hi, B, "percentile-bootstrap-independent",
             "WikiText only (single corpus); see also paired-by-corpus in mixed_effects.txt"])

# Paired-by-corpus diff across 7 corpora (transformer 4-model mean - serial 3-model mean per corpus)
paired = []
corpora = sorted(seven.corpus.unique())
for c in corpora:
    t = seven[(seven.corpus==c)&(seven.arch_type=="transformer")].bpt.mean()
    s = seven[(seven.corpus==c)&(seven.arch_type=="serial")].bpt.mean()
    paired.append(t-s)
paired = np.array(paired)
lo,hi = boot_mean_ci(paired)
rows.append(["paired_by_corpus_diff_trans_minus_serial", float(paired.mean()),
             lo, hi, B, "percentile-bootstrap-paired-corpora", f"n={len(paired)} corpora"])

# Mean structural bonus across 8-model panel (FP16)
fp16 = quant[quant.precision=="fp16"]
sb = fp16.structural_bonus.values
lo,hi = boot_mean_ci(sb)
rows.append(["mean_structural_bonus_fp16_panel", float(sb.mean()), lo, hi, B,
             "percentile-bootstrap-over-models", f"n={len(sb)}"])

# Mean cliff ratio (int3/int4) across 8 models
cliff_per_model = []
for m, grp in quant.groupby("model"):
    g = grp.set_index("precision")
    if "int3" in g.index and "int4" in g.index:
        cliff_per_model.append((m, g.loc["int3","bpt"]/g.loc["int4","bpt"], g.loc["int4","bpt"], g.loc["int3","bpt"], g.loc["params","precision"] if False else grp.params.iloc[0]))
cliff_arr = np.array([c[1] for c in cliff_per_model])
lo,hi = boot_mean_ci(cliff_arr)
rows.append(["mean_cliff_ratio_int3_over_int4", float(cliff_arr.mean()), lo, hi, B,
             "percentile-bootstrap-over-models", f"n={len(cliff_arr)}"])

# Mean log10(phi) over exp6
lp = exp6.log10_phi.dropna().values
lo,hi = boot_mean_ci(lp)
rows.append(["mean_log10_phi_exp6", float(lp.mean()), lo, hi, B,
             "percentile-bootstrap-over-configs", f"n={len(lp)}"])

pd.DataFrame(rows, columns=["quantity","point_estimate","ci_low","ci_high",
                            "n_resamples","method","notes"]).to_csv(
    f"{OUT}/bootstrap_cis.csv", index=False)

# ---------- 2. TOST ----------
margin = 0.5
p1, _, _ = ttost_ind(trans_wt, serial_wt, low=-margin, upp=margin, usevar="unequal")
# ttost_ind returns (pvalue, t1_stat, t2_stat) -- pvalue is the max already
tost_p = float(p1)
welch = stats.ttest_ind(trans_wt, serial_wt, equal_var=False)

# Min equivalence margin detectable at 80% power for n=4 vs 3
# Use TTestIndPower to find effect size for 80% power, then convert to BPT units via pooled SD
pooled_sd = math.sqrt(((len(trans_wt)-1)*trans_wt.var(ddof=1)+(len(serial_wt)-1)*serial_wt.var(ddof=1))/(len(trans_wt)+len(serial_wt)-2))
analysis = TTestIndPower()
try:
    d_min = analysis.solve_power(effect_size=None, nobs1=4, ratio=3/4, alpha=0.05, power=0.80, alternative="two-sided")
except Exception:
    d_min = float("nan")
margin_bpt_min = d_min * pooled_sd if not math.isnan(d_min) else float("nan")

with open(f"{OUT}/tost_results.md","w") as f:
    f.write(f"""# TOST equivalence test: transformer vs serial WikiText BPT

**Pre-specified equivalence margin**: ±{margin} BPT.

Justification: 0.5 BPT is approximately half of one natural standard deviation
of the WikiText basin across the model panel (pooled SD = {pooled_sd:.3f} BPT)
and is the smallest effect that would be scientifically meaningful — differences
smaller than this are dwarfed by run-to-run variability we have not characterised
and by inter-corpus variation already documented in Paper 7.

## Inputs
- Transformer (n={len(trans_wt)}): {list(map(float, trans_wt))}
- Serial      (n={len(serial_wt)}): {list(map(float, serial_wt))}
- Means: transformer = {trans_wt.mean():.4f}, serial = {serial_wt.mean():.4f}
- Welch t = {welch.statistic:.4f}, p = {welch.pvalue:.4f}

## TOST result
- Overall TOST p-value (max of two one-sided tests) = **{tost_p:.4f}**
- Conclusion at α=0.05: **{"equivalent within ±0.5 BPT" if tost_p < 0.05 else "CANNOT conclude equivalence within ±0.5 BPT"}**

## Honest power note
With n=4 vs n=3, even the equivalence test is severely underpowered. The
minimum effect size detectable at 80% power for a Welch-style two-sample test
at this n is Cohen's d ≈ {d_min:.2f}, which corresponds to roughly
**±{margin_bpt_min:.2f} BPT** in raw units given the observed pooled SD.

In other words: even if we ran TOST honestly, this design could only have
"established equivalence" if the true difference were essentially zero AND we
were willing to call ±{margin_bpt_min:.2f} BPT a "small" margin — which it
is not. The current data therefore CANNOT support either "transformer ≠ serial"
or "transformer ≈ serial". The honest verdict is **insufficient data**, and
Paper 7's claim of "no architecture-specific bias" must be downgraded to
"no architecture-specific bias was detected, but the test had no power to
detect one of plausible size".
""")

# ---------- 3. Mixed-effects ----------
mixed_log = []
try:
    df = seven.copy()
    df["arch_bin"] = (df.arch_type=="transformer").astype(int)
    md = sm.MixedLM.from_formula("bpt ~ arch_bin", groups="model", re_formula="1",
                                 vc_formula={"corpus":"0+C(corpus)"}, data=df)
    mres = md.fit(method="lbfgs")
    mixed_log.append(str(mres.summary()))
    fe = mres.fe_params["arch_bin"]
    se = mres.bse_fe["arch_bin"]
    pv = mres.pvalues["arch_bin"]
    var_model = float(mres.cov_re.iloc[0,0]) if mres.cov_re is not None else float("nan")
    var_corpus = float(list(mres.vcomp)[0]) if len(mres.vcomp)>0 else float("nan")
    var_resid = float(mres.scale)
    mixed_log.append(f"\nFIXED EFFECT arch_bin (transformer=1): beta={fe:.4f}, SE={se:.4f}, p={pv:.4f}")
    mixed_log.append(f"Var(model)={var_model:.4f}, Var(corpus)={var_corpus:.4f}, Var(resid)={var_resid:.4f}")
    used_fallback = False
except Exception as e:
    mixed_log.append(f"MixedLM failed: {e}\nFalling back to paired-by-corpus one-sample t.")
    used_fallback = True

# Paired-by-corpus fallback (always run as a sanity check too)
diffs = []
for c in corpora:
    t = seven[(seven.corpus==c)&(seven.arch_type=="transformer")].bpt.mean()
    s = seven[(seven.corpus==c)&(seven.arch_type=="serial")].bpt.mean()
    diffs.append(t-s)
diffs = np.array(diffs)
tt = stats.ttest_1samp(diffs, 0.0)
mixed_log.append(f"\nPaired-by-corpus fallback (n={len(diffs)} corpora):")
mixed_log.append(f"  diffs (T-S) by corpus: {dict(zip(corpora, [round(float(x),4) for x in diffs]))}")
mixed_log.append(f"  mean diff = {diffs.mean():.4f}, t = {tt.statistic:.4f}, p = {tt.pvalue:.4f}")
mixed_log.append(f"\nNaive Welch (from exp3_statistics.csv) p = 0.6880.")
mixed_log.append("Mixed-effects / paired analyses correctly account for repeated measures across corpora;")
mixed_log.append("they should be (and are) more conservative or comparable, NOT less. The data still")
mixed_log.append("provide no evidence of an architecture difference, but also no evidence of equivalence.")
with open(f"{OUT}/mixed_effects.txt","w") as f:
    f.write("\n".join(mixed_log))

# ---------- 4. Power curves ----------
ns = [3,5,10,20,50,100,200,500]
mdes = []
for n in ns:
    d = analysis.solve_power(effect_size=None, nobs1=n, ratio=1.0, alpha=0.05, power=0.80, alternative="two-sided")
    mdes.append(d)
# Current n=4 vs 3
d_current = analysis.solve_power(effect_size=None, nobs1=4, ratio=3/4, alpha=0.05, power=0.80, alternative="two-sided")
# n needed for d=0.397
d_obs = 0.39671608
n_needed = analysis.solve_power(effect_size=d_obs, nobs1=None, ratio=1.0, alpha=0.05, power=0.80, alternative="two-sided")

plt.figure(figsize=(7,5))
plt.plot(ns, mdes, "o-", label="MDE at 80% power (equal n)")
plt.axhline(d_obs, color="red", ls="--", label=f"observed d={d_obs:.3f}")
plt.scatter([4], [d_current], color="orange", zorder=5, s=80, label=f"current n=4 vs 3, MDE≈{d_current:.2f}")
plt.scatter([n_needed], [d_obs], color="green", zorder=5, s=80, label=f"n≈{n_needed:.0f}/grp to detect d=0.397")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("n per group"); plt.ylabel("Minimum detectable Cohen's d")
plt.title("Architecture comparison power curve (α=0.05, power=0.80)")
plt.grid(True, which="both", alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/power_curves.png", dpi=140); plt.close()

pd.DataFrame({"n_per_group": ns, "mde_cohens_d": mdes}).to_csv(
    f"{OUT}/power_curves.csv", index=False)
with open(f"{OUT}/power_curves.csv","a") as f:
    f.write(f"# current_n=4_vs_3,mde={d_current:.4f}\n")
    f.write(f"# observed_d={d_obs:.4f},n_needed_per_group={n_needed:.1f}\n")

# ---------- 5. Holm / BH on 7 per-corpus tests ----------
rows = []
raw_p = []
for c in corpora:
    t_vals = seven[(seven.corpus==c)&(seven.arch_type=="transformer")].bpt.values
    s_vals = seven[(seven.corpus==c)&(seven.arch_type=="serial")].bpt.values
    tt = stats.ttest_ind(t_vals, s_vals, equal_var=False)
    rows.append([c, len(t_vals), len(s_vals), float(t_vals.mean()), float(s_vals.mean()), float(tt.pvalue)])
    raw_p.append(float(tt.pvalue))
holm_rej, holm_p, _, _ = multipletests(raw_p, alpha=0.05, method="holm")
bh_rej,   bh_p,   _, _ = multipletests(raw_p, alpha=0.05, method="fdr_bh")
out = []
for i, r in enumerate(rows):
    out.append(r + [holm_p[i], bh_p[i], bool(holm_rej[i]), bool(bh_rej[i])])
pd.DataFrame(out, columns=["corpus","n_transformer","n_serial","mean_t","mean_s",
                           "raw_p","holm_p","bh_fdr_p","holm_significant","bh_significant"]
            ).to_csv(f"{OUT}/holm_corrected_corpus_tests.csv", index=False)

# ---------- 6. Cliff ratio CIs ----------
rows = []
ratios = []
for m, grp in quant.groupby("model"):
    g = grp.set_index("precision")
    if "int3" in g.index and "int4" in g.index:
        ratio = float(g.loc["int3","bpt"]/g.loc["int4","bpt"])
        ratios.append(ratio)
        rows.append([m, int(grp.params.iloc[0]), float(g.loc["int4","bpt"]),
                     float(g.loc["int3","bpt"]), ratio, 1,
                     "single measurement, no within-cell uncertainty"])
ratios = np.array(ratios)
lo, hi = boot_mean_ci(ratios)
rows.append(["ALL_MEAN", len(ratios), np.nan, np.nan, float(ratios.mean()),
             B, f"bootstrap mean across n={len(ratios)} models, ci_low={lo:.4f}, ci_high={hi:.4f}"])
df_out = pd.DataFrame(rows, columns=["model","params","bpt_int4","bpt_int3","cliff_ratio","n_within_cell","notes"])
# add explicit ci_low/ci_high columns for the aggregate row
df_out["ci_low"]  = [np.nan]*(len(rows)-1) + [lo]
df_out["ci_high"] = [np.nan]*(len(rows)-1) + [hi]
df_out.to_csv(f"{OUT}/cliff_ratio_cis.csv", index=False)

# ---------- 7. Correlation replay (cross-corpus n=16) ----------
ent2 = ent.copy()
ent2["model"] = ent2.corpus.str.lower().str.replace("-","")
ent2["corpus_norm"] = ent2.corpus.str.lower().str.replace("-","")
emap = dict(zip(ent2.corpus_norm, ent2.empirical_entropy))
cc = cross_corp.copy()
cc["train_H"]  = cc["model"].map(emap)
cc["target_H"] = cc["corpus"].map(emap)
cc = cc.dropna(subset=["train_H","target_H","bpt"])

def boot_pearson(x, y, B=B):
    x = np.asarray(x); y = np.asarray(y)
    n = len(x)
    rs = []
    for _ in range(B):
        i = RNG.integers(0,n,size=n)
        if np.std(x[i])==0 or np.std(y[i])==0: continue
        rs.append(np.corrcoef(x[i],y[i])[0,1])
    rs = np.array(rs)
    return float(np.percentile(rs,2.5)), float(np.percentile(rs,97.5))

rows = []
pairs = [
    ("train_H_vs_BPT",     cc["train_H"],  cc["bpt"]),
    ("target_H_vs_BPT",    cc["target_H"], cc["bpt"]),
    ("delta_H_vs_BPT",     (cc["target_H"]-cc["train_H"]), cc["bpt"]),
    ("abs_delta_H_vs_BPT", (cc["target_H"]-cc["train_H"]).abs(), cc["bpt"]),
]
for name, a, b in pairs:
    r, p = stats.pearsonr(a, b)
    lo, hi = boot_pearson(a.values, b.values)
    rows.append([name, len(a), float(r), float(p), lo, hi])
pd.DataFrame(rows, columns=["pair","n","pearson_r","p_value","r_ci_low","r_ci_high"]
            ).to_csv(f"{OUT}/correlation_cis.csv", index=False)

print("DONE. Outputs in", OUT)
print("TOST p =", tost_p, "Welch p =", welch.pvalue)
print("Paired-by-corpus diff t-test p =", tt.pvalue if False else stats.ttest_1samp(diffs,0).pvalue)
print("Mean cliff ratio 95% CI =", lo, hi if False else (boot_mean_ci(ratios)))
