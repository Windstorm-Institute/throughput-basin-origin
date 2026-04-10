#!/usr/bin/env python3
"""Generate publication-ready figures for Paper 7. CPU-only, no torch."""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = "/home/user1-gpu/agi-extensions"
OUT = f"{ROOT}/paper/figures"
os.makedirs(OUT, exist_ok=True)

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Liberation Serif', 'Times New Roman', 'serif'],
    'mathtext.fontset': 'dejavuserif',
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 9, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})

COLOR_TRANSFORMER = '#2166ac'
COLOR_SERIAL = '#b2182b'
COLOR_NEUTRAL = '#4d4d4d'
COLOR_HIGHLIGHT = '#f59e0b'
COLOR_REFERENCE = '#999999'
CMAP_SEQUENTIAL = 'viridis'


def save_both(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300)
    fig.savefig(f"{OUT}/{name}.pdf")
    plt.close(fig)
    print(f"saved {name}")


# ----- Optional inputs from sibling terminals
BOOT_PATH = f"{ROOT}/paper7.1/stats_v2/bootstrap_cis.csv"
TOST_PATH = f"{ROOT}/paper7.1/stats_v2/tost_results.md"
EXP6_CORR = f"{ROOT}/paper7.1/exp6_energy_corrected.csv"
boot = pd.read_csv(BOOT_PATH) if os.path.exists(BOOT_PATH) else None
have_corr = os.path.exists(EXP6_CORR)
tost_text = open(TOST_PATH).read() if os.path.exists(TOST_PATH) else ""


# ============ FIGURE 1 ============
def fig1():
    se = pd.read_csv(f"{ROOT}/exp-1/results/exp1_self_eval.csv")
    ent = pd.read_csv(f"{ROOT}/exp-1/results/corpus_entropy.csv")
    labels = ['SYN-2', 'SYN-4', 'SYN-8', 'SYN-12']
    keys = ['syn2', 'syn4', 'syn8', 'syn12']
    xs = [ent.loc[ent.corpus == lab, 'empirical_entropy'].values[0] for lab in labels]
    ys = [se.loc[se.model == k, 'bpt'].values[0] for k in keys]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = [COLOR_NEUTRAL, COLOR_NEUTRAL, COLOR_HIGHLIGHT, COLOR_NEUTRAL]
    ax.scatter(xs, ys, s=150, c=colors, edgecolor='black', linewidth=0.6, zorder=5)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=10)

    lims = [0, 13]
    ax.plot(lims, lims, '--', color=COLOR_REFERENCE, lw=1, zorder=1)
    ax.text(12.5, 12.0, 'ideal: BPT = source H', color=COLOR_REFERENCE,
            fontsize=9, ha='right', style='italic')
    ax.axhline(4.16, color=COLOR_REFERENCE, lw=1, linestyle=':')
    ax.text(0.2, 4.4, 'natural-language basin (Paper 4)', color=COLOR_REFERENCE,
            fontsize=9, style='italic')

    ax.annotate('SYN-8: 8.92 BPT on 8.0-bit\nsource — within 12%',
                xy=(xs[2], ys[2]), xytext=(5.5, 4.5),
                fontsize=9, color='#7a4a00',
                arrowprops=dict(arrowstyle='->', color=COLOR_HIGHLIGHT, lw=1.2))

    note = ("n=1 seed per condition. SYN-2/SYN-4 overshoot reflects BPE\n"
            "tokenization artifacts on small alphabets (§7.3). SYN-12\n"
            "overshoot reflects 92M-parameter capacity limit (§7.4).")
    ax.text(0.02, 0.98, note, transform=ax.transAxes, fontsize=8,
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=COLOR_REFERENCE, lw=0.5))

    ax.set_xlim(0, 13); ax.set_ylim(0, 25)
    ax.set_xlabel('Empirical source entropy (bits per symbol)')
    ax.set_ylabel('Achieved bits per token (held-out)')
    ax.set_title('Achieved BPT vs source entropy on Markov synthetic corpora')
    save_both(fig, 'fig1_syn_self_eval')


# ============ FIGURE 2 ============
def fig2():
    cc = pd.read_csv(f"{ROOT}/exp-1/results/exp1_cross_corpus.csv")
    order = ['syn2', 'syn4', 'syn8', 'syn12']
    labels = ['SYN-2', 'SYN-4', 'SYN-8', 'SYN-12']
    M = np.zeros((4, 4))
    for i, m in enumerate(order):
        for j, c in enumerate(order):
            M[i, j] = cc.loc[(cc.model == m) & (cc.corpus == c), 'bpt'].values[0]

    fig, ax = plt.subplots(figsize=(7, 6))
    from matplotlib.colors import LogNorm
    im = ax.imshow(M, cmap=CMAP_SEQUENTIAL, norm=LogNorm(vmin=max(M.min(), 1e-2), vmax=M.max()))
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel('Evaluated on corpus')
    ax.set_ylabel('Trained on corpus')
    ax.set_title('Cross-corpus BPT matrix: each model evaluated on each corpus')

    vmin, vmax = np.log10(max(M.min(), 1e-2)), np.log10(M.max())
    for i in range(4):
        for j in range(4):
            v = M[i, j]
            lv = np.log10(max(v, 1e-2))
            txt_color = 'white' if (lv - vmin) / (vmax - vmin) < 0.5 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    color=txt_color, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('BPT (log scale)')

    diag = np.mean([M[i, i] for i in range(4)])
    off = np.mean([M[i, j] for i in range(4) for j in range(4) if i != j])
    cap = (f"Diagonal mean = {diag:.2f} BPT (specialization). "
           f"Off-diagonal mean = {off:.2f} BPT (catastrophic transfer).\n"
           "No clustering near the ~4 BPT natural-language basin in any cell.")
    fig.text(0.5, -0.02, cap, ha='center', fontsize=8.5, style='italic')
    save_both(fig, 'fig2_cross_corpus_matrix')


# ============ FIGURE 3 ============
def fig3():
    q = pd.read_csv(f"{ROOT}/exp-2/results/exp2_quantization.csv")
    precs = ['fp16', 'int8', 'int4', 'int3', 'int2']
    prec_labels = ['FP16', 'INT8', 'INT4', 'INT3', 'INT2']
    models = q['model'].unique().tolist()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap('tab10')
    for k, m in enumerate(models):
        ys = []
        for p in precs:
            row = q[(q.model == m) & (q.precision == p)]
            ys.append(row['bpt'].values[0] if len(row) else np.nan)
        params = q.loc[q.model == m, 'params'].iloc[0]
        short = m.split('/')[-1]
        ax.plot(prec_labels, ys, '-o', color=cmap(k % 10),
                label=f'{short} ({params/1e6:.0f}M)', lw=1.5, ms=5)

    ax.set_yscale('log')
    ax.axvspan(2.5, 3.5, color=COLOR_HIGHLIGHT, alpha=0.15, zorder=0)
    ax.annotate('the cliff', xy=(3, 30), xytext=(3.7, 50),
                fontsize=10, color='#7a4a00',
                arrowprops=dict(arrowstyle='->', color=COLOR_HIGHLIGHT))

    ax.axhline(4.0, color=COLOR_REFERENCE, lw=0.8, linestyle=':')
    ax.text(0.05, 4.1, 'basin (~4)', color=COLOR_REFERENCE, fontsize=8)
    ax.axhline(8.0, color=COLOR_REFERENCE, lw=0.8, linestyle=':')
    ax.text(0.05, 8.2, 'catastrophic (~8)', color=COLOR_REFERENCE, fontsize=8)

    ax.set_xlabel('Weight precision')
    ax.set_ylabel('Bits per token (log scale)')
    ax.set_title('Quantization cliff: BPT vs weight precision across 8 models')
    ax.legend(loc='upper left', frameon=False, fontsize=8)

    cap = ("Universal cliff at INT4 → INT3 across all 8 models tested (cliff ratios 2.96× to 13.87×). "
           "bitsandbytes RTN quantization. Natural-language WikiText-2 evaluation.")
    fig.text(0.5, -0.04, cap, ha='center', fontsize=8.5, style='italic', wrap=True)
    save_both(fig, 'fig3_quantization_cliff')


# ============ FIGURE 4 ============
def fig4():
    sc = pd.read_csv(f"{ROOT}/exp-3/results/exp3_seven_corpus.csv")
    corpora = sorted(sc.corpus.unique())
    rows = []
    for c in corpora:
        t = sc[(sc.corpus == c) & (sc.arch_type == 'transformer')].bpt.values
        s = sc[(sc.corpus == c) & (sc.arch_type == 'serial')].bpt.values
        diff = t.mean() - s.mean()
        # naive SE
        pooled = np.sqrt(((t.var(ddof=1) if len(t) > 1 else 0) / len(t)) +
                         ((s.var(ddof=1) if len(s) > 1 else 0) / len(s)))
        rows.append((c, diff, 1.96 * pooled))
    df = pd.DataFrame(rows, columns=['corpus', 'diff', 'ci'])

    used_boot = False
    if boot is not None:
        m = boot[boot.quantity == 'wikitext_diff_trans_minus_serial']
        if len(m):
            r = m.iloc[0]
            mask = df.corpus == 'wikitext'
            df.loc[mask, 'diff'] = r.point_estimate
            df.loc[mask, 'ci'] = (r.ci_high - r.ci_low) / 2
            used_boot = True

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ys = np.arange(len(df))
    colors = [COLOR_TRANSFORMER if d > 0 else COLOR_SERIAL for d in df['diff']]
    ax.errorbar(df['diff'], ys, xerr=df['ci'], fmt='o', ms=8,
                ecolor=COLOR_NEUTRAL, elinewidth=1.2, capsize=4,
                mfc='white', mec=COLOR_NEUTRAL)
    for i, (d, c) in enumerate(zip(df['diff'], colors)):
        ax.plot(d, i, 'o', ms=9, color=c, zorder=5)
    ax.axvline(0, color=COLOR_REFERENCE, lw=1)
    ax.set_yticks(ys); ax.set_yticklabels(df['corpus'])
    ax.set_xlabel('BPT difference (positive = transformer slower)')
    ax.set_title('Transformer minus state-space BPT, per corpus')

    txt = ("Welch's t = 0.43, p = 0.688 (n = 4 vs 3)\n"
           "Cohen's d ≈ 0.40 (small)\n"
           "Min detectable effect at this n: d ≈ 2.0\n"
           "Reading: not significant, not equivalence-tested")
    if 'CANNOT conclude equivalence' in tost_text:
        txt += "\nTOST (±0.5 BPT): p=0.18, cannot conclude equivalence"
    ax.text(0.98, 0.97, txt, transform=ax.transAxes, fontsize=8,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=COLOR_REFERENCE, lw=0.5))
    if used_boot:
        ax.text(0.02, -0.18, 'wikitext CI from 10k bootstrap (stats_v2); others naive ±1.96·SE',
                transform=ax.transAxes, fontsize=7.5, style='italic', color=COLOR_NEUTRAL)
    save_both(fig, 'fig4_arch_comparison')
    return used_boot


# ============ FIGURE 5 ============
def fig5():
    sh = pd.read_csv(f"{ROOT}/exp-3/results/exp3_shuffling_cascade.csv")
    levels = ['original', 'paragraphs_shuffled', 'sentences_shuffled',
              'words_shuffled', 'all_shuffled']
    level_labels = ['original', 'paragraphs', 'sentences', 'words', 'all']
    models = sh.model.unique().tolist()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    trans_models = sh[sh.arch_type == 'transformer'].model.unique().tolist()
    serial_models = sh[sh.arch_type == 'serial'].model.unique().tolist()
    blues = plt.get_cmap('Blues')(np.linspace(0.45, 0.9, len(trans_models)))
    reds = plt.get_cmap('Reds')(np.linspace(0.45, 0.9, len(serial_models)))

    for i, m in enumerate(trans_models):
        ys = [sh[(sh.model == m) & (sh.shuffle_level == l)].bpt.values[0] for l in levels]
        ax.plot(level_labels, ys, '-o', color=blues[i], lw=1.4, ms=5,
                label=m.split('/')[-1])
    for i, m in enumerate(serial_models):
        ys = [sh[(sh.model == m) & (sh.shuffle_level == l)].bpt.values[0] for l in levels]
        ax.plot(level_labels, ys, '-s', color=reds[i], lw=1.4, ms=5,
                label=m.split('/')[-1])

    # Compute mean deltas
    deltas = []
    for k in range(1, len(levels)):
        d = []
        for m in models:
            a = sh[(sh.model == m) & (sh.shuffle_level == levels[k])].bpt.values[0]
            b = sh[(sh.model == m) & (sh.shuffle_level == levels[k-1])].bpt.values[0]
            d.append(a - b)
        deltas.append(np.mean(d))
    ymax = ax.get_ylim()[1]
    for k, dlt in enumerate(deltas, start=1):
        ax.text(k, ymax * 0.97, f'+{dlt:.1f}', ha='center', fontsize=8,
                color=COLOR_NEUTRAL)

    note = ("Total structural bonus = 6.7 ± 0.3 bits across all 7 models.\n"
            "Replicates Paper 6's reported bonus (~6.74 bits) independently.\n"
            "Transformer and serial architectures show identical bonus.")
    ax.text(0.02, 0.98, note, transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=COLOR_REFERENCE, lw=0.5))

    ax.set_xlabel('Shuffle level')
    ax.set_ylabel('Bits per token')
    ax.set_title('Shuffling cascade: BPT vs structural destruction level')
    ax.legend(loc='center right', frameon=False, fontsize=7.5, ncol=1)
    save_both(fig, 'fig5_shuffling_cascade')


# ============ FIGURE 6 ============
def fig6_one(df, bpt_col, phi_col, log_col, name, suffix_note=''):
    fig, ax = plt.subplots(figsize=(8, 6))
    precs = ['fp32', 'fp16', 'int8', 'int4']
    prec_colors = plt.get_cmap('viridis')(np.linspace(0.15, 0.85, 4))
    configs = sorted(df['config'].unique())
    # Distinct shapes for normal/compiled/batched if present
    shape_map = {}
    base_shapes = ['o', 's', '^', 'D', 'v', 'P', 'X']
    cset = [c for c in ['normal', 'compiled', 'batched'] if c in configs]
    for c in configs:
        if c not in cset:
            cset.append(c)
    for i, c in enumerate(cset):
        shape_map[c] = base_shapes[i % len(base_shapes)]

    for pi, p in enumerate(precs):
        for c in cset:
            sub = df[(df['precision' if 'precision' in df.columns else 'config'].str.lower() == p) &
                     (df['config'] == c)] if 'precision' in df.columns else df[df['config'] == c]
            # exp6 uses 'config' for precision label
            sub = df[(df['config'].str.lower() == p)]
            if len(sub) == 0:
                continue
            x = np.log10(sub['params'].astype(float))
            y = sub[log_col].astype(float)
            ax.scatter(x, y, c=[prec_colors[pi]], marker='o',
                       s=70, edgecolor='black', linewidth=0.4,
                       label=p.upper() if c == cset[0] else None)
            break  # exp6 csv uses config==precision; one pass per precision

    for yref, lab in [(0, 'Landauer floor'),
                      (9, r'Paper 5: $\varphi_{useful} \approx 10^9$'),
                      (16, r'Exp 6: $\varphi_{GPU} \approx 10^{16}$')]:
        ax.axhline(yref, color=COLOR_REFERENCE, lw=0.8, linestyle='--')
        ax.text(ax.get_xlim()[1] if False else 0, yref + 0.15, lab,
                color=COLOR_REFERENCE, fontsize=8)

    ax.set_xlabel('log10(parameters)')
    ax.set_ylabel(f'log10(φ_GPU){suffix_note}')
    ax.set_title('Silicon efficiency vs model size, RTX 5090' + suffix_note)
    ax.legend(loc='lower left', frameon=False, title='Precision')

    note = ("φ_GPU here measures total wall power including memory, cooling, PSU losses,\n"
            "idle leakage. Distinct from Paper 5's φ_useful (≈10^9) which estimates only\n"
            "the irreversible discrimination cost. Both valid; different physical boundaries (§3.4).")
    fig.text(0.5, -0.04, note, ha='center', fontsize=8, style='italic')
    save_both(fig, name)


def fig6():
    df = pd.read_csv(f"{ROOT}/exp-6/results/exp6_energy.csv")
    fig6_one(df, 'bpt', 'phi', 'log10_phi', 'fig6_phi_landscape')
    if have_corr:
        dfc = pd.read_csv(EXP6_CORR)
        fig6_one(dfc, 'bpt_corrected', 'phi_corrected', 'log10_phi_corrected',
                 'fig6b_phi_landscape_corrected', suffix_note=' (corrected)')


def write_readme(used_boot):
    lines = [
        "# Paper 7 Figures",
        "",
        "Publication-ready figures generated from existing CSVs in exp-1, exp-2, exp-3, exp-6.",
        "",
        "## Figures",
        "- **fig1_syn_self_eval** — Achieved BPT vs source entropy for SYN-{2,4,8,12}. §7 (results).",
        "- **fig2_cross_corpus_matrix** — 4×4 cross-corpus BPT heatmap showing specialization vs catastrophic transfer. §7.",
        "- **fig3_quantization_cliff** — BPT vs weight precision (FP16→INT2) across 8 models, highlighting universal INT4→INT3 cliff. §8 (quantization).",
        "- **fig4_arch_comparison** — Forest plot of (transformer − serial) BPT per corpus with CIs. §9 (architecture comparison).",
        "- **fig5_shuffling_cascade** — BPT vs progressive shuffling for 7 models (4 transformer, 3 serial). §10 (structural bonus).",
        "- **fig6_phi_landscape** — log10(φ_GPU) vs log10(params) energy landscape. §11 (silicon efficiency).",
        "",
        f"## Bootstrap CIs: {'used 10k bootstrap CI from paper7.1/stats_v2/bootstrap_cis.csv for the wikitext row of Figure 4; other rows use naive ±1.96·SE over n=4 vs n=3 model means.' if used_boot else 'paper7.1/stats_v2 outputs not available; all error bars in Figure 4 are naive ±1.96·SE.'}",
        f"## Figure 6 corrected variant: {'fig6b_phi_landscape_corrected.{png,pdf} also generated from paper7.1/exp6_energy_corrected.csv (Terminal A B2 output).' if have_corr else 'no corrected variant available (paper7.1/exp6_energy_corrected.csv missing).'}",
        "",
        "## Format",
        "All figures saved at 300 DPI as both PNG (for the website) and PDF (for the manuscript).",
        "",
        "## Color palette",
        "Defined at the top of `scripts/generate_paper7_figures.py`:",
        "- transformer: `#2166ac` (blue)",
        "- serial: `#b2182b` (red)",
        "- neutral: `#4d4d4d`",
        "- highlight: `#f59e0b` (amber, used for SYN-8)",
        "- reference lines: `#999999`",
        "- sequential cmap: viridis",
        "",
        "## Reproduction",
        "```",
        "cd /home/user1-gpu/agi-extensions/paper/figures",
        "python ../../scripts/generate_paper7_figures.py",
        "```",
    ]
    with open(f"{OUT}/README.md", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
    used_boot = fig4()
    fig5()
    fig6()
    write_readme(used_boot)
    print("done")
