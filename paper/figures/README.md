# Paper 7 Figures

Publication-ready figures generated from existing CSVs in exp-1, exp-2, exp-3, exp-6.

## Figures
- **fig1_syn_self_eval** — Achieved BPT vs source entropy for SYN-{2,4,8,12}. §7 (results).
- **fig2_cross_corpus_matrix** — 4×4 cross-corpus BPT heatmap showing specialization vs catastrophic transfer. §7.
- **fig3_quantization_cliff** — BPT vs weight precision (FP16→INT2) across 8 models, highlighting universal INT4→INT3 cliff. §8 (quantization).
- **fig4_arch_comparison** — Forest plot of (transformer − serial) BPT per corpus with CIs. §9 (architecture comparison).
- **fig5_shuffling_cascade** — BPT vs progressive shuffling for 7 models (4 transformer, 3 serial). §10 (structural bonus).
- **fig6_phi_landscape** — log10(φ_GPU) vs log10(params) energy landscape. §11 (silicon efficiency).

## Bootstrap CIs: used 10k bootstrap CI from paper7.1/stats_v2/bootstrap_cis.csv for the wikitext row of Figure 4; other rows use naive ±1.96·SE over n=4 vs n=3 model means.
## Figure 6 corrected variant: fig6b_phi_landscape_corrected.{png,pdf} also generated from paper7.1/exp6_energy_corrected.csv (Terminal A B2 output).

## Format
All figures saved at 300 DPI as both PNG (for the website) and PDF (for the manuscript).

## Color palette
Defined at the top of `scripts/generate_paper7_figures.py`:
- transformer: `#2166ac` (blue)
- serial: `#b2182b` (red)
- neutral: `#4d4d4d`
- highlight: `#f59e0b` (amber, used for SYN-8)
- reference lines: `#999999`
- sequential cmap: viridis

## Reproduction
```
cd /home/user1-gpu/agi-extensions/paper/figures
python ../../scripts/generate_paper7_figures.py
```
