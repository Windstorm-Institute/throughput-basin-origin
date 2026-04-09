# Conductor Status — 2026-04-09

**Conductor:** Claude Opus 4.6 (1M)
**PI:** Grant Lavell Whitmer III

## Survey of Parallel Terminal Output

| Terminal | Artifact | Status |
|---|---|---|
| T0 (overnight runner) | exp-1..6 CSVs, master summary, base draft | Committed locally; **already pushed** to `sneakyfree/agi-extensions` (HEAD `29a2265`) |
| T1 (GPU, Exp 8) | `exp-8/results/*.csv`, `exp-8/plots/*.png`, `summary.json` | **Complete** (elapsed 545s). No process running. ⚠ multimodal block "skipped" → NaN |
| T2 (paper draft) | `paper/paper7_formal_draft.md` (173 lines) | Complete. Addresses SYN-12 capacity, φ-definition split, BPE artifacts. Does **not** address §0.1, §0.2, §1.1, §3.2 of adversarial review. |
| T3 (deep analysis) | `analysis/paper7_deep_analysis.md` + 5 PNGs | Complete |
| T4 (adversarial) | `review/adversarial_review.md` | Complete. **Recommendation: Major revisions required.** |
| T5 (website) | `website/paper7_{article,publication_entry,research_arc_entry}.md` | Complete |

## Git State

- `agi-extensions`: remote = `sneakyfree/agi-extensions` (exists). HEAD already on origin/main. Untracked: `analysis/`, `paper/paper7_formal_draft.md`, `review/`, `website/`, `exp-8/`, `synthesize_results.py`, status flags.
- `fons-constraint`: remote = `Windstorm-Institute/fons-constraint` (NOT `sneakyfree`). Clean working tree.
- Website repo exists: `sneakyfree/windstorminstitute.org` (not yet cloned).

## ⛔ BLOCKING ISSUES — DO NOT PUSH UNTIL PI DECIDES

The internal adversarial review (`review/adversarial_review.md`) identifies issues that the formal draft does **not** resolve. These are not interpretive — they are arithmetic contradictions in our own CSVs:

1. **§0.1 Self-eval vs cross-corpus diagonal disagree.** Same model on same corpus reports two different BPTs (e.g., SYN-8: 8.92 vs 7.38; SYN-12: 17.40 vs 5.48). Either the eval pipelines measure different things or the cross-corpus split leaked. **The headline number for the entire paper sits in this gap.**
2. **§0.2 Exp 6 BPT ≠ Exp 2/3 BPT for the same (model, corpus).** Pythia-160m on WikiText: 3.96 (exp2/3) vs 12.10 (exp6). Every φ in Exp 6 is downstream of this. The "10^15–10^18 above Landauer" headline rides on numbers that don't reconcile with the project's own Exp 2.
3. **§1.1 BPT unit confound.** SYN-8 uses a corpus-specific BPE tokenizer (vocab=8192). The "8.92 BPT vs 8.0 source entropy" comparison is a unit error — bits-per-token is not bits-per-source-symbol. The formal draft notes BPE artifacts as a limitation but still treats 8.92 as decisive.
4. **§3.2 No learning curves.** We cannot tell whether SYN-8 plateaued at 8.92 or was still descending at the cutoff.
5. **§0.3 Mamba energy 100× higher than Pythia** in `exp3_energy.csv` — likely un-fused reference kernel. Invalidates the architecture-fairness energy comparison.

The formal draft does correctly handle: SYN-12 capacity limit (§3.1 of review), φ_GPU vs φ_useful methodology gap (§3.3), SYN-2/4 tokenization weirdness (§3.4) — but only as Limitations, not by re-running.

## ⚠ Other Issues Noted

- `exp-8/results/exp8d_multimodal.csv` is 23 bytes; `summary.json` records the multimodal condition as `"skipped"` with `bpt: NaN`. Exp 8 is incomplete on the multimodal arm.
- `fons-constraint` remote is `Windstorm-Institute/`, not `sneakyfree/`. The instructions assumed `sneakyfree/fons-constraint`. Need PI confirmation which is canonical before pushing the README link update.

## What I Have NOT Done (Awaiting PI Direction)

- ❌ Staged or pushed `analysis/`, `paper/`, `review/`, `website/`, `exp-8/` to `agi-extensions`. Pushing the formal draft as-is would publish claims the internal adversarial review explicitly flags as unsupported. The Windstorm Institute principle ("never soften a negative result, never overclaim a positive one") cuts both ways.
- ❌ Updated `fons-constraint` README with Paper 7 link.
- ❌ Cloned `sneakyfree/windstorminstitute.org` or integrated T5 website content.
- ❌ Created Paper 8 stub from Exp 8 (multimodal arm incomplete anyway).

## Decision Points for the PI

**A. How to publish given the adversarial review?**
   1. **Hold and re-run.** Address §0.1, §0.2, §1.1, §3.2 before any push. Cleanest, slowest.
   2. **Push as "preprint with internal review attached".** Push everything including `review/adversarial_review.md`, and rewrite the formal draft's abstract/conclusions to match the "Defensible claim" column from review §5. Honest, fast, costs the headline.
   3. **Push raw artifacts only** (CSVs, plots, code) to `agi-extensions`, hold the formal draft. Lets the world reproduce; defers the interpretive claims.

   My recommendation: **(2)**. It matches the institute's stated values and the adversarial review is itself a publication-grade artifact.

**B. `fons-constraint` canonical org.** Confirm: `Windstorm-Institute/fons-constraint` (current remote) or `sneakyfree/fons-constraint` (per instructions)?

**C. Exp 8 multimodal.** Re-run the skipped block, or ship Exp 8 as vision-only?

**D. Website integration.** OK to clone `sneakyfree/windstorminstitute.org` and prepare a PR-style commit on a branch for review before push?

---

## Update — execution log

PI greenlit recommendations A–D. Steps executed:

1. ✅ **Formal draft rewritten.** `paper/paper7_formal_draft.md` abstract and §7 conclusion replaced with defensible-claim language. New §5b "Internal Adversarial Review and Paper 7.1 Scope" added, listing all eight items from `review/adversarial_review.md` and the re-runs they call for.
2. ✅ **agi-extensions README updated.** Adversarial-review callout block added directly under the headline. `fons-constraint` link corrected to `Windstorm-Institute/fons-constraint`. Closing line replaced with the "defensible vs stronger claim" framing. Exp 8 vision-only status noted.
3. ✅ **Pushed to `sneakyfree/agi-extensions`.** Commit `83cf04d`, 29 files, 52K insertions. Includes `paper/paper7_formal_draft.md`, `review/adversarial_review.md`, `analysis/`, `website/`, `exp-8/`, `CONDUCTOR_STATUS.md`. `*.log` excluded by gitignore. No model weights, no credentials.
4. ✅ **fons-constraint backlink.** `Windstorm-Institute/fons-constraint` README now links to `agi-extensions`. Commit `d225b2e`, pushed to `main`.
5. 🟡 **Website branch prepared, awaiting PI review before push.** Cloned `sneakyfree/windstorminstitute.org` to `/tmp/wsi-site`, branch `paper7`. `index.html` diff staged (not committed, not pushed):
   - Research Arc subtitle: "Six papers" → "Seven papers... and now, to falsification."
   - New arc node: Paper 6 (site numbering) = "The Throughput Basin Origin" → linked to `github.com/sneakyfree/agi-extensions`.
   - New publication card #06 with the defensible abstract and explicit "published with its internal adversarial review attached" framing.
   - **Not** integrated yet: T5's long-form `paper7_article.md` as a new `articles/throughput-basin-origin.html` page. That's a follow-up if you want it.
   - **Show-stopper to verify before push:** site uses `Paper 0..5` numbering (off-by-one from manuscript numbering). Paper 7 manuscript becomes Paper 6 on the site. Confirm that's what you want, or I can renumber to Paper 7 throughout.
6. 🟡 **Exp 8 multimodal arm.** Shipped as deferred per recommendation C. Vision arm CSVs and 4 plots are in the push. Multimodal `summary.json: skipped, NaN` is preserved verbatim — not erased.

## Update 2 — execution log (steps 6–8)

7. ✅ **Long-form article ported.** `articles/throughput-basin-origin.html` ("The Mirror, Not the Wall") created on the website using `inherited-constraint.html` as scaffolding. The body is the T5 long-form, with a dedicated **callout section** ("The part where we tell on ourselves") that foregrounds the four blocking items from the adversarial review. Article-nav prev link points to `inherited-constraint.html`.
8. ✅ **index.html updates.** Research arc node now points to the article (not GitHub). Publication card #06 links to the article and includes the GitHub URL in the description. New articles-grid card added. Site keeps 0-indexed numbering with "(Paper 7 in manuscript series)" parentheticals.
9. ✅ **Branch pushed and merged.** `paper7` branch pushed to `sneakyfree/windstorminstitute.org`, fast-forwarded into `main` (`986cb40..e116c35`), pushed. **GitHub Pages now live with the article and the updated index.**
10. ✅ **Paper 7.1 tracking issue opened.** [`sneakyfree/agi-extensions#1`](https://github.com/sneakyfree/agi-extensions/issues/1) — "Paper 7.1: blocking items from internal adversarial review". 4 blocking items + 4 strongly recommended re-runs as a checklist, with section anchors back into the review.

## Final state

- `sneakyfree/agi-extensions` @ `3492ecc` — Paper 7 manuscript, adversarial review, deep analysis, Exp 8 vision arm, website source content, conductor status. Issue #1 open for Paper 7.1.
- `Windstorm-Institute/fons-constraint` @ `d225b2e` — backlink added.
- `sneakyfree/windstorminstitute.org` @ `e116c35` — Paper 6 (ms. 7) live in research arc, publications, and articles. New `articles/throughput-basin-origin.html` published.

## Outstanding decisions for the PI

- **D-followup.** Approve the website `index.html` diff (shown in conversation; also reproducible via `cd /tmp/wsi-site && git diff`). Confirm site numbering: keep as Paper 6, or renumber to Paper 7 to match manuscripts? Once approved I commit on `paper7` branch and push.
- **Article HTML.** Want me to convert `website/paper7_article.md` to a new `articles/throughput-basin-origin.html` page following the existing `articles/inherited-constraint.html` template? Adds ~30 min and a real article URL.
- **Paper 7.1 ticket.** Should I open a tracking issue on `sneakyfree/agi-extensions` enumerating the eight items from §5b so they're visible to outside readers?
- **`agi-extensions` org.** Note for future consistency: `fons-constraint` is under `Windstorm-Institute/`, but `agi-extensions` is under `sneakyfree/`. Worth a transfer at some point; not blocking.

