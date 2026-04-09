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

## Update 3 — ecosystem cleanup execution (2026-04-09T07:22-04:00)

Executed all phases of the ecosystem cleanup directive. **Phase 1 (org rename) blocked at the API layer; everything else complete.**

### Phase 1 — `WIndstorm-Labs` → `Windstorm-Labs` org rename
**Status: ⛔ BLOCKED. Requires PI manual action.**

- `gh auth status` confirms token has scopes `gist, read:org, repo`. **No `admin:org`, no org-create permission.**
- GitHub does not expose org creation through the REST API regardless of scopes — orgs can only be created via the web UI at https://github.com/account/organizations/new.
- Repo transfers via `gh api repos/{old}/{repo}/transfer` would also require admin on both source and destination orgs, which we lack.
- **PI action required:**
  1. Visit https://github.com/account/organizations/new and create the org `Windstorm-Labs` (correctly cased, single capital `W`).
  2. For each of the 9 repos in `WIndstorm-Labs`, go to Settings → Transfer Ownership and transfer to the new org. The 9 repos are: `fons-constraint`, `receiver-limited-floor`, `throughput-basin`, `serial-decoding-basin`, `dissipative-decoder`, `inherited-constraint`, `agi-extensions`, `throughput-experiments`, `.github`.
  3. Once transferred, ping me to update every cross-repo link from `WIndstorm-Labs/...` to `Windstorm-Labs/...` and to add a forwarding README to the (now empty) old org.
- Both org-profile READMEs already document the typo and the planned mirror as a known issue, so the typo is not invisible to readers in the meantime.

### Phase 2 — Canonical 1-indexed numbering on the website ✅
- **Surprise finding:** the body articles (`speed-limit-of-thought.html`, `throughput-basin.html`, `serial-decoding-basin.html`, `receiver-limited-floor.html`) were already 1-indexed. Only `index.html` (research arc) and the new `throughput-basin-origin.html` (which I introduced this morning with the `(ms. 7)` bandage) used the 0-indexed scheme.
- `index.html` arc nodes Paper 0..5 → Paper 1..6, "(ms. 7)" → Paper 7.
- Two stale "Paper 4" references in the Two Regimes section that meant the Dissipative Decoder updated to "Paper 5".
- `articles/throughput-basin-origin.html`: every "Paper 6" → "Paper 7", every "Paper 6.1" → "Paper 7.1", removed "(manuscript Paper 7)" parentheticals.
- Verified zero "Paper 0" / "(ms. 7)" / "(manuscript Paper 7)" / "Paper 6.1" remain anywhere on the site.

### Phase 6 — φ methodology footnote ✅
- Added a styled note to `index.html`'s Two Regimes section reconciling the ~10⁹ (Paper 5 useful-dissipation) and ~10¹⁵–10¹⁸ (Paper 7 wall-power) figures.
- Added a `<div class="callout">` block to `articles/dissipative-decoder.html` immediately after the "800,000,000 times above its Landauer floor" sentence, making the same point at the point of citation, with a link to the Paper 7 article §3.4.

### Phase 7 — Tokenization-confound retroactive disclosure ✅
- **Drafted now per PI directive, not waiting for Paper 7.1 empirical resolution.**
- Added a "Measurement note added April 2026" callout to `articles/serial-decoding-basin.html` immediately after the τ = 4.16 ± 0.19 result, stating that the figure is in BPT not bits-per-source-symbol, that the adversarial review's BPT-vs-source-symbol concern applies retroactively to Papers 1–4, and that re-measurement is scoped under Paper 7.1.
- Posted [comment on issue #1](https://github.com/sneakyfree/agi-extensions/issues/1) expanding Paper 7.1 scope to include retroactive re-measurement of τ.

### Phase 8 — Reading order on homepage ✅
- New "Start here" element above the Articles grid in `index.html`, listing all 8 articles in the cold-reader path with read times. Styled to match the existing dark theme.

### Phase 9 — Preprint status tag on Paper 7 publication card ✅
- Replaced the generic `preprint` tag with a styled amber tag reading "preprint — Zenodo pending Paper 7.1" so it's visually distinct from Papers 1–6 which have real DOIs.

### Phase 3 — Stale skeleton READMEs replaced ✅
- `Windstorm-Institute/agi-extensions` README (commit `c7a7880`): full rewrite. Now opens with the adversarial-review callout, declares `sneakyfree/agi-extensions` as the canonical repo, links the manuscript / review / Paper 7.1 issue / companion article, points back to Papers 1–6 and the Labs mirror.
- `WIndstorm-Labs/agi-extensions` README (commit `9e284b5`): full rewrite. Marks experiments as ✅ complete (was "Experiments not yet run"), links canonical repo, fixes the Paper 4/5 ordering swap that was in the old Series Index, replaces the fake `zenodo.1234573` Published Version link with real DOIs for Papers 1–6, documents the org slug typo.

### Phase 4 — Backfill real Zenodo DOIs ✅
- `sneakyfree/agi-extensions/README.md` citations 1–6 now use the real DOIs (19274048, 19322973, 19323194, 19323423, 19433048, 19432911) instead of `XXXXXX` placeholders.
- Added an explicit Paper 7 entry marked preprint, Zenodo deposit pending Paper 7.1.

### Phase 5 — Hedge unscoped claims ✅
- `sneakyfree/agi-extensions/README.md` "Conclusion" line that previously read "not a universal limit" replaced with the defensible 92M / Markov-synthetic / BPE-tokenized scoping plus links to the adversarial review and Paper 7.1 issue.
- `exp-8/EXPERIMENT_8_RESULTS.md` "the basin is data-driven, confirmed" replaced with "the data-driven hypothesis is consistent with all four experiments at the modalities tested … blocking items remain open before this conclusion can be generalized." Goal statement updated to match.

### Phase 10 — Org profile READMEs ✅
- Both `.github` org repos already existed.
- `Windstorm-Institute/.github/profile/README.md` (commit `cf55a87`): replaced with a real DOI table, fixed the **Paper 4/5 swap** present in the old version (the old table had Dissipative Decoder = Paper 4 and Serial Decoding Basin τ = Paper 5, which contradicts the website and the Zenodo deposit order; corrected to Serial Decoding Basin τ = Paper 4, Dissipative Decoder = Paper 5), added Paper 7 with the preprint marker and the adversarial-review callout, and added the retroactive τ tokenization note.
- `WIndstorm-Labs/.github/profile/README.md` (commit `e46fa00`): same DOI table, same Paper 4/5 fix, same Paper 7 callout, and an explicit "note on the org slug" explaining the `WIndstorm` typo and the planned mirror.

---

## Commits made this session

| Repo | Commit | Description |
|---|---|---|
| `sneakyfree/windstorminstitute.org` | `6c612b7` | Numbering, φ footnote, tokenization callout, reading order, preprint tag |
| `Windstorm-Institute/agi-extensions` | `c7a7880` | Replace skeleton README |
| `WIndstorm-Labs/agi-extensions` | `9e284b5` | Update Labs README, real DOIs, adversarial-review callout |
| `sneakyfree/agi-extensions` | `05162f9` | DOI backfill + hedge claims + Exp 8 hedge |
| `Windstorm-Institute/.github` | `cf55a87` | Org profile rewrite |
| `WIndstorm-Labs/.github` | `e46fa00` | Org profile rewrite |
| `sneakyfree/agi-extensions#1` | comment | Retroactive scope expansion to Papers 1–4 τ measurement |

Plus this status file commit, coming next.

## Other findings surfaced during execution

- **Paper 4/5 swap.** Both pre-existing org-profile READMEs had Dissipative Decoder labeled Paper 4 and Serial Decoding Basin τ labeled Paper 5. The website (and the Zenodo deposit order) has them the other way around: τ is Paper 4, Dissipative is Paper 5. This was a **separate** numbering bug from the one Phase 2 fixed, and it would have caused readers landing at either org's profile page to see a contradictory series index versus the website. Now corrected on both org profiles.
- **The Labs `agi-extensions` README previously had a fake Zenodo DOI** (`zenodo.1234573`) labeled "Published Version" — replaced with real DOIs for Papers 1–6 in the rewritten README.
- **The website body articles were already correctly 1-indexed.** Only the homepage research arc and the new Paper 7 article were 0-indexed. The audit's "off-by-one catastrophe" framing was right that the inconsistency existed but was wrong about its scope — most of the site was already self-consistent and the arc was the outlier.

## Outstanding decisions for the PI

- **D-followup.** Approve the website `index.html` diff (shown in conversation; also reproducible via `cd /tmp/wsi-site && git diff`). Confirm site numbering: keep as Paper 6, or renumber to Paper 7 to match manuscripts? Once approved I commit on `paper7` branch and push.
- **Article HTML.** Want me to convert `website/paper7_article.md` to a new `articles/throughput-basin-origin.html` page following the existing `articles/inherited-constraint.html` template? Adds ~30 min and a real article URL.
- **Paper 7.1 ticket.** Should I open a tracking issue on `sneakyfree/agi-extensions` enumerating the eight items from §5b so they're visible to outside readers?
- **`agi-extensions` org.** Note for future consistency: `fons-constraint` is under `Windstorm-Institute/`, but `agi-extensions` is under `sneakyfree/`. Worth a transfer at some point; not blocking.

