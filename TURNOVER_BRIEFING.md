# WINDSTORM INSTITUTE — FULL CONTEXT BRIEFING

You are picking up work for Grant Lavell Whitmer III, founder of the Windstorm Institute,
on his Varon-1 workstation (RTX 5090, 32 GB VRAM). He runs a multi-agent Claude Code fleet
and prefers independent verification, detailed signed patient-file-style log entries, and
direct honest answers (including telling him when something is wrong or won't survive
peer review). Call him Grant.

## 1. WHAT THE WINDSTORM INSTITUTE IS

An independent research program on information-theoretic constraints in serial decoding
systems — from ribosomes to transformers. The core empirical claim across nine papers:
serial information-processing systems across biology, neuroscience, and AI converge on a
similar throughput range, and in AI specifically the "basin" near ~1 bit per source byte
(previously reported as ~4 BPT in Papers 1–6) is driven by training-data entropy modulated
by exploitable structure, not by architecture or thermodynamics.

The refined equation from Paper 7 onward: BPT ≈ source_entropy − f(structural_depth).

## 2. THE 3-SURFACE PUBLISHING MODEL

Every paper is published across three GitHub locations plus (where assigned) Zenodo:

1. **Windstorm-Institute/<slug>** — the publication repo. Contains paper.pdf, article.html,
   paper-aap-draft.md, paper-arxiv.tex, paper-entropy-draft.md, paper-rsif-draft.md,
   LICENSE (CC BY 4.0 for papers, MIT for code), README.md. Paper 7 also contains the
   complete working tree (it doubles as the conductor repo). Papers 7, 8, 9 also contain
   grandslam_supplementary.pdf.
2. **Windstorm-Labs/<slug>** — experimental code, data, plots. Pattern inconsistency: only
   Papers 2, 7, 8, 9 have actual experiment code. Papers 1, 3, 4, 5, 6 Labs repos contain
   just PDF + LICENSE + README (the original code wasn't migrated).
3. **sneakyfree/windstorminstitute.org** — public website. Deployed live at
   windstorminstitute.org. Structure: index.html (homepage), articles/<slug>.html (one
   long-form article per paper plus "speed-limit-of-thought.html" overview),
   assets/images/, assets/documents/ (empty — the two PDFs that were there were orphans),
   README.md.

Zenodo community: zenodo.org/communities/windstorm-institute/

## 3. THE NINE PAPERS

| # | Repo slug | Title | DOI | Status |
|---|-----------|-------|-----|--------|
| 1 | fons-constraint | The Fons Constraint | 10.5281/zenodo.19274048 | Published |
| 2 | receiver-limited-floor | The Receiver-Limited Floor | 10.5281/zenodo.19322973 | Published |
| 3 | throughput-basin | The Throughput Basin | 10.5281/zenodo.19323194 | Published |
| 4 | serial-decoding-basin | The Serial Decoding Basin τ | 10.5281/zenodo.19323423 | Published |
| 5 | dissipative-decoder | The Dissipative Decoder | 10.5281/zenodo.19433048 | Published |
| 6 | inherited-constraint | The Inherited Constraint | 10.5281/zenodo.19432911 | Published |
| 7 | throughput-basin-origin | The Throughput Basin Origin | 10.5281/zenodo.19498582 (v1.4) | Published, v1.5 in repo but not on Zenodo |
| 8 | vision-basin | The Vision Basin | Preprint, DOI pending | v2.1 in repo |
| 9 | hardware-basin | The Hardware Basin | Preprint, DOI pending | v2.1 in repo |

Paper 7's old name was `agi-extensions`. A REDIRECT repo still exists at
Windstorm-Institute/agi-extensions pointing to throughput-basin-origin. Do not delete it.

## 4. LOCAL WORKSTATION LAYOUT (Varon-1)

Primary working directory: /home/user1-gpu/agi-extensions
  - This IS the throughput-basin-origin repo (remote: Windstorm-Institute/throughput-basin-origin)
  - Contains ALL the experimental code, data, and papers

Paper repo clones:
  /home/user1-gpu/vision-basin-repo       (Windstorm-Institute/vision-basin)
  /home/user1-gpu/hardware-basin-repo     (Windstorm-Institute/hardware-basin)
  /home/user1-gpu/fons-constraint         (Windstorm-Institute/fons-constraint)
  (Not all Institute repos are cloned locally; clone as needed.)

Website repo clone: /tmp/wsi-site (remote: sneakyfree/windstorminstitute.org)
  Note: /tmp gets cleared on reboot. Re-clone if missing:
    git clone https://github.com/sneakyfree/windstorminstitute.org.git /tmp/wsi-site

Paper source drafts: /home/user1-gpu/agi-extensions/paper/
  - Paper7-v1.5-draft.md (latest Paper 7 source)
  - Paper8-v2.1-draft.md (latest Paper 8 source)
  - Paper9-v2.1-draft.md (latest Paper 9 source)
  - Compiled PDFs: Paper7-Throughput-Basin-Origin-v1.5.pdf,
    Paper8-Vision-Basin-v2.1.pdf, Paper9-Hardware-Basin-v2.1.pdf

Experiment outputs (all under /home/user1-gpu/agi-extensions/weekend_experiments/):
  - decisive_round/            6 experiments (Exp 1-6, April 15)
  - robust_round/              R1-R4 (April 15)
  - grandslam/                 GS1-GS4 (April 15-16) — the bulletproof round

Chipyard/Gemmini build (for future hardware simulation):
  /home/user1-gpu/hardware-lab/chipyard — built but cycle-accurate simulation not yet run

LJ Speech dataset: /home/user1-gpu/LJSpeech-1.1/wavs/ (13,100 real speech utterances)

## 5. HARDWARE ENVIRONMENT

- NVIDIA RTX 5090 (32 GB VRAM), driver 590.48.01, CUDA 13.1
- Intel Core Ultra 9 285K, 256 GB RAM, Ubuntu 24.04
- Python 3.12 with PyTorch 2.x, Transformers 4.x, BitsAndBytes 0.49.2
- GPU is shared — if another user starts a process, yield. The robust experiment scripts
  under weekend_experiments/ include automatic GPU-yield logic (check
  weekend_experiments/grandslam.py for the pattern).
- NF4 quantization requires loading libnvJitLink.so.13 via ctypes before importing
  bitsandbytes:
    import ctypes
    ctypes.CDLL("/home/user1-gpu/miniconda3/envs/qwen/lib/python3.13/site-packages/nvidia/cu13/lib/libnvJitLink.so.13")

## 6. WHAT WAS JUST COMPLETED (April 15-16, 2026)

### The Decisive Round (Exps 1-6)
6 experiments addressing the single weakest claim of each paper. Key findings:
- Exp 1: NF4 preserves structural bonus (6.20); symmetric INT4 destroys it (0.28)
- Exp 2: Multilingual τ — English 0.95, Python 0.07, DNA 2.02 bits/char (Pythia-410M)
- Exp 3: MAE loss tracks image complexity (pretrained evaluation)
- Exp 4: Mel entropy tracks audio complexity; real LJ Speech = 0.095 bits/frame on wav2vec2
- Exp 5: PCFG depth sweep with pretrained GPT-2
- Exp 6: Lloyd-Max INT3 fails end-to-end (BPT = 11.74 vs FP16 = 4.27)

### The Robust Round (R1-R4)
Publication-quality with multiple seeds and error bars:
- R1: PCFG trained from scratch at depths 0-6 × 3 seeds. Std ±0.001. Bonus salad→depth-0 = 2.75 bits
- R2: Multilingual τ across 7 corpora × 4 models (Pythia-160M/410M/1.4B, GPT-2-medium)
- R3: ConvAE on controlled-entropy images (weak — replaced by GS1)
- R4: Structural bonus across 3 models × 4 methods × 3 seeds = 36 data points

### The Grand Slam (GS1-GS4) — the bulletproof round
GS1: 112M ViT-MAE trained from scratch on 7 entropy levels × 3 seeds (the visual SYN-8).
     Welch t = 249,994, p = 1.6e-11, Cohen's d = 204,119.
GS2: Scale invariance — Pythia 160M/410M/1.4B × 7 corpora. BPT is scale-dependent;
     bits/char is approximately scale-invariant.
GS3: Pythia-1.4B structural bonus × 4 methods × 5 seeds.
     FP16 vs symmetric-INT4: Welch t = 633.74, p = 2.84e-15, Cohen's d = 400.81.
     (The most statistically decisive result in the entire series.)
GS4: The Killer Table — unified evidence table consolidating every claim with CIs and p-values.

### Papers updated (v1.5 / v2.1)
- Paper 7 v1.5: Added §3.10 R1, §3.11 GS2, §3.12 GS3. Compiled to paper.pdf.
- Paper 8 v2.1: Added §3.6 GS1 from-scratch ViT-MAE. Compiled to paper.pdf.
- Paper 9 v2.1: Added §3.7 GS3, §3.8 Lloyd-Max end-to-end failure, §3.9 R4. Compiled to paper.pdf.

### Grand Slam Supplementary Materials PDF
Created and pushed to all three Institute repos. Formal 7-page document with methodology,
CIs, Welch t-tests, Cohen's d. Also linked from all three article footers on the website.
File: grandslam_supplementary.pdf

### Ecosystem alignment pass (the very last thing we did)
Fixed inconsistencies found in deep-inspection of all three GitHub locations:
- Homepage Paper 8/9 arc-node and pub-card links fixed (were pointing at GitHub; now
  point at local articles/vision-basin.html and articles/hardware-basin.html like Papers 1-7).
- Paper 8/9 article footers now include Zenodo line ("DOI pending") and "Download the full
  paper (PDF)" link, matching Papers 1-7 pattern.
- Added 4 submission-format paper drafts (AAP, arXiv, Entropy, RSIF) to Papers 8 and 9
  Institute repos, matching Papers 1-6 pattern.
- Removed orphaned files from the website repo: Dissipative-Decoder-FINAL.pdf,
  Inherited-Constraint-FINAL.pdf, windstorm-institute-site.zip (no HTML referenced them).
- Wrote a proper README for the website repo.

## 7. OPEN ITEMS THAT REMAIN

1. **Upload Paper 7 v1.5 to Zenodo as a new version.** The current Zenodo record
   (10.5281/zenodo.19498582) is v1.4. v1.5 adds R1, GS2, GS3 but has not been uploaded.
   Requires Grant to upload via Zenodo web form.

2. **Create Zenodo records for Papers 8 and 9.** Neither has a DOI yet. The article
   footers on the website say "DOI pending". Requires Grant to create records via Zenodo
   web form.

3. **Migrate experiment code for Papers 1, 3, 4, 5, 6 to Labs repos.** These repos
   currently contain only PDF + LICENSE + README. The original experiment code from
   early 2026 was never migrated into them. Would require archaeology through
   agi-extensions git history and old scratch directories.

4. **Cross-architecture generalization experiments.** Paper 7 v1.5 Section 5 lists as
   open: state-space models, diffusion models, mixtures of experts. Not yet run.

5. **Non-Latin scripts for multilingual τ.** Paper 7 v1.5 lists as open: Chinese, Arabic,
   Devanagari. Chinese CC-100 loader is deprecated; needs alternative source.

6. **Chipyard/Gemmini cycle-accurate simulation.** Built but not run. Would give Paper 9
   a hardware-simulation result beyond the pure-arithmetic weight-quantization evidence.

7. **article.md inconsistency.** Papers 5 and 6 Institute repos have article.md files
   (markdown sources); Papers 1, 2, 3, 4, 7, 8, 9 do not. Low-priority — could add or
   remove to normalize. Current decision: leave alone.

## 8. GRANT'S PREFERENCES (read this carefully)

- **Sign and date every patient-file-style edit.** When you touch a paper or repo,
  commit with a clear message including co-author line:
    Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
- **Ask for confirmation before destructive or public actions** (force pushes, deletions,
  sending messages). Local editing and commits within established pattern are fine.
- **Answer honestly about scientific rigor.** Grant explicitly asked whether our
  experiments would survive peer review. Give him the real answer, not reassurance.
  When something is a "toy experiment," say so. When a result is bulletproof (Cohen's
  d = 400), say so.
- **Don't dumb things down or over-explain.** He's technically sharp.
- **"Proceed"** means continue the obvious next step. If ambiguous, ask briefly.
- **Skip preamble.** Go straight to action or answer.
- He runs a multi-agent Claude Code fleet (Dr. A Kit OC1 Alpha, Dr. B Herm Zero, Dr. C
  Opus 4.6 Opus-Claw) and operates as "The Windstorm" in that fleet context.

## 9. COMMON COMMANDS YOU'LL NEED

### Check GPU state
    nvidia-smi --query-gpu=memory.free,memory.used,memory.total,utilization.gpu --format=csv,noheader

### Clone the website repo if /tmp was cleared
    git clone https://github.com/sneakyfree/windstorminstitute.org.git /tmp/wsi-site

### Compile a paper markdown draft to PDF (xelatex handles Unicode)
    cd /home/user1-gpu/agi-extensions/paper
    pandoc PaperN-vX.Y-draft.md -o PaperN-Title-vX.Y.pdf \
      --pdf-engine=xelatex -V geometry:margin=1in -V fontsize=11pt \
      -V mainfont="DejaVu Serif"

### Sync an article to its Institute repo and push (3-repo pattern)
    cp /tmp/wsi-site/articles/<slug>.html /home/user1-gpu/<slug>-repo/article.html
    cd /tmp/wsi-site && git add -A && git commit -m "..." && git push
    cd /home/user1-gpu/<slug>-repo && git add article.html && git commit -m "..." && git push

### Run a GPU experiment in background with yield-to-other-user logic
    cd /home/user1-gpu/agi-extensions
    nohup python3 -u weekend_experiments/<script>.py > weekend_experiments/<name>_nohup.out 2>&1 &

### List all repos in either org
    gh repo list Windstorm-Institute --limit 30
    gh repo list Windstorm-Labs --limit 30

## 10. KEY REFERENCE FILES

Read any of these when you need ground truth on current state:

- /home/user1-gpu/agi-extensions/weekend_experiments/grandslam/gs4_killer_table/KILLER_TABLE.md
  — The unified evidence table. Every claim, every CI, every p-value.

- /home/user1-gpu/agi-extensions/grandslam_supplementary.pdf (and the same file in
  vision-basin-repo/ and hardware-basin-repo/) — The formal statistical addendum.

- /home/user1-gpu/agi-extensions/paper/Paper7-v1.5-draft.md
- /home/user1-gpu/agi-extensions/paper/Paper8-v2.1-draft.md
- /home/user1-gpu/agi-extensions/paper/Paper9-v2.1-draft.md
  — Latest markdown sources for Papers 7, 8, 9.

- /home/user1-gpu/agi-extensions/review/adversarial_review.md
  — Paper 7's internal adversarial review (4 blocking items identified, all now resolved).

- /home/user1-gpu/agi-extensions/CONDUCTOR_STATUS.md
  — Historical log of conductor-mode actions from earlier sessions.

## 11. MEMORY SYSTEM

Grant has an auto-memory directory at:
  /home/user1-gpu/.claude/projects/-home-user1-gpu/memory/

Key memory files:
  - MEMORY.md (the index, always loaded into context)
  - user_grant_whitmer.md
  - project_windy_word_overview.md
  - project_agent_fleet_naming.md
  - feedback_patient_file_signoff.md

Check MEMORY.md first. When you learn something new about Grant's preferences or the
project state, save it as a new memory file and add a line to MEMORY.md. Do not duplicate.

## 12. DEFAULT NEXT ACTIONS ON PICKUP

When Grant says "continue" or "proceed" at the start of a fresh session, the default
reasonable actions are (in order of plausibility):

1. Ask what he wants to tackle — open items from §7, a new experiment, or something else.
2. Check GPU state and running processes before starting anything compute-heavy.
3. If he references a result from "earlier," check the Killer Table and the
   grandslam/ directory first — most recent work lives there.
4. If he says "push to GitHub," remember there are 4 relevant remotes: the website, the
   Institute repo for each paper, the Labs repo for each paper. Sync intentionally.

## 13. PHILOSOPHICAL CONTEXT YOU SHOULD INTERNALIZE

The Windstorm Institute operates on radical transparency. Papers are published together
with their internal adversarial reviews. Blocking items are listed openly. Corrections
are made in new versions with clear changelogs rather than quietly patched. The τ ≈ 4.16
number from Papers 1-6 was later re-measured to τ ≈ 1 bit per source byte — and that
correction was published openly rather than buried.

Grant's stated value: the falsification attempt should arrive at the same time as the
claim it constrains, not six months later in someone else's reply paper.

Apply this when you're evaluating claims. Don't over-smooth. Don't under-caveat.

---

Acknowledge that you have read this briefing, then wait for Grant's first instruction.
