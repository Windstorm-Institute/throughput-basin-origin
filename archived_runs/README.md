# Archived Runs — Operational Records

This directory holds operational records from Paper 7's autonomous execution phase (April 2026). These files were originally at the repo root but are archived here so the repo root reads as a publication artifact rather than a live lab notebook.

| File | Date | What it is |
|------|------|------------|
| `LIVE_STATUS.txt` | 2026-04-08 | Live status snapshot during the autonomous experiment run |
| `EXPERIMENT_STATUS.md` | 2026-04-08 | Per-experiment status snapshot during the autonomous run |
| `AUTONOMOUS_EXECUTION_REPORT.md` | 2026-04-08 | Detailed status report on the autonomous orchestration |
| `CONDUCTOR_STATUS.md` | 2026-04-09 | Conductor-mode operational record by the previous Claude Opus 4.6 (1M) session |
| `PAPER7_MASTER_SUMMARY.md` | 2026-04-09 | 700-line autonomous research execution report by Claude Sonnet 4.5 covering the full Paper 7 experimental program |

These are preserved (rather than deleted) because they are a record of an autonomous multi-agent research execution — methodologically interesting in their own right — and because the Conductor and Master Summary documents contain context about decisions made during the experimental phase.

The orchestration scripts that produced this work (`auto_orchestrator.sh`, `check_status.sh`, `run_all_experiments.sh`, `synthesize_results.py`) live in `../scripts/orchestration/`.

For the canonical scientific output of Paper 7, see the repo root: `paper.pdf`, `article.html`, `review/adversarial_review.md`, and `grandslam_supplementary.pdf`.
