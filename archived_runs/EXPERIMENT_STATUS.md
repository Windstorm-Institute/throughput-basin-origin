# Windstorm Institute Paper 7 - Live Experiment Status

**Last Updated:** 2026-04-08 13:04 EDT
**Platform:** Veron 1 (RTX 5090, 32GB VRAM)

---

## Currently Running Experiments

### ✅ Experiment 3: Recurrent vs Transformer
- **Status:** COMPLETED
- **Runtime:** ~2.2 hours
- **Completion:** 2026-04-08 13:00:51
- **Results:** 5 CSV files, 5 PNG plots, SUMMARY.md generated
- **Key Finding:** **H0 SUPPORTED** - No architectural difference in BPT convergence
  - Both transformers and Mamba converge to ~3.4 bits
  - **SURPRISE:** Transformers 100-1000× more energy efficient than Mamba!
  - Evidence for DATA-DRIVEN basin constraint

### 🔄 Experiment 2: The Quantization Cliff
- **Status:** RUNNING (started 13:03:39)
- **Current:** Testing pythia-70m @ fp16
- **Progress:** 1/40 configurations (8 models × 5 precisions)
- **Est. Completion:** ~19:00-23:00 today (6-10 hours)
- **Will Test:**
  - Models: Pythia-70m/160m/410m/1b/1.4b, GPT2/medium/large
  - Precisions: FP16, INT8, INT4, INT3, INT2
- **Key Question:** At what precision does BPT collapse out of basin?

### 🔄 Experiment 1 Phase 1: Corpus Generation
- **Status:** RUNNING (started 13:04:16)
- **Current:** Generating SYN-2 corpus
- **Progress:** 1/5 corpora
- **Est. Completion:** ~15:00-16:00 today (~2-3 hours)
- **Will Generate:**
  - SYN-2: 100M tokens, 4-symbol, ~2 bits/event
  - SYN-4: 100M tokens, 16-symbol, ~4 bits/event
  - SYN-8: 50M tokens, 256-symbol, ~8 bits/event (KEY TEST)
  - SYN-12: 20M tokens, 4096-symbol, ~12 bits/event
  - SYN-MIX: Interleaved blocks

### ⏸️ Experiment 6: Energy Survey / Thermodynamic Roadmap
- **Status:** READY TO RUN (script complete)
- **Scheduled:** After Exp 2 completes
- **Est. Runtime:** 4-6 hours
- **Will Measure:**
  - Pythia-70m through 1.4b
  - Configs: FP32, FP16, INT8, INT4, compiled, batched
  - φ (efficiency vs Landauer), gap to ribosome
  - Thermodynamic roadmap for AGI hardware

### ⏸️ Experiment 1 Phases 2-4: Training and Evaluation
- **Status:** WAITING (Phase 1 must complete first)
- **Scheduled:** After corpus generation completes (~16:00)
- **Est. Runtime:**
  - Phase 2-3 (Training): 8-10 hours
  - Phase 4 (Evaluation): 2 hours
- **Est. Completion:** Tomorrow ~04:00-08:00
- **Critical Test:** Does SYN-8 model achieve 8 BPT or compress to ~4?

---

## Experiment Queue Timeline

```
Now (13:04)
│
├─ Exp 2 RUNNING ──────────────────────────────────► (~19:00-23:00)
│
├─ Exp 1 Phase 1 RUNNING ───────► (~15:00-16:00)
│                                  │
│                                  └─ Exp 1 Phase 2-3 Training ────────────────────────────► (Tomorrow ~02:00-04:00)
│                                                                                             │
└─ After Exp 2 completes: Launch Exp 6 ──────────► (Tomorrow ~01:00-05:00)                  │
                                                                                              │
                                                                                              └─ Exp 1 Phase 4 Eval ─► (~06:00-08:00)
                                                                                                                        │
                                                                                                                        └─ SYNTHESIS
```

**Total Est. Completion:** Tomorrow (2026-04-09) ~08:00-10:00 EDT

---

## Results Summary (Completed Experiments)

### Experiment 3 Results

| Metric | Transformer | Serial (Mamba) | Verdict |
|--------|-------------|----------------|---------|
| Mean BPT | 3.495 bits | 3.346 bits | No sig. diff (p=0.69) |
| BPT Variance | 10.283 | 10.439 | Identical (p=0.98) |
| Structural Bonus | 6.839±0.15 | 6.782±0.25 | Equal |
| Energy/Token | 0.12-1.67 mJ | 69-168 mJ | **Transformer 100-1000× better!** |

**Verdict:** H0 SUPPORTED - Basin is **data-driven**, not architectural.

---

## Key Pending Questions

1. **Quantization Cliff** (Exp 2): Does collapse occur at INT4, INT3, or INT2?
2. **Energy Limits** (Exp 6): How many OOMs from RTX 5090 to ribosome?
3. **THE BIG ONE** (Exp 1): Does training on 8-bit data produce 8-bit models?
   - If YES → **Data-driven basin CONFIRMED**
   - If NO (compresses to ~4) → **Architecture-driven basin**
   - Layer entropy probe will reveal compression mechanism

---

## Monitoring Commands

```bash
# Quick status check
/home/user1-gpu/agi-extensions/check_status.sh

# Check individual logs
tail -f /home/user1-gpu/agi-extensions/exp-2/exp2.log
tail -f /home/user1-gpu/agi-extensions/exp-1/exp1_corpus_gen.log

# GPU monitoring
watch -n 1 nvidia-smi

# Disk usage (corpora will be large)
du -sh /home/user1-gpu/agi-extensions/exp-1/corpora/
```

---

## Auto-Update Script

This file can be auto-regenerated with:
```bash
python3 /home/user1-gpu/agi-extensions/synthesize_results.py
```

---

**Status:** 2/4 experiments running, infrastructure 100% complete, on track for ~24 hour completion.
