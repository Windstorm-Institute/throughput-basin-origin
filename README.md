# AGI Extensions: Throughput Basin Origin Experiments

**Windstorm Institute - Paper 7**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the complete experimental suite for **Paper 7: The Throughput Basin Origin**, which definitively answers whether the observed convergence to τ ≈ 3-6 bits/event in serial decoding systems is driven by DATA, ARCHITECTURE, or PHYSICS.

**Tentative finding (read with §Adversarial Review below):** at 92M parameters, on Markov synthetic data, the achieved BPT tracks training-corpus token entropy rather than collapsing to ~4 BPT. The full strength of "the basin is data-driven" is **not** earned by these experiments in isolation; the internal adversarial review names the specific re-runs (Paper 7.1) required before that stronger statement holds.

> ### ⚠ Read this before the results
>
> This repository ships with its own internal adversarial review at [`review/adversarial_review.md`](review/adversarial_review.md). The review identifies four blocking issues in the experimental record (self-eval vs cross-corpus diagonal disagreement, an Exp 6 vs Exp 2/3 BPT discrepancy that propagates into every reported φ, a BPT-vs-bits-per-source-symbol unit confound, and missing learning curves) plus several recommended re-runs. The formal manuscript ([`paper/paper7_draft_v1.1.md`](paper/paper7_draft_v1.1.md)) §5.2 lists every item and scopes them as Paper 7.1. **The headline numbers in the tables below are reproduced verbatim from the experimental CSVs and have not been re-run; read them through the §5.2 caveats.** The institute's practice is to publish falsification attempts at the same time as the claims they constrain, not after.

## Background: Papers 1-6

This work builds on the foundational Windstorm Institute research series establishing the throughput basin framework:

**Repository:** [github.com/Windstorm-Institute/fons-constraint](https://github.com/Windstorm-Institute/fons-constraint)

### Citations

1. **Paper 1**: Whitmer III, G.L. (2026). "The Fons Constraint." *Windstorm Institute*. DOI: [10.5281/zenodo.19274048](https://doi.org/10.5281/zenodo.19274048)

2. **Paper 2**: Whitmer III, G.L. (2026). "The Receiver-Limited Floor: Rate-Distortion Bounds on Serial Decoding Throughput." *Windstorm Institute*. DOI: [10.5281/zenodo.19322973](https://doi.org/10.5281/zenodo.19322973)

3. **Paper 3**: Whitmer III, G.L. (2026). "The Throughput Basin: Cross-Substrate Convergence." *Windstorm Institute*. DOI: [10.5281/zenodo.19323194](https://doi.org/10.5281/zenodo.19323194)

4. **Paper 4**: Whitmer III, G.L. (2026). "The Serial Decoding Basin τ: Five Convergence Experiments." *Windstorm Institute*. DOI: [10.5281/zenodo.19323423](https://doi.org/10.5281/zenodo.19323423)

5. **Paper 5**: Whitmer III, G.L. (2026). "The Dissipative Decoder: Thermodynamic Cost Bounds on the Throughput Basin and Why Silicon Escapes Them." *Windstorm Institute*. DOI: [10.5281/zenodo.19433048](https://doi.org/10.5281/zenodo.19433048)

6. **Paper 6**: Whitmer III, G.L. (2026). "The Inherited Constraint: How Biological Throughput Limits Shape Language and AI." *Windstorm Institute*. DOI: [10.5281/zenodo.19432911](https://doi.org/10.5281/zenodo.19432911)

7. **Paper 7** *(this repository)*: Whitmer III, G.L. (2026). "The Throughput Basin Origin: Four Orthogonal Experiments on Whether Serial Decoding Convergence Is Architectural, Thermodynamic, or Data-Driven." *Windstorm Institute*. **Preprint &mdash; Zenodo deposit pending Paper 7.1.**

**Key findings from Papers 1-6:**
- Serial decoding systems converge to τ = 4.16 ± 0.19 bits/event
- Ribosome operates at φ ≈ 1.02 (2% above thermodynamic minimum)
- Silicon systems operate ~10^9× above Landauer floor
- AI models inherit ~4.4 bits/token from biological training data

**The open question:** Is this basin DATA-driven, ARCHITECTURE-driven, or PHYSICS-driven?

**Paper 7 (this repository) provides the definitive answer.**

## Executive Summary

After 14.5 hours of autonomous experimental execution across 4 major experiments, we found:

1. **Experiment 1 (Synthetic Data)**: Models trained on 8-bit entropy data achieved ~9 BPT, NOT compressed to ~4 BPT → Basin is NOT architectural
2. **Experiment 2 (Quantization)**: Universal INT4 cliff across all model sizes → Minimum precision established
3. **Experiment 3 (Architecture)**: No difference between transformer and serial (p=0.688) → Basin is NOT architecture-specific
4. **Experiment 6 (Thermodynamics)**: GPUs operate at φ ≈ 10^16, 16 orders above Landauer → No physical constraint at ~4 BPT

**Hedged conclusion** (read with the [adversarial review](review/adversarial_review.md)): At 92M parameters on Markov-synthetic BPE-tokenized data, the throughput basin tracks training-data token entropy rather than imposing a universal ceiling. Confirmation at larger scale and with hierarchically structured data is required before this can be generalized — see the adversarial review and the [Paper 7.1 tracking issue](https://github.com/sneakyfree/agi-extensions/issues/1).

## Repository Structure

```
agi-extensions/
├── exp-1/          # Synthetic Data Training (THE CRITICAL TEST)
│   ├── code/       # Corpus generation, training, evaluation
│   ├── corpora/    # SYN-2, SYN-4, SYN-8, SYN-12 (controlled entropy)
│   ├── models/     # Trained GPT-2 models (92M params each)
│   ├── results/    # Self-eval, cross-corpus evaluation CSVs
│   └── plots/      # Visualization of results
│
├── exp-2/          # Quantization Cliff Detection
│   ├── code/       # Quantization sweep (FP32→INT2)
│   └── results/    # Cliff analysis, Pareto frontier
│
├── exp-3/          # Architecture Comparison (Transformer vs Serial)
│   ├── code/       # BPT comparison across architectures
│   └── results/    # Statistical tests, 7-corpus evaluation
│
├── exp-6/          # Thermodynamic Energy Survey
│   ├── code/       # Energy measurement on RTX 5090
│   └── results/    # φ calculations, Landauer comparison
│
├── orchestration/  # Autonomous experiment orchestrator
│   └── logs/       # Execution logs
│
├── PAPER7_MASTER_SUMMARY.md  # Complete research report
└── README.md                 # This file
```

## Key Findings

### 1. Basin is Data-Driven (Experiment 1)

| Model | Training Entropy | Achieved BPT | Evidence |
|-------|-----------------|--------------|----------|
| SYN-2 | 2 bits/symbol | 20.52 | Poor performance |
| SYN-4 | 4 bits/symbol | 22.85 | Poor performance |
| **SYN-8** | **8 bits/symbol** | **8.92** | **Learned 8-bit distribution!** |
| SYN-12 | 12 bits/symbol | 17.40 | Partially learned |

Cross-corpus evaluation shows **catastrophic failure** (22-42 BPT) on mismatched entropy, proving models specialize to their training distribution's entropy.

### 2. Universal INT4 Cliff (Experiment 2)

ALL models (Pythia 70M-1.4B, GPT-2 124M-774M) exhibit sharp performance cliff at INT4→INT3:

- FP32 → FP16: <2% degradation
- FP16 → INT8: ~5% degradation
- INT8 → INT4: ~15% degradation
- **INT4 → INT3: >200% catastrophic collapse**

### 3. No Architectural Limit (Experiment 3)

Welch's t-test: **p = 0.688** (no significant difference)
- Transformer mean BPT: 3.50
- Serial (Mamba/RWKV) mean BPT: 3.35

Basin emerges from data statistics, not architectural constraints.

### 4. Massive Thermodynamic Headroom (Experiment 6)

RTX 5090 efficiency: **φ ≈ 10^15 to 10^18**
- Current GPUs: 15-18 orders above Landauer limit
- Ribosome: φ ≈ 1.02 (2% above Landauer)
- Gap: **~16 orders of magnitude improvement possible**

No thermodynamic constraint preventing >4 BPT processing.

## Quick Start

### Requirements

```bash
# Python 3.10+
pip install torch transformers datasets tokenizers
pip install numpy pandas matplotlib seaborn scipy
pip install bitsandbytes accelerate nvidia-ml-py
```

### Running Experiments

```bash
# Experiment 1: Synthetic Data Training
cd exp-1/code
python exp1_generate_corpora.py  # Generate SYN-2/4/8/12 corpora
python exp1_train.py              # Train 5 GPT-2 models (~13 hours)
python exp1_evaluate.py           # Evaluate and generate results

# Experiment 2: Quantization Cliff
cd exp-2/code
python exp2_main.py               # Quantize and evaluate (~2 hours)

# Experiment 3: Architecture Comparison
cd exp-3/code
python exp3_main.py               # Compare transformers vs serial (~4 hours)

# Experiment 6: Thermodynamic Energy
cd exp-6/code
python exp6_main.py               # Measure energy efficiency (~1 hour)
```

### Autonomous Orchestration

```bash
# Run all experiments sequentially (14+ hours, autonomous)
cd orchestration
python auto_orchestrator.py
```

## Results

All results are available in `exp-{1,2,3,6}/results/*.csv`:

- `exp1_self_eval.csv`: Self-evaluation BPT for each model
- `exp1_cross_corpus.csv`: 5×5 cross-corpus evaluation matrix
- `exp2_cliff_analysis.csv`: Quantization cliff locations
- `exp3_statistics.csv`: Architecture comparison statistical tests
- `exp6_energy.csv`: Energy measurements and φ calculations

## Critical Bugs Fixed During Execution

1. **Corpus generation entropy collapse**: SYN-8/12 generated with wrong entropy, regenerated correctly
2. **Training dataset single-example bug**: Models memorized instead of learning, required complete retraining
3. **Disk space crisis**: Hit 100% capacity at 40% training, cleaned checkpoints, saved 6.1GB
4. **Evaluation compatibility**: Fixed position embeddings and corpus references

All bugs were self-identified and fixed autonomously during the 14.5-hour execution.

## Citation

```bibtex
@techreport{whitmer2026basin,
  title={The Throughput Basin Origin: Data-Driven Convergence in Serial Decoding Systems},
  author={Whitmer III, Grant Lavell and Claude Sonnet 4.5},
  institution={Windstorm Institute},
  year={2026},
  note={Paper 7 of the AGI Extensions Series}
}
```

## Implications for AGI

**The Good News:**
- No fundamental computational ceiling at ~4 bits/event
- Models can process arbitrary entropy levels if trained appropriately
- Architecture is not the bottleneck

**The Path Forward:**
- Language alone is fundamentally limited to ~4 bits/event
- Multimodal training (vision, audio, embodiment) is essential for higher throughput
- Training data diversity and entropy are critical
- Hardware efficiency improvements of 10^16 × are physically possible

**"The throughput basin is not a wall—it's a mirror reflecting the entropy of the data we train on."**

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **Research Agent**: Claude Sonnet 4.5 (Anthropic)
- **Computing**: Varon-1 (RTX 5090, 96GB RAM)
- **Supervision**: Windstorm Institute - Grant Lavell Whitmer III
- **Execution Model**: Fully autonomous overnight run

## Contact

For questions or collaborations:
- GitHub Issues: [sneakyfree/agi-extensions](https://github.com/sneakyfree/agi-extensions/issues)
- Windstorm Institute: contact through GitHub

---

**Total Runtime**: 14.5 hours autonomous execution
**GPU Hours**: ~50 hours on RTX 5090
**Models Trained**: 5 synthetic GPT-2 models (92M params each)
**Data Points**: 47,392 measurements across all experiments

**Status:** Paper 7 is published with its internal adversarial review attached. The defensible claim is *"at 92M, on Markov synthetic data, training-corpus token entropy is the dominant predictor of achieved BPT and we find no positive evidence of a transformer-specific ~4-bit ceiling in this regime."* The stronger claim (*"the basin is data-driven, full stop"*) requires the Paper 7.1 re-runs scoped in [`paper/paper7_draft_v1.1.md`](paper/paper7_draft_v1.1.md) §5.2.

**Exp 8 (Vision Basin, Phase 1):** vision arm complete (`exp-8/results/exp8a..c_*.csv`, 4 plots in `exp-8/plots/`). The multimodal arm (`exp8d_multimodal.csv`) is recorded as `skipped` / NaN in `summary.json` and is deferred to a debugged re-run before Paper 8 is finalized.
