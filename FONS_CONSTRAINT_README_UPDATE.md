# Suggested Update for fons-constraint README

## Current Status

The fons-constraint repository currently only documents Paper 1. To create proper cross-linking with Paper 7 (agi-extensions), here's the recommended README structure:

---

# The Fons Constraint

**Paper 1** — Information-Theoretic Convergence on Encoding Depth in Self-Replicating Systems

**Website:** https://windstorminstitute.org

**Published Research Paper (Zenodo):** [https://zenodo.org/records/19274048](https://zenodo.org/records/19274048)

**Windstorm Institute Community:** https://zenodo.org/communities/windstorm-institute/

**Experiments & Code:** https://github.com/Windstorm-Labs/fons-constraint

This repository contains:
- `article.html` — Accessible article from the website
- `paper.pdf` — Full academic paper

---

## Windstorm Institute Research Series

This paper is part of the ongoing Windstorm Institute research series investigating the throughput basin (τ ≈ 3-6 bits/event) convergence in serial decoding systems:

### Papers 1-6: Foundation

**Repository:** [github.com/sneakyfree/fons-constraint](https://github.com/sneakyfree/fons-constraint)

1. **Paper 1**: The Fons Constraint - Information-Theoretic Convergence (this repository)
2. **Paper 2**: Biological Serial Decoding - From DNA to Proteins
3. **Paper 3**: Technological Serial Decoding - From Telegraph to Transformers
4. **Paper 4**: The Ribosome Benchmark - φ = 1.02
5. **Paper 5**: Silicon Inefficiency - 10^9× Above Landauer
6. **Paper 6**: AI Language Models - Inherited 4.4 BPT Constraint

**Key findings:** Serial decoding systems across biology and technology converge to τ = 4.16 ± 0.19 bits/event. The ribosome operates at φ ≈ 1.02 (2% above thermodynamic minimum), while silicon systems operate ~10^9× above the Landauer floor.

### Paper 7: AGI Extensions ⭐ NEW

**Repository:** [github.com/sneakyfree/agi-extensions](https://github.com/sneakyfree/agi-extensions)

**Title:** The Throughput Basin Origin: Data-Driven Convergence in Serial Decoding Systems

**Authors:** Whitmer III, Grant Lavell; Claude Sonnet 4.5

**Year:** 2026

**Status:** ✅ Published

**The Answer:** Through 4 major autonomous experiments, Paper 7 definitively proves the throughput basin is **DATA-DRIVEN**, not architectural or physical.

**Key Results:**
- Models trained on 8-bit entropy data achieved ~9 BPT (not compressed to ~4)
- No difference between transformer and serial architectures (p=0.688)
- Universal INT4 quantization cliff across all model sizes
- GPUs operate at φ ≈ 10^16 (16 orders above Landauer limit)

**Impact:** The ~4 BPT basin in natural language reflects actual linguistic entropy (~3-4 bits/character), not a universal computational limit. Different data → different basin.

**Citation:**
```bibtex
@techreport{whitmer2026basin,
  title={The Throughput Basin Origin: Data-Driven Convergence in Serial Decoding Systems},
  author={Whitmer III, Grant Lavell and Claude Sonnet 4.5},
  institution={Windstorm Institute},
  year={2026},
  url={https://github.com/sneakyfree/agi-extensions}
}
```

---

## How to Update

You can either:

1. **Replace the entire README** with the version above
2. **Add just the "Research Series" section** after the current content
3. **Create a separate RESEARCH_SERIES.md** file and link to it from the main README

Recommendation: Option 2 (add the Research Series section) maintains the current Paper 1 focus while providing context and cross-links.
