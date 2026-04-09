# Text to Add to fons-constraint README

Add this section to the fons-constraint repository README (github.com/sneakyfree/fons-constraint):

---

## Paper 7: AGI Extensions

**Repository:** [github.com/sneakyfree/agi-extensions](https://github.com/sneakyfree/agi-extensions)

**Title:** The Throughput Basin Origin: Data-Driven Convergence in Serial Decoding Systems

**Authors:** Whitmer III, Grant Lavell; Claude Sonnet 4.5

**Year:** 2026

**DOI:** [10.5281/zenodo.XXXXXX](https://doi.org/10.5281/zenodo.XXXXXX) *(to be assigned)*

### Summary

Paper 7 definitively answers the central question raised by Papers 1-6: **Is the throughput basin (τ ≈ 3-6 bits/event) driven by DATA, ARCHITECTURE, or PHYSICS?**

**The Answer: DATA-DRIVEN**

Through 4 major experiments conducted over 14.5 hours of autonomous execution:

1. **Experiment 1 (Synthetic Data)**: Models trained on 8-bit entropy data achieved ~9 BPT, NOT compressed to ~4 BPT → Basin is NOT architectural
2. **Experiment 2 (Quantization)**: Universal INT4 cliff across all model sizes → Minimum precision established
3. **Experiment 3 (Architecture)**: No difference between transformer and serial (p=0.688) → Basin is NOT architecture-specific
4. **Experiment 6 (Thermodynamics)**: GPUs operate at φ ≈ 10^16 (16 orders above Landauer) → No physical constraint at ~4 BPT

**Key Insight:** The ~4 BPT basin in natural language reflects the actual statistical properties of natural language itself (~3-4 bits/character), not a universal computational limit.

**Implication for AGI:** Different data → different basin. To build AGI with 10-100× higher throughput than language models, we need richer, higher-entropy training data from multimodal, embodied experience.

**Citation:**
```bibtex
@techreport{whitmer2026basin,
  title={The Throughput Basin Origin: Data-Driven Convergence in Serial Decoding Systems},
  author={Whitmer III, Grant Lavell and Claude Sonnet 4.5},
  institution={Windstorm Institute},
  year={2026},
  note={Paper 7 of the AGI Extensions Series},
  url={https://github.com/sneakyfree/agi-extensions}
}
```

---

**Suggested placement:** Add this after the Papers 1-6 section in your README, creating a new "Paper 7" subsection.
