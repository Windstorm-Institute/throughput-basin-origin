# Paper 7.1 R6 — PCFG Hierarchical Structure Experiment

## Question

**Is the throughput basin about ENTROPY or STRUCTURE?**

Paper 7 showed that flat Markov data at ~8-bit entropy yields ~8–9 BPT (data
tracks entropy). Natural language has BOTH ~4-bit entropy AND deep hierarchical
structure. If we construct data with 8-bit entropy AND hierarchical structure,
does BPT follow entropy (~8) or structure (~4)?

**Prediction table:**
- ~8 BPT → entropy dominates → data-driven hypothesis strengthened
- ~4 BPT → structure dominates → data-driven hypothesis needs revision
- 5–7 BPT → mixed → new theory needed

## Setup

**PCFG grammar:** 256 terminals (bytes 0x00–0xFF) distributed across 16 lexical
categories of 16 terminals each. ~60 non-terminals, 200+ production rules,
recursive nesting (S → CL CONJ S, NP → NP PP, ADJ_P → ADJ ADJ_P, etc.) with
depth bounded at 10. Design ensures near-uniform byte distribution for high
entropy while maintaining deep syntactic hierarchy.

| Property | Value |
|---|---|
| Empirical source entropy | **7.6478 bits/symbol** (target: 7.5–8.5) |
| Unique terminals used | 256/256 |
| Corpus size | 100M characters, 25M raw bytes |
| Format | Whitespace-separated "xHH" tokens (matching SYN-8) |
| Architecture | GPT-2 (768/12/12, ~92M params), BPE vocab 8192 |
| Training | 10,000 steps, batch 32, lr 3e-4, cosine decay, fp16 |
| Seeds | 42, 137 (2 seeds for PCFG/PCFG-SHUF; 1 for SYN-8 from exp-1) |
| GPU | RTX 5090 (shared with concurrent agents during training) |

**Controls:**
- *PCFG-8-SHUFFLED:* same byte stream, randomly permuted. Preserves entropy
  exactly (7.6482 bits/symbol), destroys all hierarchical structure.
- *SYN-8:* flat 8-bit Markov corpus from exp-1 (8.0 bits/symbol, no structure).

## Results

| Model | Source H | BPT (mean ± sd) | BPSS* | Structural Bonus |
|---|---|---|---|---|
| SYN-8 (flat) | 8.000 | 9.047 | 2.262 | 0.016 |
| PCFG-8-SHUF (flat, same bytes) | 7.648 | 7.955 ± 0.231 | 1.989 | −0.006 |
| **PCFG-8 (structured)** | **7.648** | **6.594 ± 0.010** | **1.648** | **5.332 ± 0.143** |

**Structural bonus** = BPT(model on shuffled held-out) − BPT(model on original
held-out). This measures how many bits of predictive advantage the model extracts
from hierarchical structure.

## Verdict

### **MIXED: Structure pulls BPT substantially below entropy, but not to the language basin.**

PCFG-8 lands at **6.59 BPT** — firmly between the source entropy ceiling (7.65)
and the natural-language basin (4.16). The structural bonus is **5.33 bits**,
close to natural language's ~6.7. This falsifies both extremes of the prediction
table:

- ❌ ~8 BPT (entropy dominates): **Falsified.** PCFG-8 is 1.05 bits below
  source entropy — the model exploits hierarchical structure to compress below
  the byte-level entropy floor.
- ❌ ~4 BPT (structure dominates): **Falsified.** PCFG-8 is 2.43 bits above
  the natural-language basin. The PCFG's recursive grammar, while deep, is
  shallower and more regular than natural language's full syntactic-semantic-
  pragmatic hierarchy.
- ✅ **5–7 BPT (mixed):** Confirmed at 6.59.

## Detailed Analysis

### 1. Structure exploitability is real and large

The 5.33-bit structural bonus proves that the GPT-2 architecture learns to
exploit PCFG hierarchy. When fed the same bytes in shuffled order, the PCFG-
trained model's BPT jumps from 6.59 to ~11.9 — a catastrophic loss of
predictive power. Structure is not cosmetic; it is load-bearing.

For comparison:
- Natural language structural bonus: ~6.7 bits (Paper 6)
- PCFG-8 structural bonus: ~5.3 bits
- PCFG-8-SHUF structural bonus: ~0 bits (as expected)
- SYN-8 structural bonus: ~0 bits (as expected)

### 2. PCFG-SHUF confirms the flat-entropy baseline

PCFG-8-SHUFFLED achieves BPT ≈ 7.96, close to its source entropy of 7.65. The
~0.3-bit gap could be tokenizer overhead (BPE on shuffled high-entropy data
is suboptimal). Importantly, PCFG-SHUF matches the pattern of SYN-8 (BPT near
or slightly above source entropy), confirming that byte-level entropy controls
the basin for structure-free data, regardless of the specific byte distribution.

The SYN-8 BPT of 9.05 vs source entropy 8.0 shows a similar ~1-bit overhead,
consistent across two independent flat corpora.

### 3. The basin is a function of BOTH entropy and structure

The data-driven hypothesis from Paper 7 stated: "the throughput basin is
data-driven." This is correct but incomplete. The basin is determined by an
interaction:

```
BPT ≈ source_entropy − f(structural_depth)
```

where `f(structural_depth)` is the bits-per-token reduction achievable by
exploiting hierarchical patterns. For natural language, f ≈ 4–5 bits. For
our PCFG, f ≈ 1–1.5 bits (in BPT terms; the structural bonus of 5.3 is
measured differently — it's the penalty of losing structure, not the gain
from having it relative to entropy).

More precisely:
- **Flat data:** BPT ≈ source_entropy (+ tokenizer overhead)
- **Structured data:** BPT ≈ source_entropy − Δ_structure
- **Natural language:** 4.16 ≈ 4.0 − 0 (entropy ≈ 4, structure fully exploited)
- **PCFG-8:** 6.59 ≈ 7.65 − 1.06 (entropy ≈ 8, partial structure exploitation)

### 4. Implications for the inherited constraint hypothesis

The data-driven hypothesis is **refined, not refuted:**

> The throughput basin depends on the data's *compressible information content*,
> which is a function of both raw entropy and exploitable hierarchical structure.
> Different data types (natural language, PCFG, flat random) produce different
> basins based on where they sit in the entropy × structure space.

The ~4 BPT language basin is not a universal constant — it reflects the specific
combination of ~4-bit/character entropy and deep recursive syntactic-semantic
structure in human language. A hypothetical data source with 8-bit entropy and
language-depth structure would likely yield a basin between our PCFG result (6.6)
and the language basin (4.2), depending on structural depth.

### 5. Convergence

Training curves show rapid initial descent followed by plateau. At 10,000 steps:
- PCFG models plateau by step ~2000 (loss essentially flat thereafter)
- PCFG-SHUF models show a longer descent phase, plateauing by ~6000
- Final slopes are on the order of 1e-6, indicating convergence

The 10k-step budget is sufficient for these corpora. The rapid convergence of
PCFG relative to PCFG-SHUF is itself informative: structured data is "easier"
for the model to learn.

## Files

| File | Description |
|---|---|
| `corpora/pcfg_corpus.{txt,bin}` | 100M-char structured corpus |
| `corpora/pcfg_shuffled.{txt,bin}` | Shuffled control |
| `results/r6_pcfg_results.csv` | Per-seed metrics (5 rows) |
| `results/r6_summary.csv` | Aggregated by model |
| `results/curve_*.csv` | Training loss every 100 steps |
| `r6_pcfg_comparison.png` | Main bar chart (Fig. R6-1) |
| `plots/r6_curves.png` | Convergence diagnostic |

## Caveats

1. **10k steps** (not 50k). Convergence verified but longer training might
   tighten the PCFG BPT slightly.
2. **SYN-8 single-seed** from exp-1 (no error bars). Established result.
3. **PCFG source entropy 7.65** (not 8.0). The 0.35-bit gap means PCFG-8 had
   a slight "easier" starting point than SYN-8. The structure effect (1.05 bits
   below entropy) substantially exceeds this gap.
4. **2 seeds** for PCFG variants. Low variance (σ = 0.01 for PCFG) suggests
   the result is robust, but a third seed would strengthen confidence.
5. **PCFG is a simple grammar.** Natural language has deeper, more varied
   hierarchical structure. The 6.59 BPT should be treated as an upper bound
   on what structure can achieve with this entropy level — richer grammars
   would likely push BPT lower.

## Next Steps

1. **Vary structural depth:** Build PCFGs with the same entropy but different
   recursion depths (max_depth 3, 5, 10, 20). Does BPT decrease monotonically
   with depth?
2. **Match entropy exactly:** Generate a PCFG at exactly 8.0 bits/symbol for
   direct comparison with SYN-8.
3. **Scale model:** Does the structure bonus grow or shrink with model size?
   Try the same experiment with a 400M-param GPT-2.
