# Exp 5: PCFG Depth Sweep (Fixed GPT-2 Tokenizer)

## Key Question
Does BPT decrease with grammatical depth? If yes, f(structural_depth) is real.

## Results

| Depth | BPT | Source Entropy | Tokens |
|---|---|---|---|
| 0 | 4.1175 | 3.90 | 100254 |
| 1 | 4.9311 | 3.89 | 100254 |
| 2 | 5.1223 | 3.88 | 100254 |
| 3 | 5.2089 | 3.88 | 100254 |
| 4 | 5.2420 | 3.87 | 100254 |
| 5 | 5.2672 | 3.87 | 100254 |
| 6 | 5.2731 | 3.87 | 100254 |

**Random word salad (no grammar):** BPT = 8.8713

## Structural Bonus

Depth 0 → Depth 6: BPT drops by -1.1556 bits
Salad → Depth 6: BPT drops by 3.5981 bits

**Weak or no structural bonus detected.** The fixed GPT-2 tokenizer may absorb structure.
