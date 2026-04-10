# TOST equivalence test: transformer vs serial WikiText BPT

**Pre-specified equivalence margin**: ±0.5 BPT.

Justification: 0.5 BPT is approximately half of one natural standard deviation
of the WikiText basin across the model panel (pooled SD = 0.443 BPT)
and is the smallest effect that would be scientifically meaningful — differences
smaller than this are dwarfed by run-to-run variability we have not characterised
and by inter-corpus variation already documented in Paper 7.

## Inputs
- Transformer (n=4): [3.9561403074377046, 3.370045447076563, 2.98119404933696, 3.674363932264079]
- Serial      (n=3): [3.8448585892316234, 3.299700415290053, 2.894202472808137]
- Means: transformer = 3.4954, serial = 3.3463
- Welch t = 0.4314, p = 0.6880

## TOST result
- Overall TOST p-value (max of two one-sided tests) = **0.1834**
- Conclusion at α=0.05: **CANNOT conclude equivalence within ±0.5 BPT**

## Honest power note
With n=4 vs n=3, even the equivalence test is severely underpowered. The
minimum effect size detectable at 80% power for a Welch-style two-sample test
at this n is Cohen's d ≈ 2.68, which corresponds to roughly
**±1.19 BPT** in raw units given the observed pooled SD.

In other words: even if we ran TOST honestly, this design could only have
"established equivalence" if the true difference were essentially zero AND we
were willing to call ±1.19 BPT a "small" margin — which it
is not. The current data therefore CANNOT support either "transformer ≠ serial"
or "transformer ≈ serial". The honest verdict is **insufficient data**, and
Paper 7's claim of "no architecture-specific bias" must be downgraded to
"no architecture-specific bias was detected, but the test had no power to
detect one of plausible size".
