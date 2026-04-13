# P9-F1: Can NF4 Push the Cliff Below INT4?

| Model | FP16 | BNB INT8 | BNB NF4 | BNB FP4 | SYM INT8 | SYM INT4 | SYM INT3 |
|---|---|---|---|---|---|---|---|
| pythia-410m | 4.27 | — | — | — | 4.78 | 16.76 | 16.05 |
| pythia-1.4b | 3.81 | — | — | — | 3.79 | 16.87 | 15.75 |

## Key question: Does BNB NF4 survive while SYM INT4 doesn't?
If yes → the cliff is about level allocation, not bit count.
If both fail → the cliff is truly at 4 bits regardless of method.
