# R3: Visual Entropy — Trained Autoencoder

ConvAE (128-dim bottleneck), 10 epochs, 3 seeds each.

| Entropy Level | Name | MSE (mean ± std) | Bits/Pixel (mean ± std) |
|---|---|---|---|
| 0 | uniform | 0.000093 ± 0.000124 | 6.4964 ± 4.1262 |
| 1 | 4-color | 0.062710 ± 0.000071 | 24.1316 ± 0.0024 |
| 2 | 16-color-blocks | 0.017782 ± 0.000253 | 21.4039 ± 0.0307 |
| 3 | 16-color-pixels | 0.078734 ± 0.000129 | 24.6240 ± 0.0036 |
| 4 | smooth+noise | 0.008751 ± 0.000035 | 19.8698 ± 0.0087 |
| 5 | gaussian-noise | 0.039036 ± 0.000094 | 23.1058 ± 0.0052 |
| 6 | uniform-noise | 0.083297 ± 0.000004 | 24.7460 ± 0.0001 |

## Key question
Does reconstruction loss increase monotonically with entropy level?
If yes: visual throughput tracks source entropy (Paper 8 confirmed).
