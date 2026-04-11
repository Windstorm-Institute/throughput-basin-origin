# Paper 8 Experiment A: Visual Shuffling Cascade

## Question

Does the equation BPT ≈ source_entropy − f(structural_depth) hold for vision?
What is the visual structural bonus — how many bits per pixel does spatial
structure contribute to a model's predictive power?

## Method

- **Data:** CIFAR-100 (50K train, 10K eval, 32×32×3)
- **Architecture:** Autoregressive next-patch predictor (GPT-style transformer)
  - Patch size: 8×8 = 192 raw values per patch
  - 16 patches per image
  - 384d, 6 layers, 6 heads
  - 14,353,728 parameters
- **Training:** 30000 steps on ORIGINAL images only, 2 seeds
- **Evaluation:** Trained model evaluated on 5 destruction levels:
  1. Original (intact spatial structure)
  2. Quadrant-shuffled (4 quadrants permuted)
  3. Patch-shuffled (8×8 patches permuted)
  4. Row-shuffled (pixel rows permuted)
  5. Pixel-shuffled (all pixels permuted independently)
- **Metric:** Bits per pixel via rate-distortion: R(D) = 0.5 × log₂(σ²/MSE)

## Results

| Destruction level | Bits/pixel (mean±std) | Δ from original |
|---|---|---|
| original | 0.7613±0.0011 | +0.0000 |
| quadrant_shuffled | 0.3105±0.0021 | -0.4507 |
| patch_shuffled | 0.0000±0.0000 | -0.7613 |
| row_shuffled | 0.0043±0.0043 | -0.7570 |
| pixel_shuffled | 0.0000±0.0000 | -0.7613 |

## Structural Bonus

**Visual structural bonus: -0.7613 bits/pixel**

For comparison:
- Language structural bonus (Paper 6): ~6.7 bits/token
- PCFG-8 structural bonus (Paper 7 R6): ~5.3 bits/token

## Interpretation

The visual structural bonus of -0.761 bits/pixel is small,
suggesting that either (a) the model does not exploit spatial structure effectively
at this scale, (b) CIFAR-100's 32×32 resolution does not carry much spatial
hierarchy, or (c) the rate-distortion metric underestimates the structural
contribution. Higher-resolution datasets (ImageNet, 224×224) may show a larger
bonus.

## Files

- `results/shuffling_cascade.csv` — all measurements
- `results/training_curve_seed*.csv` — loss curves
- `plots/shuffling_cascade.png` — cascade plot
- `plots/delta_per_level.png` — per-level delta plot

*Experiment completed automatically. 2 seeds, 30000 training steps each.*
