# Paper 8 P8-A v2: Visual Shuffling Cascade (High-Resolution)

## Changes from v1
- STL-10 (96×96) instead of CIFAR-100 (32×32) — 9× more pixels
- 16×16 patches (36 per image) instead of 8×8 (16 per image)
- Larger model: 512d, 8L, 8H, ~33M params (was 384d, 6L, 6H, 14M)
- 50K steps (was 30K)
- Added block_4x4_shuffled level for finer spatial decomposition

## Results

| Level | Bits/pixel (mean) | Δ from original |
|---|---|---|
| original | 0.7612 | +0.0000 |
| quadrant_shuffled | 0.4186 | -0.3425 |
| block_4x4_shuffled | 0.2648 | -0.4963 |
| patch_shuffled | 0.0000 | -0.7612 |
| row_shuffled | 0.3201 | -0.4411 |
| pixel_shuffled | 0.0701 | -0.6911 |

## Structural Bonus: 0.6911 bits/pixel

Language structural bonus (Paper 6): ~6.7 bits/token
PCFG-8 structural bonus (Paper 7 R6): ~5.3 bits/token
