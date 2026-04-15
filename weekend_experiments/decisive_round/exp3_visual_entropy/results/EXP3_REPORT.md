# Exp 3: Visual Entropy Tracking Curve

## Part A: MAE Reconstruction Loss vs Image Complexity

| Image Type | MAE Loss | Bits/Pixel Est | N Images |
|---|---|---|---|
| uniform_gray | 0.000022 | 0.0000 | 1024 |
| gradient | 0.000191 | 0.0000 | 1024 |
| blocks_4x4 | 1.796778 | 2.4698 | 1024 |
| blocks_16x16 | 2.604553 | 2.7376 | 1024 |
| noise_gaussian | 1.135447 | 2.1387 | 1024 |
| noise_uniform | 1.646955 | 2.4070 | 1024 |
| cifar10_real | 0.070993 | 0.1390 | 1024 |

## Part B: ViT Output Entropy

| Model | Image Type | Output Entropy (bits) |
|---|---|---|
| ViT-B/16 | uniform_gray | 8.2188 |
| ViT-B/16 | gradient | 8.3750 |
| ViT-B/16 | blocks_4x4 | 8.0547 |
| ViT-B/16 | blocks_16x16 | 7.8008 |
| ViT-B/16 | noise_gaussian | 7.4883 |
| ViT-B/16 | noise_uniform | 6.5625 |
| ViT-B/16 | cifar10_real | 3.5977 |
| ViT-L/16 | uniform_gray | 7.7969 |
| ViT-L/16 | gradient | 7.9727 |
| ViT-L/16 | blocks_4x4 | 8.5469 |
| ViT-L/16 | blocks_16x16 | 8.3906 |
| ViT-L/16 | noise_gaussian | 8.6719 |
| ViT-L/16 | noise_uniform | 8.5703 |
| ViT-L/16 | cifar10_real | 3.2715 |

## Part C: Patch-Size Sensitivity

| Effective Patch | MAE Loss | Bits/Pixel |
|---|---|---|
| 16x16 | 0.068724 | 0.1156 |
| 32x32 | 0.076523 | 0.1931 |
| 64x64 | 0.133007 | 0.5919 |

## Interpretation

The MAE reconstruction loss should increase monotonically with image complexity (harder to predict masked patches in complex images). If it does, the visual throughput basin reflects image entropy — confirming Paper 8's central claim.
