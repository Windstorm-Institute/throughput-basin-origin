# R1: PCFG Depth Sweep — Trained From Scratch

25M-param transformer, 5 epochs, 3 seeds each.

| Depth | BPT (mean ± std) | N seeds |
|---|---|---|
| salad | 5.9681 ± 0.0018 | 3 |
| depth_0 | 3.2173 ± 0.0011 | 3 |
| depth_1 | 3.6392 ± 0.0015 | 3 |
| depth_2 | 3.7233 ± 0.0028 | 3 |
| depth_3 | 3.7571 ± 0.0008 | 3 |
| depth_4 | 3.7737 ± 0.0010 | 3 |
| depth_5 | 3.7839 ± 0.0008 | 3 |
| depth_6 | 3.7899 ± 0.0009 | 3 |

**Structural bonus (salad → depth-0):** 2.7508 bits
**Depth effect (depth-0 → depth-6):** -0.5726 bits

## Key difference from toy version
Models trained FROM SCRATCH on each depth's data — no pretrained bias.
If BPT decreases with depth, f(structural_depth) is real and learnable.
