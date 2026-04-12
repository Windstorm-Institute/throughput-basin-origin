# P8-E7: Patch Size Invariance

| Patch size | Patches/img | Bits/pixel | Bits/patch |
|---|---|---|---|
| 8×8 | 144 | 1.1850 | 910.06 |
| 12×12 | 64 | 0.9284 | 713.04 |
| 16×16 | 36 | 0.7644 | 587.07 |
| 24×24 | 16 | 0.5695 | 437.39 |
| 32×32 | 9 | 0.4470 | 343.29 |
| 48×48 | 4 | 0.2896 | 222.42 |

If bits/pixel is constant → it's a genuine per-pixel metric (like bits/source-byte for text).
If it varies → patch size acts like a tokenizer and the metric is encoding-dependent.
