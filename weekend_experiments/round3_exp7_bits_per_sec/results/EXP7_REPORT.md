# Exp 7: Cross-Modal Normalization

| Modality | Throughput | Unit | Bits/sec | Compression ratio |
|---|---|---|---|---|
| Language | 0.85 | bits/byte | 18 | 0.106 |
| Vision (next-patch) | 0.76 | bits/pixel | 630374 | 0.095 |
| Vision (MAE) | 1.39 | bits/pixel | 6277018 | 0.174 |
| Audio (speech) | 1.80 | bits/mel_dim | 9907 | 0.225 |
| Audio (noise) | 0.00 | bits/mel_dim | 0 | 0.000 |

Compression ratio = throughput / max entropy (8 bits). Higher = model extracts more.
