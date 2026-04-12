# P9-E4: Real Pythia-410M Weights Through Hardware Quantization

## Cliff Analysis

| Matrix | INT4→INT3 degradation | INT5→INT4 degradation | Cliff ratio |
|---|---|---|---|
| attn_qkv | 0.271095 | 0.070214 | 3.9× |
| attn_dense | 0.458553 | 0.099365 | 4.6× |
| mlp_dense_h_to_4h | 0.083347 | 0.017128 | 4.9× |
| mlp_dense_4h_to_h | 0.530215 | 0.117685 | 4.5× |
