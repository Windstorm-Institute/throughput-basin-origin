# Exp 4: Multiple Models Hardware Quantization

| Model | Arch | Matrix | INT5→4 deg | INT4→3 deg | Cliff ratio |
|---|---|---|---|---|---|
| pythia-160m | transformer | weight | 0.0667 | 0.2500 | 3.8× |
| pythia-160m | transformer | weight | 0.2321 | 0.4850 | 2.1× |
| pythia-160m | transformer | weight | 0.1061 | 0.3505 | 3.3× |
| pythia-1.4b | transformer | weight | 0.1492 | 0.4675 | 3.1× |
| pythia-1.4b | transformer | weight | 0.1206 | 0.5287 | 4.4× |
| pythia-1.4b | transformer | weight | 0.0720 | 0.2797 | 3.9× |
| gpt2-medium | transformer | weight | 0.1339 | 0.5131 | 3.8× |
| gpt2-medium | transformer | weight | 0.0819 | 0.0730 | 0.9× |
| gpt2-medium | transformer | weight | 0.0170 | 0.0777 | 4.6× |
| mamba-370m-hf | state-space | A_log | 0.0121 | 0.0275 | 2.3× |
| mamba-370m-hf | state-space | weight | 0.2521 | 0.5667 | 2.2× |
| mamba-370m-hf | state-space | weight | 0.0816 | 0.2660 | 3.3× |
