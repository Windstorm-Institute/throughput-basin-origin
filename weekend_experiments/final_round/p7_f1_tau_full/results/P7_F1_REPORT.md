# P7-F1: τ Re-measurement — 9 Models × 2 Corpora

| Model | Params | Wiki BPT | Wiki bits/byte | LAMBADA BPT | LAMBADA bits/byte |
|---|---|---|---|---|---|
| pythia-70m | 70M | 5.805 | 1.2997 | 6.536 | 1.5700 |
| pythia-160m | 162M | 5.001 | 1.1197 | 5.787 | 1.3901 |
| pythia-410m | 405M | 4.269 | 0.9558 | 5.115 | 1.2287 |
| pythia-1b | 1012M | 3.989 | 0.8931 | 4.913 | 1.1802 |
| pythia-1.4b | 1415M | 3.811 | 0.8534 | 4.775 | 1.1469 |
| gpt2 | 124M | 4.983 | 1.1051 | 5.709 | 1.3635 |
| gpt2-medium | 355M | 4.517 | 1.0016 | 5.288 | 1.2629 |
| gpt2-large | 774M | 4.304 | 0.9543 | 5.136 | 1.2267 |
| gpt2-xl | 1558M | 4.155 | 0.9213 | 5.034 | 1.2024 |

**WikiText-2 mean:** τ(bits/byte) = 1.0116 ± 0.1322
**LAMBADA mean:** τ(bits/byte) = 1.2857 ± 0.1258

**The 4.16 was BPT. The real basin is ~1 bit/byte, consistent across 9 models and 2 corpora.**
