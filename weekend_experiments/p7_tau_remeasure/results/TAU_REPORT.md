# τ Re-measurement in Bits Per Source Byte

The original τ = 4.16 ± 0.19 was in BPT (tokenizer-dependent).
This re-measures in bits per source character and bits per source byte.

| Model | BPT | Bits/char | Bits/byte | Bytes/token |
|---|---|---|---|---|
| EleutherAI/pythia-160m | 5.0007 | 1.1215 | 1.1197 | 4.47 |
| EleutherAI/pythia-410m | 4.2688 | 0.9573 | 0.9558 | 4.47 |
| EleutherAI/pythia-1.4b | 3.8112 | 0.8547 | 0.8534 | 4.47 |
| gpt2-medium | 4.5169 | 1.0032 | 1.0016 | 4.51 |

**Mean bits/char: 0.9842**
**Mean bits/byte: 0.9826**

For comparison: τ (BPT) = 4.16 ± 0.19
τ (bits/char) = 0.98
τ (bits/byte) = 0.98
