# Real LJ Speech Audio Experiment

## Data
- **LJ Speech:** 3,000 utterances (~5.5 hours) of single-speaker English
- **Music:** Synthetic piano (600 seconds)
- **Controls:** White noise (300s), near-silence (60s)

## Results

| Source | Real data? | Bits/mel_dim (mean±std) | Structural bonus | Duration |
|---|---|---|---|---|
| speech_ljspeech_REAL | YES | 1.8858±0.0013 | 1.8858 | 19920s |
| music_synthetic | no | 2.6811±0.0136 | 2.6811 | 600s |
| noise_control | no | 0.0000±0.0000 | 0.0000 | 300s |
| silence_control | no | 0.0000±0.0000 | 0.0000 | 60s |

## Key comparison

| Source | Bits/mel_dim |
|---|---|
| Real LJ Speech (this experiment) | 1.8858 |
| Synthetic formants (Exp 1) | 0.6300 |

## Controls
- Noise: should be ~0 (can't predict random)
- Silence: should be ~0 (nothing to predict)

## 3 seeds, 50K steps each, plateau slope verified.
