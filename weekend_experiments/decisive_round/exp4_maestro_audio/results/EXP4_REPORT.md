# Exp 4: Audio Throughput Across Complexity (Fixed)

## wav2vec2 Output Entropy (FP32)

| Audio Type | Entropy (bits/frame) | Frames |
|---|---|---|
| silence | 1.3091 | 1992 |
| pure_440hz | 2.9639 | 1992 |
| chord_3note | 0.6604 | 1992 |
| white_noise | 1.5196 | 1992 |
| pink_noise | 1.3302 | 1992 |
| am_sweep | 0.582 | 1992 |
| lj_speech_real | 0.0949 | 64287 |

## Mel-Spectrogram Entropy

| Audio Type | Entropy (bits/frame) | Frames |
|---|---|---|
| silence | 0.0 | 626 |
| pure_440hz | 5.3495 | 626 |
| chord_3note | 5.371 | 626 |
| white_noise | 6.3004 | 626 |
| pink_noise | 6.2769 | 626 |
| am_sweep | 5.2982 | 626 |
| lj_speech_real | 6.2553 | 28754 |

## Interpretation

If entropy increases with audio complexity (silence < tone < chord < speech < noise), the audio throughput basin is data-driven — confirming Paper 8.
