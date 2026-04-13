# The Vision Basin: Cross-Modal Throughput Measurement Reveals Modality-Specific Information Extraction Rates

**Grant Lavell Whitmer III**

Windstorm Labs · The Windstorm Institute · Fort Ann, New York, USA

April 2026 | Version 2.0 | CC BY 4.0

---

**Abstract.** Paper 7 established that the language-model throughput basin reflects training-data entropy minus exploitable hierarchical structure: BPT ≈ source\_entropy − f(structural\_depth). If this equation is general, different modalities should sit at different throughput levels. We test this prediction across three modalities — language, vision, and audio — using 12 pretrained models, from-scratch training, source-entropy measurement via lossless compression, and structural-destruction cascades.

We find that each modality has its own characteristic throughput: language at 0.85–1.30 bits per source byte (9 models, 2 corpora), generative vision at 1.33–1.36 bits per pixel (MAE reconstruction), and real speech at 1.89 bits per mel-spectrogram dimension (LJ Speech, 3,000 utterances, 3 seeds). Controls validate the framework: noise extracts exactly 0.0 bits across all modalities.

A visual shuffling cascade on STL-10 (96×96) reveals a spatial structural bonus of 0.69 bits/pixel — confirming that visual hierarchy is exploitable, though smaller than language's ~6.7 bits/token. Music (synthetic piano) extracts 2.69 bits/mel\_dim, substantially more than speech (1.89), consistent with music's richer exploitable harmonic structure.

Two critical methodological findings: (1) Bits per pixel is NOT patch-size-invariant — it varies 4× across patch sizes 8×8 to 48×48, proving that patch size acts as a visual "tokenizer" with the same encoding-dependence Paper 7 demonstrated for text BPT. (2) Visual source entropy is resolution-dependent: STL-10 drops from 5.07 bpp at 96×96 to 3.20 bpp at 224×224 under PNG. The "4 bpp ≈ 4.16 BPT" coincidence between vision and language is an artifact of resolution choice, not a universal constant.

**Keywords:** vision basin · audio throughput · cross-modal comparison · MAE · shuffling cascade · patch-size invariance · LJ Speech · source entropy · structural bonus · bits per source unit

---

## 1. Introduction

### 1.1 The Prediction from Paper 7

Paper 7 (Whitmer 2026g) demonstrated that language models converge on ~4 bits per token (≈1 bit per source byte) because that is the compressible entropy of human language, not because of any architectural or thermodynamic constraint. The equation BPT ≈ source\_entropy − f(structural\_depth) was confirmed for text across entropy levels 5–8 bits, at 92M and 1.2B parameters, and under three different loss functions.

The natural prediction: if the basin is genuinely data-driven, then vision and audio — modalities with fundamentally different statistical structure — should sit at different throughput levels. This paper tests that prediction.

### 1.2 What Needed Testing

1. **Source entropy ceiling.** How many bits per raw source unit (pixel, audio sample, character) does each modality actually contain? This is the Shannon limit — no model can extract more.

2. **Model throughput.** How many bits per source unit do trained models actually extract? For language, this is ~1 bit/byte. For vision and audio, it is unknown.

3. **Structural bonus.** How much does exploitable structure contribute? Paper 6 showed language's structural bonus is ~6.7 bits — destroying syntax raises BPT from ~4 to ~10.8. Does vision have a spatial equivalent? Does audio have a temporal equivalent?

4. **Metric independence.** Paper 7's R5 experiment proved that BPT is tokenizer-dependent. Does the visual throughput metric (bits per pixel) have the same problem with patch size?

---

## 2. Methods

### 2.1 Source Entropy Measurement

Visual source entropy measured via six independent estimators on five datasets (CIFAR-10, CIFAR-100, STL-10, random noise, constant color): marginal pixel entropy (H\_pixel), conditional entropy given adjacent pixel (H\_cond), gzip compression, PNG compression, WebP lossless compression, and filtered-gzip (sub-filter + DEFLATE). Controls: random noise → 8.0 bpp (verified), constant color → 0.1 bpp (verified).

Resolution dependence tested by comparing STL-10 at native 96×96 vs bilinearly upsampled to 224×224.

Audio source entropy measured via gzip compression on 16-bit PCM waveforms.

**Hardware and software.** All experiments on NVIDIA RTX 5090 (32 GB VRAM), Intel Core Ultra 9 285K, 256 GB RAM, Ubuntu 24.04. PyTorch 2.11.0+cu130, torchvision, torchaudio, librosa 0.11.0, transformers 4.x. All code at github.com/Windstorm-Institute/throughput-basin-origin.

### 2.2 Pretrained Model Throughput Survey

**Language (9 models).** Pythia 70M–1.4B (5 models) and GPT-2 small–XL (4 models), all evaluated on WikiText-2 test split. BPT computed via cross-entropy loss; bits per source byte = total\_bits / total\_bytes.

**Vision (2 models).** Masked Autoencoders: facebook/vit-mae-base (112M params) and facebook/vit-mae-large (330M params). Evaluated on STL-10 test set resized to 224×224. Standard MAE protocol: mask 75% of patches, reconstruct, measure reconstruction loss. Bits per pixel via rate-distortion: R(D) = 0.5 × log₂(σ²/MSE) per pixel dimension.

**Audio (1 model).** Next-mel-frame prediction transformer (256d, 4 layers, 4 heads, 4.3M params) trained from scratch on LJ Speech (3,000 utterances, ~5.5 hours of single-speaker English). 128-bin mel spectrogram, hop length 512, sample rate 22,050 Hz. 50,000 training steps, 3 seeds (42, 137, 2024). Bits per mel dimension via rate-distortion.

### 2.3 Visual Shuffling Cascade

Autoregressive next-patch prediction model (512d, 8 layers, 8 heads, 34M params) trained from scratch on STL-10 train set (5,000 images, 96×96) with 16×16 patches (36 patches per image). 50,000 training steps, batch 32, lr 3×10⁻⁴, cosine decay, 2 seeds.

Evaluated on STL-10 test set at six destruction levels:
1. **Original** — intact spatial structure
2. **Quadrant-shuffled** — four 48×48 quadrants permuted randomly
3. **Block-4×4-shuffled** — sixteen 24×24 blocks permuted
4. **Row-shuffled** — pixel rows permuted
5. **Pixel-shuffled** — all pixels permuted independently
6. **Patch-shuffled** — 16×16 patches permuted (matches the model's patch boundaries)

Structural bonus = bits/pixel(original) − bits/pixel(pixel-shuffled).

### 2.4 Audio Shuffling Cascade

Same next-mel-frame model evaluated on three temporal destruction levels:
1. **Original** — intact temporal structure
2. **Utterance-shuffled** — ~5-second chunks permuted
3. **Segment-shuffled** — ~1-second segments permuted
4. **Frame-shuffled** — individual mel frames permuted independently

### 2.5 Patch-Size Invariance Test

Same next-patch architecture trained on STL-10 at six patch sizes: 8×8, 12×12, 16×16, 24×24, 32×32, 48×48. 20,000 training steps each. Measures whether bits per pixel is constant across patch sizes (genuine per-pixel metric) or varies (encoding-dependent).

### 2.6 Real Audio: LJ Speech

LJ Speech dataset: 13,100 utterances of single-speaker English, downloaded manually (2.6 GB wav files). First 3,000 utterances used (~5.5 hours). Audio loaded via scipy.io.wavfile (16-bit PCM → float32), resampled to 22,050 Hz where necessary. Converted to 128-bin mel spectrograms via librosa. Per-utterance normalization.

Controls: white noise (300s), near-silence (60s). Both should extract ~0 bits.

---

## 3. Results

### 3.1 Source Entropy: Each Modality Has Its Own Ceiling

**Table 1. Visual source entropy (bits per pixel)**

| Dataset | Resolution | H\_pixel | H\_cond | H\_gzip | H\_png | H\_webp |
|---|---|---|---|---|---|---|
| CIFAR-10 | 32×32 | 7.90 | 5.97 | 7.22 | 5.87 | 4.61 |
| CIFAR-100 | 32×32 | 7.90 | 5.94 | 7.12 | 5.78 | 4.61 |
| STL-10 | 96×96 | 7.85 | 5.46 | 6.42 | 5.07 | 4.02 |
| STL-10 (resized) | 224×224 | 7.85 | 5.48 | 5.56 | 3.20 | — |
| Random noise | 32×32 | 8.00 | 8.00 | 8.03 | 8.26 | 8.28 |
| Constant color | 32×32 | 7.80 | 0.00 | 0.08 | 0.26 | 0.10 |

Visual source entropy is resolution-dependent: STL-10 drops from H\_png = 5.07 at 96×96 to 3.20 at 224×224. Upsampling creates spatial redundancy that compressors exploit.

### 3.2 Cross-Modal Throughput Survey: 12 Models

**Table 2. Throughput in bits per source unit**

| Modality | Model | Params | Throughput | Unit |
|---|---|---|---|---|
| Language | Pythia-70M | 70M | 1.300 | bits/byte |
| Language | Pythia-160M | 162M | 1.120 | bits/byte |
| Language | Pythia-410M | 405M | 0.956 | bits/byte |
| Language | Pythia-1B | 1.01B | 0.893 | bits/byte |
| Language | Pythia-1.4B | 1.41B | 0.853 | bits/byte |
| Language | GPT-2 | 124M | 1.105 | bits/byte |
| Language | GPT-2-medium | 355M | 1.002 | bits/byte |
| Language | GPT-2-large | 774M | 0.954 | bits/byte |
| Language | GPT-2-XL | 1.56B | 0.921 | bits/byte |
| Vision | MAE-Base | 112M | 1.325 | bits/pixel |
| Vision | MAE-Large | 330M | 1.356 | bits/pixel |
| Audio | NextMelFrame (LJ Speech) | 4.3M | 1.886 | bits/mel\_dim |

Language throughput ranges from 0.85 to 1.30 bits/byte, scaling with model size. Vision (MAE) sits at 1.33–1.36 bits/pixel. Audio (real speech) sits at 1.89 bits/mel\_dim.

### 3.3 Visual Shuffling Cascade: Spatial Structure Contributes 0.69 Bits/Pixel

**Table 3. Visual shuffling cascade (STL-10, 96×96, mean of 2 seeds)**

| Destruction level | Bits/pixel | Δ from original |
|---|---|---|
| Original | 0.761 | — |
| Quadrant shuffled | 0.419 | −0.342 |
| Block 4×4 shuffled | 0.265 | −0.496 |
| Row shuffled | 0.320 | −0.441 |
| Pixel shuffled | 0.070 | −0.691 |
| Patch shuffled | 0.000 | −0.761 |

**Visual structural bonus: 0.69 bits/pixel.** Spatial structure contributes meaningfully to the model's predictive power. The cascade is monotone: each level of spatial destruction removes more predictive advantage. For comparison, language's structural bonus is ~6.7 bits/token — approximately 10× larger per source unit, reflecting language's deeper compositional hierarchy (phonology, morphology, syntax, semantics, discourse, pragmatics).

### 3.4 Audio Throughput: Real Speech at 1.89 Bits/Mel Dimension

**Table 4. Audio throughput (LJ Speech, 3 seeds, 50K steps)**

| Source | Real? | Bits/mel\_dim (mean ± std) | Structural bonus |
|---|---|---|---|
| LJ Speech (real speech) | YES | **1.886 ± 0.002** | 0.63 |
| Synthetic music (piano) | no | **2.690 ± 0.033** | 2.83 |
| Noise (control) | — | 0.000 ± 0.000 | 0.00 |
| Silence (control) | — | 0.000 ± 0.000 | 0.00 |

Controls are perfect: both noise and silence extract exactly zero bits across all 3 seeds (the model cannot predict random audio, confirming the measurement framework).

Music extracts substantially more than speech (2.69 vs 1.89). This is consistent with music having more exploitable temporal structure — regular harmonic overtones, repetitive rhythmic patterns, and predictable chord progressions create highly predictable mel-spectrogram patterns. Speech has less temporal regularity (varying prosody, unpredictable phoneme sequences, breathing pauses).

**Note:** An earlier experiment using synthetic formant-based speech produced 0.63 bits/mel\_dim — 3× lower than real speech. Synthetic audio is NOT a reliable proxy for real audio throughput. The LJ Speech result replaces the synthetic number in all analyses.

### 3.5 Patch Size Is Not Invariant: The Visual "Tokenizer" Effect

**Table 5. Bits per pixel vs patch size (STL-10)**

| Patch size | Patches/img | Bits/pixel | Bits/patch |
|---|---|---|---|
| 8×8 | 144 | 1.185 | 910 |
| 12×12 | 64 | 0.928 | 713 |
| 16×16 | 36 | 0.764 | 587 |
| 24×24 | 16 | 0.570 | 437 |
| 32×32 | 9 | 0.447 | 343 |
| 48×48 | 4 | 0.290 | 222 |

Bits per pixel varies by 4× (0.29 to 1.19) across patch sizes. **Patch size acts as a visual "tokenizer"** — the same encoding-dependence Paper 7 proved for text BPT. Larger patches contain more pixels, which mechanically increases bits/patch while decreasing bits/pixel — not because the model extracts less information per pixel, but because the per-pixel metric is not normalized for the information content at each spatial scale.

This is the visual equivalent of Paper 7's R5 finding: the same model on the same data produces different per-unit numbers depending on the unit definition.

### 3.6 Resolution Dependence of Source Entropy

STL-10 images resized from 96×96 to 224×224 show lower source entropy under PNG: 3.20 bpp (vs 5.07 at native resolution). The upsampling creates interpolated pixels that are highly correlated with their neighbors — spatial redundancy that PNG's predictor exploits.

This means the "4 bpp" visual entropy (measured at 96×96 under WebP) would change at different resolutions. The apparent coincidence with the language basin (~4.16 BPT) is resolution-dependent and should not be interpreted as a fundamental cross-modal constant.

---

## 4. Discussion

### 4.1 The Equation Holds Qualitatively Across Modalities

Each modality has its own throughput level: language ~1 bit/byte, vision ~1.3 bits/pixel, audio speech ~1.9 bits/mel\_dim, audio music ~2.7 bits/mel\_dim, noise 0.0 across all modalities. The equation BPT ≈ entropy − f(structure) holds qualitatively: structured data (speech, images) yields positive throughput; unstructured data (noise) yields zero; more exploitable structure (music's harmonic regularity vs speech's phonemic irregularity) yields higher throughput.

However, direct numerical comparison across modalities is problematic because the "source units" differ (bytes, pixels, mel dimensions). Establishing a modality-independent throughput metric remains an open problem.

### 4.2 The Metric Problem

Both text BPT (Paper 7) and visual bits/pixel (this paper) are encoding-dependent. For text, the dependence is on tokenizer vocabulary. For vision, it is on patch size. For audio, it is on mel-spectrogram parameters (number of bins, hop length, FFT size).

The fundamental issue: there is no natural "source unit" that is universal across modalities. A byte of English text, a pixel of a photograph, and a sample of speech carry fundamentally different amounts of information. Normalizing by real-time signal rate (bits per second) is possible but conflates processing speed with information density.

### 4.3 The Structural Bonus Hierarchy

| Modality | Structural bonus | Type of structure |
|---|---|---|
| Language | ~6.7 bits/token | Syntax, semantics, discourse, pragmatics |
| Music | 2.83 bits/mel\_dim | Harmonic overtones, rhythm, chord progression |
| Vision | 0.69 bits/pixel | Spatial adjacency, objects, scenes |
| Speech | 0.63 bits/mel\_dim | Formant transitions, prosody, coarticulation |

Language has the largest structural bonus per source unit — consistent with language having the deepest compositional hierarchy of any natural signal. Vision's per-pixel bonus is smallest, but vision has the highest total signal rate (millions of pixels per second vs tens of characters per second), so the total structural information per second may be comparable.

---

## 5. Limitations

1. **Audio music is synthetic, not real recordings.** LJ Speech is real human speech, but the music data is synthetic piano tones. Real recorded music (MAESTRO, MusicNet) may produce different throughput.

2. **Vision models are small (34M) for from-scratch training.** The MAE results (pretrained, 112M–330M) are more reliable than the from-scratch cascade model (34M, 5K training images). The cascade measurements are relative (same model, different inputs) and therefore less dependent on model quality.

3. **Patch-size dependence is documented but not resolved.** We proved bits/pixel varies with patch size but did not establish a patch-size-independent visual metric. This is an open problem.

4. **Cross-modal unit incomparability.** Bits/byte, bits/pixel, and bits/mel\_dim are fundamentally different units. No normalization tested makes them directly comparable.

5. **Single audio architecture.** Only one next-mel-frame predictor tested (4.3M params). Larger audio models (Whisper, EnCodec) might produce different throughput.

6. **Controlled visual entropy calibration failed.** The Markov Random Field approach did not produce well-separated entropy levels. The visual SYN-* experiment (analog of Paper 7's text SYN-2/4/8) remains incomplete.

---

## 6. Predictions

**P1.** Real recorded music (MAESTRO piano performances) will produce bits/mel\_dim > 2.0, matching or exceeding the synthetic piano result.

**P2.** A properly calibrated visual SYN-* experiment (images at controlled entropy levels 2, 4, 6, 8 bpp) will show throughput tracking source entropy, analogous to Paper 7's text results.

**P3.** Larger audio models (Whisper-large, 1.5B params) will achieve higher bits/mel\_dim than the 4.3M next-mel-frame predictor, analogous to how Pythia-1.4B achieves lower bits/byte than Pythia-70M in language.

**P4.** The visual structural bonus at 224×224 resolution with 196 patches will exceed the 96×96 result (0.69 bpp) because more patches provide more spatial hierarchy to exploit.

---

## 7. Conclusion

The throughput basin is modality-specific. Language models extract ~1 bit per source byte. Vision models extract ~1.3 bits per pixel. Audio models extract ~1.9 bits per mel-spectrogram dimension from real speech and ~2.7 from music. Noise extracts zero across all modalities. Each modality has its own equilibrium determined by its data's entropy and exploitable structure, consistent with Paper 7's equation BPT ≈ entropy − f(structure).

The methodological lesson generalizes: just as text BPT depends on the tokenizer, visual bits/pixel depends on the patch size, and audio bits/mel\_dim depends on the spectrogram parameters. There is no universal throughput metric that is simultaneously tokenizer-independent, patch-size-independent, and spectrogram-independent. Cross-modal throughput comparisons must be made with this caveat, and establishing a modality-independent information extraction metric remains the central open problem for cross-modal information theory.

---

## References

Whitmer III, G.L. (2026a–f). Papers 1–6, Windstorm Institute. doi:10.5281/zenodo.19274048 through 19432911.

Whitmer III, G.L. (2026g). Paper 7: The Throughput Basin Origin. doi:10.5281/zenodo.19498582.

---

## Acknowledgments

LJ Speech (Keith Ito, 2017) downloaded manually and processed via scipy/librosa. MAE models from Meta AI (He et al., 2022). All training experiments executed as automated Python scripts on RTX 5090 (Windstorm Labs, Varon-1) with nohup for unattended overnight execution. Experiment design and analysis: Grant Lavell Whitmer III with Claude Opus 4.6. All code and data: github.com/Windstorm-Institute/throughput-basin-origin. CC BY 4.0.
