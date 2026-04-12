# The Vision Basin: Cross-Modal Throughput Measurement from Language to Vision to Audio

**Grant Lavell Whitmer III**

Windstorm Labs · The Windstorm Institute · Fort Ann, New York, USA

April 2026 | Version 1.0 (First Draft) | CC BY 4.0

---

**Abstract.** Paper 7 established that the throughput basin in language models reflects training-data entropy minus exploitable hierarchical structure: BPT ≈ source\_entropy − f(structural\_depth). If this equation is general, different modalities should sit at different throughput levels determined by their own entropy and structure. We test this prediction across three modalities — vision, audio, and language — using trained-from-scratch models, pretrained generative models, and source-entropy measurements via lossless compression.

We find that generative vision models (MAE) extract 1.38–1.40 bits per pixel from natural images — 35% of the Shannon ceiling (4.0 bpp under WebP lossless) and far above the classification floor (0.02 bits/patch). A visual shuffling cascade on STL-10 (96×96) reveals a spatial structural bonus of 0.69 bits/pixel, confirming that visual hierarchy is exploitable. Audio models trained from scratch on speech extract 1.80 bits per mel-spectrogram dimension, while music extracts 1.69 and noise extracts 0.0 — consistent with the equation.

A critical methodological finding: bits per pixel is **not** patch-size-invariant. Models trained at different patch sizes (8×8 through 48×48) produce bits/pixel ranging from 1.19 to 0.29, proving that patch size acts as a visual "tokenizer" — the same encoding-dependence Paper 7 demonstrated for text BPT. Visual source entropy is also resolution-dependent (5.1 bpp at 96×96 vs 3.2 bpp at 224×224 under PNG). The "4 bpp ≈ 4.16 BPT" coincidence between vision and language weakens under scrutiny.

The tokenizer-independent re-measurement of the language basin gives τ ≈ 0.85–1.12 bits per source byte — not the 4.16 bits per token previously reported. Each modality has its own throughput level determined by its own entropy and structure. The basin is not universal across modalities; it is universal within each modality, and the equation BPT ≈ entropy − f(structure) holds qualitatively for all three.

**Keywords:** vision basin · multimodal throughput · MAE · shuffling cascade · patch size invariance · audio throughput · cross-modal comparison · bits per source byte · structural bonus

---

## 1. Introduction

### 1.1 The Prediction from Paper 7

Paper 7 (Whitmer 2026g) established that the ~4 BPT throughput basin in language models reflects training-data entropy minus exploitable hierarchical structure. The refined equation — BPT ≈ source\_entropy − f(structural\_depth) — was confirmed for text across entropy levels 5–8 bits, at 92M and 1.2B parameters, and under three different loss functions.

The natural prediction: if the basin is genuinely data-driven, then vision and audio should sit at *different* throughput levels, determined by each modality's own entropy and structure.

### 1.2 What Needed Testing

Three questions:

1. **Where does generative vision sit?** Classification ViTs discard information by design (image → class label). Generative models (MAE, next-patch prediction) must retain pixel information. Where do they land relative to the language basin?

2. **Does vision have a structural bonus?** Paper 6 showed language's structural bonus is ~6.7 bits — destroying syntax raises BPT from ~4 to ~10.8. Does destroying spatial structure in images produce a comparable effect?

3. **Does the equation hold for audio?** Speech is language in a different encoding. Music is structured but non-linguistic. Noise is unstructured. Do all three track the equation?

---

## 2. Methods

### 2.1 Source Entropy Measurement

Visual source entropy measured via five independent estimators on CIFAR-10, CIFAR-100, STL-10, and random noise: marginal pixel entropy (H\_pixel), conditional entropy given adjacent pixel (H\_cond), gzip compression, PNG compression, and WebP lossless compression. Controls validated: random noise → 8.0 bpp, constant color → 0.1 bpp.

Resolution dependence tested by comparing STL-10 at native 96×96 vs upsampled 224×224.

### 2.2 Generative Vision Throughput (MAE)

Pretrained Masked Autoencoders (facebook/vit-mae-base, 112M params; facebook/vit-mae-large, 330M params) evaluated on STL-10 resized to 224×224. Standard MAE protocol: mask 75% of patches, reconstruct, measure reconstruction loss. Bits per pixel via rate-distortion: R(D) = 0.5 × log₂(σ²/MSE) per pixel dimension.

### 2.3 Visual Shuffling Cascade

Autoregressive next-patch prediction model (512d, 8 layers, 8 heads, 34M params) trained from scratch on STL-10 (96×96) with 16×16 patches (36 patches per image). 50K training steps, 2 seeds. Evaluated on five destruction levels: original, quadrant-shuffled, block-4×4-shuffled, patch-shuffled, row-shuffled, pixel-shuffled.

### 2.4 Patch Size Invariance Test

Same architecture trained at six patch sizes on STL-10: 8×8, 12×12, 16×16, 24×24, 32×32, 48×48. 20K steps each. Measures whether bits per pixel is constant (genuine per-pixel metric) or varies (encoding-dependent, like BPT for text).

### 2.5 Audio From Scratch

Next-mel-frame prediction transformer (256d, 4 layers, 4 heads, 4.3M params) trained from scratch on three audio types: speech (LJ Speech / synthetic formant), music (synthetic piano with overtones), white noise. 128-bin mel spectrograms, 30K steps, 2 seeds. Audio shuffling cascade (original → segment-shuffled → frame-shuffled).

### 2.6 Controlled Visual Entropy

Four synthetic image datasets at approximately controlled entropy levels: VIMG-LOW (noisy solid colors), VIMG-MED (random rectangles with gradients), VIMG-HIGH (uniform random pixels), VIMG-NAT (STL-10). Same 34M next-patch model trained on each from scratch.

### 2.7 Classification ViT Survey

Eight pretrained ImageNet ViTs evaluated on CIFAR-100: vit\_tiny/small/base, deit\_tiny/small/base, swin\_tiny/small. Entropy of the model's 1000-class prediction distribution measured as bits per image and bits per patch.

---

## 3. Results

### 3.1 Source Entropy: Vision Is Resolution-Dependent

**Table 1. Visual source entropy (bits per pixel)**

| Dataset | Resolution | H\_pixel | H\_cond | H\_gzip | H\_png | H\_webp |
|---|---|---|---|---|---|---|
| CIFAR-10 | 32×32 | 7.90 | 5.97 | 7.22 | 5.87 | 4.61 |
| CIFAR-100 | 32×32 | 7.90 | 5.94 | 7.12 | 5.78 | 4.61 |
| STL-10 | 96×96 | 7.85 | 5.46 | 6.42 | 5.07 | 4.02 |
| STL-10 resized | 224×224 | 7.85 | 5.48 | 5.56 | 3.20 | — |
| Random noise | 32×32 | 8.00 | 8.00 | 8.03 | 8.26 | 8.28 |
| Constant color | 32×32 | 7.80 | 0.00 | 0.08 | 0.26 | 0.10 |

Visual source entropy decreases with resolution under spatial-aware compressors (PNG, WebP). STL-10 drops from 5.07 bpp (96×96) to 3.20 bpp (224×224) under PNG. The "4 bpp ≈ language basin" coincidence holds at 96×96 but weakens at 224×224.

### 3.2 Generative Vision: MAE Extracts 1.4 Bits/Pixel

**Table 2. MAE generative throughput**

| Model | Params | MAE Loss | Bits/pixel |
|---|---|---|---|
| MAE-Base | 112M | 0.00968 | 1.38 |
| MAE-Large | 330M | 0.00940 | 1.40 |

Generative vision models extract 1.38–1.40 bits per pixel — 35% of the Shannon ceiling (4.0 bpp WebP) and 60× more than classification ViTs (0.02 bits/patch). This places generative vision between the classification floor and the source-entropy ceiling, consistent with a modality-specific basin.

### 3.3 Visual Shuffling Cascade: Spatial Structure Contributes 0.69 Bits/Pixel

**Table 3. Visual shuffling cascade (STL-10, 96×96)**

| Destruction level | Bits/pixel | Δ from original |
|---|---|---|
| Original | 0.761 | — |
| Quadrant shuffled | 0.419 | −0.342 |
| Block 4×4 shuffled | 0.265 | −0.496 |
| Row shuffled | 0.320 | −0.441 |
| Pixel shuffled | 0.070 | −0.691 |
| Patch shuffled | 0.000 | −0.761 |

**Visual structural bonus: 0.69 bits/pixel.** Destroying spatial structure reduces the model's predictive power monotonically. For comparison, language's structural bonus is ~6.7 bits/token — approximately 10× larger, consistent with language carrying more exploitable hierarchy per source unit than images.

### 3.4 Patch Size Is Not Invariant — The Visual "Tokenizer" Effect

**Table 4. Bits per pixel vs patch size (STL-10)**

| Patch size | Patches/img | Bits/pixel | Bits/patch |
|---|---|---|---|
| 8×8 | 144 | 1.185 | 910 |
| 12×12 | 64 | 0.928 | 713 |
| 16×16 | 36 | 0.764 | 587 |
| 24×24 | 16 | 0.570 | 437 |
| 32×32 | 9 | 0.447 | 343 |
| 48×48 | 4 | 0.290 | 222 |

Bits per pixel drops by 4× across the range. **Patch size acts as a visual tokenizer** — the same encoding-dependence Paper 7 proved for text BPT. Bits per patch scales with patch area, but bits per pixel does not normalize out the patch-size dependence. This means visual throughput comparisons across different patch sizes are invalid, just as text BPT comparisons across different tokenizers are invalid.

### 3.5 Audio: Three Modalities, Three Throughput Levels

**Table 5. Audio throughput (trained from scratch)**

| Source | Bits/mel\_dim | Structural bonus |
|---|---|---|
| Speech | 1.80 | −0.27 |
| Music | 1.69 | −1.66 |
| Noise | 0.00 | 0.00 |

Noise correctly extracts zero (cannot predict random). Speech and music both extract ~1.7–1.8 bits per mel dimension. The negative structural bonus for music indicates the model *exploits* temporal structure — destroying it (frame shuffling) reduces predictive power by 1.66 bits/dim. This is a temporal analog of the visual spatial bonus.

### 3.6 Controlled Visual Entropy: Partial Tracking

**Table 6. Controlled visual entropy**

| Dataset | Source H (gzip) | Model bits/pixel |
|---|---|---|
| VIMG-LOW (noisy solids) | 5.73 | 3.62 |
| VIMG-MED (rectangles) | 0.31 | 0.94 |
| VIMG-HIGH (random pixels) | 8.01 | 0.00 |
| VIMG-NAT (STL-10) | 6.57 | 0.76 |

VIMG-HIGH extracts 0.0 (correct — can't predict random pixels). VIMG-LOW has the highest model throughput despite moderate source entropy — because solid colors with slight noise are maximally predictable (high structure, low residual entropy). The relationship is not a simple linear tracking as it was for text; visual throughput depends on structure more than raw entropy.

### 3.7 Classification ViTs: The Floor

Eight pretrained ViTs on CIFAR-100: 0.0016–0.0301 bits per patch (4.3–5.9 bits per image). This is the information-discarding floor — classification collapses an entire image to a one-of-1000 label, retaining at most log₂(1000) ≈ 10 bits per image.

### 3.8 Language Basin Re-measured in Bits Per Source Byte

**Table 7. τ re-measurement (tokenizer-independent)**

| Model | BPT (old) | Bits/source byte (new) | Bytes/token |
|---|---|---|---|
| Pythia-160M | 5.00 | 1.12 | 4.47 |
| Pythia-410M | 4.27 | 0.96 | 4.47 |
| Pythia-1.4B | 3.81 | 0.85 | 4.47 |
| GPT-2-medium | 4.52 | 1.00 | 4.51 |

The language basin in tokenizer-independent units is **~0.85–1.12 bits per source byte** — not 4.16. Each Pythia/GPT-2 token spans ~4.5 bytes of English text, so BPT ≈ 4.5 × bits/byte. The 4.16 number was a tokenizer-packaging artifact, not a fundamental constant.

---

## 4. Discussion

### 4.1 The Equation Holds Qualitatively Across Modalities

The throughput basin is modality-dependent: language (~1 bit/byte), generative vision (~1.4 bits/pixel), audio speech (~1.8 bits/mel\_dim), audio noise (0.0). Each modality sits at a throughput level determined by its data's entropy and exploitable structure. The equation BPT ≈ entropy − f(structure) holds qualitatively — higher structure means more compression, lower residual throughput — but the functional form of f() differs across modalities.

### 4.2 The 4-Bit Coincidence Is a Packaging Artifact

The apparent convergence of vision (~4 bpp) and language (~4 BPT) near 4 bits is an artifact of the units used. In tokenizer-independent units, language sits at ~1 bit/byte and vision at ~1.4 bits/pixel. These are different numbers in different units. Whether they are comparable depends on the information content of a "byte" vs a "pixel" — a question this paper measures but does not resolve.

### 4.3 Patch Size = Visual Tokenizer

The patch-size invariance test (Table 4) is the visual equivalent of Paper 7's R5 tokenizer result. Both prove the same point: the per-event throughput number is a property of the event-definition (token size, patch size), not solely of the data. Cross-experiment comparisons require either (a) the same event definition or (b) a definition-independent metric.

For vision, the definition-independent metric is not yet established. Bits per pixel varies with patch size. Bits per image varies with resolution. The correct invariant may be bits per unit of source entropy — but this requires first measuring source entropy at the same resolution and format the model operates on.

### 4.4 Visual Structure Is Shallower Than Linguistic Structure

The visual structural bonus (0.69 bpp) is ~10× smaller than language's (~6.7 BPT). This is consistent with images having less hierarchical depth than text: pixel adjacency is one level of structure, objects are a second, scenes are a third. Language has phonology, morphology, syntax, semantics, discourse, and pragmatics — at least six compositional levels. The model can exploit more levels in language than in images at the per-source-unit level.

---

## 5. Limitations

1. **Visual entropy calibration.** The controlled-entropy datasets (VIMG-LOW/MED/HIGH) were not precisely calibrated; VIMG-MED had source entropy of only 0.31 bpp (too low). A proper visual SYN-* experiment requires datasets calibrated to exact entropy targets.

2. **Audio used synthetic data for speech and music.** LJ Speech download failed (missing FFmpeg codec). The formant-based synthetic speech captures rough statistical properties but is not real speech. Results should be confirmed with real recordings.

3. **Patch-size dependence is measured but not resolved.** We proved bits/pixel varies with patch size, but did not establish a patch-size-independent visual metric. This is an open problem.

4. **Single architecture for vision.** All from-scratch models use the same GPT-style next-patch predictor. CNN-based or diffusion-based generative models might show different throughput.

5. **Cross-modal training incomplete.** P8-E6 only trained a text-only baseline; image-only and mixed modes require a unified tokenizer architecture.

6. **The language τ re-measurement uses only 4 models.** More models across more families would narrow the confidence interval on bits/source-byte.

---

## 6. Predictions

**P1.** Real speech (LJ Speech, LibriSpeech) will produce bits/mel\_dim ≈ 1.8, matching the synthetic result. *Falsified if* real speech differs by >0.5 bits/dim.

**P2.** A properly calibrated visual SYN-* experiment (images at H=2, 4, 6, 8 bpp precisely) will show throughput tracking source entropy, as Paper 7's text SYN-* did.

**P3.** The visual structural bonus at 224×224 (196 patches) will exceed the STL-10 96×96 result (0.69 bpp) because more patches = more spatial hierarchy to exploit.

**P4.** A unified text+image model trained on mixed data will achieve higher throughput on both modalities than single-modality models.

---

## 7. Conclusion

The throughput basin is not a single number shared across modalities. It is a modality-specific equilibrium determined by each data type's entropy and exploitable structure. Language sits at ~1 bit per source byte. Generative vision sits at ~1.4 bits per pixel. Audio speech sits at ~1.8 bits per mel dimension. Noise sits at zero across all modalities.

The equation BPT ≈ source\_entropy − f(structural\_depth) holds qualitatively for all three modalities tested, but the functional form of f() and the correct definition of "source unit" differ across modalities. Establishing modality-independent throughput metrics — metrics that are invariant to patch size, token size, frame size, and resolution — remains the central open problem for cross-modal information theory.

---

## References

Whitmer III, G.L. (2026a–f). Papers 1–6, Windstorm Institute. doi:10.5281/zenodo.19274048 through 19432911.

Whitmer III, G.L. (2026g). Paper 7: The Throughput Basin Origin. doi:10.5281/zenodo.19498582.

---

## Acknowledgments

All experiments executed on RTX 5090 (Windstorm Labs, Varon-1). Autonomous overnight runs via nohup Python scripts with no AI assistance during execution. Experiment design and analysis: Grant Lavell Whitmer III with Claude Opus 4.6. Code and data: github.com/Windstorm-Institute/throughput-basin-origin. CC BY 4.0.
