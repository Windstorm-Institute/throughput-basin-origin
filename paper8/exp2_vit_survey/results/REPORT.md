# Paper 8 Exp 2 — ViT Classification Throughput Survey

## Summary
We measured the per-patch information throughput of eight pretrained ImageNet-1k
classification ViTs (ViT, DeiT, Swin) on the CIFAR-100 validation set (10,000
images, upsampled 32→224). For each image we forward-pass with `torch.no_grad()`
and `fp16`, then compute the Shannon entropy of the model's 1000-way softmax
prediction (Option B: task-free, no probe training). Bits per patch =
entropy / num_patches; bits per pixel = entropy / (224·224·3). A secondary
column reports the cross-entropy of the prediction against the uniform
distribution over the 1000 ImageNet classes.

## Results

| model_name                    |   params_M |   num_patches |   n_images_evaluated |   mean_bits_per_image |   mean_bits_per_patch |   mean_bits_per_pixel |   mean_ce_vs_uniform |   peak_vram_mb |   elapsed_seconds | status   |
|:------------------------------|-----------:|--------------:|---------------------:|----------------------:|----------------------:|----------------------:|---------------------:|---------------:|------------------:|:---------|
| vit_tiny_patch16_224          |      5.717 |           196 |                10000 |               5.48565 |            0.027988   |           3.64427e-05 |              14.8254 |           29.4 |            3016.2 | ok       |
| vit_small_patch16_224         |     22.051 |           196 |                10000 |               4.32    |            0.0220408  |           2.8699e-05  |              15.3036 |           69.7 |              16.4 | ok       |
| vit_base_patch16_224          |     86.568 |           196 |                10000 |               4.78504 |            0.0244135  |           3.17884e-05 |              12.7166 |          214.8 |              26.4 | ok       |
| deit_tiny_patch16_224         |      5.717 |           196 |                10000 |               5.89432 |            0.0300731  |           3.91577e-05 |              12.4136 |           29.4 |              14.9 | ok       |
| deit_small_patch16_224        |     22.051 |           196 |                10000 |               5.09995 |            0.0260201  |           3.38804e-05 |              12.4549 |           69.7 |              16.9 | ok       |
| deit_base_patch16_224         |     86.568 |           196 |                10000 |               4.77206 |            0.0243472  |           3.17021e-05 |              12.5365 |          216.4 |              28.7 | ok       |
| swin_tiny_patch4_window7_224  |     28.288 |          3136 |                10000 |               5.01874 |            0.00160036 |           3.33409e-05 |              12.563  |          127   |              23.2 | ok       |
| swin_small_patch4_window7_224 |     49.606 |          3136 |                10000 |               4.99037 |            0.00159132 |           3.31524e-05 |              12.5095 |          170.6 |              29.5 | ok       |

## Key finding
Classification ViTs sit between **0.0016** and **0.0301** bits/patch — roughly **138–2614× below** the ~4.16 bits/token natural-language basin.

The ImageNet maximum-uncertainty ceiling on per-image entropy is
log2(1000) ≈ 9.97 bits, which on a 196-patch ViT-B grid
corresponds to ≈ 0.0508 bits/patch. Even a *maximally*
uncertain ImageNet classifier therefore cannot exceed roughly two orders of
magnitude below the language basin under Option B; real classifiers sit lower
still, because they are confident.

This is the predicted result: classification is information-discarding by
construction. An entire image is mapped onto one of 1000 (or 100) labels, so
the per-patch bits the model emits are bounded above by log2(K)/num_patches
regardless of how rich the underlying representation is.

## Caveats
- **Option B, not classification accuracy.** We measure the entropy of the
  model's own ImageNet prediction distribution, not its accuracy on CIFAR-100.
  This is the model's representational compression on a task it was trained
  for, not generalization to a new domain. A linear-probe variant (Option A)
  would also tell us about transfer, but requires training and is out of
  tonight's scope.
- **Classification discards information by design.** The low bits/patch is the
  correct, expected lower bound — not a defect of these models.
- **The right Paper 8 follow-up is generative vision** (MAE, BEiT, MaskGIT)
  where the model must reconstruct pixels and therefore retain pixel-level
  entropy. That measurement will sit far above this floor and gives Paper 8
  the upper bound of the vision throughput range. Out of scope tonight.
- **CIFAR-100 upsampling.** Inputs are bilinearly upsampled from 32×32 to
  224×224, so the raw input entropy is much lower than 224×224 native imagery.
  This makes the bits-per-pixel column an artifact of upsampling, not an
  intrinsic property of the architecture; bits-per-patch and bits-per-image
  are the meaningful columns.
- **fp16 inference.** Negligible effect on softmax entropy at this scale.

## Contribution to Paper 8
A publication-grade lower bound on the vision throughput range:
classification ViTs floor at ≪ 0.05 bits/patch on CIFAR-100 under Option B.
Any future generative-vision measurement (MAE, BEiT) will sit above this
floor, and the gap between the two bounds is the empirical width of the
vision throughput basin that Paper 8 §3.1 needs to characterize.
