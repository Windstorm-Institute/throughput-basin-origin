#!/usr/bin/env python3
"""
Paper 8 Experiment A: Visual Shuffling Cascade
The visual equivalent of Paper 6's text shuffling cascade.

Trains an autoregressive next-patch prediction model on natural images
from scratch, then evaluates on progressively destroyed versions to
measure the structural bonus at each spatial scale.

This directly measures f(visual_structure) in the equation:
    BPT ≈ source_entropy − f(structural_depth)

If the visual structural bonus is comparable to language's ~6.7 bits,
the equation holds across modalities.

Designed for quality, not speed. Full training, multiple seeds.
Saves incrementally. Auto-commits when done.
"""

import os, sys, time, math, csv, json, subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

OUTDIR = "/home/user1-gpu/agi-extensions/paper8/p8a_visual_shuffling_cascade"
REPO = "/home/user1-gpu/agi-extensions"
SEEDS = [42, 137]

# Model config — small but real transformer for next-patch prediction
PATCH_SIZE = 8          # 8×8 pixel patches
IMG_SIZE = 32           # CIFAR-100 is 32×32
PATCHES_PER_IMG = (IMG_SIZE // PATCH_SIZE) ** 2  # 16 patches per image
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 3          # 192 raw values per patch
EMBED_DIM = 384
N_LAYERS = 6
N_HEADS = 6
TOTAL_STEPS = 30000
BATCH_SIZE = 64
LR = 3e-4
EVAL_EVERY = 5000
LOG_EVERY = 200
WARMUP_STEPS = 1000

# Shuffling levels
SHUFFLE_LEVELS = [
    'original',
    'quadrant_shuffled',    # swap 4 quadrants randomly
    'patch_shuffled',       # shuffle 8×8 patches within image
    'row_shuffled',         # shuffle pixel rows
    'pixel_shuffled',       # shuffle all pixels independently
]

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(f"{OUTDIR}/run.log", "a") as f:
        f.write(line + "\n")

# =====================================================================
# Dataset
# =====================================================================

class PatchDataset(Dataset):
    """Convert images to sequences of patches for autoregressive prediction."""

    def __init__(self, images, patch_size, shuffle_mode='original'):
        """
        images: numpy array (N, H, W, C) uint8
        shuffle_mode: one of SHUFFLE_LEVELS
        """
        self.images = images.astype(np.float32) / 255.0  # normalize to [0,1]
        self.patch_size = patch_size
        self.shuffle_mode = shuffle_mode
        self.h_patches = images.shape[1] // patch_size
        self.w_patches = images.shape[2] // patch_size
        self.n_patches = self.h_patches * self.w_patches

    def __len__(self):
        return len(self.images)

    def _shuffle_image(self, img):
        """Apply the specified shuffle to a single image (H, W, C)."""
        if self.shuffle_mode == 'original':
            return img

        elif self.shuffle_mode == 'quadrant_shuffled':
            h, w = img.shape[:2]
            mh, mw = h // 2, w // 2
            quads = [img[:mh, :mw], img[:mh, mw:], img[mh:, :mw], img[mh:, mw:]]
            perm = np.random.permutation(4)
            result = np.zeros_like(img)
            positions = [(0, 0), (0, mw), (mh, 0), (mh, mw)]
            for i, p in enumerate(perm):
                r, c = positions[i]
                result[r:r+mh, c:c+mw] = quads[p]
            return result

        elif self.shuffle_mode == 'patch_shuffled':
            ps = self.patch_size
            patches = []
            for i in range(self.h_patches):
                for j in range(self.w_patches):
                    patches.append(img[i*ps:(i+1)*ps, j*ps:(j+1)*ps].copy())
            np.random.shuffle(patches)
            result = np.zeros_like(img)
            idx = 0
            for i in range(self.h_patches):
                for j in range(self.w_patches):
                    result[i*ps:(i+1)*ps, j*ps:(j+1)*ps] = patches[idx]
                    idx += 1
            return result

        elif self.shuffle_mode == 'row_shuffled':
            perm = np.random.permutation(img.shape[0])
            return img[perm]

        elif self.shuffle_mode == 'pixel_shuffled':
            flat = img.reshape(-1, img.shape[-1])
            perm = np.random.permutation(len(flat))
            return flat[perm].reshape(img.shape)

        return img

    def __getitem__(self, idx):
        img = self._shuffle_image(self.images[idx].copy())
        ps = self.patch_size
        patches = []
        for i in range(self.h_patches):
            for j in range(self.w_patches):
                patch = img[i*ps:(i+1)*ps, j*ps:(j+1)*ps]
                patches.append(patch.flatten())
        return torch.tensor(np.array(patches), dtype=torch.float32)

# =====================================================================
# Model: Autoregressive Patch Predictor
# =====================================================================

class PatchPredictor(nn.Module):
    """GPT-style transformer that predicts next patch from previous patches."""

    def __init__(self, patch_dim, embed_dim, n_layers, n_heads, n_patches):
        super().__init__()
        self.patch_dim = patch_dim
        self.embed_dim = embed_dim
        self.n_patches = n_patches

        # Input projection: raw patch pixels → embedding
        self.input_proj = nn.Linear(patch_dim, embed_dim)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, embed_dim) * 0.02)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output projection: embedding → predicted next patch pixels
        self.output_proj = nn.Linear(embed_dim, patch_dim)

        # Causal mask
        self.register_buffer('causal_mask',
            nn.Transformer.generate_square_subsequent_mask(n_patches))

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, patches):
        """
        patches: (B, N, patch_dim) — sequence of flattened patches
        Returns: (B, N-1, patch_dim) — predicted next patches
        """
        B, N, D = patches.shape

        # Embed patches + positional encoding
        x = self.input_proj(patches) + self.pos_embed[:, :N]

        # Dummy memory (decoder-only, self-attention via causal mask)
        memory = torch.zeros(B, 1, self.embed_dim, device=x.device)

        # Causal self-attention
        mask = self.causal_mask[:N, :N].to(x.device)
        x = self.transformer(x, memory, tgt_mask=mask)

        # Predict next patch from each position
        predictions = self.output_proj(x[:, :-1])  # predict patches 1..N from 0..N-1

        return predictions

# =====================================================================
# Training
# =====================================================================

def compute_bits_per_pixel(model, dataloader, device):
    """
    Compute bits per pixel using MSE → rate-distortion conversion.
    R(D) = 0.5 * log2(variance / MSE) per dimension, summed over patch dims.
    Also compute raw MSE for reference.
    """
    model.eval()
    total_mse = 0.0
    total_pixels = 0
    total_patches = 0

    # Compute global pixel variance from first few batches
    pixel_values = []

    with torch.no_grad():
        for batch_idx, patches in enumerate(dataloader):
            patches = patches.to(device)
            B, N, D = patches.shape

            if batch_idx < 10:
                pixel_values.append(patches.cpu().numpy().flatten())

            with autocast():
                predictions = model(patches)
                targets = patches[:, 1:]  # next patches
                mse = F.mse_loss(predictions, targets, reduction='sum')

            total_mse += mse.item()
            total_patches += B * (N - 1)
            total_pixels += B * (N - 1) * D

    avg_mse_per_dim = total_mse / total_pixels

    # Estimate pixel variance
    all_pixels = np.concatenate(pixel_values)
    pixel_var = np.var(all_pixels)

    if pixel_var < 1e-10:
        pixel_var = 1e-10

    # Rate-distortion: bits = 0.5 * log2(var/MSE) per dimension
    # Clamp to avoid negative bits (when MSE > var, model is worse than random)
    if avg_mse_per_dim < pixel_var:
        bits_per_dim = 0.5 * math.log2(pixel_var / avg_mse_per_dim)
    else:
        bits_per_dim = 0.0

    bits_per_patch = bits_per_dim * PATCH_DIM
    bits_per_pixel = bits_per_dim  # since each dim IS a pixel channel value

    model.train()

    return {
        'mse_per_dim': avg_mse_per_dim,
        'pixel_variance': pixel_var,
        'bits_per_dim': bits_per_dim,
        'bits_per_patch': bits_per_patch,
        'bits_per_pixel': bits_per_pixel,
        'total_patches': total_patches,
    }

def train_model(seed, train_images, eval_images):
    """Train one model from scratch on original images."""
    log(f"\n{'='*60}")
    log(f"Training next-patch predictor (seed={seed})")
    log(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create dataset with original images only for training
    train_dataset = PatchDataset(train_images, PATCH_SIZE, 'original')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True)

    # Build model
    model = PatchPredictor(PATCH_DIM, EMBED_DIM, N_LAYERS, N_HEADS, PATCHES_PER_IMG)
    n_params = sum(p.numel() for p in model.parameters())
    model = model.cuda()
    log(f"Model: {EMBED_DIM}d, {N_LAYERS}L, {N_HEADS}H, {n_params:,} params")
    log(f"Patches per image: {PATCHES_PER_IMG} ({PATCH_SIZE}×{PATCH_SIZE} patches)")
    log(f"Patch dim: {PATCH_DIM} (raw pixel values)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scaler = GradScaler()

    start_time = time.time()
    step = 0
    losses = []

    while step < TOTAL_STEPS:
        for patches in train_loader:
            if step >= TOTAL_STEPS:
                break
            step += 1

            # LR schedule
            if step < WARMUP_STEPS:
                lr = LR * step / WARMUP_STEPS
            else:
                progress = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
                lr = LR * 0.5 * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            patches = patches.cuda()
            optimizer.zero_grad()

            with autocast():
                predictions = model(patches)
                targets = patches[:, 1:]
                loss = F.mse_loss(predictions, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            losses.append((step, loss.item()))

            if step % LOG_EVERY == 0:
                log(f"  step {step}/{TOTAL_STEPS}, loss={loss.item():.6f}, lr={lr:.6f}")

            if step % EVAL_EVERY == 0:
                # Quick eval on original
                eval_dataset = PatchDataset(eval_images[:2000], PATCH_SIZE, 'original')
                eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE,
                                        num_workers=2, pin_memory=True)
                metrics = compute_bits_per_pixel(model, eval_loader, 'cuda')
                log(f"  EVAL step {step}: MSE={metrics['mse_per_dim']:.6f}, "
                    f"bits/pixel={metrics['bits_per_pixel']:.4f}")

    elapsed = time.time() - start_time
    log(f"Training complete: {elapsed/3600:.2f}h, {n_params:,} params")

    return model, losses, elapsed, n_params

def evaluate_shuffling_cascade(model, eval_images, seed):
    """Evaluate the trained model on all shuffling levels."""
    log(f"\nEvaluating shuffling cascade (seed={seed})...")
    results = []

    for level in SHUFFLE_LEVELS:
        np.random.seed(seed + hash(level) % 10000)  # deterministic shuffling

        dataset = PatchDataset(eval_images, PATCH_SIZE, level)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                           num_workers=2, pin_memory=True)

        metrics = compute_bits_per_pixel(model, loader, 'cuda')

        result = {
            'shuffle_level': level,
            'seed': seed,
            'mse_per_dim': metrics['mse_per_dim'],
            'pixel_variance': metrics['pixel_variance'],
            'bits_per_pixel': metrics['bits_per_pixel'],
            'bits_per_patch': metrics['bits_per_patch'],
            'total_patches': metrics['total_patches'],
        }
        results.append(result)

        log(f"  {level:25s}: MSE={metrics['mse_per_dim']:.6f}, "
            f"bits/pixel={metrics['bits_per_pixel']:.4f}, "
            f"bits/patch={metrics['bits_per_patch']:.2f}")

    return results

# =====================================================================
# Main
# =====================================================================

def main():
    os.makedirs(f"{OUTDIR}/results", exist_ok=True)
    os.makedirs(f"{OUTDIR}/plots", exist_ok=True)

    log("="*60)
    log("PAPER 8 EXPERIMENT A: VISUAL SHUFFLING CASCADE")
    log("="*60)

    # Load CIFAR-100
    log("\nLoading CIFAR-100...")
    from torchvision.datasets import CIFAR100

    train_set = CIFAR100(root='/tmp/cifar100', train=True, download=True)
    test_set = CIFAR100(root='/tmp/cifar100', train=False, download=True)

    train_images = np.array([np.array(img) for img, _ in train_set])  # (50000, 32, 32, 3)
    eval_images = np.array([np.array(img) for img, _ in test_set])    # (10000, 32, 32, 3)

    log(f"Train: {train_images.shape}, Eval: {eval_images.shape}")

    # Measure raw pixel entropy for reference
    pixel_var = np.var(train_images.astype(np.float32) / 255.0)
    max_bits_per_pixel = 0.5 * math.log2(pixel_var / (1.0 / (12 * 255**2)))  # vs uniform quantization noise
    log(f"Pixel variance: {pixel_var:.6f}")
    log(f"Theoretical max bits/pixel (vs quantization noise): {max_bits_per_pixel:.2f}")

    all_results = []

    for seed in SEEDS:
        # Train model on original images
        model, losses, elapsed, n_params = train_model(seed, train_images, eval_images)

        # Save training curve
        curve_path = f"{OUTDIR}/results/training_curve_seed{seed}.csv"
        with open(curve_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['step', 'loss'])
            w.writerows(losses)

        # Run the shuffling cascade
        cascade_results = evaluate_shuffling_cascade(model, eval_images, seed)

        # Compute structural bonus
        original_bpp = [r for r in cascade_results if r['shuffle_level'] == 'original'][0]['bits_per_pixel']
        pixel_shuf_bpp = [r for r in cascade_results if r['shuffle_level'] == 'pixel_shuffled'][0]['bits_per_pixel']
        structural_bonus = pixel_shuf_bpp - original_bpp

        for r in cascade_results:
            r['structural_bonus_total'] = structural_bonus
            r['delta_from_original'] = r['bits_per_pixel'] - original_bpp
            r['n_params'] = n_params
            r['train_time_hrs'] = elapsed / 3600
            all_results.append(r)

        log(f"\n  STRUCTURAL BONUS (seed={seed}): {structural_bonus:.4f} bits/pixel")
        log(f"  (Language structural bonus from Paper 6: ~6.7 bits/token)")

        # Save incrementally
        csv_path = f"{OUTDIR}/results/shuffling_cascade.csv"
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        with open(csv_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            if not file_exists:
                w.writeheader()
            for r in cascade_results:
                w.writerow(r)

        # Clean up GPU
        del model
        torch.cuda.empty_cache()

        # Git commit after each seed
        try:
            subprocess.run(['git', 'add', 'paper8/p8a_visual_shuffling_cascade/'],
                          cwd=REPO, capture_output=True)
            subprocess.run(['git', 'commit', '-m',
                f'Paper 8 P8-A: visual shuffling cascade seed={seed}\n\n'
                f'Structural bonus: {structural_bonus:.4f} bits/pixel\n'
                f'Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>'],
                cwd=REPO, capture_output=True)
            subprocess.run(['git', 'push'], cwd=REPO, capture_output=True)
            log(f"Git push: seed {seed} committed")
        except Exception as e:
            log(f"Git error: {e}")

    # =========================================================
    # Generate plots and report
    # =========================================================
    log("\nGenerating plots...")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.DataFrame(all_results)

        # Plot 1: Shuffling cascade (bits/pixel vs destruction level)
        fig, ax = plt.subplots(figsize=(10, 6))

        level_order = SHUFFLE_LEVELS
        level_labels = ['Original', 'Quadrants\nshuffled', 'Patches\nshuffled',
                       'Rows\nshuffled', 'Pixels\nshuffled']

        for seed in SEEDS:
            seed_data = df[df['seed'] == seed]
            bpp_values = [seed_data[seed_data['shuffle_level'] == l]['bits_per_pixel'].values[0]
                         for l in level_order]
            ax.plot(range(len(level_order)), bpp_values, 'o-',
                   label=f'Seed {seed}', markersize=8, linewidth=2)

        ax.set_xticks(range(len(level_order)))
        ax.set_xticklabels(level_labels, fontsize=10)
        ax.set_ylabel('Bits per pixel (rate-distortion)', fontsize=12)
        ax.set_xlabel('Destruction level', fontsize=12)
        ax.set_title('Visual Shuffling Cascade — CIFAR-100\n'
                     '(trained on original, evaluated on destroyed versions)', fontsize=13)
        ax.legend(fontsize=10)

        # Annotate structural bonus
        mean_bonus = df.groupby('seed')['structural_bonus_total'].first().mean()
        ax.annotate(f'Structural bonus: {mean_bonus:.3f} bits/pixel\n'
                   f'(Language: ~6.7 bits/token for comparison)',
                   xy=(0.98, 0.98), xycoords='axes fraction',
                   ha='right', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        ax.grid(True, alpha=0.3)
        fig.savefig(f"{OUTDIR}/plots/shuffling_cascade.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{OUTDIR}/plots/shuffling_cascade.pdf", bbox_inches='tight')
        plt.close(fig)
        log("Cascade plot saved")

        # Plot 2: Delta from original at each level
        fig, ax = plt.subplots(figsize=(10, 6))

        for seed in SEEDS:
            seed_data = df[df['seed'] == seed]
            deltas = [seed_data[seed_data['shuffle_level'] == l]['delta_from_original'].values[0]
                     for l in level_order]
            ax.bar([x + (0.35 if seed == SEEDS[1] else 0) for x in range(len(level_order))],
                  deltas, width=0.35, label=f'Seed {seed}', alpha=0.8)

        ax.set_xticks([x + 0.175 for x in range(len(level_order))])
        ax.set_xticklabels(level_labels, fontsize=10)
        ax.set_ylabel('Δ bits/pixel from original', fontsize=12)
        ax.set_title('Information lost at each destruction level', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        fig.savefig(f"{OUTDIR}/plots/delta_per_level.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        log("Delta plot saved")

    except Exception as e:
        log(f"Plot error: {e}")

    # =========================================================
    # Write report
    # =========================================================
    log("\nWriting report...")

    # Compute summary stats
    mean_by_level = {}
    for level in SHUFFLE_LEVELS:
        level_data = [r for r in all_results if r['shuffle_level'] == level]
        mean_bpp = np.mean([r['bits_per_pixel'] for r in level_data])
        std_bpp = np.std([r['bits_per_pixel'] for r in level_data]) if len(level_data) > 1 else 0
        mean_by_level[level] = (mean_bpp, std_bpp)

    original_bpp = mean_by_level['original'][0]
    pixel_shuf_bpp = mean_by_level['pixel_shuffled'][0]
    mean_bonus = pixel_shuf_bpp - original_bpp

    report = f"""# Paper 8 Experiment A: Visual Shuffling Cascade

## Question

Does the equation BPT ≈ source_entropy − f(structural_depth) hold for vision?
What is the visual structural bonus — how many bits per pixel does spatial
structure contribute to a model's predictive power?

## Method

- **Data:** CIFAR-100 (50K train, 10K eval, 32×32×3)
- **Architecture:** Autoregressive next-patch predictor (GPT-style transformer)
  - Patch size: {PATCH_SIZE}×{PATCH_SIZE} = {PATCH_DIM} raw values per patch
  - {PATCHES_PER_IMG} patches per image
  - {EMBED_DIM}d, {N_LAYERS} layers, {N_HEADS} heads
  - {all_results[0]['n_params']:,} parameters
- **Training:** {TOTAL_STEPS} steps on ORIGINAL images only, {len(SEEDS)} seeds
- **Evaluation:** Trained model evaluated on 5 destruction levels:
  1. Original (intact spatial structure)
  2. Quadrant-shuffled (4 quadrants permuted)
  3. Patch-shuffled ({PATCH_SIZE}×{PATCH_SIZE} patches permuted)
  4. Row-shuffled (pixel rows permuted)
  5. Pixel-shuffled (all pixels permuted independently)
- **Metric:** Bits per pixel via rate-distortion: R(D) = 0.5 × log₂(σ²/MSE)

## Results

| Destruction level | Bits/pixel (mean±std) | Δ from original |
|---|---|---|
"""
    for level in SHUFFLE_LEVELS:
        mean_bpp, std_bpp = mean_by_level[level]
        delta = mean_bpp - original_bpp
        report += f"| {level} | {mean_bpp:.4f}±{std_bpp:.4f} | {delta:+.4f} |\n"

    report += f"""
## Structural Bonus

**Visual structural bonus: {mean_bonus:.4f} bits/pixel**

For comparison:
- Language structural bonus (Paper 6): ~6.7 bits/token
- PCFG-8 structural bonus (Paper 7 R6): ~5.3 bits/token

## Interpretation

"""
    if mean_bonus > 0.1:
        report += f"""The visual structural bonus of {mean_bonus:.3f} bits/pixel confirms that
spatial structure is exploitable by the model — destroying it reduces predictive
power. The equation BPT ≈ entropy − f(structure) holds qualitatively for vision.

"""
    else:
        report += f"""The visual structural bonus of {mean_bonus:.3f} bits/pixel is small,
suggesting that either (a) the model does not exploit spatial structure effectively
at this scale, (b) CIFAR-100's 32×32 resolution does not carry much spatial
hierarchy, or (c) the rate-distortion metric underestimates the structural
contribution. Higher-resolution datasets (ImageNet, 224×224) may show a larger
bonus.

"""

    report += f"""## Files

- `results/shuffling_cascade.csv` — all measurements
- `results/training_curve_seed*.csv` — loss curves
- `plots/shuffling_cascade.png` — cascade plot
- `plots/delta_per_level.png` — per-level delta plot

*Experiment completed automatically. {len(SEEDS)} seeds, {TOTAL_STEPS} training steps each.*
"""

    with open(f"{OUTDIR}/results/P8A_REPORT.md", 'w') as f:
        f.write(report)
    log("Report written")

    # Final git commit
    try:
        subprocess.run(['git', 'add', 'paper8/p8a_visual_shuffling_cascade/'],
                      cwd=REPO, capture_output=True)
        subprocess.run(['git', 'commit', '-m',
            f'Paper 8 P8-A: visual shuffling cascade complete\n\n'
            f'Visual structural bonus: {mean_bonus:.4f} bits/pixel\n'
            f'(Language reference: ~6.7 bits/token)\n'
            f'Trained from scratch, {len(SEEDS)} seeds, {TOTAL_STEPS} steps\n\n'
            f'Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>'],
            cwd=REPO, capture_output=True)
        subprocess.run(['git', 'push'], cwd=REPO, capture_output=True)
        log("Final git push complete")
    except Exception as e:
        log(f"Git error: {e}")

    log("\n" + "="*60)
    log("VISUAL SHUFFLING CASCADE — Complete")
    log(f"Structural bonus: {mean_bonus:.4f} bits/pixel")
    log("="*60)

if __name__ == "__main__":
    main()
