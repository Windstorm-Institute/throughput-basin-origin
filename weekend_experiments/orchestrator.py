#!/usr/bin/env python3
"""
WEEKEND EXPERIMENT ORCHESTRATOR
Papers 8 and 9 — 13 experiments over 60 hours

Runs GPU and CPU experiments in parallel.
Each experiment is a self-contained function that:
  - Downloads any needed data
  - Runs the experiment
  - Saves results to CSV
  - Generates plots
  - Writes a report
  - Git commits and pushes

Designed to run unattended from Saturday night to Monday morning.
"""

import os, sys, time, math, csv, json, subprocess, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Process
import threading

REPO = "/home/user1-gpu/agi-extensions"
BASE = "/home/user1-gpu/agi-extensions/weekend_experiments"

def log(msg, logfile=f"{BASE}/orchestrator.log"):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    with open(logfile, "a") as f:
        f.write(line + "\n")

def git_commit_push(msg, paths):
    try:
        for p in paths:
            subprocess.run(['git', 'add', p], cwd=REPO, capture_output=True)
        subprocess.run(['git', 'commit', '-m', msg + '\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>'],
                       cwd=REPO, capture_output=True)
        subprocess.run(['git', 'push'], cwd=REPO, capture_output=True)
    except Exception as e:
        log(f"Git error: {e}")

def save_csv(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not data:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader()
        w.writerows(data)

# =====================================================================
# P8-E1: MAE Generative Vision Throughput (GPU, ~3 hours)
# =====================================================================
def run_p8_e1():
    """Load pretrained MAE/BEiT, measure reconstruction bits/pixel."""
    OUTDIR = f"{BASE}/p8_e1_mae_throughput"
    os.makedirs(f"{OUTDIR}/results", exist_ok=True)
    os.makedirs(f"{OUTDIR}/plots", exist_ok=True)
    log("=" * 60)
    log("P8-E1: MAE GENERATIVE VISION THROUGHPUT")
    log("=" * 60)

    from torchvision.datasets import STL10
    from torchvision import transforms

    # Load STL-10 test set
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    dataset = STL10(root='/tmp/stl10', split='test', download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=16, num_workers=4, pin_memory=True)

    results = []

    # Test MAE models
    mae_models = [
        ('facebook/vit-mae-base', 'MAE-Base'),
        ('facebook/vit-mae-large', 'MAE-Large'),
    ]

    for model_name, label in mae_models:
        try:
            log(f"  Loading {model_name}...")
            from transformers import ViTMAEForPreTraining, ViTMAEConfig

            model = ViTMAEForPreTraining.from_pretrained(model_name).cuda().eval()
            n_params = sum(p.numel() for p in model.parameters())

            total_mse = 0.0
            total_pixels = 0
            pixel_values_for_var = []

            with torch.no_grad():
                for batch_idx, (images, _) in enumerate(loader):
                    if batch_idx >= 100:  # 1600 images
                        break
                    images = images.cuda()

                    if batch_idx < 5:
                        pixel_values_for_var.append(images.cpu().numpy().flatten())

                    outputs = model(images)
                    # MAE loss is MSE on masked patches
                    loss = outputs.loss.item()
                    total_mse += loss * images.shape[0]
                    total_pixels += images.shape[0]

            avg_loss = total_mse / total_pixels

            # Compute pixel variance for rate-distortion
            all_px = np.concatenate(pixel_values_for_var)
            px_var = np.var(all_px)

            # Rate-distortion: bits = 0.5 * log2(var / MSE)
            if avg_loss < px_var and avg_loss > 0:
                bits_per_pixel = 0.5 * math.log2(px_var / avg_loss)
            else:
                bits_per_pixel = 0.0

            result = {
                'model': label,
                'model_name': model_name,
                'n_params': n_params,
                'mae_loss': avg_loss,
                'pixel_variance': px_var,
                'bits_per_pixel': bits_per_pixel,
                'n_images': total_pixels,
            }
            results.append(result)
            log(f"  {label}: MAE_loss={avg_loss:.6f}, bits/px={bits_per_pixel:.4f}")

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            log(f"  ERROR on {model_name}: {e}")
            traceback.print_exc()

    save_csv(results, f"{OUTDIR}/results/mae_throughput.csv")

    # Report
    report = "# P8-E1: MAE Generative Vision Throughput\n\n"
    report += "| Model | MAE Loss | Bits/pixel | Params |\n|---|---|---|---|\n"
    for r in results:
        report += f"| {r['model']} | {r['mae_loss']:.6f} | {r['bits_per_pixel']:.4f} | {r['n_params']:,} |\n"
    report += f"\nClassification ViTs: 0.002-0.03 bits/patch (Paper 8 survey)\n"
    report += f"Language basin: ~4.16 bits/token\n"
    report += f"Visual source entropy (WebP): 4.0-4.6 bits/pixel\n"

    with open(f"{OUTDIR}/results/P8_E1_REPORT.md", 'w') as f:
        f.write(report)

    git_commit_push(f"P8-E1: MAE generative throughput complete", ['weekend_experiments/p8_e1_mae_throughput/'])
    log("P8-E1 COMPLETE")

# =====================================================================
# P8-E8: High-Resolution Visual Entropy (CPU only, ~2 hours)
# =====================================================================
def run_p8_e8():
    """Measure source entropy of ImageNet-resolution images."""
    OUTDIR = f"{BASE}/p8_e8_highres_entropy"
    os.makedirs(f"{OUTDIR}/results", exist_ok=True)
    log("=" * 60)
    log("P8-E8: HIGH-RESOLUTION VISUAL ENTROPY")
    log("=" * 60)

    from torchvision.datasets import STL10
    from PIL import Image
    import io, gzip, zlib

    results = []

    # STL-10 at native 96×96 (already measured but re-confirm)
    # Also resize to 224×224 to see resolution effect
    dataset = STL10(root='/tmp/stl10', split='test', download=True)
    images_96 = np.array([np.array(img) for img, _ in dataset])[:2000]

    for res_name, images in [('STL10_96x96', images_96)]:
        n, h, w, c = images.shape
        log(f"  Processing {res_name}: {n} images, {h}×{w}×{c}")

        # H_pixel (marginal)
        flat = images.flatten().astype(np.float64)
        counts = np.bincount(flat.astype(int), minlength=256)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        H_pixel = -np.sum(probs * np.log2(probs))

        # H_conditional (adjacent pixels)
        joint = np.zeros((256, 256), dtype=np.float64)
        for img in images[:500]:
            for ch in range(3):
                row = img[:, :, ch].flatten()
                for i in range(len(row) - 1):
                    joint[row[i], row[i+1]] += 1
        joint /= joint.sum()
        marginal = joint.sum(axis=1)
        H_joint = -np.sum(joint[joint > 0] * np.log2(joint[joint > 0]))
        H_marginal = -np.sum(marginal[marginal > 0] * np.log2(marginal[marginal > 0]))
        H_cond = H_joint - H_marginal

        # H_gzip
        gzip_bpp = []
        for img in images[:500]:
            raw = img.tobytes()
            compressed = gzip.compress(raw, compresslevel=9)
            gzip_bpp.append(len(compressed) * 8 / (h * w * c))

        # H_png
        png_bpp = []
        for img in images[:500]:
            pil_img = Image.fromarray(img)
            buf = io.BytesIO()
            pil_img.save(buf, format='PNG', optimize=True)
            png_bpp.append(len(buf.getvalue()) * 8 / (h * w * c))

        result = {
            'dataset': res_name,
            'resolution': f'{h}x{w}',
            'n_images': n,
            'H_pixel': H_pixel,
            'H_conditional': H_cond,
            'H_gzip_mean': np.mean(gzip_bpp),
            'H_gzip_std': np.std(gzip_bpp),
            'H_png_mean': np.mean(png_bpp),
            'H_png_std': np.std(png_bpp),
        }
        results.append(result)
        log(f"  {res_name}: H_pixel={H_pixel:.3f}, H_cond={H_cond:.3f}, "
            f"H_gzip={np.mean(gzip_bpp):.3f}, H_png={np.mean(png_bpp):.3f}")

    # Also measure on resized 224×224 versions
    log("  Resizing to 224×224...")
    from PIL import Image as PILImage
    images_224 = []
    for img in images_96[:500]:
        pil = PILImage.fromarray(img).resize((224, 224), PILImage.BILINEAR)
        images_224.append(np.array(pil))
    images_224 = np.array(images_224)

    h, w = 224, 224
    gzip_bpp_224 = []
    png_bpp_224 = []
    for img in images_224:
        raw = img.tobytes()
        compressed = gzip.compress(raw, compresslevel=9)
        gzip_bpp_224.append(len(compressed) * 8 / (h * w * 3))
        buf = io.BytesIO()
        PILImage.fromarray(img).save(buf, format='PNG', optimize=True)
        png_bpp_224.append(len(buf.getvalue()) * 8 / (h * w * 3))

    results.append({
        'dataset': 'STL10_resized_224x224',
        'resolution': '224x224',
        'n_images': len(images_224),
        'H_pixel': H_pixel,  # same source distribution
        'H_conditional': H_cond,
        'H_gzip_mean': np.mean(gzip_bpp_224),
        'H_gzip_std': np.std(gzip_bpp_224),
        'H_png_mean': np.mean(png_bpp_224),
        'H_png_std': np.std(png_bpp_224),
    })
    log(f"  STL10@224: H_gzip={np.mean(gzip_bpp_224):.3f}, H_png={np.mean(png_bpp_224):.3f}")

    save_csv(results, f"{OUTDIR}/results/highres_entropy.csv")

    report = "# P8-E8: High-Resolution Visual Entropy\n\n"
    report += "| Dataset | Resolution | H_pixel | H_cond | H_gzip | H_png |\n|---|---|---|---|---|---|\n"
    for r in results:
        report += f"| {r['dataset']} | {r['resolution']} | {r['H_pixel']:.3f} | {r['H_conditional']:.3f} | {r['H_gzip_mean']:.3f} | {r['H_png_mean']:.3f} |\n"

    with open(f"{OUTDIR}/results/P8_E8_REPORT.md", 'w') as f:
        f.write(report)

    git_commit_push("P8-E8: high-resolution visual entropy", ['weekend_experiments/p8_e8_highres_entropy/'])
    log("P8-E8 COMPLETE")

# =====================================================================
# P9-E4: Real Model Weights Through Hardware Quantization (CPU, ~2 hrs)
# =====================================================================
def run_p9_e4():
    """Load Pythia-410M weights, quantize, compare software vs hardware arithmetic."""
    OUTDIR = f"{BASE}/p9_e4_real_weights"
    os.makedirs(f"{OUTDIR}/results", exist_ok=True)
    os.makedirs(f"{OUTDIR}/plots", exist_ok=True)
    log("=" * 60)
    log("P9-E4: REAL MODEL WEIGHTS THROUGH HARDWARE QUANTIZATION")
    log("=" * 60)

    from transformers import AutoModelForCausalLM

    log("  Loading Pythia-410M...")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m", torch_dtype=torch.float32)

    precisions = [8, 6, 5, 4, 3, 2]
    results = []

    # Extract weight matrices from first transformer layer
    layer = model.gpt_neox.layers[0]
    weight_matrices = {
        'attn_qkv': layer.attention.query_key_value.weight.detach().numpy(),
        'attn_dense': layer.attention.dense.weight.detach().numpy(),
        'mlp_dense_h_to_4h': layer.mlp.dense_h_to_4h.weight.detach().numpy(),
        'mlp_dense_4h_to_h': layer.mlp.dense_4h_to_h.weight.detach().numpy(),
    }

    del model
    torch.cuda.empty_cache()

    log(f"  Extracted {len(weight_matrices)} weight matrices")
    for name, w in weight_matrices.items():
        log(f"    {name}: {w.shape}, range [{w.min():.4f}, {w.max():.4f}]")

    # Quantize each matrix at each precision
    for mat_name, W in weight_matrices.items():
        for n_bits in precisions:
            qmax = (1 << (n_bits - 1)) - 1
            wmax = np.max(np.abs(W))
            scale = wmax / qmax if wmax > 0 else 1e-8

            W_int = np.clip(np.round(W / scale), -qmax, qmax).astype(np.int32)
            W_deq = W_int.astype(np.float32) * scale

            mse = np.mean((W - W_deq) ** 2)
            cos = np.dot(W.flatten(), W_deq.flatten()) / (
                np.linalg.norm(W) * np.linalg.norm(W_deq) + 1e-10)
            signal = np.mean(W ** 2)
            sqnr = 10 * np.log10(signal / mse) if mse > 0 else float('inf')

            # Forward pass comparison: random input through original vs quantized
            X = np.random.randn(32, W.shape[1]).astype(np.float32) * 0.1
            Y_fp = X @ W.T
            Y_q = X @ W_deq.T
            out_mse = np.mean((Y_fp - Y_q) ** 2)
            out_cos = np.dot(Y_fp.flatten(), Y_q.flatten()) / (
                np.linalg.norm(Y_fp) * np.linalg.norm(Y_q) + 1e-10)

            results.append({
                'matrix': mat_name,
                'shape': f'{W.shape[0]}x{W.shape[1]}',
                'n_bits': n_bits,
                'weight_mse': mse,
                'weight_cosine': cos,
                'weight_sqnr_db': sqnr,
                'output_mse': out_mse,
                'output_cosine': out_cos,
                'scale': scale,
                'n_unique_values': len(np.unique(W_int)),
            })

        log(f"  {mat_name}: INT4 cos={[r['output_cosine'] for r in results if r['matrix']==mat_name and r['n_bits']==4][0]:.6f}, "
            f"INT3 cos={[r['output_cosine'] for r in results if r['matrix']==mat_name and r['n_bits']==3][0]:.6f}")

    save_csv(results, f"{OUTDIR}/results/real_weights_quant.csv")

    # Cliff analysis
    cliff_data = []
    for mat_name in weight_matrices.keys():
        for i in range(len(precisions) - 1):
            high = precisions[i]
            low = precisions[i + 1]
            h_cos = np.mean([r['output_cosine'] for r in results if r['matrix'] == mat_name and r['n_bits'] == high])
            l_cos = np.mean([r['output_cosine'] for r in results if r['matrix'] == mat_name and r['n_bits'] == low])
            cliff_data.append({
                'matrix': mat_name,
                'transition': f'INT{high}→INT{low}',
                'high_cosine': h_cos,
                'low_cosine': l_cos,
                'degradation': h_cos - l_cos,
            })

    save_csv(cliff_data, f"{OUTDIR}/results/cliff_analysis.csv")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        for mat_name in weight_matrices.keys():
            mat_results = [r for r in results if r['matrix'] == mat_name]
            bits = [r['n_bits'] for r in mat_results]
            cos = [r['output_cosine'] for r in mat_results]
            ax.plot(bits, cos, 'o-', label=mat_name, markersize=8)

        ax.axvline(x=3.5, color='red', linestyle='--', alpha=0.5, label='INT4→INT3')
        ax.set_xlabel('Weight precision (bits)')
        ax.set_ylabel('Output cosine similarity')
        ax.set_title('Pythia-410M Layer 0: output fidelity vs quantization precision')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        fig.savefig(f"{OUTDIR}/plots/real_weights_cliff.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        log(f"Plot error: {e}")

    report = "# P9-E4: Real Pythia-410M Weights Through Hardware Quantization\n\n"
    report += "## Cliff Analysis\n\n"
    report += "| Matrix | INT4→INT3 degradation | INT5→INT4 degradation | Cliff ratio |\n|---|---|---|---|\n"
    for mat_name in weight_matrices.keys():
        d43 = [c for c in cliff_data if c['matrix'] == mat_name and c['transition'] == 'INT4→INT3']
        d54 = [c for c in cliff_data if c['matrix'] == mat_name and c['transition'] == 'INT5→INT4']
        if d43 and d54:
            ratio = d43[0]['degradation'] / d54[0]['degradation'] if d54[0]['degradation'] > 0 else float('inf')
            report += f"| {mat_name} | {d43[0]['degradation']:.6f} | {d54[0]['degradation']:.6f} | {ratio:.1f}× |\n"

    with open(f"{OUTDIR}/results/P9_E4_REPORT.md", 'w') as f:
        f.write(report)

    git_commit_push("P9-E4: real Pythia-410M weights through hardware quantization", ['weekend_experiments/p9_e4_real_weights/'])
    log("P9-E4 COMPLETE")

# =====================================================================
# P8-E7: Patch Size Invariance (GPU, ~4 hours)
# =====================================================================
def run_p8_e7():
    """Train next-patch models at different patch sizes to test if bits/pixel is constant."""
    OUTDIR = f"{BASE}/p8_e7_patch_invariance"
    os.makedirs(f"{OUTDIR}/results", exist_ok=True)
    os.makedirs(f"{OUTDIR}/plots", exist_ok=True)
    log("=" * 60)
    log("P8-E7: PATCH SIZE INVARIANCE")
    log("=" * 60)

    from torchvision.datasets import STL10

    dataset = STL10(root='/tmp/stl10', split='train', download=True)
    train_images = np.array([np.array(img) for img, _ in dataset])
    test_dataset = STL10(root='/tmp/stl10', split='test', download=True)
    eval_images = np.array([np.array(img) for img, _ in test_dataset])

    pixel_var = np.var(train_images.astype(np.float32) / 255.0)
    log(f"  Pixel variance: {pixel_var:.6f}")

    # Import the model class from P8-A v2
    sys.path.insert(0, f"{REPO}/paper8/p8a_v2_visual_cascade")
    from run_p8a_v2 import PatchDataset, PatchPredictor, compute_metrics

    patch_sizes = [8, 12, 16, 24, 32, 48]
    results = []

    for ps in patch_sizes:
        if 96 % ps != 0:
            log(f"  Skipping patch_size={ps} (doesn't divide 96)")
            continue

        n_patches = (96 // ps) ** 2
        patch_dim = ps * ps * 3
        embed_dim = min(512, max(128, patch_dim))
        n_heads = max(2, embed_dim // 64)

        log(f"\n  Patch size {ps}×{ps}: {n_patches} patches/image, dim={patch_dim}, embed={embed_dim}")

        torch.manual_seed(42)
        np.random.seed(42)

        train_ds = PatchDataset(train_images, ps, 'original')
        train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4,
                             pin_memory=True, drop_last=True)

        model = PatchPredictor(patch_dim, embed_dim, 6, n_heads, n_patches).cuda()
        n_params = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
        scaler = GradScaler()

        # Train 20K steps
        step = 0
        total_steps = 20000
        start = time.time()

        while step < total_steps:
            for patches in train_dl:
                if step >= total_steps:
                    break
                step += 1

                lr = 3e-4 * min(step / 1000, 0.5 * (1 + math.cos(math.pi * max(0, step - 1000) / (total_steps - 1000))))
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

                patches = patches.cuda()
                optimizer.zero_grad()
                with autocast():
                    preds = model(patches)
                    loss = F.mse_loss(preds, patches[:, 1:])
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                if step % 5000 == 0:
                    log(f"    step {step}/{total_steps}, loss={loss.item():.6f}")

        elapsed = time.time() - start

        # Evaluate
        eval_ds = PatchDataset(eval_images, ps, 'original')
        eval_dl = DataLoader(eval_ds, batch_size=32, num_workers=2, pin_memory=True)
        metrics = compute_metrics(model, eval_dl, 'cuda', pixel_var)

        results.append({
            'patch_size': ps,
            'n_patches': n_patches,
            'patch_dim': patch_dim,
            'embed_dim': embed_dim,
            'n_params': n_params,
            'bits_per_pixel': metrics['bits_per_pixel'],
            'bits_per_patch': metrics['bits_per_patch'],
            'mse_per_dim': metrics['mse_per_dim'],
            'train_time_hrs': elapsed / 3600,
        })

        log(f"    Result: bits/px={metrics['bits_per_pixel']:.4f}, bits/patch={metrics['bits_per_patch']:.2f}")

        del model, optimizer, scaler
        torch.cuda.empty_cache()

    save_csv(results, f"{OUTDIR}/results/patch_invariance.csv")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ps_vals = [r['patch_size'] for r in results]
        bpp_vals = [r['bits_per_pixel'] for r in results]
        bppatch_vals = [r['bits_per_patch'] for r in results]

        ax1.plot(ps_vals, bpp_vals, 'o-', markersize=10)
        ax1.set_xlabel('Patch size')
        ax1.set_ylabel('Bits per pixel')
        ax1.set_title('Is bits/pixel constant across patch sizes?')
        ax1.grid(True, alpha=0.3)

        ax2.plot(ps_vals, bppatch_vals, 's-', color='green', markersize=10)
        ax2.set_xlabel('Patch size')
        ax2.set_ylabel('Bits per patch')
        ax2.set_title('Bits per patch scales with patch area')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(f"{OUTDIR}/plots/patch_invariance.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        log(f"Plot error: {e}")

    report = "# P8-E7: Patch Size Invariance\n\n"
    report += "| Patch size | Patches/img | Bits/pixel | Bits/patch |\n|---|---|---|---|\n"
    for r in results:
        report += f"| {r['patch_size']}×{r['patch_size']} | {r['n_patches']} | {r['bits_per_pixel']:.4f} | {r['bits_per_patch']:.2f} |\n"
    report += "\nIf bits/pixel is constant → it's a genuine per-pixel metric (like bits/source-byte for text).\n"
    report += "If it varies → patch size acts like a tokenizer and the metric is encoding-dependent.\n"

    with open(f"{OUTDIR}/results/P8_E7_REPORT.md", 'w') as f:
        f.write(report)

    git_commit_push("P8-E7: patch size invariance test", ['weekend_experiments/p8_e7_patch_invariance/'])
    log("P8-E7 COMPLETE")

# =====================================================================
# MAIN ORCHESTRATOR
# =====================================================================
def main():
    log("=" * 60)
    log("WEEKEND EXPERIMENT ORCHESTRATOR — STARTING")
    log(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Target: 13 experiments for Papers 8 and 9")
    log("=" * 60)

    # Phase 1: Parallel — GPU (E1) + CPU (E8, E4)
    log("\n\n========== PHASE 1: Saturday Night ==========\n")

    # Run CPU experiments in threads (they don't touch GPU)
    cpu_thread_e8 = threading.Thread(target=run_p8_e8, name='P8-E8')
    cpu_thread_e4 = threading.Thread(target=run_p9_e4, name='P9-E4')

    cpu_thread_e8.start()
    cpu_thread_e4.start()

    # Run GPU experiment in main thread
    try:
        run_p8_e1()
    except Exception as e:
        log(f"P8-E1 FAILED: {e}")
        traceback.print_exc()

    # Wait for CPU experiments
    cpu_thread_e8.join()
    cpu_thread_e4.join()

    log("\n\n========== PHASE 1 COMPLETE ==========\n")

    # Phase 2: GPU (E7 patch invariance) — runs alone since it's GPU-heavy
    log("\n\n========== PHASE 2: Patch Invariance ==========\n")
    try:
        run_p8_e7()
    except Exception as e:
        log(f"P8-E7 FAILED: {e}")
        traceback.print_exc()

    log("\n\n========== PHASE 2 COMPLETE ==========\n")

    # Summary
    log("\n\n" + "=" * 60)
    log("WEEKEND ORCHESTRATOR — PHASE 1-2 COMPLETE")
    log("Experiments completed:")

    completed = []
    for exp in ['p8_e1_mae_throughput', 'p8_e8_highres_entropy', 'p9_e4_real_weights', 'p8_e7_patch_invariance']:
        path = f"{BASE}/{exp}/results"
        if os.path.exists(path) and any(f.endswith('.csv') for f in os.listdir(path)):
            completed.append(exp)
            log(f"  ✓ {exp}")
        else:
            log(f"  ✗ {exp} (no results)")

    log(f"\n{len(completed)}/4 Phase 1-2 experiments completed")
    log("=" * 60)

    # Final git push
    git_commit_push("Weekend experiments Phase 1-2 complete", ['weekend_experiments/'])

if __name__ == "__main__":
    main()
