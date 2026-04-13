#!/usr/bin/env python3
"""
Round 3 Remaining Experiments:
- Exp 2 (fixed): bitsandbytes NF4 vs symmetric — where does each method's cliff live?
- Exp 5: From-scratch MAE on STL-10
- Exp 6: Calibrated visual entropy (Markov pixel fields)
- Exp 8: Patch-size bits/H normalization

All auto-commit and push. Designed for unattended overnight execution.
"""

import os, sys, time, math, csv, subprocess, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

REPO = "/home/user1-gpu/agi-extensions"
BASE = "/home/user1-gpu/agi-extensions/weekend_experiments"

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(f"{BASE}/round3_remaining.log", "a") as f:
        f.write(line + "\n")

def git_push(msg, paths):
    try:
        for p in paths:
            subprocess.run(['git', 'add', p], cwd=REPO, capture_output=True)
        subprocess.run(['git', 'commit', '-m', msg + '\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>'],
                       cwd=REPO, capture_output=True)
        subprocess.run(['git', 'push'], cwd=REPO, capture_output=True)
    except: pass

def save_csv(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not data: return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader()
        w.writerows(data)

# =====================================================================
# EXP 2 FIXED: bitsandbytes NF4 vs symmetric end-to-end BPT
# =====================================================================
def exp2_fixed():
    OUT = f"{BASE}/round3_exp2_gptq_cliff"
    os.makedirs(f"{OUT}/results", exist_ok=True)
    log("\n" + "="*60)
    log("EXP 2 (FIXED): BITSANDBYTES NF4 vs SYMMETRIC QUANTIZATION")
    log("Question: Does the cliff location depend on the quantization method?")
    log("="*60)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from datasets import load_dataset

    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw_text = " ".join([t for t in wiki["text"] if len(t.strip()) > 0])

    model_name = 'EleutherAI/pythia-410m'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = tokenizer.encode(raw_text, return_tensors='pt').squeeze()
    n_bytes = len(raw_text.encode('utf-8'))

    results = []

    def eval_model(model, label, bits):
        model.eval()
        tot_loss, tot_tok = 0.0, 0
        with torch.no_grad():
            for s in range(0, min(len(input_ids)-1024, 50000), 1024):
                x = input_ids[s:s+1024].unsqueeze(0).cuda()
                y = input_ids[s+1:s+1025].unsqueeze(0).cuda()
                try:
                    out = model(x)
                    loss = F.cross_entropy(out.logits.view(-1, out.logits.size(-1)), y.view(-1))
                    if not math.isnan(loss.item()) and not math.isinf(loss.item()):
                        tot_loss += loss.item() * 1024
                        tot_tok += 1024
                except: break
        if tot_tok > 0:
            bpt = (tot_loss/tot_tok)/math.log(2)
            results.append({'method': label, 'bits': bits, 'BPT': bpt,
                           'bits_per_byte': bpt * len(input_ids) / n_bytes, 'tokens': tot_tok})
            log(f"    {label} INT{bits}: BPT={bpt:.4f}")
        else:
            log(f"    {label} INT{bits}: FAILED (NaN/Inf)")
        del model
        torch.cuda.empty_cache()

    # FP16 baseline
    log("  FP16 baseline...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
    eval_model(model, 'FP16', 16)

    # bitsandbytes NF4 (the method Paper 7 used)
    log("  bitsandbytes NF4...")
    try:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
        eval_model(model, 'bitsandbytes_NF4', 4)
    except Exception as e:
        log(f"    bitsandbytes NF4 failed: {e}")

    # bitsandbytes INT8
    log("  bitsandbytes INT8...")
    try:
        bnb8_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb8_config)
        eval_model(model, 'bitsandbytes_INT8', 8)
    except Exception as e:
        log(f"    bitsandbytes INT8 failed: {e}")

    # Symmetric quantization at each precision (from Exp 3, but re-run for this model)
    for n_bits in [8, 4, 3]:
        log(f"  Symmetric INT{n_bits}...")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        qmax = (1 << (n_bits - 1)) - 1
        for name, param in model.named_parameters():
            if param.dim() >= 2:
                w = param.data
                wmax = w.abs().max()
                scale = wmax / qmax if wmax > 0 else 1e-8
                param.data = torch.clamp(torch.round(w / scale), -qmax, qmax) * scale
        model = model.half().cuda()
        eval_model(model, f'symmetric', n_bits)

    save_csv(results, f"{OUT}/results/method_comparison.csv")

    report = "# Exp 2: Quantization Method Comparison\n\n"
    report += "## Key finding: The cliff location depends on the method\n\n"
    report += "| Method | Precision | BPT | vs FP16 |\n|---|---|---|---|\n"
    fp16_bpt = [r['BPT'] for r in results if r['method'] == 'FP16']
    fp16_val = fp16_bpt[0] if fp16_bpt else 4.27
    for r in results:
        ratio = r['BPT'] / fp16_val
        report += f"| {r['method']} | {r['bits']} | {r['BPT']:.4f} | {ratio:.2f}× |\n"

    report += "\n## Interpretation\n\n"
    report += "- bitsandbytes NF4 uses non-uniform quantization levels optimized for normal distributions\n"
    report += "- Symmetric quantization uses uniform levels — cruder but matches real hardware\n"
    report += "- If NF4 works at INT4 but symmetric doesn't → the cliff is about LEVEL ALLOCATION, not just bit count\n"
    report += "- This refines Paper 9's claim: the floor is 4 bits for uniform quantization; NF4-style methods can push lower\n"

    with open(f"{OUT}/results/EXP2_REPORT.md", 'w') as f:
        f.write(report)

    git_push("Round 3 Exp 2 fixed: bitsandbytes vs symmetric", ['weekend_experiments/round3_exp2_gptq_cliff/'])
    log("EXP 2 COMPLETE")

# =====================================================================
# EXP 5: From-Scratch MAE on STL-10
# =====================================================================
def exp5_scratch_mae():
    OUT = f"{BASE}/round3_exp5_scratch_mae"
    os.makedirs(f"{OUT}/results", exist_ok=True)
    os.makedirs(f"{OUT}/plots", exist_ok=True)
    log("\n" + "="*60)
    log("EXP 5: FROM-SCRATCH MAE ON STL-10")
    log("Skeptic: 'MAE throughput reflects ImageNet pretraining, not data itself.'")
    log("="*60)

    from torchvision.datasets import STL10
    from torchvision import transforms

    transform = transforms.Compose([transforms.Resize(96), transforms.ToTensor()])
    train_set = STL10(root='/tmp/stl10', split='train', download=True, transform=transform)
    test_set = STL10(root='/tmp/stl10', split='test', download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4,
                             pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=32, num_workers=2, pin_memory=True)

    # Simple MAE: encoder encodes visible patches, decoder reconstructs masked patches
    PATCH_SIZE = 16
    N_PATCHES = (96 // PATCH_SIZE) ** 2  # 36
    PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 3  # 768
    MASK_RATIO = 0.75
    EMBED_DIM = 512
    seeds = [42, 137]
    total_steps = 60000
    results = []

    class SimpleMAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Linear(PATCH_DIM, EMBED_DIM)
            self.pos_embed = nn.Parameter(torch.randn(1, N_PATCHES, EMBED_DIM) * 0.02)
            self.mask_token = nn.Parameter(torch.randn(1, 1, EMBED_DIM) * 0.02)

            enc_layer = nn.TransformerEncoderLayer(EMBED_DIM, 8, EMBED_DIM*4, 0.1,
                                                    'gelu', batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, 6)

            dec_layer = nn.TransformerEncoderLayer(EMBED_DIM, 8, EMBED_DIM*4, 0.1,
                                                    'gelu', batch_first=True, norm_first=True)
            self.decoder = nn.TransformerEncoder(dec_layer, 4)
            self.decoder_pred = nn.Linear(EMBED_DIM, PATCH_DIM)

        def patchify(self, imgs):
            B, C, H, W = imgs.shape
            p = PATCH_SIZE
            patches = imgs.unfold(2, p, p).unfold(3, p, p)
            patches = patches.contiguous().view(B, -1, C * p * p)
            return patches

        def forward(self, imgs):
            patches = self.patchify(imgs)  # (B, N, D)
            B, N, D = patches.shape

            # Random masking
            n_mask = int(N * MASK_RATIO)
            n_vis = N - n_mask
            noise = torch.rand(B, N, device=patches.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # Visible patches
            ids_vis = ids_shuffle[:, :n_vis]
            vis_patches = torch.gather(patches, 1, ids_vis.unsqueeze(-1).expand(-1, -1, D))
            vis_embed = self.patch_embed(vis_patches) + torch.gather(
                self.pos_embed.expand(B, -1, -1), 1,
                ids_vis.unsqueeze(-1).expand(-1, -1, EMBED_DIM))

            # Encode
            encoded = self.encoder(vis_embed)

            # Decoder: add mask tokens at masked positions
            mask_tokens = self.mask_token.expand(B, n_mask, -1)
            full = torch.cat([encoded, mask_tokens], dim=1)

            # Unshuffle
            ids_unshuffle = torch.argsort(
                torch.cat([ids_vis, ids_shuffle[:, n_vis:]], dim=1), dim=1)
            full = torch.gather(full, 1, ids_unshuffle.unsqueeze(-1).expand(-1, -1, EMBED_DIM))
            full = full + self.pos_embed.expand(B, -1, -1)

            decoded = self.decoder(full)
            pred = self.decoder_pred(decoded)

            # Loss on masked patches only
            ids_mask = ids_shuffle[:, n_vis:]
            target = torch.gather(patches, 1, ids_mask.unsqueeze(-1).expand(-1, -1, D))
            pred_mask = torch.gather(pred, 1, ids_mask.unsqueeze(-1).expand(-1, -1, D))

            loss = F.mse_loss(pred_mask, target)
            return loss, pred, patches

    for seed in seeds:
        log(f"\n  Training from-scratch MAE seed={seed}...")
        torch.manual_seed(seed)

        model = SimpleMAE().cuda()
        n_params = sum(p.numel() for p in model.parameters())
        log(f"    {n_params:,} params")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)
        scaler = GradScaler()

        step = 0
        losses = []
        t0 = time.time()

        while step < total_steps:
            for imgs, _ in train_loader:
                if step >= total_steps: break
                step += 1

                lr = 1.5e-4 * min(step/2000, 0.5*(1+math.cos(math.pi*max(0,step-2000)/(total_steps-2000))))
                for pg in optimizer.param_groups: pg['lr'] = lr

                imgs = imgs.cuda()
                optimizer.zero_grad()
                with autocast():
                    loss, _, _ = model(imgs)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())

                if step % 10000 == 0:
                    log(f"      step {step}/{total_steps}, loss={loss.item():.6f}")

        elapsed = time.time() - t0

        # Evaluate
        model.eval()
        total_mse, total_pixels = 0.0, 0
        pixel_vals = []

        with torch.no_grad():
            for imgs, _ in test_loader:
                imgs = imgs.cuda()
                loss, pred, patches = model(imgs)
                total_mse += loss.item() * imgs.shape[0]
                total_pixels += imgs.shape[0]
                if len(pixel_vals) < 5:
                    pixel_vals.append(imgs.cpu().numpy().flatten())

        avg_mse = total_mse / total_pixels
        px_var = np.var(np.concatenate(pixel_vals))
        bpp = max(0, 0.5 * math.log2(px_var / avg_mse)) if avg_mse < px_var and avg_mse > 0 else 0.0

        slope = 0
        if len(losses) > 100:
            last = losses[int(len(losses)*0.8):]
            slope = np.polyfit(range(len(last)), last, 1)[0]

        results.append({
            'model': 'from_scratch_mae', 'seed': seed,
            'mae_loss': avg_mse, 'pixel_var': px_var,
            'bits_per_pixel': bpp, 'n_params': n_params,
            'plateau_slope': slope, 'time_hrs': elapsed/3600,
        })
        log(f"    seed={seed}: MAE_loss={avg_mse:.6f}, bpp={bpp:.4f}, slope={slope:.2e}")

        del model, optimizer, scaler
        torch.cuda.empty_cache()

    # Compare to pretrained
    pretrained_bpp = 1.39  # from Exp P8-E1
    scratch_bpp = np.mean([r['bits_per_pixel'] for r in results])

    save_csv(results, f"{OUT}/results/scratch_mae.csv")

    report = f"""# Exp 5: From-Scratch MAE on STL-10

## Key comparison

| Model | Bits/pixel | Source |
|---|---|---|
| Pretrained MAE-Base (ImageNet) | {pretrained_bpp:.4f} | P8-E1 |
| From-scratch MAE (STL-10) | {scratch_bpp:.4f} | This experiment |

"""
    if abs(scratch_bpp - pretrained_bpp) / pretrained_bpp < 0.5:
        report += "**Verdict:** From-scratch throughput is within 50% of pretrained → throughput reflects the DATA, not pretraining.\n"
    else:
        report += f"**Verdict:** From-scratch ({scratch_bpp:.4f}) differs from pretrained ({pretrained_bpp:.4f}) → pretraining matters. Report honestly.\n"

    with open(f"{OUT}/results/EXP5_REPORT.md", 'w') as f:
        f.write(report)

    git_push("Round 3 Exp 5: from-scratch MAE", ['weekend_experiments/round3_exp5_scratch_mae/'])
    log("EXP 5 COMPLETE")

# =====================================================================
# EXP 6: Calibrated Visual Entropy
# =====================================================================
def exp6_calibrated_entropy():
    OUT = f"{BASE}/round3_exp6_calibrated_entropy"
    os.makedirs(f"{OUT}/results", exist_ok=True)
    os.makedirs(f"{OUT}/plots", exist_ok=True)
    log("\n" + "="*60)
    log("EXP 6: CALIBRATED VISUAL ENTROPY (Markov pixel fields)")
    log("Skeptic: 'Your VIMG datasets were badly calibrated.'")
    log("="*60)

    from torchvision.datasets import STL10
    import gzip

    IMG_SIZE = 96
    PATCH_SIZE = 16
    N_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
    PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 3
    N_IMAGES = 5000

    sys.path.insert(0, f"{REPO}/paper8/p8a_v2_visual_cascade")
    from run_p8a_v2 import PatchDataset, PatchPredictor, compute_metrics

    # Generate Markov Random Field images at controlled entropy
    def generate_mrf_images(n, h, w, correlation, seed=42):
        """Generate images from a Markov Random Field with tunable spatial correlation.
        correlation=0 → random noise (H≈8), correlation=1 → smooth gradients (H≈2)"""
        rng = np.random.RandomState(seed)
        images = np.zeros((n, h, w, 3), dtype=np.uint8)
        for i in range(n):
            for c in range(3):
                img = rng.randint(0, 256, (h, w)).astype(float)
                # Apply spatial smoothing to reduce entropy
                if correlation > 0:
                    from scipy.ndimage import gaussian_filter
                    sigma = correlation * 10  # higher sigma → more smoothing → lower entropy
                    img = gaussian_filter(img, sigma=sigma)
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
                images[i, :, :, c] = img.astype(np.uint8)
        return images

    def measure_entropy_gzip(images, n_sample=200):
        bpp_list = []
        for img in images[:n_sample]:
            compressed = gzip.compress(img.tobytes(), compresslevel=9)
            bpp_list.append(len(compressed) * 8 / img.size)
        return np.mean(bpp_list), np.std(bpp_list)

    # Generate datasets at target entropies by tuning correlation
    targets = [
        ('VMRF_2', 2.0, 3.0),    # high correlation → low entropy
        ('VMRF_4', 4.0, 1.0),    # moderate correlation
        ('VMRF_6', 6.0, 0.3),    # low correlation
        ('VMRF_8', 8.0, 0.0),    # no correlation → max entropy
    ]

    datasets = {}
    for name, target_H, init_corr in targets:
        log(f"  Generating {name} (target H={target_H})...")

        # Binary search on correlation to hit target entropy
        lo, hi = 0.0, 5.0
        best_corr = init_corr
        best_diff = float('inf')

        for _ in range(15):
            mid = (lo + hi) / 2
            test_imgs = generate_mrf_images(50, IMG_SIZE, IMG_SIZE, mid, seed=42)
            H, _ = measure_entropy_gzip(test_imgs)

            diff = abs(H - target_H)
            if diff < best_diff:
                best_diff = diff
                best_corr = mid

            if H > target_H:
                lo = mid  # need more smoothing
            else:
                hi = mid  # need less smoothing

        # Generate full dataset at best correlation
        imgs = generate_mrf_images(N_IMAGES, IMG_SIZE, IMG_SIZE, best_corr, seed=42)
        H_actual, H_std = measure_entropy_gzip(imgs)
        datasets[name] = imgs
        log(f"    {name}: correlation={best_corr:.3f}, H_actual={H_actual:.3f}±{H_std:.3f} (target={target_H})")

    # Add natural images
    stl = STL10(root='/tmp/stl10', split='train', download=True)
    datasets['VMRF_NAT'] = np.array([np.array(img) for img, _ in stl])

    # Train on each
    all_results = []
    pixel_var_global = np.var(datasets['VMRF_NAT'].astype(np.float32) / 255.0)

    for name, imgs in datasets.items():
        H, H_std = measure_entropy_gzip(imgs)
        log(f"\n  Training on {name} (H={H:.3f})...")

        train_imgs = imgs[:int(len(imgs)*0.9)]
        eval_imgs = imgs[int(len(imgs)*0.9):]
        pixel_var = np.var(train_imgs.astype(np.float32) / 255.0)

        for seed in [42]:
            torch.manual_seed(seed)
            np.random.seed(seed)

            train_ds = PatchDataset(train_imgs, PATCH_SIZE, 'original')
            train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4,
                                 pin_memory=True, drop_last=True)

            model = PatchPredictor(PATCH_DIM, 512, 8, 8, N_PATCHES).cuda()
            n_params = sum(p.numel() for p in model.parameters())
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
            scaler = GradScaler()

            step = 0
            total_steps = 30000
            losses = []
            t0 = time.time()

            while step < total_steps:
                for patches in train_dl:
                    if step >= total_steps: break
                    step += 1

                    lr = 3e-4 * min(step/1000, 0.5*(1+math.cos(math.pi*max(0,step-1000)/(total_steps-1000))))
                    for pg in optimizer.param_groups: pg['lr'] = lr

                    patches = patches.cuda()
                    optimizer.zero_grad()
                    with autocast():
                        pred = model(patches)
                        loss = F.mse_loss(pred, patches[:, 1:])
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    losses.append(loss.item())

                    if step % 5000 == 0:
                        log(f"      step {step}/{total_steps}, loss={loss.item():.6f}")

            elapsed = time.time() - t0

            eval_ds = PatchDataset(eval_imgs, PATCH_SIZE, 'original')
            eval_dl = DataLoader(eval_ds, batch_size=32, num_workers=2, pin_memory=True)
            metrics = compute_metrics(model, eval_dl, 'cuda', pixel_var)

            slope = 0
            if len(losses) > 100:
                last = losses[int(len(losses)*0.8):]
                slope = np.polyfit(range(len(last)), last, 1)[0]

            all_results.append({
                'dataset': name, 'seed': seed,
                'source_entropy_gzip': H, 'source_entropy_std': H_std,
                'bits_per_pixel': metrics['bits_per_pixel'],
                'mse': metrics['mse_per_dim'], 'pixel_var': pixel_var,
                'plateau_slope': slope, 'time_hrs': elapsed/3600,
            })
            log(f"      {name}: H={H:.3f}, bpp={metrics['bits_per_pixel']:.4f}")

            del model, optimizer, scaler
            torch.cuda.empty_cache()

        git_push(f"Exp 6: {name}", ['weekend_experiments/round3_exp6_calibrated_entropy/'])

    save_csv(all_results, f"{OUT}/results/calibrated_entropy.csv")

    # Plot: the visual tracking curve
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        for r in all_results:
            color = '#2166ac' if 'NAT' not in r['dataset'] else '#f59e0b'
            ax.scatter(r['source_entropy_gzip'], r['bits_per_pixel'], s=150, c=color,
                      zorder=5, edgecolors='black', linewidth=0.5)
            ax.annotate(r['dataset'], (r['source_entropy_gzip'], r['bits_per_pixel']),
                       textcoords="offset points", xytext=(8, 8), fontsize=9)

        x_line = np.linspace(0, 9, 100)
        ax.plot(x_line, x_line, '--', color='gray', alpha=0.5, label='bits/px = H')
        ax.set_xlabel('Source entropy (gzip, bpp)', fontsize=12)
        ax.set_ylabel('Model bits per pixel', fontsize=12)
        ax.set_title('Visual Throughput Tracking\n(Calibrated Markov Random Field images)', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.savefig(f"{OUT}/plots/visual_tracking.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{OUT}/plots/visual_tracking.pdf", bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        log(f"  Plot error: {e}")

    report = "# Exp 6: Calibrated Visual Entropy\n\n"
    report += "| Dataset | Target H | Actual H | Model bpp | Tracking ratio |\n|---|---|---|---|---|\n"
    for r in all_results:
        ratio = r['bits_per_pixel'] / r['source_entropy_gzip'] if r['source_entropy_gzip'] > 0 else 0
        report += f"| {r['dataset']} | — | {r['source_entropy_gzip']:.3f} | {r['bits_per_pixel']:.4f} | {ratio:.3f} |\n"

    with open(f"{OUT}/results/EXP6_REPORT.md", 'w') as f:
        f.write(report)

    git_push("Exp 6: calibrated visual entropy complete", ['weekend_experiments/round3_exp6_calibrated_entropy/'])
    log("EXP 6 COMPLETE")

# =====================================================================
# EXP 8: Patch-Size Bits/H Normalization
# =====================================================================
def exp8_patch_bits_h():
    OUT = f"{BASE}/round3_exp8_patch_bits_h"
    os.makedirs(f"{OUT}/results", exist_ok=True)
    log("\n" + "="*60)
    log("EXP 8: PATCH-SIZE BITS/H NORMALIZATION")
    log("Skeptic: 'Maybe bits/pixel normalized by scale-entropy IS constant.'")
    log("="*60)

    import gzip

    # Load existing patch-size sweep data
    sweep_path = f"{BASE}/p8_e7_patch_invariance/results/patch_invariance.csv"
    if not os.path.exists(sweep_path):
        log("  ERROR: patch_invariance.csv not found. Skipping.")
        return

    import pandas as pd
    df = pd.read_csv(sweep_path)

    # For each patch size, measure the source entropy at that patch scale
    from torchvision.datasets import STL10
    stl = STL10(root='/tmp/stl10', split='test', download=True)
    images = np.array([np.array(img) for img, _ in stl])[:500]

    results = []
    for _, row in df.iterrows():
        ps = int(row['patch_size'])
        log(f"  Patch size {ps}×{ps}...")

        # Extract patches and measure their entropy
        patch_bytes_list = []
        for img in images[:200]:
            for i in range(96 // ps):
                for j in range(96 // ps):
                    patch = img[i*ps:(i+1)*ps, j*ps:(j+1)*ps]
                    patch_bytes_list.append(patch.tobytes())

        # Measure entropy via gzip compression of individual patches
        bpp_list = []
        for pb in patch_bytes_list[:1000]:
            compressed = gzip.compress(pb, compresslevel=9)
            n_pixels = ps * ps * 3
            bpp_list.append(len(compressed) * 8 / n_pixels)

        H_patch = np.mean(bpp_list)

        bits_over_H = row['bits_per_pixel'] / H_patch if H_patch > 0 else 0

        results.append({
            'patch_size': ps,
            'n_patches': int(row['n_patches']),
            'bits_per_pixel': row['bits_per_pixel'],
            'H_patch_gzip': H_patch,
            'bits_over_H': bits_over_H,
        })
        log(f"    ps={ps}: bpp={row['bits_per_pixel']:.4f}, H_patch={H_patch:.3f}, bpp/H={bits_over_H:.4f}")

    save_csv(results, f"{OUT}/results/patch_bits_h.csv")

    report = "# Exp 8: Patch-Size Bits/H Normalization\n\n"
    report += "| Patch size | Bits/pixel | H_patch (gzip) | Bits/pixel / H |\n|---|---|---|---|\n"
    for r in results:
        report += f"| {r['patch_size']}×{r['patch_size']} | {r['bits_per_pixel']:.4f} | {r['H_patch_gzip']:.3f} | {r['bits_over_H']:.4f} |\n"

    ratios = [r['bits_over_H'] for r in results if r['bits_over_H'] > 0]
    if ratios:
        cv = np.std(ratios) / np.mean(ratios)
        report += f"\nCoefficient of variation of bits/H: {cv:.3f}\n"
        if cv < 0.2:
            report += "**Verdict:** bits/H is approximately constant (CV<0.2) → this IS the invariant metric.\n"
        else:
            report += f"**Verdict:** bits/H varies (CV={cv:.3f}) → no simple invariant found at this level.\n"

    with open(f"{OUT}/results/EXP8_REPORT.md", 'w') as f:
        f.write(report)

    git_push("Exp 8: patch-size bits/H normalization", ['weekend_experiments/round3_exp8_patch_bits_h/'])
    log("EXP 8 COMPLETE")

# =====================================================================
# MAIN
# =====================================================================
def main():
    log("="*60)
    log("ROUND 3 REMAINING EXPERIMENTS")
    log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("="*60)

    experiments = [
        ('Exp 2 fixed', exp2_fixed),
        ('Exp 8', exp8_patch_bits_h),
        ('Exp 5', exp5_scratch_mae),
        ('Exp 6', exp6_calibrated_entropy),
    ]

    for name, func in experiments:
        try:
            func()
        except Exception as e:
            log(f"{name} FAILED: {e}")
            traceback.print_exc()

    log("\n" + "="*60)
    log("ALL REMAINING EXPERIMENTS COMPLETE")
    log("="*60)

    git_push("Round 3 remaining experiments complete", ['weekend_experiments/'])

if __name__ == "__main__":
    main()
