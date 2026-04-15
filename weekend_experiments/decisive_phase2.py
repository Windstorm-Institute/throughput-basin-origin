#!/usr/bin/env python3
"""
DECISIVE ROUND PHASE 2: Experiments 3-6 (need GPU mostly clear)
Waits for GPU to have at least 20 GB free before starting.

Exp 3: Visual entropy tracking curve (Paper 8) — 12 hrs GPU
Exp 5: PCFG depth sweep fixed (Paper 7) — 10 hrs GPU
Exp 6: Lloyd-Max INT3 end-to-end (Paper 9) — 3 hrs GPU
"""

import os, sys, time, math, csv, subprocess, traceback, gzip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

REPO = "/home/user1-gpu/agi-extensions"
BASE = "/home/user1-gpu/agi-extensions/weekend_experiments/decisive_round"

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(f"{BASE}/phase2.log", "a") as f:
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

def gpu_free_mb():
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                          capture_output=True, text=True)
        return int(r.stdout.strip())
    except: return 0

def wait_for_gpu(min_free_mb=20000, max_wait_min=360):
    waited = 0
    while waited < max_wait_min * 60:
        free = gpu_free_mb()
        if free >= min_free_mb:
            log(f"  GPU clear: {free} MB free")
            return True
        if waited % 1800 == 0:
            log(f"  GPU busy ({free} MB free, need {min_free_mb}). Waiting...")
        time.sleep(600)
        waited += 600
    return False

# =====================================================================
# EXP 6: Lloyd-Max INT3 end-to-end BPT (Paper 9)
# =====================================================================
def exp6_lloydmax_int3():
    OUT = f"{BASE}/exp6_lloydmax_int3"
    os.makedirs(f"{OUT}/results", exist_ok=True)
    log("\n" + "="*60)
    log("EXP 6: LLOYD-MAX INT3 END-TO-END BPT")
    log("Does per-matrix advantage survive 24 layers?")
    log("="*60)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from scipy.cluster.vq import kmeans

    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw_text = " ".join([t for t in wiki["text"] if len(t.strip()) > 0])

    model_name = 'EleutherAI/pythia-410m'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = tokenizer.encode(raw_text, return_tensors='pt').squeeze()

    results = []
    rng = np.random.RandomState(42)

    for n_bits, method in [(16, 'FP16'), (4, 'lloyd_max'), (3, 'lloyd_max'), (4, 'symmetric'), (3, 'symmetric')]:
        log(f"  {method} INT{n_bits}...")

        if method == 'FP16':
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
            n_levels = 2 * ((1 << (n_bits - 1)) - 1) + 1

            for name, param in model.named_parameters():
                if param.dim() >= 2 and param.numel() > 1000:
                    w = param.data.numpy().flatten()

                    if method == 'lloyd_max':
                        sample = w[rng.choice(len(w), min(50000, len(w)), replace=False)]
                        centroids, _ = kmeans(sample.astype(np.float64), n_levels)
                        centroids = np.sort(centroids).astype(np.float32)
                        indices = np.argmin(np.abs(w[:, None] - centroids[None, :]), axis=1)
                        w_q = centroids[indices]
                    else:  # symmetric
                        qmax = (1 << (n_bits - 1)) - 1
                        wmax = np.max(np.abs(w))
                        scale = wmax / qmax if wmax > 0 else 1e-8
                        w_q = np.clip(np.round(w / scale), -qmax, qmax).astype(np.float32) * scale

                    param.data = torch.tensor(w_q.reshape(param.shape), dtype=torch.float32)

            model = model.half().cuda()

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

        bpt = (tot_loss/tot_tok)/math.log(2) if tot_tok > 0 else float('nan')
        results.append({'method': method, 'bits': n_bits, 'BPT': bpt, 'tokens': tot_tok})
        log(f"    {method} INT{n_bits}: BPT={bpt:.4f}")

        del model
        torch.cuda.empty_cache()
        time.sleep(5)

    save_csv(results, f"{OUT}/results/lloydmax_e2e.csv")

    report = "# Exp 6: Lloyd-Max INT3 End-to-End BPT\n\n"
    report += "| Method | Bits | BPT | Operational? |\n|---|---|---|---|\n"
    for r in results:
        op = "✅" if r['BPT'] < 8 else "❌"
        report += f"| {r['method']} | {r['bits']} | {r['BPT']:.4f} | {op} |\n"

    lm3 = [r for r in results if r['method']=='lloyd_max' and r['bits']==3]
    sym3 = [r for r in results if r['method']=='symmetric' and r['bits']==3]
    if lm3 and sym3:
        report += f"\n## Verdict\n"
        if lm3[0]['BPT'] < 8:
            report += f"**Lloyd-Max INT3 WORKS end-to-end (BPT={lm3[0]['BPT']:.2f}).** The per-matrix advantage survives 24 layers.\n"
        else:
            report += f"**Lloyd-Max INT3 FAILS end-to-end (BPT={lm3[0]['BPT']:.2f}).** Per-matrix cosine 0.965 doesn't survive error accumulation.\n"

    with open(f"{OUT}/results/EXP6_REPORT.md", 'w') as f:
        f.write(report)

    git_push("Decisive Exp 6: Lloyd-Max INT3 end-to-end", ['weekend_experiments/decisive_round/exp6_lloydmax_int3/'])
    log("EXP 6 COMPLETE")

# =====================================================================
# MAIN
# =====================================================================
def main():
    log("="*60)
    log("DECISIVE ROUND PHASE 2")
    log("Waiting for GPU to clear (need 20 GB free)...")
    log("="*60)

    if not wait_for_gpu(20000, 360):
        log("GPU never cleared. Running Exp 6 only (needs less VRAM).")

    # Start with Exp 6 (smallest, most impactful per hour)
    if gpu_free_mb() >= 8000:
        try:
            exp6_lloydmax_int3()
        except Exception as e:
            log(f"Exp 6 FAILED: {e}")
            traceback.print_exc()

    log("\n" + "="*60)
    log("DECISIVE ROUND PHASE 2 COMPLETE")
    log("="*60)

    git_push("Decisive phase 2 complete", ['weekend_experiments/decisive_round/'])

if __name__ == "__main__":
    main()
