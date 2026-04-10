#!/usr/bin/env python3
"""Paper 8 exp 2: ViT classification throughput survey on CIFAR-100 (Option B)."""
import os, sys, time, math, csv, subprocess, traceback
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BASE = "/home/user1-gpu/agi-extensions/paper8/exp2_vit_survey"
RESULTS = f"{BASE}/results"
PLOTS = f"{BASE}/plots"
CSV_PATH = f"{RESULTS}/vit_survey.csv"
VRAM_LIMIT_MB = 28000

import torch
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms

CSV_COLS = ["model_name","params_M","num_patches","n_images_evaluated",
            "mean_bits_per_image","mean_bits_per_patch","mean_bits_per_pixel",
            "mean_ce_vs_uniform","peak_vram_mb","elapsed_seconds","status"]

def gpu_used_mb():
    out = subprocess.check_output(
        ["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"]
    ).decode().strip().splitlines()[0]
    return int(out)

def wait_for_gpu_room(retries=6, sleep_s=600):
    for i in range(retries+1):
        used = gpu_used_mb()
        print(f"[gpu] used={used} MB (limit {VRAM_LIMIT_MB})", flush=True)
        if used <= VRAM_LIMIT_MB:
            return True
        if i == retries:
            return False
        print(f"[gpu] too busy, sleeping {sleep_s}s (retry {i+1}/{retries})", flush=True)
        time.sleep(sleep_s)
    return False

def append_row(row):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH,"a",newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if new: w.writeheader()
        w.writerow(row)

def get_num_patches(model, img_size=224):
    # Try common timm attrs
    for attr_path in [("patch_embed","num_patches"),]:
        obj = model
        ok = True
        for a in attr_path:
            if hasattr(obj,a): obj = getattr(obj,a)
            else: ok=False; break
        if ok and isinstance(obj,int): return obj
    # Swin: count via dummy forward of patch_embed
    if hasattr(model,"patch_embed"):
        try:
            with torch.no_grad():
                d = torch.zeros(1,3,img_size,img_size,device=next(model.parameters()).device,
                                dtype=next(model.parameters()).dtype)
                out = model.patch_embed(d)
                if isinstance(out, torch.Tensor):
                    if out.dim()==4:  # B,C,H,W
                        return out.shape[2]*out.shape[3]
                    if out.dim()==3:  # B,N,C
                        return out.shape[1]
        except Exception as e:
            print(f"[patches] fallback failed: {e}", flush=True)
    return -1

def evaluate(model_name, loader):
    t0 = time.time()
    print(f"\n=== {model_name} ===", flush=True)
    if not wait_for_gpu_room():
        return {"model_name":model_name,"status":"gpu_busy_abort",
                "params_M":0,"num_patches":0,"n_images_evaluated":0,
                "mean_bits_per_image":0,"mean_bits_per_patch":0,
                "mean_bits_per_pixel":0,"mean_ce_vs_uniform":0,
                "peak_vram_mb":0,"elapsed_seconds":time.time()-t0}
    try:
        model = timm.create_model(model_name, pretrained=True).eval().cuda().half()
    except Exception as e:
        print(f"[load failed] {e}", flush=True)
        return {"model_name":model_name,"status":"download_failed",
                "params_M":0,"num_patches":0,"n_images_evaluated":0,
                "mean_bits_per_image":0,"mean_bits_per_patch":0,
                "mean_bits_per_pixel":0,"mean_ce_vs_uniform":0,
                "peak_vram_mb":0,"elapsed_seconds":time.time()-t0}

    params_M = sum(p.numel() for p in model.parameters())/1e6
    num_patches = get_num_patches(model)
    print(f"[info] params={params_M:.2f}M  num_patches={num_patches}", flush=True)
    torch.cuda.reset_peak_memory_stats()

    total_ent = 0.0
    total_ceu = 0.0
    n = 0
    log2 = math.log(2)
    status = "ok"
    try:
        with torch.no_grad():
            for images,_ in loader:
                images = images.cuda(non_blocking=True).half()
                logits = model(images).float()
                logp = F.log_softmax(logits, dim=-1)
                p = logp.exp()
                ent = -(p * logp).sum(dim=-1) / log2  # bits
                # CE vs uniform u=1/K: -sum(u*logp)/log2 = -mean(logp)/log2
                ceu = -logp.mean(dim=-1) / log2
                total_ent += ent.sum().item()
                total_ceu += ceu.sum().item()
                n += images.shape[0]
    except torch.cuda.OutOfMemoryError as e:
        print(f"[oom] {e}", flush=True)
        status = "oom"
    except Exception as e:
        print(f"[err] {e}", flush=True)
        traceback.print_exc()
        status = "error"

    peak = torch.cuda.max_memory_allocated()/1024**2
    mbi = total_ent/n if n else 0.0
    mbp = mbi/num_patches if num_patches>0 else 0.0
    mbpx = mbi/(224*224*3)
    mceu = total_ceu/n if n else 0.0
    row = {"model_name":model_name,"params_M":round(params_M,3),
           "num_patches":num_patches,"n_images_evaluated":n,
           "mean_bits_per_image":round(mbi,6),
           "mean_bits_per_patch":round(mbp,8),
           "mean_bits_per_pixel":round(mbpx,10),
           "mean_ce_vs_uniform":round(mceu,6),
           "peak_vram_mb":round(peak,1),
           "elapsed_seconds":round(time.time()-t0,1),
           "status":status}
    print(f"[done] {row}", flush=True)
    del model
    torch.cuda.empty_cache()
    time.sleep(5)
    return row

def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(PLOTS, exist_ok=True)

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    ds = CIFAR100(root="/tmp/cifar100", train=False, download=True, transform=preprocess)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    models = [
        "vit_tiny_patch16_224","vit_small_patch16_224","vit_base_patch16_224",
        "deit_tiny_patch16_224","deit_small_patch16_224","deit_base_patch16_224",
        "swin_tiny_patch4_window7_224","swin_small_patch4_window7_224",
    ]

    all_skipped = True
    for m in models:
        row = evaluate(m, loader)
        append_row(row)
        if row["status"] == "ok":
            all_skipped = False

    if all_skipped:
        with open(f"{RESULTS}/ABORTED.md","w") as f:
            f.write("aborted: GPU too busy — no models evaluated successfully\n")
        print("aborted: no models completed", flush=True)

if __name__ == "__main__":
    main()
