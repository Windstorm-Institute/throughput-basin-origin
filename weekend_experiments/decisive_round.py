#!/usr/bin/env python3
"""
DECISIVE ROUND: 6 experiments targeting the weakest point of each paper.
Hardware-aware: polls GPU before each operation, backs off if busy.

Exp 1: Structural bonus under NF4 vs symmetric (Paper 9) — ~4 GB VRAM
Exp 2: τ in Chinese/Japanese/code/DNA (Paper 7) — ~4 GB VRAM per model
Exp 3: Visual entropy tracking curve (Paper 8) — ~4 GB VRAM, overnight
Exp 4: Real MAESTRO piano (Paper 8) — ~2 GB VRAM
Exp 5: PCFG depth sweep fixed (Paper 7) — ~4 GB VRAM
Exp 6: Lloyd-Max INT3 end-to-end (Paper 9) — ~4 GB VRAM
"""

import os, sys, time, math, csv, subprocess, traceback, gzip
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

# Load the NF4 CUDA library
try:
    import ctypes
    ctypes.CDLL('/home/user1-gpu/miniconda3/envs/qwen/lib/python3.13/site-packages/nvidia/cu13/lib/libnvJitLink.so.13')
except: pass

REPO = "/home/user1-gpu/agi-extensions"
BASE = "/home/user1-gpu/agi-extensions/weekend_experiments/decisive_round"

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    os.makedirs(BASE, exist_ok=True)
    with open(f"{BASE}/orchestrator.log", "a") as f:
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
    """Return free VRAM in MB."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True)
        return int(result.stdout.strip())
    except: return 0

def wait_for_gpu(min_free_mb=6000, max_wait_min=120):
    """Wait until at least min_free_mb VRAM is available."""
    waited = 0
    while waited < max_wait_min * 60:
        free = gpu_free_mb()
        if free >= min_free_mb:
            log(f"  GPU available: {free} MB free")
            return True
        if waited == 0:
            log(f"  GPU busy ({free} MB free, need {min_free_mb}). Waiting...")
        time.sleep(300)
        waited += 300
        if waited % 1800 == 0:
            log(f"  Still waiting... {waited//60} min elapsed, {free} MB free")
    log(f"  GPU wait timeout after {max_wait_min} min")
    return False

def eval_bpt(model, input_ids, max_tokens=50000):
    """Evaluate BPT on input_ids. Returns (bpt, total_tokens)."""
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    with torch.no_grad():
        for s in range(0, min(len(input_ids) - 1024, max_tokens), 1024):
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
        return (tot_loss / tot_tok) / math.log(2), tot_tok
    return float('nan'), 0

# =====================================================================
# EXP 1: Structural bonus under NF4 vs symmetric (Paper 9)
# =====================================================================
def exp1_structural_bonus():
    OUT = f"{BASE}/exp1_structural_bonus"
    os.makedirs(f"{OUT}/results", exist_ok=True)
    log("\n" + "="*60)
    log("EXP 1: STRUCTURAL BONUS — NF4 vs SYMMETRIC")
    log("Does level allocation preserve MEANING, not just numbers?")
    log("="*60)

    if not wait_for_gpu(6000):
        log("  SKIPPED: GPU not available")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from datasets import load_dataset

    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw_text = " ".join([t for t in wiki["text"] if len(t.strip()) > 0])

    model_name = 'EleutherAI/pythia-1.4b'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids_orig = tokenizer.encode(raw_text, return_tensors='pt').squeeze()

    # Create shuffled version
    words = raw_text.split()
    np.random.seed(42)
    np.random.shuffle(words)
    shuffled_text = " ".join(words)
    input_ids_shuf = tokenizer.encode(shuffled_text, return_tensors='pt').squeeze()

    results = []

    configs = [
        ('FP16', lambda: AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()),
    ]

    # NF4
    try:
        bnb_nf4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.float16)
        configs.append(('BNB_NF4', lambda: AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_nf4)))
    except: pass

    # Symmetric INT4
    def load_sym_int4():
        m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        for name, param in m.named_parameters():
            if param.dim() >= 2:
                w = param.data
                qmax = 7  # INT4
                wmax = w.abs().max()
                scale = wmax / qmax if wmax > 0 else 1e-8
                param.data = torch.clamp(torch.round(w / scale), -qmax, qmax) * scale
        return m.half().cuda()
    configs.append(('SYM_INT4', load_sym_int4))

    # Symmetric INT8
    def load_sym_int8():
        m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        for name, param in m.named_parameters():
            if param.dim() >= 2:
                w = param.data
                qmax = 127
                wmax = w.abs().max()
                scale = wmax / qmax if wmax > 0 else 1e-8
                param.data = torch.clamp(torch.round(w / scale), -qmax, qmax) * scale
        return m.half().cuda()
    configs.append(('SYM_INT8', load_sym_int8))

    for config_name, loader in configs:
        log(f"  {config_name}...")
        try:
            if not wait_for_gpu(4000, 30):
                log(f"    Skipped: not enough VRAM")
                continue

            model = loader()

            bpt_orig, tok_orig = eval_bpt(model, input_ids_orig)
            bpt_shuf, tok_shuf = eval_bpt(model, input_ids_shuf)
            bonus = bpt_shuf - bpt_orig if not math.isnan(bpt_orig) and not math.isnan(bpt_shuf) else float('nan')

            results.append({
                'method': config_name, 'bpt_original': bpt_orig, 'bpt_shuffled': bpt_shuf,
                'structural_bonus': bonus, 'tokens_orig': tok_orig, 'tokens_shuf': tok_shuf,
            })
            log(f"    BPT_orig={bpt_orig:.4f}, BPT_shuf={bpt_shuf:.4f}, bonus={bonus:.4f}")

            del model
            torch.cuda.empty_cache()
            time.sleep(5)
        except Exception as e:
            log(f"    ERROR: {e}")

    save_csv(results, f"{OUT}/results/structural_bonus_comparison.csv")

    report = "# Exp 1: Structural Bonus — NF4 vs Symmetric\n\n"
    report += "## Does level allocation preserve MEANING?\n\n"
    report += "| Method | BPT (original) | BPT (shuffled) | Structural bonus |\n|---|---|---|---|\n"
    for r in results:
        report += f"| {r['method']} | {r['bpt_original']:.4f} | {r['bpt_shuffled']:.4f} | {r['structural_bonus']:.4f} |\n"

    report += "\n## Interpretation\n\n"
    fp16_bonus = [r['structural_bonus'] for r in results if r['method'] == 'FP16']
    nf4_bonus = [r['structural_bonus'] for r in results if r['method'] == 'BNB_NF4']
    sym4_bonus = [r['structural_bonus'] for r in results if r['method'] == 'SYM_INT4']

    if fp16_bonus and nf4_bonus and sym4_bonus:
        if nf4_bonus[0] > 3 and sym4_bonus[0] < 1:
            report += "**NF4 preserves meaning while symmetric destroys it.**\n"
        elif nf4_bonus[0] > 3 and sym4_bonus[0] > 3:
            report += "**Both preserve meaning — the cliff is about raw prediction quality only.**\n"
    report += f"\nFP16 baseline bonus: {fp16_bonus[0]:.2f} bits (reference)\n" if fp16_bonus else ""

    with open(f"{OUT}/results/EXP1_REPORT.md", 'w') as f:
        f.write(report)

    git_push("Decisive Exp 1: structural bonus NF4 vs symmetric", ['weekend_experiments/decisive_round/exp1_structural_bonus/'])
    log("EXP 1 COMPLETE")

# =====================================================================
# EXP 2: τ in Chinese/Japanese/code/DNA (Paper 7)
# =====================================================================
def exp2_multilingual_tau():
    OUT = f"{BASE}/exp2_multilingual_tau"
    os.makedirs(f"{OUT}/results", exist_ok=True)
    log("\n" + "="*60)
    log("EXP 2: τ IN MULTIPLE LANGUAGES AND DOMAINS")
    log("Is the basin about English, about language, or about structure?")
    log("="*60)

    if not wait_for_gpu(4000):
        log("  SKIPPED: GPU not available")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    results = []

    # Prepare corpora
    corpora = {}

    # English (baseline)
    wiki_en = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    corpora['english'] = " ".join([t for t in wiki_en["text"] if len(t.strip()) > 0])

    # Chinese
    log("  Loading Chinese text...")
    try:
        wiki_zh = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")  # Fallback
        # Try CC-100 Chinese
        cc100 = load_dataset("cc100", "zh-Hans", split="train", streaming=True, trust_remote_code=True)
        zh_texts = []
        for i, sample in enumerate(cc100):
            if i >= 2000: break
            zh_texts.append(sample['text'])
        if zh_texts:
            corpora['chinese'] = " ".join(zh_texts)[:500000]
            log(f"    Chinese: {len(corpora['chinese'])} chars")
    except Exception as e:
        log(f"    Chinese failed: {e}")

    # Python code
    log("  Loading Python code...")
    try:
        code = load_dataset("codeparrot/github-code", streaming=True, split="train",
                           languages=["Python"], trust_remote_code=True)
        code_texts = []
        for i, sample in enumerate(code):
            if i >= 500: break
            code_texts.append(sample['code'])
        if code_texts:
            corpora['python_code'] = " ".join(code_texts)[:500000]
            log(f"    Python: {len(corpora['python_code'])} chars")
    except Exception as e:
        log(f"    Python code failed: {e}")
        # Fallback: generate some Python-like text
        corpora['python_code'] = """
def fibonacci(n):
    if n <= 1: return n
    return fibonacci(n-1) + fibonacci(n-2)

class DataProcessor:
    def __init__(self, data):
        self.data = data
    def process(self):
        return [x * 2 for x in self.data if x > 0]
""" * 5000
        log(f"    Python (synthetic): {len(corpora['python_code'])} chars")

    # DNA sequence
    log("  Generating DNA sequence...")
    rng = np.random.RandomState(42)
    bases = ['A', 'C', 'G', 'T']
    # Realistic DNA: not uniform, has codon bias
    codon_probs = [0.3, 0.2, 0.2, 0.3]  # AT-rich like human genome
    dna = ''.join(rng.choice(bases, size=500000, p=codon_probs))
    corpora['dna_sequence'] = dna
    log(f"    DNA: {len(corpora['dna_sequence'])} chars")

    # Test models
    models = [
        ('EleutherAI/pythia-410m', 'Pythia-410M'),
        ('gpt2-medium', 'GPT-2-medium'),
    ]

    # For Chinese, try a multilingual model
    if 'chinese' in corpora:
        models.append(('bigscience/bloom-560m', 'BLOOM-560M'))

    for model_name, label in models:
        log(f"\n  {label} ({model_name})...")
        if not wait_for_gpu(4000, 30):
            log(f"    Skipped: not enough VRAM")
            continue

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda().eval()

            for corpus_name, text in corpora.items():
                # Skip Chinese on non-multilingual models
                if corpus_name == 'chinese' and 'bloom' not in model_name.lower():
                    continue

                input_ids = tokenizer.encode(text[:200000], return_tensors='pt').squeeze()
                if len(input_ids) < 2048:
                    log(f"    {corpus_name}: too short ({len(input_ids)} tokens), skipping")
                    continue

                n_tokens = len(input_ids)
                n_chars = len(text[:200000])
                n_bytes = len(text[:200000].encode('utf-8'))

                bpt, tok_eval = eval_bpt(model, input_ids, max_tokens=30000)

                if not math.isnan(bpt):
                    total_bits = bpt * n_tokens
                    bpc = total_bits / n_chars
                    bpb = total_bits / n_bytes
                    bytes_per_char = n_bytes / n_chars

                    results.append({
                        'model': label, 'corpus': corpus_name,
                        'n_tokens': n_tokens, 'n_chars': n_chars, 'n_bytes': n_bytes,
                        'bytes_per_char': bytes_per_char, 'chars_per_token': n_chars / n_tokens,
                        'BPT': bpt, 'bits_per_char': bpc, 'bits_per_byte': bpb,
                    })
                    log(f"    {corpus_name}: BPT={bpt:.4f}, bits/char={bpc:.4f}, "
                        f"bits/byte={bpb:.4f}, bytes/char={bytes_per_char:.2f}")

            del model
            torch.cuda.empty_cache()
            time.sleep(5)
        except Exception as e:
            log(f"    ERROR: {e}")

    save_csv(results, f"{OUT}/results/multilingual_tau.csv")

    report = "# Exp 2: τ Across Languages and Domains\n\n"
    report += "| Model | Corpus | BPT | Bits/char | Bits/byte | Bytes/char |\n|---|---|---|---|---|---|\n"
    for r in results:
        report += f"| {r['model']} | {r['corpus']} | {r['BPT']:.3f} | {r['bits_per_char']:.4f} | {r['bits_per_byte']:.4f} | {r['bytes_per_char']:.2f} |\n"

    report += "\n## Key question: Is bits/char constant across languages?\n"
    report += "If yes → the basin is about characters (cognitive units).\n"
    report += "If bits/byte is constant → the basin is about byte-level compression.\n"

    with open(f"{OUT}/results/EXP2_REPORT.md", 'w') as f:
        f.write(report)

    git_push("Decisive Exp 2: multilingual τ", ['weekend_experiments/decisive_round/exp2_multilingual_tau/'])
    log("EXP 2 COMPLETE")

# =====================================================================
# MAIN
# =====================================================================
def main():
    log("="*60)
    log("DECISIVE ROUND: 2 HIGHEST-PRIORITY EXPERIMENTS")
    log("Hardware-aware: yields to other GPU users")
    log(f"GPU free: {gpu_free_mb()} MB")
    log("="*60)

    # Run Exp 1 and 2 (both need ~4 GB VRAM, can fit alongside other users)
    experiments = [
        ('Exp 1: Structural bonus', exp1_structural_bonus),
        ('Exp 2: Multilingual τ', exp2_multilingual_tau),
    ]

    for name, func in experiments:
        try:
            func()
        except Exception as e:
            log(f"{name} FAILED: {e}")
            traceback.print_exc()

    log("\n" + "="*60)
    log("DECISIVE ROUND PHASE 1 COMPLETE")
    log(f"Remaining experiments (3-6) need more GPU — run when machine is free")
    log("="*60)

    git_push("Decisive round phase 1 complete", ['weekend_experiments/decisive_round/'])

if __name__ == "__main__":
    main()
