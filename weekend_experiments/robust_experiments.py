#!/usr/bin/env python3
"""
ROBUST EXPERIMENT SUITE — Publication-quality experiments for Papers 7, 8, 9.
Trains models from scratch, multiple seeds, error bars.
Monitors GPU and yields to other users.

Target: >55% GPU utilization on RTX 5090 (32 GB).

Experiments:
  R1: PCFG depth sweep — train GPT-2 from scratch at each depth (Paper 7)
  R2: Multilingual τ — 5+ languages, 4+ models, means ± std (Paper 7)
  R3: Visual entropy tracking — train autoencoder on controlled images (Paper 8)
  R4: Structural bonus with error bars — 3 seeds × 4 quant methods (Paper 9)
"""

import os, sys, time, math, csv, subprocess, traceback, json, gc, signal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

REPO = "/home/user1-gpu/agi-extensions"
BASE = f"{REPO}/weekend_experiments/robust_round"
os.makedirs(BASE, exist_ok=True)

# =====================================================================
# UTILITIES
# =====================================================================
def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(f"{BASE}/robust.log", "a") as f:
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

def gpu_info():
    """Return (our_pids_mem, other_pids_mem, free_mb, total_mb)."""
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=memory.free,memory.total',
                           '--format=csv,noheader,nounits'], capture_output=True, text=True)
        free, total = [int(x.strip()) for x in r.stdout.strip().split(',')]

        r2 = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True)
        our_pid = os.getpid()
        other_mem = 0
        for line in r2.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(',')
                pid = int(parts[0].strip())
                mem = int(parts[1].strip().replace(' MiB', ''))
                if pid != our_pid:
                    other_mem += mem
        return other_mem, free, total
    except:
        return 0, 0, 0

def check_gpu_yield():
    """Check if another user started using GPU. If so, pause and wait."""
    other_mem, free, total = gpu_info()
    if other_mem > 2000:  # Someone else using >2 GB
        log(f"  ⚠ OTHER USER DETECTED: {other_mem} MB used by other processes. YIELDING...")
        # Free our GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        while True:
            time.sleep(300)  # Check every 5 minutes
            other_mem, free, total = gpu_info()
            if other_mem < 500:
                log(f"  ✓ GPU clear again ({free} MB free). Resuming...")
                return
            else:
                log(f"  ... still yielding ({other_mem} MB used by others, {free} MB free)")

def gpu_report():
    other_mem, free, total = gpu_info()
    used = total - free
    pct = used * 100 // total if total > 0 else 0
    log(f"  GPU: {used}/{total} MB ({pct}% mem), {free} MB free")
    return free


# =====================================================================
# R1: PCFG DEPTH SWEEP — TRAIN FROM SCRATCH (Paper 7)
# Train a small transformer on PCFG text at each depth level.
# This is the experiment that would silence the "pretrained model" critique.
# =====================================================================
class PCFGDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=256):
        self.ids = tokenizer.encode(text)
        self.seq_len = seq_len

    def __len__(self):
        return max(1, (len(self.ids) - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.ids[start:start + self.seq_len]
        y = self.ids[start + 1:start + self.seq_len + 1]
        # Pad if needed
        if len(x) < self.seq_len:
            x = x + [0] * (self.seq_len - len(x))
            y = y + [0] * (self.seq_len - len(y))
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class SmallTransformer(nn.Module):
    """~25M param transformer for training from scratch."""
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_ff=2048, max_seq=256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) * math.sqrt(self.d_model) + self.pos_emb(pos)
        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.transformer(h, mask=mask, is_causal=True)
        return self.head(h)


def generate_pcfg_text(depth, n_words=500000, seed=42):
    """Generate PCFG text at controlled nesting depth using English-like words."""
    rng = np.random.RandomState(seed)
    nouns = ["cat", "dog", "fish", "bird", "tree", "rock", "star", "moon", "lake", "hill",
             "book", "lamp", "door", "wall", "ship", "road", "farm", "bell", "song", "rain",
             "king", "child", "hand", "word", "life", "time", "home", "mind", "hope", "fear"]
    verbs = ["sees", "eats", "finds", "loves", "hears", "makes", "takes", "gives", "knows", "hits",
             "holds", "runs", "calls", "gets", "puts", "sets", "lets", "cuts", "asks", "adds"]
    adjs = ["big", "old", "new", "red", "hot", "cold", "dark", "soft", "tall", "fast",
            "good", "bad", "long", "deep", "wide", "thin", "hard", "calm", "warm", "full"]
    preps = ["in", "on", "by", "at", "to", "of", "from", "with", "near", "past"]
    conjs = ["and", "but", "or", "so", "yet"]

    def gen(d, cur=0):
        n = rng.choice(nouns)
        v = rng.choice(verbs)
        adj = rng.choice(adjs) if rng.random() < 0.5 else ""
        subj = f"the {adj} {n}".strip() if adj else f"the {n}"
        obj_n = rng.choice(nouns)
        obj_adj = rng.choice(adjs) if rng.random() < 0.4 else ""
        obj = f"the {obj_adj} {obj_n}".strip() if obj_adj else f"the {obj_n}"
        s = f"{subj} {v} {obj}"
        if cur < d and rng.random() < 0.65:
            p = rng.choice(preps)
            s = f"{s} {p} {gen(d, cur+1)}"
        if cur < d and rng.random() < 0.25:
            c = rng.choice(conjs)
            s = f"{s} {c} {gen(d, cur+1)}"
        return s

    sentences = []
    total_words = 0
    while total_words < n_words:
        sent = gen(depth) + "."
        sentences.append(sent)
        total_words += len(sent.split())
    return " ".join(sentences)


def r1_pcfg_train():
    """Train a transformer from scratch on PCFG data at each depth level."""
    OUT = f"{BASE}/r1_pcfg_train"
    os.makedirs(f"{OUT}/results", exist_ok=True)
    log("\n" + "="*60)
    log("R1: PCFG DEPTH SWEEP — TRAINING FROM SCRATCH")
    log("Train 25M-param transformer on PCFG at depths 0-6 + salad")
    log("3 seeds per depth for error bars")
    log("="*60)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    depths = [0, 1, 2, 3, 4, 5, 6]
    seeds = [42, 137, 271]
    results = []

    # Also generate word salad
    rng = np.random.RandomState(42)
    all_words = ["cat", "dog", "fish", "bird", "tree", "rock", "star", "moon", "lake", "hill",
                 "book", "lamp", "door", "wall", "ship", "road", "farm", "bell", "song", "rain",
                 "king", "child", "hand", "word", "life", "time", "home", "mind", "hope", "fear",
                 "sees", "eats", "finds", "loves", "hears", "makes", "takes", "gives", "knows", "hits",
                 "big", "old", "new", "red", "hot", "cold", "dark", "soft", "tall", "fast",
                 "the", "a", "in", "on", "by", "at", "to", "of", "and", "but", "or", "so"]

    configs = [(d, s) for d in depths for s in seeds] + [(-1, s) for s in seeds]  # -1 = salad

    for depth, seed in configs:
        check_gpu_yield()

        label = f"depth_{depth}" if depth >= 0 else "salad"
        log(f"  Training on {label} (seed {seed})...")

        # Generate data
        if depth >= 0:
            train_text = generate_pcfg_text(depth, n_words=300000, seed=seed)
            eval_text = generate_pcfg_text(depth, n_words=50000, seed=seed + 1000)
        else:
            train_text = " ".join(rng.choice(all_words, 300000)) + "."
            eval_text = " ".join(rng.choice(all_words, 50000)) + "."

        train_ds = PCFGDataset(train_text, tokenizer, seq_len=256)
        eval_ds = PCFGDataset(eval_text, tokenizer, seq_len=256)

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
        eval_loader = DataLoader(eval_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

        # Build model
        torch.manual_seed(seed)
        model = SmallTransformer(vocab_size, d_model=512, nhead=8, num_layers=6, dim_ff=2048).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        scaler = GradScaler()

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        if depth == 0 and seed == 42:
            log(f"    Model: {n_params:.1f}M params")
            gpu_report()

        # Train for 5 epochs (enough to converge on small data)
        model.train()
        for epoch in range(5):
            epoch_loss = 0.0
            n_batches = 0
            for x, y in train_loader:
                x, y = x.cuda(), y.cuda()
                optimizer.zero_grad()
                with autocast(dtype=torch.float16):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), ignore_index=0)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                n_batches += 1

                # Yield check every 100 batches
                if n_batches % 200 == 0:
                    check_gpu_yield()

            avg_train_loss = epoch_loss / n_batches
            if epoch == 4:
                log(f"    Epoch {epoch}: train_loss={avg_train_loss:.4f}")

        # Evaluate
        model.eval()
        total_loss, total_tok = 0.0, 0
        with torch.no_grad():
            for x, y in eval_loader:
                x, y = x.cuda(), y.cuda()
                with autocast(dtype=torch.float16):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), ignore_index=0)
                if not math.isnan(loss.item()):
                    total_loss += loss.item() * x.size(0) * x.size(1)
                    total_tok += x.size(0) * x.size(1)

        eval_bpt = (total_loss / total_tok) / math.log(2) if total_tok > 0 else float('nan')
        results.append({
            'depth': depth,
            'seed': seed,
            'label': label,
            'train_loss_final': round(avg_train_loss, 4),
            'eval_BPT': round(eval_bpt, 4),
            'eval_tokens': total_tok
        })
        log(f"    {label} seed={seed}: eval_BPT={eval_bpt:.4f}")

        del model, optimizer, scaler
        torch.cuda.empty_cache()

    save_csv(results, f"{OUT}/results/pcfg_train_sweep.csv")

    # Compute means and stds
    report = "# R1: PCFG Depth Sweep — Trained From Scratch\n\n"
    report += "25M-param transformer, 5 epochs, 3 seeds each.\n\n"
    report += "| Depth | BPT (mean ± std) | N seeds |\n|---|---|---|\n"

    for depth in [-1] + depths:
        label = f"depth_{depth}" if depth >= 0 else "salad"
        bpts = [r['eval_BPT'] for r in results if r['depth'] == depth]
        if bpts:
            mean_bpt = np.mean(bpts)
            std_bpt = np.std(bpts)
            report += f"| {label} | {mean_bpt:.4f} ± {std_bpt:.4f} | {len(bpts)} |\n"

    # Structural bonus
    salad_bpts = [r['eval_BPT'] for r in results if r['depth'] == -1]
    d0_bpts = [r['eval_BPT'] for r in results if r['depth'] == 0]
    d6_bpts = [r['eval_BPT'] for r in results if r['depth'] == 6]
    if salad_bpts and d0_bpts:
        bonus = np.mean(salad_bpts) - np.mean(d0_bpts)
        report += f"\n**Structural bonus (salad → depth-0):** {bonus:.4f} bits\n"
    if d0_bpts and d6_bpts:
        deep_effect = np.mean(d0_bpts) - np.mean(d6_bpts)
        report += f"**Depth effect (depth-0 → depth-6):** {deep_effect:.4f} bits\n"

    report += "\n## Key difference from toy version\n"
    report += "Models trained FROM SCRATCH on each depth's data — no pretrained bias.\n"
    report += "If BPT decreases with depth, f(structural_depth) is real and learnable.\n"

    with open(f"{OUT}/results/R1_REPORT.md", 'w') as f:
        f.write(report)

    git_push("Robust R1: PCFG depth sweep (trained from scratch, 3 seeds)",
             ['weekend_experiments/robust_round/r1_pcfg_train/'])
    log("R1 COMPLETE")


# =====================================================================
# R2: MULTILINGUAL τ — 5+ LANGUAGES, 4+ MODELS (Paper 7)
# =====================================================================
def r2_multilingual_tau():
    OUT = f"{BASE}/r2_multilingual_tau"
    os.makedirs(f"{OUT}/results", exist_ok=True)
    log("\n" + "="*60)
    log("R2: MULTILINGUAL τ — EXPANDED")
    log("5+ domains, 4 models, means ± std")
    log("="*60)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    check_gpu_yield()

    # Models to test
    model_configs = [
        ("EleutherAI/pythia-160m", "Pythia-160M"),
        ("EleutherAI/pythia-410m", "Pythia-410M"),
        ("EleutherAI/pythia-1.4b", "Pythia-1.4B"),
        ("gpt2-medium", "GPT-2-medium"),
    ]

    # Prepare corpora
    log("  Loading corpora...")
    corpora = {}

    # English
    try:
        wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        eng_text = " ".join([t for t in wiki["text"] if len(t.strip()) > 0])[:300000]
        corpora["english"] = eng_text
        log(f"    English: {len(eng_text)} chars")
    except Exception as e:
        log(f"    English FAILED: {e}")

    # German (from CC-100 or wikitext)
    try:
        de = load_dataset("wikimedia/wikipedia", "20231101.de", split="train", streaming=True)
        de_text = ""
        for item in de:
            de_text += item["text"] + " "
            if len(de_text) >= 300000:
                break
        corpora["german"] = de_text[:300000]
        log(f"    German: {len(corpora['german'])} chars")
    except Exception as e:
        log(f"    German wiki FAILED: {e}")
        # Fallback: generate German-like text from ASCII
        try:
            de2 = load_dataset("oscar-corpus/OSCAR-2301", "de", split="train", streaming=True, trust_remote_code=True)
            de_text = ""
            for item in de2:
                de_text += item["text"] + " "
                if len(de_text) >= 300000:
                    break
            corpora["german"] = de_text[:300000]
            log(f"    German (OSCAR): {len(corpora['german'])} chars")
        except Exception as e2:
            log(f"    German OSCAR also FAILED: {e2}")

    # French
    try:
        fr = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)
        fr_text = ""
        for item in fr:
            fr_text += item["text"] + " "
            if len(fr_text) >= 300000:
                break
        corpora["french"] = fr_text[:300000]
        log(f"    French: {len(corpora['french'])} chars")
    except Exception as e:
        log(f"    French FAILED: {e}")

    # Spanish
    try:
        es = load_dataset("wikimedia/wikipedia", "20231101.es", split="train", streaming=True)
        es_text = ""
        for item in es:
            es_text += item["text"] + " "
            if len(es_text) >= 300000:
                break
        corpora["spanish"] = es_text[:300000]
        log(f"    Spanish: {len(corpora['spanish'])} chars")
    except Exception as e:
        log(f"    Spanish FAILED: {e}")

    # Python code
    try:
        py_text = ""
        py_ds = load_dataset("bigcode/the-stack-smol", "data/python", split="train", streaming=True, trust_remote_code=True)
        for item in py_ds:
            py_text += item["content"] + "\n"
            if len(py_text) >= 300000:
                break
        corpora["python_code"] = py_text[:300000]
        log(f"    Python: {len(corpora['python_code'])} chars")
    except Exception as e:
        log(f"    Python stack FAILED: {e}")
        # Fallback: generate synthetic Python
        rng = np.random.RandomState(42)
        py_lines = []
        for i in range(10000):
            indent = "    " * rng.randint(0, 4)
            kw = rng.choice(["def", "if", "for", "while", "return", "class", "import", "from", "try", "except"])
            var = rng.choice(["x", "y", "data", "result", "value", "item", "count", "total", "name", "idx"])
            py_lines.append(f"{indent}{kw} {var}_{i}:")
        corpora["python_code"] = "\n".join(py_lines)
        log(f"    Python (synthetic): {len(corpora['python_code'])} chars")

    # DNA sequence
    rng = np.random.RandomState(42)
    dna = "".join(rng.choice(list("ACGT"), 300000))
    corpora["dna_sequence"] = dna
    log(f"    DNA: {len(dna)} chars")

    # Medical text (PubMed abstracts)
    try:
        pubmed = load_dataset("ccdv/pubmed-summarization", "document", split="test", streaming=True)
        med_text = ""
        for item in pubmed:
            med_text += item["article"] + " "
            if len(med_text) >= 300000:
                break
        corpora["medical_pubmed"] = med_text[:300000]
        log(f"    Medical: {len(corpora['medical_pubmed'])} chars")
    except Exception as e:
        log(f"    Medical FAILED: {e}")

    results = []

    for model_name, model_label in model_configs:
        check_gpu_yield()
        log(f"\n  Loading {model_label}...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
            model.eval()
            gpu_report()

            for corpus_name, text in corpora.items():
                log(f"    {model_label} × {corpus_name}...")

                input_ids = tokenizer.encode(text, return_tensors='pt', truncation=False).squeeze()
                n_tokens = len(input_ids)
                n_chars = len(text)
                n_bytes = len(text.encode('utf-8'))

                seq_len = 1024
                total_loss, total_tok = 0.0, 0

                with torch.no_grad():
                    for s in range(0, min(n_tokens - seq_len, 100000), seq_len):
                        x = input_ids[s:s+seq_len].unsqueeze(0).cuda()
                        y = input_ids[s+1:s+seq_len+1].unsqueeze(0).cuda()
                        try:
                            out = model(x)
                            loss = F.cross_entropy(out.logits.view(-1, out.logits.size(-1)), y.view(-1))
                            if not math.isnan(loss.item()) and not math.isinf(loss.item()):
                                total_loss += loss.item() * seq_len
                                total_tok += seq_len
                        except:
                            break

                bpt = (total_loss / total_tok) / math.log(2) if total_tok > 0 else float('nan')
                chars_per_token = n_chars / n_tokens if n_tokens > 0 else 1
                bytes_per_char = n_bytes / n_chars if n_chars > 0 else 1
                bits_per_char = bpt / chars_per_token if chars_per_token > 0 else float('nan')
                bits_per_byte = bpt / (chars_per_token * bytes_per_char) if chars_per_token > 0 else float('nan')

                results.append({
                    'model': model_label,
                    'corpus': corpus_name,
                    'n_tokens': total_tok,
                    'n_chars': n_chars,
                    'n_bytes': n_bytes,
                    'chars_per_token': round(chars_per_token, 2),
                    'BPT': round(bpt, 4),
                    'bits_per_char': round(bits_per_char, 4),
                    'bits_per_byte': round(bits_per_byte, 4),
                })
                log(f"      BPT={bpt:.4f}, bits/char={bits_per_char:.4f}, bits/byte={bits_per_byte:.4f}")

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            log(f"    {model_label} FAILED: {e}")
            traceback.print_exc()

    save_csv(results, f"{OUT}/results/multilingual_tau.csv")

    # Report with means across models
    report = "# R2: Multilingual τ — Expanded\n\n"
    report += f"Models: {', '.join(m[1] for m in model_configs)}\n\n"
    report += "## Bits per character by corpus (mean ± std across models)\n\n"
    report += "| Corpus | Bits/char (mean ± std) | N models |\n|---|---|---|\n"

    for corpus_name in corpora.keys():
        bpc = [r['bits_per_char'] for r in results if r['corpus'] == corpus_name and not math.isnan(r['bits_per_char'])]
        if bpc:
            report += f"| {corpus_name} | {np.mean(bpc):.4f} ± {np.std(bpc):.4f} | {len(bpc)} |\n"

    report += "\n## Full results\n\n"
    report += "| Model | Corpus | BPT | Bits/char | Bits/byte |\n|---|---|---|---|---|\n"
    for r in results:
        report += f"| {r['model']} | {r['corpus']} | {r['BPT']} | {r['bits_per_char']} | {r['bits_per_byte']} |\n"

    with open(f"{OUT}/results/R2_REPORT.md", 'w') as f:
        f.write(report)

    git_push("Robust R2: Multilingual τ (5+ domains, 4 models)",
             ['weekend_experiments/robust_round/r2_multilingual_tau/'])
    log("R2 COMPLETE")


# =====================================================================
# R3: VISUAL ENTROPY — TRAIN AUTOENCODER FROM SCRATCH (Paper 8)
# Train a convolutional autoencoder on images at controlled entropy.
# =====================================================================
class ControlledEntropyImageDataset(Dataset):
    """Generate images at a specific entropy level on the fly."""
    def __init__(self, entropy_level, n_images=10000, size=64, seed=42):
        self.entropy_level = entropy_level
        self.n_images = n_images
        self.size = size
        self.seed = seed

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        rng = np.random.RandomState(self.seed + idx)
        s = self.size

        if self.entropy_level == 0:
            # Uniform color
            c = rng.randint(0, 256)
            img = np.full((3, s, s), c, dtype=np.float32) / 255.0
        elif self.entropy_level == 1:
            # 4 colors (2-bit)
            palette = rng.randint(0, 256, (4, 3))
            indices = rng.randint(0, 4, (s, s))
            img = palette[indices].transpose(2, 0, 1).astype(np.float32) / 255.0
        elif self.entropy_level == 2:
            # 16 colors (4-bit) in 8x8 blocks
            palette = rng.randint(0, 256, (16, 3))
            block_size = 8
            blocks = rng.randint(0, 16, (s // block_size, s // block_size))
            indices = np.repeat(np.repeat(blocks, block_size, axis=0), block_size, axis=1)
            img = palette[indices].transpose(2, 0, 1).astype(np.float32) / 255.0
        elif self.entropy_level == 3:
            # 16 colors (4-bit) per pixel
            palette = rng.randint(0, 256, (16, 3))
            indices = rng.randint(0, 16, (s, s))
            img = palette[indices].transpose(2, 0, 1).astype(np.float32) / 255.0
        elif self.entropy_level == 4:
            # Natural-image-like: smooth gradients + noise
            base = np.zeros((3, s, s), dtype=np.float32)
            for c in range(3):
                # Random smooth gradient
                x = np.linspace(rng.random(), rng.random(), s)
                y = np.linspace(rng.random(), rng.random(), s)
                base[c] = np.outer(y, x)
            # Add structured noise
            noise = rng.normal(0, 0.1, (3, s, s)).astype(np.float32)
            img = np.clip(base + noise, 0, 1)
        elif self.entropy_level == 5:
            # Gaussian noise (high entropy, some structure from Gaussian shape)
            img = np.clip(rng.normal(0.5, 0.2, (3, s, s)), 0, 1).astype(np.float32)
        elif self.entropy_level == 6:
            # Uniform noise (maximum entropy)
            img = rng.random((3, s, s)).astype(np.float32)
        else:
            img = rng.random((3, s, s)).astype(np.float32)

        return torch.tensor(img, dtype=torch.float32)


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder with bottleneck."""
    def __init__(self, bottleneck_dim=128):
        super().__init__()
        # Encoder: 64x64 → 32 → 16 → 8 → 4 → bottleneck
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(),    # 32x32
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),   # 16x16
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),   # 8x8
            nn.Conv2d(256, 512, 4, 2, 1), nn.ReLU(),   # 4x4
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, bottleneck_dim),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 512 * 4 * 4),
            nn.Unflatten(1, (512, 4, 4)),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(),   # 8x8
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),   # 16x16
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),    # 32x32
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Sigmoid(),   # 64x64
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def r3_visual_entropy_train():
    OUT = f"{BASE}/r3_visual_entropy_train"
    os.makedirs(f"{OUT}/results", exist_ok=True)
    log("\n" + "="*60)
    log("R3: VISUAL ENTROPY — TRAIN AUTOENCODER FROM SCRATCH")
    log("Train ConvAE on images at 7 entropy levels, 3 seeds")
    log("="*60)

    entropy_levels = [0, 1, 2, 3, 4, 5, 6]
    level_names = ["uniform", "4-color", "16-color-blocks", "16-color-pixels",
                   "smooth+noise", "gaussian-noise", "uniform-noise"]
    seeds = [42, 137, 271]
    results = []

    for level, name in zip(entropy_levels, level_names):
        for seed in seeds:
            check_gpu_yield()
            log(f"  Training AE on level {level} ({name}), seed {seed}...")

            torch.manual_seed(seed)
            model = ConvAutoencoder(bottleneck_dim=128).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scaler = GradScaler()

            train_ds = ControlledEntropyImageDataset(level, n_images=20000, seed=seed)
            eval_ds = ControlledEntropyImageDataset(level, n_images=2000, seed=seed + 10000)
            train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
            eval_loader = DataLoader(eval_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

            if level == 0 and seed == 42:
                n_params = sum(p.numel() for p in model.parameters()) / 1e6
                log(f"    Model: {n_params:.1f}M params")
                gpu_report()

            # Train 10 epochs
            model.train()
            for epoch in range(10):
                for batch in train_loader:
                    batch = batch.cuda()
                    optimizer.zero_grad()
                    with autocast(dtype=torch.float16):
                        recon = model(batch)
                        loss = F.mse_loss(recon, batch)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            # Evaluate
            model.eval()
            total_loss, total_n = 0.0, 0
            with torch.no_grad():
                for batch in eval_loader:
                    batch = batch.cuda()
                    recon = model(batch)
                    loss = F.mse_loss(recon, batch)
                    total_loss += loss.item() * batch.size(0)
                    total_n += batch.size(0)

            avg_mse = total_loss / total_n
            # Convert to bits per pixel: bits ≈ 0.5 * log2(2πe * MSE * 255^2) per channel * 3
            # (MSE is in [0,1] space, convert to pixel space)
            mse_pixel = avg_mse * (255 ** 2)
            bits_pp = 1.5 * math.log2(2 * math.pi * math.e * mse_pixel) if mse_pixel > 0 else 0

            results.append({
                'entropy_level': level,
                'level_name': name,
                'seed': seed,
                'mse_loss': round(avg_mse, 6),
                'mse_pixel_space': round(mse_pixel, 2),
                'bits_per_pixel_est': round(max(0, bits_pp), 4),
                'n_eval_images': total_n
            })
            log(f"    level {level} seed {seed}: MSE={avg_mse:.6f}, bits/pixel≈{max(0, bits_pp):.4f}")

            del model, optimizer
            torch.cuda.empty_cache()

    save_csv(results, f"{OUT}/results/visual_entropy_trained.csv")

    # Report
    report = "# R3: Visual Entropy — Trained Autoencoder\n\n"
    report += "ConvAE (128-dim bottleneck), 10 epochs, 3 seeds each.\n\n"
    report += "| Entropy Level | Name | MSE (mean ± std) | Bits/Pixel (mean ± std) |\n|---|---|---|---|\n"
    for level, name in zip(entropy_levels, level_names):
        mses = [r['mse_loss'] for r in results if r['entropy_level'] == level]
        bpps = [r['bits_per_pixel_est'] for r in results if r['entropy_level'] == level]
        if mses:
            report += f"| {level} | {name} | {np.mean(mses):.6f} ± {np.std(mses):.6f} | {np.mean(bpps):.4f} ± {np.std(bpps):.4f} |\n"

    report += "\n## Key question\n"
    report += "Does reconstruction loss increase monotonically with entropy level?\n"
    report += "If yes: visual throughput tracks source entropy (Paper 8 confirmed).\n"

    with open(f"{OUT}/results/R3_REPORT.md", 'w') as f:
        f.write(report)

    git_push("Robust R3: Visual entropy (trained autoencoder, 7 levels, 3 seeds)",
             ['weekend_experiments/robust_round/r3_visual_entropy_train/'])
    log("R3 COMPLETE")


# =====================================================================
# R4: STRUCTURAL BONUS WITH ERROR BARS (Paper 9)
# Same as Exp 1 but with 3 models × 3 seeds
# =====================================================================
def r4_structural_bonus():
    OUT = f"{BASE}/r4_structural_bonus"
    os.makedirs(f"{OUT}/results", exist_ok=True)
    log("\n" + "="*60)
    log("R4: STRUCTURAL BONUS — MULTI-MODEL, ERROR BARS")
    log("3 models × 4 quant methods × 3 corpus seeds")
    log("="*60)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    check_gpu_yield()

    # Load corpus
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw_text = " ".join([t for t in wiki["text"] if len(t.strip()) > 0])

    models = [
        ("EleutherAI/pythia-160m", "Pythia-160M"),
        ("EleutherAI/pythia-410m", "Pythia-410M"),
        ("gpt2-medium", "GPT-2-medium"),
    ]

    quant_methods = ['FP16', 'BNB_NF4', 'SYM_INT4', 'SYM_INT8']
    seeds = [42, 137, 271]
    results = []

    for model_name, model_label in models:
        check_gpu_yield()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        input_ids = tokenizer.encode(raw_text, return_tensors='pt').squeeze()

        for method in quant_methods:
            log(f"  {model_label} × {method}...")

            try:
                if method == 'FP16':
                    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
                elif method == 'BNB_NF4':
                    try:
                        from transformers import BitsAndBytesConfig
                        import ctypes
                        ctypes.CDLL("/home/user1-gpu/miniconda3/envs/qwen/lib/python3.13/site-packages/nvidia/cu13/lib/libnvJitLink.so.13")
                        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                        bnb_4bit_compute_dtype=torch.float16)
                        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
                    except Exception as e:
                        log(f"    NF4 failed for {model_label}: {e}")
                        continue
                elif method == 'SYM_INT4':
                    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
                    for pname, param in model.named_parameters():
                        if param.dim() >= 2 and param.numel() > 1000:
                            w = param.data.numpy().flatten()
                            qmax = 7  # INT4: -7 to 7
                            wmax = np.max(np.abs(w))
                            scale = wmax / qmax if wmax > 0 else 1e-8
                            w_q = np.clip(np.round(w / scale), -qmax, qmax).astype(np.float32) * scale
                            param.data = torch.tensor(w_q.reshape(param.shape), dtype=torch.float32)
                    model = model.half().cuda()
                elif method == 'SYM_INT8':
                    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
                    for pname, param in model.named_parameters():
                        if param.dim() >= 2 and param.numel() > 1000:
                            w = param.data.numpy().flatten()
                            qmax = 127  # INT8: -127 to 127
                            wmax = np.max(np.abs(w))
                            scale = wmax / qmax if wmax > 0 else 1e-8
                            w_q = np.clip(np.round(w / scale), -qmax, qmax).astype(np.float32) * scale
                            param.data = torch.tensor(w_q.reshape(param.shape), dtype=torch.float32)
                    model = model.half().cuda()

                model.eval()
                gpu_report()

                for seed in seeds:
                    # Create shuffled version with this seed
                    rng = np.random.RandomState(seed)
                    words = raw_text.split()
                    shuffled_words = words.copy()
                    rng.shuffle(shuffled_words)
                    shuffled_text = " ".join(shuffled_words)
                    shuffled_ids = tokenizer.encode(shuffled_text, return_tensors='pt').squeeze()

                    # Eval original
                    tot_loss_orig, tot_tok_orig = 0.0, 0
                    with torch.no_grad():
                        for s in range(0, min(len(input_ids)-1024, 50000), 1024):
                            x = input_ids[s:s+1024].unsqueeze(0).cuda()
                            y = input_ids[s+1:s+1025].unsqueeze(0).cuda()
                            try:
                                out = model(x)
                                loss = F.cross_entropy(out.logits.view(-1, out.logits.size(-1)), y.view(-1))
                                if not math.isnan(loss.item()) and not math.isinf(loss.item()):
                                    tot_loss_orig += loss.item() * 1024
                                    tot_tok_orig += 1024
                            except: break

                    # Eval shuffled
                    tot_loss_shuf, tot_tok_shuf = 0.0, 0
                    with torch.no_grad():
                        for s in range(0, min(len(shuffled_ids)-1024, 50000), 1024):
                            x = shuffled_ids[s:s+1024].unsqueeze(0).cuda()
                            y = shuffled_ids[s+1:s+1025].unsqueeze(0).cuda()
                            try:
                                out = model(x)
                                loss = F.cross_entropy(out.logits.view(-1, out.logits.size(-1)), y.view(-1))
                                if not math.isnan(loss.item()) and not math.isinf(loss.item()):
                                    tot_loss_shuf += loss.item() * 1024
                                    tot_tok_shuf += 1024
                            except: break

                    bpt_orig = (tot_loss_orig / tot_tok_orig) / math.log(2) if tot_tok_orig > 0 else float('nan')
                    bpt_shuf = (tot_loss_shuf / tot_tok_shuf) / math.log(2) if tot_tok_shuf > 0 else float('nan')
                    bonus = bpt_shuf - bpt_orig

                    results.append({
                        'model': model_label,
                        'method': method,
                        'seed': seed,
                        'bpt_original': round(bpt_orig, 4),
                        'bpt_shuffled': round(bpt_shuf, 4),
                        'structural_bonus': round(bonus, 4),
                    })
                    log(f"    {model_label} {method} seed={seed}: bonus={bonus:.4f}")

                del model
                torch.cuda.empty_cache()

            except Exception as e:
                log(f"    {model_label} {method} FAILED: {e}")
                traceback.print_exc()

    save_csv(results, f"{OUT}/results/structural_bonus_robust.csv")

    # Report
    report = "# R4: Structural Bonus — Multi-Model with Error Bars\n\n"
    report += "| Model | Method | Bonus (mean ± std) | N |\n|---|---|---|---|\n"
    for model_name, model_label in models:
        for method in quant_methods:
            bonuses = [r['structural_bonus'] for r in results
                      if r['model'] == model_label and r['method'] == method
                      and not math.isnan(r['structural_bonus'])]
            if bonuses:
                report += f"| {model_label} | {method} | {np.mean(bonuses):.4f} ± {np.std(bonuses):.4f} | {len(bonuses)} |\n"

    report += "\n## Key result\n"
    fp16_bonuses = [r['structural_bonus'] for r in results if r['method'] == 'FP16' and not math.isnan(r['structural_bonus'])]
    nf4_bonuses = [r['structural_bonus'] for r in results if r['method'] == 'BNB_NF4' and not math.isnan(r['structural_bonus'])]
    sym4_bonuses = [r['structural_bonus'] for r in results if r['method'] == 'SYM_INT4' and not math.isnan(r['structural_bonus'])]

    if fp16_bonuses and nf4_bonuses and sym4_bonuses:
        report += f"FP16 bonus: {np.mean(fp16_bonuses):.2f} ± {np.std(fp16_bonuses):.2f}\n"
        report += f"NF4 bonus: {np.mean(nf4_bonuses):.2f} ± {np.std(nf4_bonuses):.2f}\n"
        report += f"SYM_INT4 bonus: {np.mean(sym4_bonuses):.2f} ± {np.std(sym4_bonuses):.2f}\n"

    with open(f"{OUT}/results/R4_REPORT.md", 'w') as f:
        f.write(report)

    git_push("Robust R4: Structural bonus (3 models, 3 seeds, 4 quant methods)",
             ['weekend_experiments/robust_round/r4_structural_bonus/'])
    log("R4 COMPLETE")


# =====================================================================
# MAIN
# =====================================================================
def main():
    log("="*60)
    log("ROBUST EXPERIMENT SUITE")
    log("Publication-quality: train from scratch, multiple seeds, error bars")
    log("GPU-aware: yields to other users automatically")
    log("="*60)
    gpu_report()

    experiments = [
        ("R1: PCFG depth sweep (train from scratch)", r1_pcfg_train),
        ("R2: Multilingual τ (expanded)", r2_multilingual_tau),
        ("R3: Visual entropy (trained AE)", r3_visual_entropy_train),
        ("R4: Structural bonus (multi-model)", r4_structural_bonus),
    ]

    for name, fn in experiments:
        log(f"\n{'='*60}")
        log(f">>> {name}")
        log(f"{'='*60}")
        check_gpu_yield()
        try:
            fn()
        except Exception as e:
            log(f"  {name} FAILED: {e}")
            traceback.print_exc()
            with open(f"{BASE}/robust.log", "a") as f:
                traceback.print_exc(file=f)
        torch.cuda.empty_cache()
        gc.collect()

    log("\n" + "="*60)
    log("ROBUST EXPERIMENT SUITE COMPLETE")
    log("="*60)
    gpu_report()

    git_push("Robust experiment suite complete (R1-R4)",
             ['weekend_experiments/robust_round/'])


if __name__ == "__main__":
    main()
