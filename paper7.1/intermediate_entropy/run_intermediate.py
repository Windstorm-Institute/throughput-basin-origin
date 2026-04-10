#!/usr/bin/env python3
"""
Intermediate Entropy Sweep: SYN-5, SYN-6, SYN-7
Fills the gap between SYN-4 (H=3.7) and SYN-8 (H=8.0).

Waits for GPU to free up (R5 runs first), generates corpora on CPU while waiting.
Saves results incrementally. Git commits when done.
"""

import os, sys, time, math, csv, subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
from scipy.optimize import brentq

OUTDIR = "/home/user1-gpu/agi-extensions/paper7.1/intermediate_entropy"
REPO = "/home/user1-gpu/agi-extensions"
SEEDS = [42, 137]
VOCAB_SIZE = 8192
SEQ_LEN = 512
TOTAL_STEPS = 40000
EVAL_EVERY = 5000
LOG_EVERY = 200
CORPUS_SIZE = 100_000_000
BATCH_SIZE = 16

CORPORA = [
    ('syn5', 32, 5.0),
    ('syn6', 64, 6.0),
    ('syn7', 128, 7.0),
]

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(f"{OUTDIR}/run.log", "a") as f:
        f.write(line + "\n")

def entropy_of_zipf(alpha, K):
    ranks = np.arange(1, K+1, dtype=np.float64)
    probs = ranks ** (-alpha)
    probs /= probs.sum()
    return -np.sum(probs * np.log2(probs + 1e-15))

def find_alpha(target_H, K):
    max_H = math.log2(K)
    if target_H >= max_H - 0.01:
        return 0.001
    f = lambda a: entropy_of_zipf(a, K) - target_H
    try:
        return brentq(f, 0.001, 20.0)
    except:
        return 0.5

def generate_corpus(name, K, target_H):
    """Generate a Markov-0 corpus with controlled entropy."""
    path = f"{OUTDIR}/corpora/{name}.bin"
    if os.path.exists(path) and os.path.getsize(path) >= CORPUS_SIZE:
        log(f"{name} corpus exists ({os.path.getsize(path)} bytes)")
        data = np.fromfile(path, dtype=np.uint8)[:10_000_000]
        counts = Counter(data.tolist())
        total = sum(counts.values())
        H = -sum((c/total) * math.log2(c/total) for c in counts.values())
        return path, H

    log(f"Generating {name}: K={K}, target H={target_H}")
    alpha = find_alpha(target_H, K)
    ranks = np.arange(1, K+1, dtype=np.float64)
    probs = ranks ** (-alpha)
    probs /= probs.sum()
    H_theoretical = -np.sum(probs * np.log2(probs))
    log(f"  Zipf alpha={alpha:.4f}, theoretical H={H_theoretical:.4f}")

    rng = np.random.default_rng(42)
    data = rng.choice(K, size=CORPUS_SIZE, p=probs).astype(np.uint8)
    data.tofile(path)

    # Verify
    counts = Counter(data[:10_000_000].tolist())
    total = sum(counts.values())
    H_emp = -sum((c/total) * math.log2(c/total) for c in counts.values())
    log(f"  {name} generated: H_empirical={H_emp:.4f}, unique={len(counts)}")
    return path, H_emp

def build_tokenizer(name, corpus_path):
    """Build BPE tokenizer for this corpus."""
    tok_dir = f"{OUTDIR}/tokenizers/{name}"
    tok_file = f"{tok_dir}/tokenizer.json"
    if os.path.exists(tok_file):
        log(f"Tokenizer for {name} exists")
        from transformers import PreTrainedTokenizerFast
        return PreTrainedTokenizerFast(tokenizer_file=tok_file)

    log(f"Training tokenizer for {name}")
    data = np.fromfile(corpus_path, dtype=np.uint8)[:50_000_000]
    text = ' '.join(f'x{b:02x}' for b in data)
    chunks = [text[i:i+10000] for i in range(0, len(text), 10000)]

    from tokenizers import ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(chunks, vocab_size=VOCAB_SIZE, min_frequency=2,
                                   special_tokens=["<pad>", "<eos>", "<unk>"])
    os.makedirs(tok_dir, exist_ok=True)
    tokenizer.save(tok_file)

    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file=tok_file)
    tok.pad_token = "<pad>"
    log(f"  Tokenizer trained: vocab={tok.vocab_size}")
    return tok

def tokenize_corpus(corpus_path, tokenizer):
    """Tokenize and split."""
    data = np.fromfile(corpus_path, dtype=np.uint8)
    text = ' '.join(f'x{b:02x}' for b in data)

    split = int(len(text) * 0.9)
    train_text = text[:split]
    eval_text = text[split:]

    train_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    eval_ids = torch.tensor(tokenizer.encode(eval_text), dtype=torch.long)
    n_eval_chars = len(eval_text)

    log(f"  Train: {len(train_ids)} tokens, Eval: {len(eval_ids)} tokens, Eval chars: {n_eval_chars}")
    return train_ids, eval_ids, n_eval_chars

def wait_for_gpu(max_mb=5000, max_wait_min=360):
    """Wait until GPU has less than max_mb used."""
    waited = 0
    while waited < max_wait_min * 60:
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            used = int(result.stdout.strip())
            if used < max_mb:
                log(f"GPU free ({used} MB used). Proceeding.")
                return True
        except:
            pass
        if waited == 0:
            log(f"GPU busy. Waiting for <{max_mb} MB used...")
        time.sleep(600)
        waited += 600
        log(f"  Still waiting... ({waited//60} min elapsed)")
    log("GPU wait timeout. Proceeding anyway.")
    return False

def get_random_batch(token_ids, batch_size, seq_len, rng):
    max_start = len(token_ids) - seq_len - 1
    if max_start <= 0:
        max_start = 1
    starts = rng.integers(0, max_start, size=batch_size)
    x = torch.stack([token_ids[s:s+seq_len] for s in starts])
    y = torch.stack([token_ids[s+1:s+seq_len+1] for s in starts])
    return x, y

def evaluate(model, eval_ids, n_eval_chars):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, len(eval_ids) - SEQ_LEN, SEQ_LEN):
            x = eval_ids[start:start+SEQ_LEN].unsqueeze(0).cuda()
            y = eval_ids[start+1:start+SEQ_LEN+1].unsqueeze(0).cuda()
            with autocast():
                out = model(x)
                logits = out.logits if hasattr(out, 'logits') else out[0]
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item() * SEQ_LEN
            total_tokens += SEQ_LEN
    bpt = (total_loss / total_tokens) / math.log(2)
    bpss = (bpt * total_tokens) / n_eval_chars if n_eval_chars > 0 else float('nan')
    model.train()
    return bpt, bpss

def train_model(name, seed, train_ids, eval_ids, n_eval_chars, vocab_size, source_H):
    log(f"\n{'='*60}")
    log(f"Training {name} seed={seed}")
    log(f"{'='*60}")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    from transformers import GPT2Config, GPT2LMHeadModel
    config = GPT2Config(vocab_size=vocab_size, n_embd=768, n_layer=12, n_head=12,
                         n_positions=SEQ_LEN, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    model = GPT2LMHeadModel(config).cuda()
    model.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  {n_params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scaler = GradScaler()

    start_time = time.time()
    losses = []

    for step in range(1, TOTAL_STEPS + 1):
        if step < 1000:
            lr = 3e-4 * step / 1000
        else:
            lr = 3e-4 * 0.5 * (1 + math.cos(math.pi * (step - 1000) / (TOTAL_STEPS - 1000)))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        x, y = get_random_batch(train_ids, BATCH_SIZE, SEQ_LEN, rng)
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        with autocast():
            out = model(x)
            logits = out.logits if hasattr(out, 'logits') else out[0]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        losses.append((step, loss.item()))
        if step % LOG_EVERY == 0:
            log(f"  step {step}/{TOTAL_STEPS}, loss={loss.item():.4f}")

        if step % EVAL_EVERY == 0 or step == TOTAL_STEPS:
            bpt, bpss = evaluate(model, eval_ids, n_eval_chars)
            log(f"  EVAL step {step}: BPT={bpt:.4f}, BPSS*={bpss:.4f}")

    elapsed = time.time() - start_time
    final_bpt, final_bpss = evaluate(model, eval_ids, n_eval_chars)
    final_bpss_over_h = final_bpss / source_H if source_H > 0 else float('nan')

    # Plateau slope
    if len(losses) > 100:
        last = losses[int(len(losses)*0.8):]
        slope = np.polyfit([s for s,l in last], [l for s,l in last], 1)[0]
    else:
        slope = float('nan')

    log(f"  FINAL: BPT={final_bpt:.4f}, BPSS*={final_bpss:.4f}, BPSS*/H={final_bpss_over_h:.4f}, slope={slope:.8f}")

    del model, optimizer, scaler
    torch.cuda.empty_cache()

    return {
        'corpus': name, 'seed': seed, 'target_H': source_H,
        'empirical_H': source_H, 'BPT': final_bpt, 'BPSS_star': final_bpss,
        'BPSS_over_H': final_bpss_over_h, 'train_loss': losses[-1][1],
        'plateau_slope': slope, 'steps': TOTAL_STEPS,
        'time_sec': elapsed, 'n_params': n_params,
    }

def main():
    log("="*60)
    log("INTERMEDIATE ENTROPY SWEEP — Starting")
    log("="*60)

    # Phase 1: Generate corpora (CPU only)
    log("\nPhase 1: Generating corpora (CPU)")
    corpus_info = {}
    for name, K, target_H in CORPORA:
        path, H_emp = generate_corpus(name, K, target_H)
        corpus_info[name] = (path, H_emp)

    # Save entropy measurements
    with open(f"{OUTDIR}/results/corpus_entropy.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['corpus', 'target_H', 'empirical_H', 'alphabet_size', 'n_chars'])
        for name, K, target_H in CORPORA:
            _, H = corpus_info[name]
            w.writerow([name, target_H, H, K, CORPUS_SIZE])
    log("Entropy measurements saved")

    # Phase 2: Build tokenizers (CPU only)
    log("\nPhase 2: Building tokenizers (CPU)")
    tokenizers = {}
    for name, K, target_H in CORPORA:
        path, _ = corpus_info[name]
        tokenizers[name] = build_tokenizer(name, path)

    # Phase 3: Wait for GPU
    log("\nPhase 3: Waiting for GPU...")
    wait_for_gpu(max_mb=5000)

    # Phase 4: Train models
    log("\nPhase 4: Training models")
    all_results = []

    # Priority order: SYN-5 first (closest to basin)
    for name, K, target_H in CORPORA:
        path, H_emp = corpus_info[name]
        tok = tokenizers[name]

        log(f"\nTokenizing {name}...")
        train_ids, eval_ids, n_eval_chars = tokenize_corpus(path, tok)

        for seed in SEEDS:
            result = train_model(name, seed, train_ids, eval_ids, n_eval_chars,
                               tok.vocab_size, H_emp)
            all_results.append(result)

            # Save incrementally
            csv_path = f"{OUTDIR}/results/intermediate_entropy_results.csv"
            file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
            with open(csv_path, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(result.keys()))
                if not file_exists:
                    w.writeheader()
                w.writerow(result)

        # Commit after each corpus
        try:
            subprocess.run(['git', 'add', 'paper7.1/intermediate_entropy/'], cwd=REPO, capture_output=True)
            subprocess.run(['git', 'commit', '-m', f'Paper 7.1 intermediate entropy: {name} complete'],
                          cwd=REPO, capture_output=True)
            subprocess.run(['git', 'push'], cwd=REPO, capture_output=True)
            log(f"Git push: {name} results committed")
        except:
            pass

    # Phase 5: Generate plots
    log("\nPhase 5: Generating plots")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Combine with existing data
        existing = [
            ('SYN-2', 1.382, 20.866),
            ('SYN-4', 3.675, 22.819),
            ('SYN-8', 7.9997, 9.063),
            ('SYN-8 (B4)', 7.9997, 8.000),
            ('SYN-12', 11.985, 17.546),
        ]
        new_means = {}
        for r in all_results:
            key = r['corpus']
            if key not in new_means:
                new_means[key] = {'H': r['empirical_H'], 'bpts': []}
            new_means[key]['bpts'].append(r['BPT'])

        fig, ax = plt.subplots(figsize=(10, 7))

        # Existing points
        for name, H, bpt in existing:
            marker = 'o' if 'SYN-8' in name else 's'
            color = '#999' if name in ['SYN-2', 'SYN-4', 'SYN-12'] else '#f59e0b'
            alpha = 0.5 if name in ['SYN-2', 'SYN-4', 'SYN-12'] else 1.0
            ax.scatter(H, bpt, s=100, c=color, marker=marker, alpha=alpha, zorder=5)
            ax.annotate(name, (H, bpt), textcoords="offset points",
                       xytext=(8, 8), fontsize=8, color=color)

        # New points
        for name, info in new_means.items():
            H = info['H']
            mean_bpt = np.mean(info['bpts'])
            std_bpt = np.std(info['bpts']) if len(info['bpts']) > 1 else 0
            ax.errorbar(H, mean_bpt, yerr=std_bpt, fmt='o', color='#2166ac',
                       markersize=10, capsize=5, zorder=10)
            ax.annotate(name.upper(), (H, mean_bpt), textcoords="offset points",
                       xytext=(8, 8), fontsize=9, fontweight='bold', color='#2166ac')

        # Reference lines
        x_line = np.linspace(0, 13, 100)
        ax.plot(x_line, x_line, '--', color='gray', alpha=0.5, label='BPT = H (perfect tracking)')
        ax.axhline(y=4.16, color='red', linestyle=':', alpha=0.5, label='Language basin (4.16)')
        ax.axvspan(3.7, 8.0, alpha=0.05, color='blue', label='Gap region (3.7–8.0)')

        ax.set_xlabel('Source entropy H (bits/symbol)', fontsize=12)
        ax.set_ylabel('Achieved BPT', fontsize=12)
        ax.set_title('BPT vs Source Entropy — Full Sweep', fontsize=14)
        ax.legend(loc='upper left')
        ax.set_xlim(0, 13)
        ax.set_ylim(0, 25)

        fig.savefig(f"{OUTDIR}/plots/bpt_vs_entropy_full.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{OUTDIR}/plots/bpt_vs_entropy_full.pdf", bbox_inches='tight')
        plt.close(fig)
        log("Plot saved: bpt_vs_entropy_full")
    except Exception as e:
        log(f"Plot error: {e}")

    # Phase 6: Write report
    log("\nPhase 6: Writing report")
    report = "# Intermediate Entropy Sweep Report\n\n"
    report += "## Results\n\n"
    report += "| Corpus | Target H | Empirical H | BPT (mean±std) | BPSS* | BPSS*/H |\n"
    report += "|---|---|---|---|---|---|\n"
    for name, K, target_H in CORPORA:
        results_for = [r for r in all_results if r['corpus'] == name]
        if results_for:
            bpts = [r['BPT'] for r in results_for]
            bpss = [r['BPSS_star'] for r in results_for]
            boh = [r['BPSS_over_H'] for r in results_for]
            report += f"| {name.upper()} | {target_H} | {results_for[0]['empirical_H']:.4f} | {np.mean(bpts):.3f}±{np.std(bpts):.3f} | {np.mean(bpss):.3f} | {np.mean(boh):.3f} |\n"

    report += "\n## Key Finding\n\n"
    syn5 = [r for r in all_results if r['corpus'] == 'syn5']
    if syn5:
        mean_bpt = np.mean([r['BPT'] for r in syn5])
        report += f"SYN-5 (H≈5.0) achieves BPT={mean_bpt:.3f}. "
        if abs(mean_bpt - 5.0) < 1.5:
            report += "This tracks source entropy through the basin range, confirming the data-driven hypothesis.\n"
        else:
            report += f"This deviates from source entropy by {abs(mean_bpt-5.0):.1f} bits.\n"

    with open(f"{OUTDIR}/results/INTERMEDIATE_ENTROPY_REPORT.md", 'w') as f:
        f.write(report)

    # Final commit
    try:
        subprocess.run(['git', 'add', 'paper7.1/intermediate_entropy/'], cwd=REPO, capture_output=True)
        subprocess.run(['git', 'commit', '-m',
            'Paper 7.1: intermediate entropy sweep complete — SYN-5/6/7 with plots and report\n\n'
            'Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>'],
            cwd=REPO, capture_output=True)
        subprocess.run(['git', 'push'], cwd=REPO, capture_output=True)
        log("Final git push complete")
    except:
        pass

    log("="*60)
    log("INTERMEDIATE ENTROPY SWEEP — Complete")
    log("="*60)

if __name__ == "__main__":
    main()
