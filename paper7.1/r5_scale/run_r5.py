#!/usr/bin/env python3
"""
R5 Scale Test: Train a 1B model from scratch on SYN-8.
The experiment that could kill the thesis.

Saves results incrementally. Git commits when done.
Designed to run unattended via nohup.
"""

import os, sys, time, math, json, csv, subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

OUTDIR = "/home/user1-gpu/agi-extensions/paper7.1/r5_scale"
REPO = "/home/user1-gpu/agi-extensions"
SEEDS = [42, 137]
VOCAB_SIZE = 8192
SEQ_LEN = 512  # conservative for VRAM
TOTAL_STEPS_1B = 25000
TOTAL_STEPS_92M = 40000
EVAL_EVERY = 2000
LOG_EVERY = 50
CORPUS_SIZE = 200_000_000

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(f"{OUTDIR}/r5_run.log", "a") as f:
        f.write(line + "\n")

def generate_syn8():
    """Generate or locate SYN-8 corpus."""
    corpus_path = f"{OUTDIR}/syn8_corpus.bin"
    if os.path.exists(corpus_path) and os.path.getsize(corpus_path) >= CORPUS_SIZE:
        log(f"SYN-8 corpus exists: {corpus_path} ({os.path.getsize(corpus_path)} bytes)")
        return corpus_path

    # Check existing locations
    for p in [
        "/home/user1-gpu/agi-extensions/paper7.1/corpora/syn8.bin",
        "/home/user1-gpu/agi-extensions/exp-1/corpora/syn8.txt",
    ]:
        if os.path.exists(p) and os.path.getsize(p) > 10_000_000:
            log(f"Found existing SYN-8: {p}")
            return p

    log(f"Generating SYN-8 corpus: {CORPUS_SIZE} bytes, uniform 256-symbol Markov-0")
    rng = np.random.default_rng(42)
    data = rng.integers(0, 256, size=CORPUS_SIZE, dtype=np.uint8)
    data.tofile(corpus_path)

    # Verify entropy
    from collections import Counter
    counts = Counter(data[:10_000_000].tolist())
    total = sum(counts.values())
    H = -sum((c/total) * math.log2(c/total) for c in counts.values())
    log(f"SYN-8 verified: H={H:.4f} bits/byte, unique={len(counts)}")
    return corpus_path

def build_tokenizer(corpus_path):
    """Build or load BPE tokenizer."""
    tok_path = f"{OUTDIR}/tokenizer"
    if os.path.exists(f"{tok_path}/tokenizer.json"):
        log("Tokenizer exists, loading")
        from transformers import PreTrainedTokenizerFast
        return PreTrainedTokenizerFast(tokenizer_file=f"{tok_path}/tokenizer.json")

    # Check if exp-1 tokenizer exists
    for p in ["/home/user1-gpu/agi-extensions/paper7.1/tokenizers/syn8"]:
        tok_json = os.path.join(p, "tokenizer.json")
        if os.path.exists(tok_json):
            log(f"Found existing tokenizer: {tok_json}")
            from transformers import PreTrainedTokenizerFast
            return PreTrainedTokenizerFast(tokenizer_file=tok_json)

    log("Training BPE tokenizer (vocab=8192)")
    from tokenizers import ByteLevelBPETokenizer

    # Read corpus as text chunks
    if corpus_path.endswith('.bin'):
        data = np.fromfile(corpus_path, dtype=np.uint8)[:50_000_000]
        text = ' '.join(f'x{b:02x}' for b in data)
    else:
        with open(corpus_path) as f:
            text = f.read()[:50_000_000]

    chunks = [text[i:i+10000] for i in range(0, len(text), 10000)]

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(chunks, vocab_size=VOCAB_SIZE, min_frequency=2,
                                   special_tokens=["<pad>", "<eos>", "<unk>"])
    os.makedirs(tok_path, exist_ok=True)
    tokenizer.save(f"{tok_path}/tokenizer.json")

    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file=f"{tok_path}/tokenizer.json")
    tok.pad_token = "<pad>"
    log(f"Tokenizer trained: vocab_size={tok.vocab_size}")
    return tok

def tokenize_corpus(corpus_path, tokenizer):
    """Tokenize entire corpus and split into train/eval."""
    log("Tokenizing corpus...")
    if corpus_path.endswith('.bin'):
        data = np.fromfile(corpus_path, dtype=np.uint8)
        text = ' '.join(f'x{b:02x}' for b in data)
    else:
        with open(corpus_path) as f:
            text = f.read()

    n_chars = len(text)
    split = int(n_chars * 0.9)

    train_text = text[:split]
    eval_text = text[split:]

    log(f"Train: {len(train_text)} chars, Eval: {len(eval_text)} chars")

    train_ids = tokenizer.encode(train_text)
    eval_ids = tokenizer.encode(eval_text)

    log(f"Train tokens: {len(train_ids)}, Eval tokens: {len(eval_ids)}")
    return torch.tensor(train_ids, dtype=torch.long), torch.tensor(eval_ids, dtype=torch.long), len(eval_text)

def build_model(embed_dim, n_layers, n_heads, vocab_size):
    """Build a GPT-2 style model."""
    from transformers import GPT2Config, GPT2LMHeadModel
    config = GPT2Config(
        vocab_size=vocab_size,
        n_embd=embed_dim,
        n_layer=n_layers,
        n_head=n_heads,
        n_positions=SEQ_LEN,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model: {embed_dim}d, {n_layers}L, {n_heads}H, {n_params:,} params")
    return model, n_params

def get_random_batch(token_ids, batch_size, seq_len, rng):
    """Get a batch of random-offset windows."""
    max_start = len(token_ids) - seq_len - 1
    starts = rng.integers(0, max_start, size=batch_size)
    x = torch.stack([token_ids[s:s+seq_len] for s in starts])
    y = torch.stack([token_ids[s+1:s+seq_len+1] for s in starts])
    return x, y

def evaluate(model, eval_ids, n_eval_chars):
    """Evaluate model on held-out data. Returns BPT, BPSS*."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for start in range(0, len(eval_ids) - SEQ_LEN, SEQ_LEN):
            x = eval_ids[start:start+SEQ_LEN].unsqueeze(0).cuda()
            y = eval_ids[start+1:start+SEQ_LEN+1].unsqueeze(0).cuda()
            with autocast():
                outputs = model(x)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item() * SEQ_LEN
            total_tokens += SEQ_LEN

    avg_loss_nats = total_loss / total_tokens
    bpt = avg_loss_nats / math.log(2)
    total_bits = bpt * total_tokens
    bpss = total_bits / n_eval_chars if n_eval_chars > 0 else float('nan')

    model.train()
    return bpt, bpss

def train_model(name, embed_dim, n_layers, n_heads, total_steps, seed,
                train_ids, eval_ids, n_eval_chars, vocab_size):
    """Train a model and return results."""
    log(f"\n{'='*60}")
    log(f"Training {name} (seed={seed})")
    log(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model, n_params = build_model(embed_dim, n_layers, n_heads, vocab_size)
    model.cuda()
    model.gradient_checkpointing_enable()

    # Determine batch size by trying
    batch_size = 8 if n_params < 500_000_000 else 4

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scaler = GradScaler()

    # Cosine decay with warmup
    warmup_steps = 1000

    start_time = time.time()
    losses = []
    eval_results = []
    peak_vram = 0

    for step in range(1, total_steps + 1):
        # LR schedule
        if step < warmup_steps:
            lr = 3e-4 * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = 3e-4 * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        try:
            x, y = get_random_batch(train_ids, batch_size, SEQ_LEN, rng)
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            with autocast():
                outputs = model(x)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.item()
            losses.append((step, loss_val))

            cur_vram = torch.cuda.max_memory_allocated() / 1024**2
            peak_vram = max(peak_vram, cur_vram)

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size > 1:
                batch_size //= 2
                log(f"OOM at step {step}, reducing batch to {batch_size}")
                continue
            else:
                log(f"OOM at batch_size=1, aborting")
                break

        if step % LOG_EVERY == 0:
            log(f"  step {step}/{total_steps}, loss={loss_val:.4f}, lr={lr:.6f}, bs={batch_size}, vram={cur_vram:.0f}MB")

        if step % EVAL_EVERY == 0 or step == total_steps:
            bpt, bpss = evaluate(model, eval_ids, n_eval_chars)
            bpss_over_h = bpss / 8.0 if bpss > 0 else float('nan')
            eval_results.append((step, bpt, bpss, bpss_over_h))
            log(f"  EVAL step {step}: BPT={bpt:.4f}, BPSS*={bpss:.4f}, BPSS*/H={bpss_over_h:.4f}")

    elapsed = time.time() - start_time

    # Final eval
    final_bpt, final_bpss = evaluate(model, eval_ids, n_eval_chars)
    final_bpss_over_h = final_bpss / 8.0
    final_loss = losses[-1][1] if losses else float('nan')

    # Plateau slope (last 20% of logged losses)
    if len(losses) > 100:
        last_20pct = losses[int(len(losses)*0.8):]
        steps_arr = np.array([s for s,l in last_20pct])
        loss_arr = np.array([l for s,l in last_20pct])
        if len(steps_arr) > 1:
            slope = np.polyfit(steps_arr, loss_arr, 1)[0]
        else:
            slope = 0
    else:
        slope = float('nan')

    log(f"\n  FINAL: BPT={final_bpt:.4f}, BPSS*={final_bpss:.4f}, BPSS*/H={final_bpss_over_h:.4f}")
    log(f"  Time: {elapsed/3600:.2f}h, Peak VRAM: {peak_vram:.0f}MB, Plateau slope: {slope:.8f}")

    del model, optimizer, scaler
    torch.cuda.empty_cache()

    return {
        'model': name, 'params': n_params if 'n_params' in dir() else 0,
        'seed': seed, 'steps': total_steps,
        'BPT': final_bpt, 'BPSS_star': final_bpss, 'BPSS_over_H': final_bpss_over_h,
        'train_loss': final_loss, 'plateau_slope': slope,
        'time_hrs': elapsed/3600, 'peak_vram_mb': peak_vram,
        'batch_size': batch_size, 'seq_len': SEQ_LEN,
        'n_params': n_params,
    }, losses, eval_results

def save_result(result, losses, eval_results):
    """Save one result row to CSV."""
    csv_path = f"{OUTDIR}/r5_results.csv"
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(result.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(result)

    # Save learning curve
    curve_path = f"{OUTDIR}/curve_{result['model']}_seed{result['seed']}.csv"
    with open(curve_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'train_loss'])
        w.writerows(losses)

    # Save eval trajectory
    eval_path = f"{OUTDIR}/eval_{result['model']}_seed{result['seed']}.csv"
    with open(eval_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'BPT', 'BPSS_star', 'BPSS_over_H'])
        w.writerows(eval_results)

    log(f"Saved results for {result['model']} seed={result['seed']}")

def write_report(results):
    """Write the final report."""
    report = f"""# R5 Scale Test Report

## Verdict

"""
    # Determine verdict
    results_1b = [r for r in results if '1b' in r['model'].lower() or r['n_params'] > 500_000_000]
    results_92m = [r for r in results if r['n_params'] < 500_000_000]

    if results_1b:
        mean_1b_bpt = np.mean([r['BPT'] for r in results_1b])
        if mean_1b_bpt > 7.5:
            verdict = f"THESIS CONFIRMED AT SCALE. 1B SYN-8 BPT = {mean_1b_bpt:.3f} (> 7.5). Capacity does not compress SYN-8 toward the basin."
        elif mean_1b_bpt > 5.0:
            verdict = f"MIXED RESULT. 1B SYN-8 BPT = {mean_1b_bpt:.3f}. Partial compression observed — more capacity reduces BPT but does not reach the basin."
        else:
            verdict = f"THESIS FALSIFIED AT SCALE. 1B SYN-8 BPT = {mean_1b_bpt:.3f} (< 5.0). The architectural hypothesis re-enters."
    else:
        verdict = "1B training did not complete. Only 92M results available."

    report += verdict + "\n\n## Results\n\n"
    report += "| Model | Params | Seed | BPT | BPSS* | BPSS*/H | Time (hrs) | Peak VRAM |\n"
    report += "|---|---|---|---|---|---|---|---|\n"
    for r in results:
        report += f"| {r['model']} | {r['n_params']:,} | {r['seed']} | {r['BPT']:.4f} | {r['BPSS_star']:.4f} | {r['BPSS_over_H']:.4f} | {r['time_hrs']:.2f} | {r['peak_vram_mb']:.0f} MB |\n"

    report += f"\n## Interpretation\n\n{verdict}\n\n"
    report += "Source entropy H = 8.0 bits/symbol. If BPT/H ≈ 1.0, the model perfectly tracks source entropy.\n"

    with open(f"{OUTDIR}/R5_SCALE_REPORT.md", 'w') as f:
        f.write(report)
    log("Report written")

def git_commit():
    """Commit and push results."""
    try:
        subprocess.run(['git', 'add', 'paper7.1/r5_scale/'], cwd=REPO, capture_output=True)
        subprocess.run(['git', 'commit', '-m',
            'Paper 7.1 R5: 1B-parameter SYN-8 scale test (automated overnight)\n\n'
            'Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>'],
            cwd=REPO, capture_output=True)
        subprocess.run(['git', 'push'], cwd=REPO, capture_output=True)
        log("Git commit and push complete")
    except Exception as e:
        log(f"Git error: {e}")

def main():
    log("="*60)
    log("R5 SCALE TEST — Starting")
    log("="*60)

    # Generate/find corpus
    corpus_path = generate_syn8()

    # Build tokenizer
    tokenizer = build_tokenizer(corpus_path)
    actual_vocab = tokenizer.vocab_size

    # Tokenize
    train_ids, eval_ids, n_eval_chars = tokenize_corpus(corpus_path, tokenizer)

    all_results = []

    # Train 92M baseline first (faster, validates the pipeline)
    for seed in SEEDS:
        result, losses, evals = train_model(
            "gpt2_92m", 768, 12, 12, TOTAL_STEPS_92M, seed,
            train_ids, eval_ids, n_eval_chars, actual_vocab
        )
        all_results.append(result)
        save_result(result, losses, evals)

    # Train 1B model
    for seed in SEEDS:
        result, losses, evals = train_model(
            "gpt2_1b", 2048, 24, 16, TOTAL_STEPS_1B, seed,
            train_ids, eval_ids, n_eval_chars, actual_vocab
        )
        all_results.append(result)
        save_result(result, losses, evals)

    # Write report
    write_report(all_results)

    # Generate plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        for result in all_results:
            curve_path = f"{OUTDIR}/eval_{result['model']}_seed{result['seed']}.csv"
            if os.path.exists(curve_path):
                import pandas as pd
                df = pd.read_csv(curve_path)
                label = f"{result['model']} (seed={result['seed']})"
                style = '-' if '1b' in result['model'] else '--'
                ax.plot(df['step'], df['BPT'], style, label=label)

        ax.axhline(y=8.0, color='gray', linestyle=':', label='Source entropy (8.0)')
        ax.axhline(y=4.16, color='red', linestyle=':', label='Language basin (4.16)')
        ax.set_xlabel('Training step')
        ax.set_ylabel('Eval BPT')
        ax.set_title('R5: 92M vs 1B on SYN-8 — does scale change the basin?')
        ax.legend()
        fig.savefig(f"{OUTDIR}/r5_learning_curves.png", dpi=200, bbox_inches='tight')
        plt.close(fig)
        log("Plot saved")
    except Exception as e:
        log(f"Plot error: {e}")

    # Commit
    git_commit()

    log("="*60)
    log("R5 SCALE TEST — Complete")
    log("="*60)

if __name__ == "__main__":
    main()
