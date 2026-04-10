#!/usr/bin/env python3
"""
Paper 7.1 R9 — Loss Function Specificity.

Train GPT-2 (92M) on SYN-8 and WikiText-2 with 3 different losses:
  A) CE   (cross-entropy)
  B) MSE  (mean squared error on logits vs one-hot targets)
  C) LS   (label-smoothed CE, smoothing=0.3)

Three seeds each, then evaluate ALL with CE/BPT on the held-out split.
Question: does BPT depend on the loss function or only on the data?
"""

import os, sys, math, json, time, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast,
    PreTrainedTokenizerFast, get_linear_schedule_with_warmup,
)
from datasets import load_dataset

ROOT = Path("/home/user1-gpu/agi-extensions/paper7.1")
RES_DIR = ROOT / "results"; RES_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = RES_DIR / "r9_loss_function.csv"
LOG_PATH = ROOT / "r9_run.log"

device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True

# --- Hyperparams (reduced from Exp1's 50K to fit overnight budget) ---
N_EMBD, N_LAYER, N_HEAD = 768, 12, 12
SEQ_LEN = 256
BATCH = 4           # GPU is shared with other Paper 7.1 jobs (~5 GB available)
LR = 3e-4
WARMUP = 200
SYN_STEPS = 4000    # SYN-8
WIKI_STEPS = 2500   # WikiText
LOG_EVERY = 200

LOSSES = ["CE", "MSE", "LS"]
SEEDS = [42, 137, 2024]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ---------- Data prep ----------

def load_syn8():
    tok_path = "/home/user1-gpu/agi-extensions/exp-1/tokenizers/syn8"
    tok = PreTrainedTokenizerFast.from_pretrained(tok_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "[PAD]"
    text = Path("/home/user1-gpu/agi-extensions/exp-1/corpora/syn8.txt").read_text()
    n = len(text); split = int(n * 0.9)
    train_text, eval_text = text[:split], text[split:]
    return tok, train_text, eval_text


def load_wikitext():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_text = "\n".join(t for t in ds["train"]["text"] if t.strip())
    eval_text  = "\n".join(t for t in ds["validation"]["text"] if t.strip())
    return tok, train_text, eval_text


def tokenize_to_blocks(tok, text, seq_len=SEQ_LEN, max_tokens=None):
    """Tokenize a single long string and pack into [N, seq_len] blocks."""
    ids = tok(text, return_tensors=None, add_special_tokens=False)["input_ids"]
    if max_tokens is not None:
        ids = ids[:max_tokens]
    n_blocks = len(ids) // seq_len
    ids = ids[: n_blocks * seq_len]
    arr = torch.tensor(ids, dtype=torch.long).view(n_blocks, seq_len)
    return arr


# ---------- Losses ----------

def compute_loss(logits, labels, loss_type):
    # logits [B, T, V], labels [B, T]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    V = shift_logits.size(-1)
    flat_logits = shift_logits.view(-1, V)
    flat_labels = shift_labels.view(-1)
    if loss_type == "CE":
        return F.cross_entropy(flat_logits, flat_labels)
    if loss_type == "LS":
        return F.cross_entropy(flat_logits, flat_labels, label_smoothing=0.3)
    if loss_type == "MSE":
        # Memory-efficient MSE vs one-hot:
        #   mse = mean_i mean_v (logit_iv - onehot_iv)^2
        #       = mean_i [ (sum_v logit_iv^2) - 2*logit_i,label_i + 1 ] / V
        sq_sum = (flat_logits.float() ** 2).sum(dim=-1)
        true_logit = flat_logits.float().gather(-1, flat_labels.unsqueeze(-1)).squeeze(-1)
        return ((sq_sum - 2.0 * true_logit + 1.0) / V).mean()
    raise ValueError(loss_type)


# ---------- Eval (always CE/BPT) ----------

@torch.no_grad()
def eval_ce(model, blocks, batch=BATCH):
    model.eval()
    total_nll = 0.0
    total_tok = 0
    for i in range(0, blocks.size(0), batch):
        b = blocks[i:i+batch].to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(b)
            logits = out.logits.float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = b[..., 1:].contiguous()
        nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_nll += nll.item()
        total_tok += shift_labels.numel()
    mean_nll = total_nll / total_tok  # nats per token
    bpt = mean_nll / math.log(2)
    return mean_nll, bpt, total_nll, total_tok


# ---------- Train one model ----------

def train_one(corpus, loss_type, seed, tok, train_blocks, eval_blocks, eval_text_len_chars, max_steps):
    set_seed(seed)
    cfg = GPT2Config(
        vocab_size=len(tok),
        n_embd=N_EMBD, n_layer=N_LAYER, n_head=N_HEAD,
        n_positions=SEQ_LEN,
        bos_token_id=tok.bos_token_id if tok.bos_token_id is not None else 0,
        eos_token_id=tok.eos_token_id if tok.eos_token_id is not None else 0,
    )
    model = GPT2LMHeadModel(cfg).to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, WARMUP, max_steps)
    scaler = torch.amp.GradScaler("cuda")

    N = train_blocks.size(0)
    g = torch.Generator().manual_seed(seed)

    model.train()
    t0 = time.time()
    last_loss = float("nan")
    for step in range(1, max_steps + 1):
        idx = torch.randint(0, N, (BATCH,), generator=g)
        batch = train_blocks[idx].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(batch)
            loss = compute_loss(out.logits.float(), batch, loss_type)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        last_loss = loss.item()
        if step % LOG_EVERY == 0 or step == 1:
            log(f"  {corpus}/{loss_type}/seed{seed} step {step}/{max_steps} loss={last_loss:.4f} elapsed={time.time()-t0:.0f}s")

    # Eval (always CE)
    mean_nll, bpt, total_nll, total_tok = eval_ce(model, eval_blocks)
    bpss_star = (total_nll / math.log(2)) / max(eval_text_len_chars, 1)

    del model
    torch.cuda.empty_cache()

    return {
        "corpus": corpus,
        "loss_function": loss_type,
        "seed": seed,
        "train_steps": max_steps,
        "eval_CE_loss": mean_nll,
        "eval_BPT": bpt,
        "eval_BPSS_star": bpss_star,
        "final_train_loss": last_loss,
        "n_params": nparams,
    }


def main():
    log("=== R9 loss-function specificity ===")
    rows = []
    if CSV_PATH.exists():
        rows = pd.read_csv(CSV_PATH).to_dict("records")
        log(f"Resuming with {len(rows)} existing rows")
    done = {(r["corpus"], r["loss_function"], int(r["seed"])) for r in rows}

    # SYN-8
    log("Loading SYN-8 ...")
    tok_s, tr_text_s, ev_text_s = load_syn8()
    tr_blocks_s = tokenize_to_blocks(tok_s, tr_text_s, max_tokens=2_000_000)
    ev_blocks_s = tokenize_to_blocks(tok_s, ev_text_s, max_tokens=200_000)
    log(f"SYN-8 train blocks {tr_blocks_s.shape}, eval blocks {ev_blocks_s.shape}, vocab {len(tok_s)}")

    for loss_type in LOSSES:
        for seed in SEEDS:
            key = ("SYN-8", loss_type, seed)
            if key in done:
                log(f"skip {key}"); continue
            log(f"START {key}")
            r = train_one("SYN-8", loss_type, seed, tok_s, tr_blocks_s, ev_blocks_s, len(ev_text_s), SYN_STEPS)
            log(f"DONE {key} -> BPT={r['eval_BPT']:.3f}")
            rows.append(r)
            pd.DataFrame(rows).to_csv(CSV_PATH, index=False)

    del tr_blocks_s, ev_blocks_s
    torch.cuda.empty_cache()

    # WikiText
    log("Loading WikiText-2 ...")
    tok_w, tr_text_w, ev_text_w = load_wikitext()
    tr_blocks_w = tokenize_to_blocks(tok_w, tr_text_w, max_tokens=2_000_000)
    ev_blocks_w = tokenize_to_blocks(tok_w, ev_text_w, max_tokens=200_000)
    log(f"WT train blocks {tr_blocks_w.shape}, eval blocks {ev_blocks_w.shape}, vocab {len(tok_w)}")

    for loss_type in LOSSES:
        for seed in SEEDS:
            key = ("WikiText", loss_type, seed)
            if key in done:
                log(f"skip {key}"); continue
            log(f"START {key}")
            r = train_one("WikiText", loss_type, seed, tok_w, tr_blocks_w, ev_blocks_w, len(ev_text_w), WIKI_STEPS)
            log(f"DONE {key} -> BPT={r['eval_BPT']:.3f}")
            rows.append(r)
            pd.DataFrame(rows).to_csv(CSV_PATH, index=False)

    log("ALL DONE")


if __name__ == "__main__":
    main()
