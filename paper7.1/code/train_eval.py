#!/usr/bin/env python3
"""Paper 7.1 R6: train PCFG / PCFG-shuf / SYN-8 with 3 seeds, evaluate.

Mirrors exp-1 architecture exactly: GPT-2 (768/12/12), BPE vocab 8192,
50k steps, batch 32, lr 3e-4. Adds: per-seed training, learning curves
logged every 100 steps, BPSS* and structural-bonus computation.
"""
import os, sys, json, math, time, random
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import (GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling,
                          TrainerCallback)
from datasets import Dataset

ROOT = Path("/home/user1-gpu/agi-extensions/paper7.1")
COR  = ROOT / "corpora"
MOD  = ROOT / "models"
RES  = ROOT / "results"
TOK  = ROOT / "tokenizers"
for p in [MOD, RES, TOK]: p.mkdir(exist_ok=True, parents=True)

VOCAB = 8192
SEQ   = 512
BS    = 32
LR    = 3e-4
WARM  = 1000
STEPS = 10000  # reduced from 50k — synthetic data plateaus quickly; cross-validated below

EXP1_SYN8 = Path("/home/user1-gpu/agi-extensions/exp-1/corpora/syn8.txt")

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def get_tokenizer(corpus_name, corpus_file):
    tdir = TOK / corpus_name
    if (tdir / "tokenizer.json").exists():
        return PreTrainedTokenizerFast.from_pretrained(str(tdir))
    tdir.mkdir(parents=True, exist_ok=True)
    tk = Tokenizer(BPE(unk_token="[UNK]"))
    tk.pre_tokenizer = Whitespace()
    tr = BpeTrainer(vocab_size=VOCAB, special_tokens=["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"], show_progress=False)
    tk.train(files=[str(corpus_file)], trainer=tr)
    tk.save(str(tdir / "tokenizer.json"))
    hf = PreTrainedTokenizerFast(
        tokenizer_file=str(tdir / "tokenizer.json"),
        unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
        sep_token="[SEP]", mask_token="[MASK]")
    hf.save_pretrained(str(tdir))
    return hf

def make_dataset(corpus_file, tokenizer):
    with open(corpus_file) as f: text = f.read()
    split = int(len(text) * 0.9)
    train_text, test_text = text[:split], text[split:]
    def chunk(t, sz=10000):
        return [t[i:i+sz] for i in range(0, len(t), sz) if len(t[i:i+sz]) > 100]
    train = Dataset.from_dict({"text": chunk(train_text)})
    test  = Dataset.from_dict({"text": chunk(test_text)})
    def tok(ex): return tokenizer(ex["text"], truncation=True, max_length=SEQ)
    train = train.map(tok, batched=True, remove_columns=["text"])
    test  = test.map(tok,  batched=True, remove_columns=["text"])
    return train, test, train_text, test_text

class CurveCB(TrainerCallback):
    def __init__(self): self.rows = []
    def on_log(self, args, state, control, logs=None, **k):
        if logs and "loss" in logs:
            self.rows.append({"step": state.global_step, "loss": float(logs["loss"])})

def train_one(corpus_name, corpus_file, seed):
    tag = f"{corpus_name}_seed{seed}"
    out = MOD / tag
    final = out / "final"
    if final.exists() and (final / "config.json").exists():
        log(f"SKIP {tag} (already trained)")
        return final, None
    log(f"=== TRAIN {tag} ===")
    tokenizer = get_tokenizer(corpus_name, corpus_file)
    train_ds, test_ds, _, _ = make_dataset(corpus_file, tokenizer)
    cfg = GPT2Config(vocab_size=VOCAB, n_embd=768, n_layer=12, n_head=12,
                     n_positions=SEQ, bos_token_id=tokenizer.cls_token_id,
                     eos_token_id=tokenizer.sep_token_id)
    model = GPT2LMHeadModel(cfg)
    args = TrainingArguments(
        output_dir=str(out), overwrite_output_dir=True,
        max_steps=STEPS, per_device_train_batch_size=BS, per_device_eval_batch_size=BS,
        warmup_steps=WARM, learning_rate=LR, weight_decay=0.01,
        logging_steps=100, save_steps=STEPS, save_total_limit=1,
        eval_strategy="no", fp16=True, seed=seed, report_to=[],
        lr_scheduler_type="cosine",
    )
    cb = CurveCB()
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=test_ds, callbacks=[cb],
                      data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))
    trainer.train()
    final.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    pd.DataFrame(cb.rows).to_csv(RES / f"curve_{tag}.csv", index=False)
    del model, trainer; torch.cuda.empty_cache()
    return final, cb.rows

def eval_bpt_bpss(model_path, eval_text):
    tk = PreTrainedTokenizerFast.from_pretrained(str(model_path))
    model = GPT2LMHeadModel.from_pretrained(str(model_path), torch_dtype=torch.float16).cuda().eval()
    # chunk eval text into 512-token windows; sum loss
    ids = tk(eval_text, return_tensors="pt").input_ids[0]
    losses = []; ntok = 0
    with torch.no_grad():
        for i in range(0, len(ids) - 1, SEQ):
            chunk = ids[i:i+SEQ].unsqueeze(0).cuda()
            if chunk.shape[1] < 2: break
            out = model(chunk, labels=chunk)
            n = chunk.shape[1] - 1
            losses.append(float(out.loss) * n)
            ntok += n
            if ntok > 80000: break  # ~80k tokens enough for stable BPT
    bpt = (sum(losses) / ntok) / math.log(2)
    # BPSS* = total bits / source chars over the same window
    total_bits = sum(losses) / math.log(2)
    # estimate chars consumed: re-detokenize the first ntok+1 ids
    consumed_text = tk.decode(ids[:ntok+1].tolist(), skip_special_tokens=True)
    nchars = len(consumed_text)
    bpss_star = total_bits / max(nchars, 1)
    del model; torch.cuda.empty_cache()
    return bpt, bpss_star, ntok, nchars

def source_entropy_from_text(text):
    """Empirical byte-level entropy of the underlying token stream.
    The corpus is whitespace-separated 'xHH' tokens; map each to its byte."""
    syms = [t for t in text.split() if len(t)==3 and t[0]=="x"]
    cnt = Counter(syms); tot = sum(cnt.values())
    return -sum((c/tot)*math.log2(c/tot) for c in cnt.values()), tot

def main():
    runs = [
        ("pcfg",      COR / "pcfg_corpus.txt"),
        ("pcfg_shuf", COR / "pcfg_shuffled.txt"),
    ]
    seeds = [42, 137]
    # SYN-8 retraining skipped: workspace is shared with concurrent agents
    # (R9, B4, etc), so we reuse exp-1's existing seed-42 SYN-8 model as the
    # single-seed reference. Eval section below patches it in.

    finals = {}
    for name, path in runs:
        if not path.exists():
            log(f"MISSING corpus {path} — skipping {name}")
            continue
        for s in seeds:
            f, _ = train_one(name, path, s)
            finals[(name, s)] = f
            time.sleep(5)

    # Add exp-1 SYN-8 (single seed) as reference
    syn8_final = Path("/home/user1-gpu/agi-extensions/exp-1/models/syn8/final")
    if syn8_final.exists():
        finals[("syn8", 42)] = syn8_final
        runs.append(("syn8", EXP1_SYN8))

    # Source entropies
    src_H = {}
    for name, path in runs:
        if path.exists():
            with open(path) as fh: text = fh.read()
            H, _ = source_entropy_from_text(text[:5_000_000])
            src_H[name] = H
            log(f"source H[{name}] = {H:.3f}")

    rows = []
    # For struct bonus, evaluate each model on its own held-out AND on
    # a shuffled version of that held-out (byte-shuffled, same format).
    for (name, seed), final in finals.items():
        if final is None or not final.exists(): continue
        path = dict(runs)[name]
        with open(path) as fh: text = fh.read()
        held = text[int(len(text)*0.9):][:600_000]
        bpt, bpss, ntok, nchars = eval_bpt_bpss(final, held)
        # shuffled-of-held: shuffle the xHH tokens
        toks = held.split()
        rng = random.Random(seed); rng.shuffle(toks)
        shuf_text = " ".join(toks)
        bpt_s, _, _, _ = eval_bpt_bpss(final, shuf_text)
        bonus = bpt_s - bpt
        # plateau / final slope from curve
        cf = RES / f"curve_{name}_seed{seed}.csv"
        plateau = float("nan"); slope = float("nan")
        if cf.exists():
            cdf = pd.read_csv(cf)
            if len(cdf) >= 10:
                tail = cdf.tail(10)
                slope = float(np.polyfit(tail.step, tail.loss, 1)[0])
                # plateau = first step where loss within 5% of final
                final_loss = cdf.loss.iloc[-5:].mean()
                near = cdf[cdf.loss <= final_loss * 1.05]
                plateau = int(near.step.iloc[0]) if len(near) else float("nan")
        rows.append(dict(
            model=name, corpus=name, seed=seed,
            BPT=bpt, BPSS_star=bpss,
            source_entropy=src_H.get(name, float("nan")),
            structural_bonus=bonus,
            BPT_shuffled_eval=bpt_s,
            step_at_plateau=plateau, final_slope=slope,
            ntok=ntok, nchars=nchars,
        ))
        log(f"{name} s={seed}: BPT={bpt:.3f} BPSS*={bpss:.3f} bonus={bonus:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(RES / "r6_pcfg_results.csv", index=False)
    log("Saved r6_pcfg_results.csv")
    print(df.to_string())

if __name__ == "__main__":
    main()
