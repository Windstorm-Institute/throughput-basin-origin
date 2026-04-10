# Terminal R5 — paste this into a fresh Claude Code terminal on Varon-1

```
SYSTEM: You are running the R5 scale test on Varon-1 (RTX 5090, 32GB VRAM).
This is the single most important experiment in the Paper 7 series. It is
the one result that could collapse the central thesis.

You have the FULL GPU for the next 10 hours. No other GPU jobs are running.
Use all 32GB of VRAM. Take your time. Get this right.

THE QUESTION

Paper 7 shows that at 92M parameters, SYN-8 (8-bit entropy Markov data)
achieves ~9 BPT — tracking source entropy, not compressing to the ~4 BPT
natural-language basin. But the original basin was observed at 70M to 175B
parameters. A critic can say: "Maybe at 1B, the model DOES compress SYN-8
toward ~4 BPT, and the 92M result just reflects undercapacity."

If the 1B model still sits at ~8 BPT → the data-driven thesis hardens
massively and Paper 7's headline is rock-solid.

If the 1B model drops toward ~4 BPT → the thesis collapses and we must
report this honestly as a falsification of our own central claim.

Either result is a publication. The Windstorm Institute leads with
falsifications.

======================================================================
STEP 1: LOCATE OR GENERATE THE SYN-8 CORPUS
======================================================================

Check if SYN-8 data exists from prior experiments:
  find /home/user1-gpu -path "*/corpora/*syn*8*" -o -path "*/syn8*" 2>/dev/null | head -10
  find /home/user1-gpu -name "*.txt" -path "*syn*8*" 2>/dev/null | head -5

If it exists, verify:
  - At least 100M characters (ideally 200M for the bigger model)
  - 256-symbol alphabet
  - Empirical entropy ~8.0 bits/symbol
  Verify with:
    python3 -c "
    from collections import Counter
    import math
    data = open('PATH_TO_SYN8', 'rb').read()[:10000000]
    counts = Counter(data)
    total = sum(counts.values())
    H = -sum((c/total) * math.log2(c/total) for c in counts.values())
    print(f'H = {H:.4f} bits/byte, len = {len(data)}, unique = {len(counts)}')
    "

If it does NOT exist or is too small, generate fresh:
  python3 -c "
  import numpy as np
  rng = np.random.default_rng(42)
  data = rng.integers(0, 256, size=200_000_000, dtype=np.uint8)
  data.tofile('/home/user1-gpu/agi-extensions/paper7.1/r5_scale/syn8_corpus.bin')
  print(f'Generated 200M bytes, H = 8.0 bits/byte (uniform)')
  "

======================================================================
STEP 2: LOCATE OR BUILD THE TOKENIZER
======================================================================

Check if the SYN-8 BPE tokenizer (vocab 8192) exists:
  find /home/user1-gpu -path "*tokenizer*" -name "*.json" 2>/dev/null | grep -i syn8 | head -5

If it exists, reuse it for comparability with the 92M results.
If not, train a new one:
  from tokenizers import ByteLevelBPETokenizer
  tokenizer = ByteLevelBPETokenizer()
  tokenizer.train_from_iterator(
      [chunk for chunk in chunks_of_corpus],
      vocab_size=8192,
      min_frequency=2,
      special_tokens=["<pad>", "<eos>", "<unk>"]
  )
  tokenizer.save("tokenizer.json")

======================================================================
STEP 3: TRAIN A 1B MODEL FROM SCRATCH ON SYN-8
======================================================================

THIS IS THE CLEAN TEST. Do not fine-tune a pretrained model — that
conflates language priors with capacity. Train from scratch so the
ONLY thing the model sees is SYN-8 data.

Architecture: GPT-2 style, ~1B parameters
  - Embedding dim: 2048
  - Layers: 24
  - Heads: 16
  - Vocab: 8192 (SYN-8 BPE tokenizer)
  - Sequence length: 1024
  - Total params: ~1.0-1.2B (verify after construction)

VRAM management (critical for 32GB):
  - Use PyTorch AMP (torch.cuda.amp.autocast) for FP16 mixed precision
  - Enable gradient checkpointing:
      model.gradient_checkpointing_enable()
  - Start with batch_size=4, sequence_length=1024
  - If OOM: reduce batch to 2 and use gradient accumulation steps=4
    to maintain effective batch of 8
  - If still OOM: reduce sequence_length to 512
  - Do NOT use DeepSpeed unless absolutely necessary (adds complexity)
  - Clear cache between eval runs:
      torch.cuda.empty_cache()

Training schedule:
  - Optimizer: AdamW, lr=3e-4, weight_decay=0.01
  - Warmup: 1000 steps linear
  - Decay: cosine to 0 over total steps
  - Total steps: 30,000 (may need to reduce if too slow — but try
    for at least 20,000)
  - Log training loss every 50 steps
  - Evaluate on held-out data every 2,000 steps

Seeds: 42 and 137 (2 seeds minimum). If time permits, add seed 2024.

DATA LOADING:
  - Split corpus: first 90% for training, last 10% for eval
  - Within training split: random-offset 1024-token windows
    (NOT sequential chunks — last night's B4 showed random-offset
    converges much faster)
  - Within eval split: non-overlapping 1024-token windows, sum CE
    in bits, divide by total tokens = BPT

SAVE CHECKPOINTS:
  Save a checkpoint every 5,000 steps to:
    /home/user1-gpu/agi-extensions/paper7.1/r5_scale/checkpoints/
  These are large (~4GB each). Keep only the last 3.
  This way if training is interrupted, you can resume.

======================================================================
STEP 4: EVALUATE
======================================================================

At the final step, evaluate the 1B model using the SAME unified
harness protocol as B1:
  - Held-out data: last 10% of SYN-8 corpus (provably disjoint)
  - Model's own tokenizer
  - Non-overlapping 1024-token windows
  - Sum CE in bits / total tokens = BPT
  - BPSS* = total bits / total source characters (bytes)

Record:
  - BPT
  - BPSS*
  - BPSS*/H (ratio to source entropy — should be ~1.0 if tracking)
  - Total training time
  - Final training loss
  - Peak VRAM usage

======================================================================
STEP 5: ALSO RUN THE 92M BASELINE WITH THE SAME PROTOCOL
======================================================================

For a clean apples-to-apples comparison, also train a 92M model on
the same SYN-8 corpus with the same tokenizer and the same random-
offset data loading. Same seeds, same eval protocol.

This is important because the original 92M result (9.063 BPT) used
a different tokenizer, different data loading (chunked vs random-offset),
and different eval harness. The B4 retrain showed 8.0 BPT with random-
offset. We need the 92M and 1B numbers to be directly comparable.

92M architecture:
  - Embedding dim: 768
  - Layers: 12
  - Heads: 12
  - Same tokenizer as the 1B model
  - Same schedule but 50,000 steps (it converges faster)
  - 2 seeds (42, 137)

======================================================================
STEP 6: THE CRITICAL COMPARISON TABLE
======================================================================

Save to: /home/user1-gpu/agi-extensions/paper7.1/r5_scale/r5_results.csv

| model | params | seed | steps | BPT | BPSS_star | BPSS_over_H | train_loss | time_hrs | peak_vram_mb |

The critical numbers:
  - 92M SYN-8 BPT: expected ~8.0 (based on B4)
  - 1B SYN-8 BPT: ???

INTERPRETATION GUIDE:
  BPT > 7.5 at 1B → "Thesis confirmed at scale. Capacity does not
    compress SYN-8 toward the basin. The basin is data-driven."
  BPT 5.0-7.5 at 1B → "Partial compression observed. More capacity
    reduces BPT but does not reach the basin. Mixed result."
  BPT < 5.0 at 1B → "THESIS FALSIFIED AT SCALE. Report this honestly
    and prominently. The architectural hypothesis re-enters."

======================================================================
STEP 7: LEARNING CURVES
======================================================================

Plot: /home/user1-gpu/agi-extensions/paper7.1/r5_scale/r5_learning_curves.png

  X-axis: training step
  Y-axis: eval BPT on SYN-8 held-out
  Two lines: 92M (dashed) and 1B (solid)
  Shaded region: ±1 std across seeds
  Horizontal lines: source entropy (8.0) and basin (4.16)

KEY QUESTION: Does the 1B model plateau at the same BPT as the 92M,
or does it continue descending past where the 92M stopped?

If they plateau at the same level → capacity doesn't matter, only data
If 1B plateaus lower → capacity partially determines the basin

======================================================================
STEP 8: REPORT
======================================================================

Write: /home/user1-gpu/agi-extensions/paper7.1/r5_scale/R5_SCALE_REPORT.md

Structure:
  1. One-sentence verdict (thesis confirmed / mixed / falsified)
  2. The comparison table
  3. Learning curves description
  4. Whether the 1B model plateaued or was still descending
  5. Plateau slope at final 2K steps (same protocol as B4)
  6. Honest interpretation: what this means for Paper 7's claims
  7. If the thesis is falsified, what the alternative explanation is

======================================================================
STEP 9: COMMIT AND PUSH
======================================================================

  cd /home/user1-gpu/agi-extensions
  git add paper7.1/r5_scale/
  git commit -m "Paper 7.1 R5: 1B-parameter SYN-8 scale test

  The experiment that could have killed the thesis.
  Result: [FILL IN — confirmed/mixed/falsified]
  1B SYN-8 BPT = [FILL IN]
  92M SYN-8 BPT = [FILL IN] (same protocol)"
  git push

======================================================================
CONSTRAINTS
======================================================================

- You have the FULL GPU. Use it. Do not limit batch size unnecessarily.
- If training is going to take more than 10 hours, reduce total steps
  to whatever fits. 20,000 steps minimum for the 1B model.
- Save results INCREMENTALLY. Write the CSV after each seed completes.
  Write the report after all seeds are done. If the machine reboots,
  we keep partial results.
- Do not modify any existing files outside paper7.1/r5_scale/.
- BE HONEST. If the result kills the thesis, say so in the first
  sentence of the report. Do not bury it. Do not hedge. The
  Windstorm Institute's credibility depends on reporting falsifications
  as prominently as confirmations.

BEGIN. This is the most important experiment in the series.
```
