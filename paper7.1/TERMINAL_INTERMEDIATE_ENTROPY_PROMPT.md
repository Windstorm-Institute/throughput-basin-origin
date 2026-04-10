# Terminal Intermediate Entropy — paste this into a fresh Claude Code terminal on Varon-1

```
SYSTEM: You are filling the entropy gaps in Paper 7's synthetic-data
experiment on Varon-1 (RTX 5090). This produces the paper's potential
cover figure: BPT vs source entropy with 7 data points.

IMPORTANT GPU SCHEDULING: Another terminal (R5 scale test) is training
a 1B model and is using the full GPU. You should:
1. Generate all corpora and tokenizers NOW (pure CPU, no GPU needed)
2. Wait for the GPU to free up before training
3. Check GPU before each training run:
   while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 5000 ]; do
     echo "GPU busy ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) MB used). Waiting 10 min... $(date)"
     sleep 600
   done
   echo "GPU free. Starting training."

THE QUESTION

Paper 7 tested SYN-2 (H=1.4), SYN-4 (H=3.7), SYN-8 (H=8.0), SYN-12
(H=12.0). There is a gap from 3.7 to 8.0 bits — exactly the range
where the language basin (4.16) lives. If BPT tracks entropy linearly
through this range, the thesis is airtight. If something weird happens
near 4 bits (a kink, a plateau, a compression attractor), the thesis
needs revision.

======================================================================
PHASE 1: CORPUS GENERATION (CPU only — do this immediately)
======================================================================

mkdir -p /home/user1-gpu/agi-extensions/paper7.1/intermediate_entropy/corpora
mkdir -p /home/user1-gpu/agi-extensions/paper7.1/intermediate_entropy/results
mkdir -p /home/user1-gpu/agi-extensions/paper7.1/intermediate_entropy/plots
mkdir -p /home/user1-gpu/agi-extensions/paper7.1/intermediate_entropy/tokenizers

Generate three new corpora with controlled entropy:

SYN-5: Target H ≈ 5.0 bits/symbol
  - Alphabet: 32 symbols (2^5)
  - Markov-0 with non-uniform probabilities tuned to hit H=5.0
  - Strategy: use a Zipf-like distribution over 32 symbols,
    then adjust the Zipf exponent until empirical H ≈ 5.0 ± 0.1
  - 100M characters
  
SYN-6: Target H ≈ 6.0 bits/symbol
  - Alphabet: 64 symbols (2^6)
  - Markov-0 with non-uniform probabilities tuned to hit H=6.0
  - Same Zipf-tuning strategy
  - 100M characters
  
SYN-7: Target H ≈ 7.0 bits/symbol
  - Alphabet: 128 symbols (2^7)
  - Markov-0 with non-uniform probabilities tuned to hit H=7.0
  - 100M characters

ENTROPY TUNING ALGORITHM:
  import numpy as np
  from scipy.optimize import brentq
  
  def entropy_of_zipf(alpha, K):
      """Entropy of Zipf distribution over K symbols with exponent alpha."""
      ranks = np.arange(1, K+1, dtype=np.float64)
      probs = ranks ** (-alpha)
      probs /= probs.sum()
      return -np.sum(probs * np.log2(probs + 1e-15))
  
  def find_alpha(target_H, K):
      """Find Zipf exponent that produces target entropy for K symbols."""
      # alpha=0 → uniform → H=log2(K), alpha→∞ → degenerate → H=0
      f = lambda a: entropy_of_zipf(a, K) - target_H
      return brentq(f, 0.001, 10.0)
  
  # Example for SYN-5:
  alpha = find_alpha(5.0, 32)
  ranks = np.arange(1, 33, dtype=np.float64)
  probs = ranks ** (-alpha)
  probs /= probs.sum()
  # Verify
  H = -np.sum(probs * np.log2(probs))
  print(f"SYN-5: alpha={alpha:.4f}, H={H:.4f}")
  # Generate corpus
  rng = np.random.default_rng(42)
  data = rng.choice(32, size=100_000_000, p=probs).astype(np.uint8)
  data.tofile('corpora/syn5.bin')

Repeat for SYN-6 (K=64) and SYN-7 (K=128).

VERIFY each corpus:
  from collections import Counter
  data = np.fromfile('corpora/synN.bin', dtype=np.uint8)
  counts = Counter(data.tolist())
  total = sum(counts.values())
  H = -sum((c/total) * math.log2(c/total) for c in counts.values())
  print(f"SYN-N: H_empirical = {H:.4f}, target = N.0, unique = {len(counts)}")

Save entropy measurements to:
  results/corpus_entropy.csv
  Columns: corpus, target_H, empirical_H, alphabet_size, markov_order, n_chars

======================================================================
PHASE 2: TOKENIZER TRAINING (CPU only)
======================================================================

Train a BPE tokenizer (vocab 8192) for each new corpus.

Use the tokenizers library:
  from tokenizers import ByteLevelBPETokenizer
  
  for name in ['syn5', 'syn6', 'syn7']:
      tokenizer = ByteLevelBPETokenizer()
      # Read corpus as text (convert bytes to hex representation
      # matching the format used in Exp 1)
      tokenizer.train_from_iterator(
          corpus_chunks,  # 10K-char chunks
          vocab_size=8192,
          min_frequency=2,
          special_tokens=["<pad>", "<eos>", "<unk>"]
      )
      tokenizer.save(f"tokenizers/{name}_tokenizer.json")

OR: If the original Exp 1 used a different text encoding (e.g.,
whitespace-separated hex tokens like "xHH"), match that format exactly.
Read the original exp-1 code to determine the encoding:
  cat /home/user1-gpu/agi-extensions/exp-1/code/exp1_generate_corpora.py | head -100

Match whatever format Exp 1 used so the tokenizer behavior is comparable.

======================================================================
PHASE 3: WAIT FOR GPU, THEN TRAIN (GPU needed)
======================================================================

Before starting any training, check GPU availability:
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  
If > 5000 MB used, wait. The R5 terminal is training a 1B model.
Use the polling loop from the scheduling section above.

Once GPU is free:

Architecture: GPT-2, 92M params (768 dim, 12 layers, 12 heads)
  — matching the original Exp 1 exactly

Training: 
  - 40,000 steps per model (B4 showed plateau by step 2K with
    random-offset loading, so 40K is more than enough)
  - Batch size 16, sequence length 512
  - Random-offset window sampling (NOT sequential chunks)
  - lr 3e-4, AdamW, cosine decay, warmup 1000
  - FP16 mixed precision
  - 2 seeds each (42, 137)
  - Log training loss every 200 steps
  - Evaluate on held-out (last 10%) every 5,000 steps

Train 3 models: SYN-5, SYN-6, SYN-7
Total: 6 training runs (3 corpora × 2 seeds)
Estimated time: ~3-4 hours at 92M params

SAVE RESULTS INCREMENTALLY — write each model's eval to the CSV
as soon as it finishes, not at the end. Commit after each corpus.

======================================================================
PHASE 4: EVALUATE
======================================================================

For each trained model on its own held-out data:
  - BPT (CE loss in bits / num tokens)
  - BPSS* (total bits / total source characters)
  - BPSS*/H (ratio — how close to source entropy?)
  - Final training loss
  - Plateau slope over last 5K steps

Save to: results/intermediate_entropy_results.csv
Columns: corpus, seed, target_H, empirical_H, BPT, BPSS_star,
         BPSS_over_H, train_loss, plateau_slope, steps, time_sec

======================================================================
PHASE 5: THE COVER FIGURE
======================================================================

This is potentially the most important figure in the entire paper.

Combine the new data with existing data to create ONE plot with all
available data points:

  Existing (from Exp 1 / B1 / B4):
    SYN-2:  H=1.382,  BPT=20.866 (unified B1), BPSS*=3.351
    SYN-4:  H=3.675,  BPT=22.819 (unified B1), BPSS*=6.412
    SYN-8:  H=7.9997, BPT=9.063  (unified B1), BPSS*=2.260
    SYN-8:  H=7.9997, BPT=8.000  (B4 random-offset retrain)
    SYN-12: H=11.985, BPT=17.546 (unified B1), BPSS*=2.917
  
  New (this experiment):
    SYN-5: H≈5.0, BPT=???, BPSS*=???
    SYN-6: H≈6.0, BPT=???, BPSS*=???
    SYN-7: H≈7.0, BPT=???, BPSS*=???

PLOT A (BPT vs Source Entropy):
  plots/bpt_vs_entropy_full.png (and .pdf)
  
  X-axis: Source entropy H (bits/symbol), range [0, 13]
  Y-axis: BPT, range [0, 25]
  
  - Plot all data points as scatter with error bars (±1 std across seeds)
  - Draw the identity line y=x (dashed gray, labeled "BPT = H, perfect tracking")
  - Draw horizontal line at y=4.16 (dotted, labeled "natural language basin")
  - Mark SYN-2 and SYN-4 with open circles and a note "BPE pathological"
  - Mark SYN-12 with open circle and note "capacity-limited (92M)"
  - Mark SYN-5, 6, 7, 8 with filled circles (the clean data points)
  - Color the 4-8 bit region with a light shaded band labeled
    "region where the basin lives"

  If the filled circles trace the identity line through [5, 6, 7, 8]
  with no kink near 4 → thesis is airtight, this is the cover figure.

PLOT B (BPSS* vs Source Entropy):
  plots/bpss_vs_entropy_full.png (and .pdf)
  
  Same layout as Plot A but Y-axis = BPSS* instead of BPT.
  This is the tokenizer-independent version.
  The identity line is BPSS* = H.
  
  BPSS* should be better behaved than BPT for low-entropy corpora
  because it removes the BPE inflation artifact.

PLOT C (BPSS*/H ratio):
  plots/tracking_ratio.png
  
  X-axis: Source entropy H
  Y-axis: BPSS*/H (ratio to source entropy)
  Horizontal line at 1.0 (perfect tracking)
  
  If all points sit near 1.0 → the model perfectly tracks source
  entropy at all tested levels. This is the cleanest visual test
  of the data-driven hypothesis.

======================================================================
PHASE 6: FIT THE TRACKING RELATIONSHIP
======================================================================

Using the clean data points only (SYN-5, 6, 7, 8 — exclude SYN-2/4
for BPE pathology and SYN-12 for capacity limitation), fit:

  BPT = a * H + b  (linear)
  
If a ≈ 1.0 and b ≈ 0 → perfect linear tracking
If a < 0.9 → partial compression (architectural component exists)
If a > 1.1 → overhead (tokenizer inflation)

Report:
  - a, b, R², p-value
  - 95% CI on a (does the CI include 1.0?)
  - 95% CI on b (does the CI include 0.0?)

Also fit BPSS* = a' * H + b' (tokenizer-independent version).

======================================================================
PHASE 7: REPORT
======================================================================

Write: results/INTERMEDIATE_ENTROPY_REPORT.md

Structure:
  1. One-sentence verdict: does BPT track entropy linearly through
     the basin range?
  2. The combined data table (all 7 entropy levels)
  3. The three plots (described above)
  4. The linear fit results
  5. Whether SYN-5 (H≈5.0) sits at BPT≈5.0 — this single data
     point is the most important because it's closest to the basin
  6. Whether the BPE pathology observed in SYN-2/4 appears in
     SYN-5/6/7 (it shouldn't — larger alphabets have more
     information per BPE merge)
  7. Honest caveats: still 92M only, still Markov-0 only, still
     single-architecture

======================================================================
PHASE 8: COMMIT AND PUSH
======================================================================

  cd /home/user1-gpu/agi-extensions
  git add paper7.1/intermediate_entropy/
  git commit -m "Paper 7.1: intermediate entropy sweep SYN-5/6/7

  Fills the 3.7→8.0 gap in the BPT vs entropy curve.
  SYN-5 (H≈5.0) BPT = [FILL IN]
  SYN-6 (H≈6.0) BPT = [FILL IN]
  SYN-7 (H≈7.0) BPT = [FILL IN]
  Linear fit: BPT = [a]H + [b], R² = [FILL IN]"
  git push

======================================================================
CONSTRAINTS
======================================================================

- DO NOT start GPU training while another terminal is using >5GB VRAM.
  Generate corpora and tokenizers on CPU first, then wait.
- The 92M models are small (~2-4 GB VRAM during training). Once the
  GPU is free, they train fast (~40 min each).
- Save incrementally. Commit after each corpus finishes training.
- Do not modify any existing files outside paper7.1/intermediate_entropy/.
- If training time is short (GPU freed late), prioritize SYN-5 first.
  SYN-5 (H≈5.0) is the single most important data point because it's
  closest to the basin. SYN-6 and SYN-7 are valuable but secondary.

BEGIN. Start with Phase 1 (CPU corpus generation) immediately.
The GPU wait is expected — use the time productively.
```
