#!/usr/bin/env python3
"""Build PCFG corpus with ~8-bit entropy AND deep hierarchical structure.

Strategy:
- 256 terminals (bytes 0x00-0xFF), distributed across 16 lexical categories
  of 16 terminals each. Within each category, sample uniformly → balanced
  byte distribution → ~8-bit byte entropy.
- ~60 non-terminals; ~200 production rules; recursive.
- Bounded recursion depth to avoid runaway / collapse.
- Output: whitespace-separated "xHH" tokens, sentences joined by "\\n"
  (matching exp-1 SYN-8 format exactly so the tokenizer is comparable).
"""
import os, sys, random, math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/user1-gpu/agi-extensions/paper7.1")
COR = ROOT / "corpora"; COR.mkdir(exist_ok=True, parents=True)

random.seed(0)

# 16 categories x 16 terminals = 256
N_CAT = 16
PER_CAT = 16
CAT_NAMES = [f"C{i:02d}" for i in range(N_CAT)]

# Assign bytes to categories (interleaved so categories are visually balanced)
TERMINALS = {c: [] for c in CAT_NAMES}
for b in range(256):
    TERMINALS[CAT_NAMES[b % N_CAT]].append(b)

def fmt(b): return f"x{b:02X}"

# ---------------- Grammar ----------------
# Non-terminals beyond the lexical categories — these create hierarchy.
# Productions: list of (rhs_tuple, weight). RHS items are either category
# names (CAT_NAMES) or non-terminal names below.
GRAMMAR = {}

def add(lhs, rhs, w=1.0):
    GRAMMAR.setdefault(lhs, []).append((tuple(rhs), float(w)))

# Top-level sentence
add("S",  ["CL"], 6)
add("S",  ["CL", "C00", "S"], 2)        # CONJ-recursion (use C00 as CONJ-like)
add("S",  ["ADV_P", "CL"], 2)
add("S",  ["CL", "C01", "CL"], 2)

# Clause
add("CL", ["NP", "VP"], 5)
add("CL", ["NP", "VP", "PP"], 3)
add("CL", ["ADV_P", "NP", "VP"], 2)
add("CL", ["NP", "AUX_P", "VP"], 2)
add("CL", ["NP", "VP", "OBJ"], 4)
add("CL", ["SUB_CL", "NP", "VP"], 1)

# Subordinate clause (recursive)
add("SUB_CL", ["C02", "CL"], 3)
add("SUB_CL", ["C03", "NP", "VP"], 2)

# Noun phrase (recursive via NP → ADJ_P NP)
add("NP", ["DET", "N"], 4)
add("NP", ["DET", "ADJ_P", "N"], 3)
add("NP", ["N"], 2)
add("NP", ["NP", "PP"], 2)
add("NP", ["NP", "REL"], 1)
add("NP", ["DET", "N", "N"], 1)  # compound

# Adjective phrase (recursive)
add("ADJ_P", ["ADJ"], 5)
add("ADJ_P", ["ADJ", "ADJ_P"], 2)
add("ADJ_P", ["ADV", "ADJ"], 2)

# Adverb phrase
add("ADV_P", ["ADV"], 4)
add("ADV_P", ["ADV", "ADV"], 1)
add("ADV_P", ["PP"], 2)

# Verb phrase
add("VP", ["V"], 3)
add("VP", ["V", "NP"], 5)
add("VP", ["V", "PP"], 2)
add("VP", ["V", "NP", "PP"], 3)
add("VP", ["AUX", "V", "NP"], 2)
add("VP", ["V", "SUB_CL"], 1)

# Prepositional phrase
add("PP", ["PREP", "NP"], 5)
add("PP", ["PREP", "DET", "N"], 3)

# Object (alt NP-like)
add("OBJ", ["NP"], 3)
add("OBJ", ["PP"], 1)

# Aux phrase
add("AUX_P", ["AUX"], 1)

# Relative clause
add("REL", ["C04", "VP"], 2)
add("REL", ["C04", "NP", "VP"], 1)

# Lexical non-terminals → category terminals
add("N",    ["NOUN_C"], 1)
add("V",    ["VERB_C"], 1)
add("ADJ",  ["ADJ_C"], 1)
add("ADV",  ["ADV_C"], 1)
add("DET",  ["DET_C"], 1)
add("PREP", ["PREP_C"], 1)
add("AUX",  ["AUX_C"], 1)

# Map "lexical pools" to actual byte categories. Some categories serve as
# more than one pool to keep byte distribution balanced even though some
# pools are sampled more often than others.
LEX = {
    "NOUN_C": ["C05", "C06", "C07", "C08"],   # 64 noun-bytes
    "VERB_C": ["C09", "C10", "C11"],          # 48 verb-bytes
    "ADJ_C":  ["C12", "C13"],                 # 32
    "ADV_C":  ["C14"],                        # 16
    "DET_C":  ["C15"],                        # 16
    "PREP_C": ["C02", "C03"],                 # 32 (also used as SUB_CL markers)
    "AUX_C":  ["C04"],                        # 16 (also REL marker)
}
# Note: C00, C01 are used as conjunction markers in S productions.
# This means *every* byte is reachable. The marker categories appear in
# fixed positions while the lexical pools contribute the bulk of bytes.

# Normalize weights
def normalize():
    for lhs, prods in GRAMMAR.items():
        Z = sum(w for _, w in prods)
        GRAMMAR[lhs] = [(rhs, w / Z) for rhs, w in prods]
normalize()

MAX_DEPTH = 10

def choose(prods, depth):
    # at high depth, prefer non-recursive productions
    if depth >= MAX_DEPTH:
        # find shortest production
        return min(prods, key=lambda x: len(x[0]))[0]
    r = random.random(); acc = 0
    for rhs, p in prods:
        acc += p
        if r <= acc: return rhs
    return prods[-1][0]

def expand(sym, depth, out):
    if sym in LEX:
        cat = random.choice(LEX[sym])
        b = random.choice(TERMINALS[cat])
        out.append(b); return
    if sym in CAT_NAMES:
        b = random.choice(TERMINALS[sym])
        out.append(b); return
    if sym in GRAMMAR:
        rhs = choose(GRAMMAR[sym], depth)
        for s in rhs:
            expand(s, depth + 1, out)
        return
    raise ValueError(f"unknown symbol {sym}")

def gen_sentence():
    out = []
    expand("S", 0, out)
    return out

def main():
    target_chars = 100_000_000
    out_path = COR / "pcfg_corpus.txt"
    sys.stdout.write(f"Generating PCFG corpus → {out_path}\n"); sys.stdout.flush()
    bytes_collected = []
    chars_written = 0
    sent_count = 0
    with open(out_path, "w") as f:
        buf = []
        while chars_written < target_chars:
            sent = gen_sentence()
            if len(sent) < 4:
                continue
            bytes_collected.extend(sent)
            line = " ".join(fmt(b) for b in sent)
            buf.append(line)
            chars_written += len(line) + 1
            sent_count += 1
            if len(buf) >= 5000:
                f.write("\n".join(buf) + "\n")
                buf = []
                if sent_count % 50000 == 0:
                    print(f"  sent={sent_count} chars={chars_written:,}", flush=True)
        if buf:
            f.write("\n".join(buf) + "\n")
    print(f"Done. {sent_count} sentences, {chars_written:,} chars")

    # Compute byte-level entropy on the underlying 256-symbol stream
    cnt = Counter(bytes_collected)
    total = sum(cnt.values())
    H = -sum((c/total) * math.log2(c/total) for c in cnt.values())
    print(f"Empirical entropy (256-symbol stream): {H:.4f} bits/symbol")
    print(f"Unique terminals used: {len(cnt)}/256")

    # Save the raw byte stream as well (for the "shuffled" version)
    raw_path = COR / "pcfg_corpus.bin"
    with open(raw_path, "wb") as f:
        f.write(bytes(bytes_collected))
    print(f"Raw byte stream → {raw_path}  ({len(bytes_collected):,} bytes)")

    # Shuffled byte stream (preserves entropy exactly)
    arr = list(bytes_collected)
    rng = random.Random(12345)
    rng.shuffle(arr)
    shuf_bin = COR / "pcfg_shuffled.bin"
    with open(shuf_bin, "wb") as f:
        f.write(bytes(arr))

    # Format the shuffled stream as a text corpus matching the SYN-8 layout:
    # whitespace-separated "xHH" with synthetic newlines every ~30 tokens.
    shuf_txt = COR / "pcfg_shuffled.txt"
    with open(shuf_txt, "w") as f:
        line = []
        line_len_target = 30
        for i, b in enumerate(arr):
            line.append(fmt(b))
            if len(line) >= line_len_target:
                f.write(" ".join(line) + "\n")
                line = []
                line_len_target = rng.randint(20, 40)
        if line: f.write(" ".join(line) + "\n")
    print(f"Shuffled corpus → {shuf_txt}")

    # Verify shuffled entropy = same as original (it must, by construction)
    cnt2 = Counter(arr)
    H2 = -sum((c/total) * math.log2(c/total) for c in cnt2.values())
    print(f"Shuffled empirical entropy: {H2:.4f} bits/symbol (should equal {H:.4f})")

    with open(ROOT / "pcfg_entropy.txt", "w") as f:
        f.write(f"pcfg_corpus_entropy_bits_per_symbol: {H:.6f}\n")
        f.write(f"pcfg_shuffled_entropy_bits_per_symbol: {H2:.6f}\n")
        f.write(f"unique_terminals: {len(cnt)}\n")
        f.write(f"total_bytes: {total}\n")

if __name__ == "__main__":
    main()
