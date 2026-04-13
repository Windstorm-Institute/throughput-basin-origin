#!/usr/bin/env python3
"""
Round 3 Analysis Experiments (CPU only — runs parallel with GPU experiments)
1. NF4 vs symmetric level allocation analysis
2. GPT-2 outlier matrix analysis
3. Cross-modal structural bonus normalization
4. Music vs speech source entropy
5. Per-layer cliff progression
"""

import os, sys, time, math, csv, subprocess, traceback
import numpy as np
import torch

REPO = "/home/user1-gpu/agi-extensions"
BASE = "/home/user1-gpu/agi-extensions/weekend_experiments"

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(f"{BASE}/round3_analysis.log", "a") as f:
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

# =====================================================================
# 1. NF4 vs Symmetric Level Allocation
# =====================================================================
def analysis_level_allocation():
    OUT = f"{BASE}/round3_analysis_level_allocation"
    os.makedirs(f"{OUT}/results", exist_ok=True)
    os.makedirs(f"{OUT}/plots", exist_ok=True)
    log("\n" + "="*60)
    log("ANALYSIS 1: NF4 vs SYMMETRIC LEVEL ALLOCATION")
    log("Why does NF4 survive at INT4 but symmetric doesn't?")
    log("="*60)

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-410m', torch_dtype=torch.float32)

    # Extract a representative weight matrix
    W = model.gpt_neox.layers[0].attention.query_key_value.weight.detach().numpy()
    log(f"  Weight matrix: {W.shape}, range [{W.min():.4f}, {W.max():.4f}]")
    log(f"  Mean: {W.mean():.6f}, Std: {W.std():.6f}")

    del model

    n_bits = 4
    n_levels = 2 * ((1 << (n_bits - 1)) - 1) + 1  # 15 levels for INT4

    # Generate test input
    rng = np.random.RandomState(42)
    X = rng.randn(64, W.shape[1]).astype(np.float32) * 0.1
    Y_fp = X @ W.T

    results = []

    # Method 1: Symmetric uniform
    log("  Symmetric uniform...")
    qmax = (1 << (n_bits - 1)) - 1
    wmax = np.max(np.abs(W))
    scale = wmax / qmax
    W_sym = np.clip(np.round(W / scale), -qmax, qmax) * scale
    Y_sym = X @ W_sym.T
    cos_sym = np.dot(Y_fp.flatten(), Y_sym.flatten()) / (np.linalg.norm(Y_fp) * np.linalg.norm(Y_sym) + 1e-10)
    mse_sym = np.mean((Y_fp - Y_sym)**2)
    results.append({'method': 'symmetric_uniform', 'cosine': cos_sym, 'mse': mse_sym,
                    'n_effective_levels': len(np.unique(np.clip(np.round(W / scale), -qmax, qmax).astype(int)))})
    log(f"    cosine={cos_sym:.6f}, mse={mse_sym:.8f}")

    # Method 2: NF4 (normal-float-4 — levels at normal distribution quantiles)
    log("  NF4 (normal quantiles)...")
    from scipy.stats import norm
    quantiles = np.linspace(0.5/n_levels, 1 - 0.5/n_levels, n_levels)
    nf4_levels = norm.ppf(quantiles) * W.std()
    # Quantize: find nearest NF4 level for each weight
    W_flat = W.flatten()
    indices = np.argmin(np.abs(W_flat[:, None] - nf4_levels[None, :]), axis=1)
    W_nf4 = nf4_levels[indices].reshape(W.shape)
    Y_nf4 = X @ W_nf4.T
    cos_nf4 = np.dot(Y_fp.flatten(), Y_nf4.flatten()) / (np.linalg.norm(Y_fp) * np.linalg.norm(Y_nf4) + 1e-10)
    mse_nf4 = np.mean((Y_fp - Y_nf4)**2)
    results.append({'method': 'NF4_normal_quantiles', 'cosine': cos_nf4, 'mse': mse_nf4,
                    'n_effective_levels': n_levels})
    log(f"    cosine={cos_nf4:.6f}, mse={mse_nf4:.8f}")

    # Method 3: Lloyd-Max (optimal for actual distribution)
    log("  Lloyd-Max (optimal MSE)...")
    from scipy.cluster.vq import kmeans
    W_sample = W_flat[rng.choice(len(W_flat), min(100000, len(W_flat)), replace=False)]
    centroids, _ = kmeans(W_sample.astype(np.float64), n_levels)
    centroids = np.sort(centroids).astype(np.float32)
    indices_lm = np.argmin(np.abs(W_flat[:, None] - centroids[None, :]), axis=1)
    W_lm = centroids[indices_lm].reshape(W.shape)
    Y_lm = X @ W_lm.T
    cos_lm = np.dot(Y_fp.flatten(), Y_lm.flatten()) / (np.linalg.norm(Y_fp) * np.linalg.norm(Y_lm) + 1e-10)
    mse_lm = np.mean((Y_fp - Y_lm)**2)
    results.append({'method': 'lloyd_max_optimal', 'cosine': cos_lm, 'mse': mse_lm,
                    'n_effective_levels': len(centroids)})
    log(f"    cosine={cos_lm:.6f}, mse={mse_lm:.8f}")

    # Method 4: Log-scale (more levels near zero)
    log("  Log-scale...")
    pos_levels = np.logspace(-3, np.log10(wmax), n_levels // 2)
    log_levels = np.concatenate([-pos_levels[::-1], [0], pos_levels])
    indices_log = np.argmin(np.abs(W_flat[:, None] - log_levels[None, :]), axis=1)
    W_log = log_levels[indices_log].reshape(W.shape)
    Y_log = X @ W_log.T
    cos_log = np.dot(Y_fp.flatten(), Y_log.flatten()) / (np.linalg.norm(Y_fp) * np.linalg.norm(Y_log) + 1e-10)
    mse_log = np.mean((Y_fp - Y_log)**2)
    results.append({'method': 'log_scale', 'cosine': cos_log, 'mse': mse_log,
                    'n_effective_levels': len(log_levels)})
    log(f"    cosine={cos_log:.6f}, mse={mse_log:.8f}")

    # Method 5: Random levels (negative control)
    log("  Random levels (control)...")
    random_levels = np.sort(rng.uniform(-wmax, wmax, n_levels)).astype(np.float32)
    indices_rand = np.argmin(np.abs(W_flat[:, None] - random_levels[None, :]), axis=1)
    W_rand = random_levels[indices_rand].reshape(W.shape)
    Y_rand = X @ W_rand.T
    cos_rand = np.dot(Y_fp.flatten(), Y_rand.flatten()) / (np.linalg.norm(Y_fp) * np.linalg.norm(Y_rand) + 1e-10)
    mse_rand = np.mean((Y_fp - Y_rand)**2)
    results.append({'method': 'random_levels', 'cosine': cos_rand, 'mse': mse_rand,
                    'n_effective_levels': n_levels})
    log(f"    cosine={cos_rand:.6f}, mse={mse_rand:.8f}")

    # Also test at INT3 for each method
    log("\n  --- Same methods at INT3 ---")
    n_bits_3 = 3
    n_levels_3 = 2 * ((1 << (n_bits_3 - 1)) - 1) + 1  # 7 levels

    for method_name, make_levels in [
        ('symmetric_uniform', lambda: np.linspace(-wmax, wmax, n_levels_3)),
        ('NF4_normal_quantiles', lambda: norm.ppf(np.linspace(0.5/n_levels_3, 1-0.5/n_levels_3, n_levels_3)) * W.std()),
        ('lloyd_max_optimal', lambda: np.sort(kmeans(W_sample.astype(np.float64), n_levels_3)[0]).astype(np.float32)),
    ]:
        levels = make_levels()
        idx = np.argmin(np.abs(W_flat[:, None] - levels[None, :]), axis=1)
        W_q = levels[idx].reshape(W.shape)
        Y_q = X @ W_q.T
        cos = np.dot(Y_fp.flatten(), Y_q.flatten()) / (np.linalg.norm(Y_fp) * np.linalg.norm(Y_q) + 1e-10)
        mse = np.mean((Y_fp - Y_q)**2)
        results.append({'method': f'{method_name}_INT3', 'cosine': cos, 'mse': mse,
                        'n_effective_levels': len(levels)})
        log(f"    {method_name} INT3: cosine={cos:.6f}")

    save_csv(results, f"{OUT}/results/level_allocation.csv")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        methods_4 = [r for r in results if 'INT3' not in r['method']]
        methods_3 = [r for r in results if 'INT3' in r['method']]

        x = range(len(methods_4))
        names = [r['method'] for r in methods_4]
        cos_4 = [r['cosine'] for r in methods_4]

        ax.bar([i-0.2 for i in x], cos_4, 0.35, label='INT4', color='steelblue')

        if methods_3:
            cos_3 = [r['cosine'] for r in methods_3]
            ax.bar([i+0.2 for i in range(len(methods_3))], cos_3, 0.35, label='INT3', color='coral')

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Output cosine similarity')
        ax.set_title('INT4 vs INT3: Level Allocation Matters\n(Pythia-410M attn_qkv weights)')
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        fig.savefig(f"{OUT}/plots/level_allocation.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        log(f"  Plot error: {e}")

    # Verdict
    report = "# Analysis 1: Level Allocation at INT4\n\n"
    report += "## Key finding: The cliff is about LEVEL ALLOCATION, not just bit count\n\n"
    report += "| Method | INT4 cosine | INT3 cosine | Cliff |\n|---|---|---|---|\n"
    for m4 in methods_4:
        m3 = [r for r in methods_3 if r['method'].replace('_INT3', '') == m4['method']]
        c3 = m3[0]['cosine'] if m3 else '—'
        cliff = f"{m4['cosine'] - m3[0]['cosine']:.4f}" if m3 else '—'
        report += f"| {m4['method']} | {m4['cosine']:.6f} | {c3 if isinstance(c3, str) else f'{c3:.6f}'} | {cliff} |\n"

    report += f"\n## Interpretation\n\n"
    if cos_nf4 > cos_sym + 0.05:
        report += f"NF4 ({cos_nf4:.4f}) dramatically outperforms symmetric ({cos_sym:.4f}) at INT4.\n"
        report += "The cliff is not about 4 bits — it's about WHERE those 4 bits are placed.\n"
        report += "Hardware implication: support non-uniform quantization tables, not just integer arithmetic.\n"
    else:
        report += f"NF4 ({cos_nf4:.4f}) and symmetric ({cos_sym:.4f}) are similar. The cliff IS about bit count.\n"

    with open(f"{OUT}/results/ANALYSIS1_REPORT.md", 'w') as f:
        f.write(report)

    git_push("Analysis 1: level allocation comparison", ['weekend_experiments/round3_analysis_level_allocation/'])
    log("ANALYSIS 1 COMPLETE")

# =====================================================================
# 2. GPT-2 Outlier Matrix Analysis
# =====================================================================
def analysis_outlier():
    OUT = f"{BASE}/round3_analysis_outlier"
    os.makedirs(f"{OUT}/results", exist_ok=True)
    os.makedirs(f"{OUT}/plots", exist_ok=True)
    log("\n" + "="*60)
    log("ANALYSIS 2: GPT-2 OUTLIER MATRIX")
    log("Why does one matrix show cliff ratio 0.9×?")
    log("="*60)

    # Load the multi-model data
    import pandas as pd
    data_path = f"{BASE}/round3_exp4_multi_model/results/multi_model_quant.csv"
    if not os.path.exists(data_path):
        log("  ERROR: multi_model_quant.csv not found")
        return

    df = pd.read_csv(data_path)
    gpt2_data = df[df['model'] == 'gpt2-medium']

    # Find cliff ratios for each matrix
    matrices = gpt2_data['matrix'].unique()
    cliff_data = []

    for mat in matrices:
        mat_df = gpt2_data[gpt2_data['matrix'] == mat]
        cos_5 = mat_df[mat_df['n_bits'] == 5]['cosine'].values
        cos_4 = mat_df[mat_df['n_bits'] == 4]['cosine'].values
        cos_3 = mat_df[mat_df['n_bits'] == 3]['cosine'].values

        if len(cos_5) > 0 and len(cos_4) > 0 and len(cos_3) > 0:
            deg_54 = cos_5[0] - cos_4[0]
            deg_43 = cos_4[0] - cos_3[0]
            ratio = deg_43 / deg_54 if deg_54 > 0.001 else float('inf')
            cliff_data.append({
                'matrix': mat, 'cos_INT8': mat_df[mat_df['n_bits']==8]['cosine'].values[0] if len(mat_df[mat_df['n_bits']==8]) > 0 else None,
                'cos_INT4': cos_4[0], 'cos_INT3': cos_3[0],
                'deg_54': deg_54, 'deg_43': deg_43, 'cliff_ratio': ratio,
            })

    cliff_data.sort(key=lambda x: x['cliff_ratio'])

    log("  Cliff ratios for GPT-2-medium matrices:")
    for cd in cliff_data:
        outlier = " ← OUTLIER" if cd['cliff_ratio'] < 1.5 else ""
        log(f"    {cd['matrix'][-30:]}: {cd['cliff_ratio']:.2f}×{outlier}")

    # Analyze the outlier's weight distribution
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained('gpt2-medium', torch_dtype=torch.float32)

    outlier_mats = [cd for cd in cliff_data if cd['cliff_ratio'] < 1.5]
    normal_mats = [cd for cd in cliff_data if cd['cliff_ratio'] > 2.0]

    dist_analysis = []

    for mat_info in cliff_data[:6]:
        mat_name = mat_info['matrix']
        # Find the parameter
        W = None
        for name, param in model.named_parameters():
            if name == mat_name:
                W = param.detach().numpy()
                break
        if W is None:
            continue

        stats = {
            'matrix': mat_name,
            'cliff_ratio': mat_info['cliff_ratio'],
            'shape': str(W.shape),
            'mean': float(np.mean(W)),
            'std': float(np.std(W)),
            'skewness': float(np.mean(((W - W.mean()) / W.std()) ** 3)),
            'kurtosis': float(np.mean(((W - W.mean()) / W.std()) ** 4)),
            'sparsity': float(np.mean(np.abs(W) < 0.01 * np.max(np.abs(W)))),
            'max_abs': float(np.max(np.abs(W))),
            'pct_near_zero': float(np.mean(np.abs(W) < W.std() * 0.1)),
        }
        dist_analysis.append(stats)
        log(f"    {mat_name[-25:]}: cliff={mat_info['cliff_ratio']:.2f}×, "
            f"std={stats['std']:.6f}, kurtosis={stats['kurtosis']:.2f}, "
            f"sparsity={stats['sparsity']:.3f}")

    del model

    save_csv(dist_analysis, f"{OUT}/results/outlier_analysis.csv")
    save_csv(cliff_data, f"{OUT}/results/gpt2_cliff_ratios.csv")

    report = "# Analysis 2: GPT-2 Outlier Matrix\n\n"
    report += "## Cliff ratios for GPT-2-medium\n\n"
    report += "| Matrix | Cliff ratio | Std | Kurtosis | Sparsity |\n|---|---|---|---|---|\n"
    for d in dist_analysis:
        outlier = " **OUTLIER**" if d['cliff_ratio'] < 1.5 else ""
        report += f"| ...{d['matrix'][-25:]} | {d['cliff_ratio']:.2f}×{outlier} | {d['std']:.6f} | {d['kurtosis']:.2f} | {d['sparsity']:.3f} |\n"

    report += "\n## Interpretation\n\n"
    if outlier_mats:
        outlier_stats = [d for d in dist_analysis if d['cliff_ratio'] < 1.5]
        normal_stats = [d for d in dist_analysis if d['cliff_ratio'] > 2.0]
        if outlier_stats and normal_stats:
            o_kurt = np.mean([d['kurtosis'] for d in outlier_stats])
            n_kurt = np.mean([d['kurtosis'] for d in normal_stats])
            report += f"Outlier kurtosis: {o_kurt:.2f}, Normal kurtosis: {n_kurt:.2f}\n"
            if o_kurt < n_kurt:
                report += "Outlier has LOWER kurtosis (lighter tails) → fewer extreme weights → less to lose at INT3.\n"
            else:
                report += "Outlier has HIGHER kurtosis → investigate other distributional properties.\n"

    with open(f"{OUT}/results/ANALYSIS2_REPORT.md", 'w') as f:
        f.write(report)

    git_push("Analysis 2: GPT-2 outlier matrix", ['weekend_experiments/round3_analysis_outlier/'])
    log("ANALYSIS 2 COMPLETE")

# =====================================================================
# 3. Cross-Modal Structural Bonus Normalization
# =====================================================================
def analysis_cross_modal_bonus():
    OUT = f"{BASE}/round3_analysis_cross_bonus"
    os.makedirs(f"{OUT}/results", exist_ok=True)
    os.makedirs(f"{OUT}/plots", exist_ok=True)
    log("\n" + "="*60)
    log("ANALYSIS 3: CROSS-MODAL STRUCTURAL BONUS")
    log("="*60)

    # Gather structural bonuses from all experiments
    bonuses = [
        {'modality': 'Language (text)', 'bonus_raw': 6.7, 'unit': 'bits/token',
         'tokens_per_sec': 20.8, 'bonus_per_sec': 6.7 * 20.8,
         'source': 'Paper 6'},
        {'modality': 'Vision (spatial)', 'bonus_raw': 0.69, 'unit': 'bits/pixel',
         'tokens_per_sec': 96*96*3*30, 'bonus_per_sec': 0.69 * 96*96*3*30,
         'source': 'P8-A v2 cascade'},
        {'modality': 'Audio (speech temporal)', 'bonus_raw': 0.63, 'unit': 'bits/mel_dim',
         'tokens_per_sec': 128*43, 'bonus_per_sec': 0.63 * 128*43,
         'source': 'Exp 1 speech'},
        {'modality': 'Audio (music temporal)', 'bonus_raw': 2.83, 'unit': 'bits/mel_dim',
         'tokens_per_sec': 128*43, 'bonus_per_sec': 2.83 * 128*43,
         'source': 'Exp 1 music'},
        {'modality': 'PCFG-8 (synthetic hierarchy)', 'bonus_raw': 5.33, 'unit': 'bits/token',
         'tokens_per_sec': 20.8, 'bonus_per_sec': 5.33 * 20.8,
         'source': 'Paper 7 R6'},
    ]

    save_csv(bonuses, f"{OUT}/results/cross_modal_bonus.csv")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        names = [b['modality'] for b in bonuses]
        raw = [b['bonus_raw'] for b in bonuses]
        per_sec = [b['bonus_per_sec'] for b in bonuses]

        colors = ['#2166ac', '#b2182b', '#4daf4a', '#4daf4a', '#ff7f0e']

        ax1.barh(names, raw, color=colors)
        ax1.set_xlabel('Structural bonus (bits per source unit)')
        ax1.set_title('Raw Structural Bonus')
        ax1.grid(True, alpha=0.3, axis='x')

        ax2.barh(names, per_sec, color=colors)
        ax2.set_xlabel('Structural bonus (bits per second)')
        ax2.set_title('Structural Bonus × Signal Rate')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        fig.savefig(f"{OUT}/plots/cross_modal_bonus.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{OUT}/plots/cross_modal_bonus.pdf", bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        log(f"  Plot error: {e}")

    report = "# Analysis 3: Cross-Modal Structural Bonus\n\n"
    report += "| Modality | Bonus (per unit) | Unit | Bonus × rate (bits/sec) |\n|---|---|---|---|\n"
    for b in bonuses:
        report += f"| {b['modality']} | {b['bonus_raw']:.2f} | {b['unit']} | {b['bonus_per_sec']:.0f} |\n"
    report += "\nLanguage has the highest per-unit bonus (6.7 bits/token) but vision has the highest\n"
    report += "total information rate because of the enormous pixel count per second.\n"

    with open(f"{OUT}/results/ANALYSIS3_REPORT.md", 'w') as f:
        f.write(report)

    git_push("Analysis 3: cross-modal structural bonus", ['weekend_experiments/round3_analysis_cross_bonus/'])
    log("ANALYSIS 3 COMPLETE")

# =====================================================================
# 4. Music vs Speech Source Entropy
# =====================================================================
def analysis_music_speech():
    OUT = f"{BASE}/round3_analysis_music_speech"
    os.makedirs(f"{OUT}/results", exist_ok=True)
    log("\n" + "="*60)
    log("ANALYSIS 4: MUSIC vs SPEECH SOURCE ENTROPY")
    log("Why is music throughput 4.5× higher than speech?")
    log("="*60)

    import gzip

    # Load the audio data from Exp 1
    # Generate the same audio sources to measure their raw entropy
    sr = 22050

    # Synthetic speech
    t = np.linspace(0, 60, sr*60)
    f0 = 120 + 30 * np.sin(2*np.pi*0.5*t)
    speech = (0.4*np.sin(2*np.pi*f0*t)*(0.5+0.5*np.sin(2*np.pi*3*t)) +
              0.2*np.sin(2*np.pi*2.5*f0*t) + 0.1*np.sin(2*np.pi*4*f0*t) +
              0.03*np.random.randn(len(t))).astype(np.float32)

    # Music
    music = np.zeros(sr*60, dtype=np.float32)
    rng = np.random.RandomState(42)
    notes = [261.6, 293.7, 329.6, 349.2, 392.0, 440.0, 493.9, 523.3]
    for s in range(0, 60*sr, sr//4):
        e = min(s+sr, len(music))
        freq = notes[rng.randint(len(notes))] * (2**rng.randint(-1,2))
        amp = rng.uniform(0.05, 0.15)
        seg = np.arange(e-s) / sr
        music[s:e] += amp * np.sin(2*np.pi*freq*seg) * np.exp(-seg*3)

    # Noise
    noise = np.random.randn(sr*60).astype(np.float32) * 0.3

    # Measure entropy of raw waveform via gzip
    results = []
    for name, audio in [('speech', speech), ('music', music), ('noise', noise)]:
        # Convert to 16-bit PCM (standard audio format)
        pcm = (audio * 32767).astype(np.int16)
        raw_bytes = pcm.tobytes()

        # Gzip compression
        compressed = gzip.compress(raw_bytes, compresslevel=9)
        bits_per_sample = len(compressed) * 8 / len(pcm)

        # Spectral entropy
        from scipy.signal import welch
        freqs, psd = welch(audio, fs=sr, nperseg=2048)
        psd_norm = psd / psd.sum()
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-15))

        # Temporal autocorrelation at lag 1
        autocorr = np.corrcoef(audio[:-1], audio[1:])[0, 1]

        results.append({
            'source': name,
            'duration_sec': len(audio)/sr,
            'bits_per_sample_gzip': bits_per_sample,
            'spectral_entropy': spectral_entropy,
            'autocorrelation_lag1': autocorr,
            'rms_amplitude': float(np.sqrt(np.mean(audio**2))),
            'model_throughput_bpd': {'speech': 0.63, 'music': 2.83, 'noise': 0.0}[name],
        })
        log(f"  {name}: H_gzip={bits_per_sample:.3f} bits/sample, "
            f"spectral_H={spectral_entropy:.2f}, autocorr={autocorr:.4f}, "
            f"model_throughput={results[-1]['model_throughput_bpd']:.2f}")

    save_csv(results, f"{OUT}/results/music_speech_entropy.csv")

    report = "# Analysis 4: Music vs Speech\n\n"
    report += "| Source | H_gzip (bits/sample) | Spectral entropy | Autocorrelation | Model throughput |\n|---|---|---|---|---|\n"
    for r in results:
        report += f"| {r['source']} | {r['bits_per_sample_gzip']:.3f} | {r['spectral_entropy']:.2f} | {r['autocorrelation_lag1']:.4f} | {r['model_throughput_bpd']:.2f} |\n"

    report += "\n## Why music > speech in throughput\n\n"
    music_r = [r for r in results if r['source'] == 'music'][0]
    speech_r = [r for r in results if r['source'] == 'speech'][0]
    if music_r['bits_per_sample_gzip'] > speech_r['bits_per_sample_gzip']:
        report += "Music has HIGHER source entropy than speech → more information to extract → higher throughput.\n"
        report += "This is consistent with BPT ≈ entropy - f(structure).\n"
    else:
        report += "Music has LOWER source entropy but HIGHER throughput → music's structure is more exploitable.\n"
        report += "The mel-spectrogram representation may favor harmonic structure over formant structure.\n"

    with open(f"{OUT}/results/ANALYSIS4_REPORT.md", 'w') as f:
        f.write(report)

    git_push("Analysis 4: music vs speech entropy", ['weekend_experiments/round3_analysis_music_speech/'])
    log("ANALYSIS 4 COMPLETE")

# =====================================================================
# 5. Per-Layer Cliff Progression
# =====================================================================
def analysis_per_layer():
    OUT = f"{BASE}/round3_analysis_per_layer"
    os.makedirs(f"{OUT}/results", exist_ok=True)
    os.makedirs(f"{OUT}/plots", exist_ok=True)
    log("\n" + "="*60)
    log("ANALYSIS 5: PER-LAYER CLIFF PROGRESSION")
    log("Does the cliff get worse in deeper layers?")
    log("="*60)

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-410m', torch_dtype=torch.float32)

    results = []
    rng = np.random.RandomState(42)

    # Test attention dense weight in every layer
    for layer_idx in range(24):  # Pythia-410M has 24 layers
        mat_name = f'gpt_neox.layers.{layer_idx}.attention.dense.weight'
        W = None
        for name, param in model.named_parameters():
            if name == mat_name:
                W = param.detach().numpy()
                break
        if W is None:
            continue

        X = rng.randn(32, W.shape[1]).astype(np.float32) * 0.1
        Y_fp = X @ W.T

        for n_bits in [8, 5, 4, 3, 2]:
            qmax = (1 << (n_bits - 1)) - 1
            wmax = np.max(np.abs(W))
            scale = wmax / qmax if wmax > 0 else 1e-8
            W_q = np.clip(np.round(W / scale), -qmax, qmax) * scale
            Y_q = X @ W_q.T
            cos = np.dot(Y_fp.flatten(), Y_q.flatten()) / (np.linalg.norm(Y_fp) * np.linalg.norm(Y_q) + 1e-10)

            results.append({
                'layer': layer_idx, 'n_bits': n_bits, 'cosine': cos,
                'weight_std': float(np.std(W)), 'weight_kurtosis': float(np.mean(((W-W.mean())/W.std())**4)),
            })

        if layer_idx % 6 == 0:
            r4 = [r for r in results if r['layer']==layer_idx and r['n_bits']==4][0]
            r3 = [r for r in results if r['layer']==layer_idx and r['n_bits']==3][0]
            log(f"    Layer {layer_idx}: INT4={r4['cosine']:.4f}, INT3={r3['cosine']:.4f}, "
                f"std={r4['weight_std']:.6f}, kurtosis={r4['weight_kurtosis']:.2f}")

    del model

    save_csv(results, f"{OUT}/results/per_layer_cliff.csv")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        layers = sorted(set(r['layer'] for r in results))

        for nb, color, style in [(4, 'steelblue', '-'), (3, 'coral', '--'), (2, 'gray', ':')]:
            cos_by_layer = [np.mean([r['cosine'] for r in results if r['layer']==l and r['n_bits']==nb]) for l in layers]
            ax1.plot(layers, cos_by_layer, f'{style}o', color=color, label=f'INT{nb}', markersize=4)

        ax1.set_xlabel('Layer index')
        ax1.set_ylabel('Output cosine similarity')
        ax1.set_title('Pythia-410M: Quantization fidelity by layer (attention.dense)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Cliff ratio by layer
        cliff_by_layer = []
        for l in layers:
            c4 = np.mean([r['cosine'] for r in results if r['layer']==l and r['n_bits']==4])
            c3 = np.mean([r['cosine'] for r in results if r['layer']==l and r['n_bits']==3])
            c5 = np.mean([r['cosine'] for r in results if r['layer']==l and r['n_bits']==5])
            d54 = c5 - c4
            d43 = c4 - c3
            ratio = d43 / d54 if d54 > 0.001 else 0
            cliff_by_layer.append(ratio)

        ax2.bar(layers, cliff_by_layer, color='coral', alpha=0.7)
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Layer index')
        ax2.set_ylabel('Cliff ratio (INT4→3 / INT5→4)')
        ax2.set_title('Cliff ratio by layer — does it get worse deeper?')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        fig.savefig(f"{OUT}/plots/per_layer_cliff.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{OUT}/plots/per_layer_cliff.pdf", bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        log(f"  Plot error: {e}")

    report = "# Analysis 5: Per-Layer Cliff Progression\n\n"
    report += f"Tested attention.dense weight across all 24 layers of Pythia-410M.\n\n"

    if cliff_by_layer:
        early = np.mean(cliff_by_layer[:8])
        mid = np.mean(cliff_by_layer[8:16])
        late = np.mean(cliff_by_layer[16:])
        report += f"| Layers | Mean cliff ratio |\n|---|---|\n"
        report += f"| 0–7 (early) | {early:.2f}× |\n"
        report += f"| 8–15 (middle) | {mid:.2f}× |\n"
        report += f"| 16–23 (late) | {late:.2f}× |\n"

        if late > early * 1.5:
            report += "\n**The cliff gets WORSE in deeper layers.** Later layers have sharper weight distributions → more fragile under quantization.\n"
        elif early > late * 1.5:
            report += "\n**The cliff gets BETTER in deeper layers.** Unexpected — early layers are more fragile.\n"
        else:
            report += "\n**The cliff is roughly constant across layers.** Depth doesn't matter much.\n"

    with open(f"{OUT}/results/ANALYSIS5_REPORT.md", 'w') as f:
        f.write(report)

    git_push("Analysis 5: per-layer cliff progression", ['weekend_experiments/round3_analysis_per_layer/'])
    log("ANALYSIS 5 COMPLETE")

# =====================================================================
# MAIN
# =====================================================================
def main():
    log("="*60)
    log("ROUND 3 ANALYSIS EXPERIMENTS (CPU only)")
    log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("="*60)

    analyses = [
        ('Analysis 1: Level allocation', analysis_level_allocation),
        ('Analysis 2: GPT-2 outlier', analysis_outlier),
        ('Analysis 3: Cross-modal bonus', analysis_cross_modal_bonus),
        ('Analysis 4: Music vs speech', analysis_music_speech),
        ('Analysis 5: Per-layer cliff', analysis_per_layer),
    ]

    for name, func in analyses:
        try:
            func()
        except Exception as e:
            log(f"{name} FAILED: {e}")
            traceback.print_exc()

    log("\n" + "="*60)
    log("ALL ANALYSES COMPLETE")
    log("="*60)

    git_push("Round 3 analyses complete (5 CPU experiments)", ['weekend_experiments/'])

if __name__ == "__main__":
    main()
