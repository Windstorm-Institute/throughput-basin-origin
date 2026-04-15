#!/usr/bin/env python3
"""
Exp 4 Fixed: Audio throughput using wav2vec2 in FP32 (FP16 caused NaN).
Also uses LJ Speech real audio and mel-spectrogram based throughput.
"""
import os, sys, time, math, csv, subprocess, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = "/home/user1-gpu/agi-extensions"
BASE = "/home/user1-gpu/agi-extensions/weekend_experiments/decisive_round"

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(f"{BASE}/phase2_full.log", "a") as f:
        f.write(line + "\n")

def save_csv(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not data: return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader()
        w.writerows(data)

def gpu_mem_report():
    r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                       '--format=csv,noheader,nounits'], capture_output=True, text=True)
    parts = r.stdout.strip().split(', ')
    used, total, util = int(parts[0]), int(parts[1]), int(parts[2])
    log(f"  GPU: {used}/{total} MB ({used*100//total}% mem), {util}% compute")

def git_push(msg, paths):
    try:
        for p in paths:
            subprocess.run(['git', 'add', p], cwd=REPO, capture_output=True)
        subprocess.run(['git', 'commit', '-m', msg + '\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>'],
                       cwd=REPO, capture_output=True)
        subprocess.run(['git', 'push'], cwd=REPO, capture_output=True)
    except: pass


def main():
    OUT = f"{BASE}/exp4_maestro_audio"
    os.makedirs(f"{OUT}/results", exist_ok=True)
    log("\n" + "="*60)
    log("EXP 4 (FIXED): AUDIO THROUGHPUT — FP32 + MEL SPECTROGRAM")
    log("="*60)

    from transformers import Wav2Vec2Processor, AutoModelForCTC
    import torchaudio

    # Load wav2vec2 in FP32 to avoid NaN
    log("  Loading wav2vec2-base-960h in FP32...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h", torch_dtype=torch.float32).cuda()
    model.eval()
    gpu_mem_report()

    sr = 16000
    duration = 10
    n_samples = sr * duration
    results = []

    # --- Synthetic audio at controlled complexity ---
    def pink_noise(n):
        white = np.random.RandomState(42).normal(0, 1, n)
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n)
        freqs[0] = 1
        fft = fft / np.sqrt(freqs)
        pink = np.fft.irfft(fft, n=n)
        return (pink / (np.max(np.abs(pink)) + 1e-8) * 0.3).astype(np.float32)

    def am_sweep(n, sr):
        t = np.arange(n) / sr
        carrier = np.sin(2 * np.pi * (200 + 800 * t / t[-1]) * t)
        modulator = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
        return (carrier * modulator * 0.5).astype(np.float32)

    audio_types = [
        ("silence", np.zeros(n_samples, dtype=np.float32)),
        ("pure_440hz", (0.5 * np.sin(2 * np.pi * 440 * np.arange(n_samples) / sr)).astype(np.float32)),
        ("chord_3note", (0.3 * (np.sin(2 * np.pi * 440 * np.arange(n_samples) / sr) +
                                 np.sin(2 * np.pi * 554 * np.arange(n_samples) / sr) +
                                 np.sin(2 * np.pi * 659 * np.arange(n_samples) / sr))).astype(np.float32)),
        ("white_noise", np.random.RandomState(42).normal(0, 0.3, n_samples).astype(np.float32)),
        ("pink_noise", pink_noise(n_samples)),
        ("am_sweep", am_sweep(n_samples, sr)),
    ]

    # --- Method 1: wav2vec2 output entropy (FP32) ---
    log("  === Method 1: wav2vec2 output entropy (FP32) ===")
    for name, waveform in audio_types:
        log(f"    {name}...")
        chunk_len = sr * 5
        total_entropy, total_frames = 0.0, 0

        for rep in range(8):
            if rep > 0:
                wv = waveform + np.random.RandomState(42 + rep).normal(0, 0.01, len(waveform)).astype(np.float32)
            else:
                wv = waveform

            for start in range(0, len(wv) - chunk_len, chunk_len):
                chunk = wv[start:start + chunk_len]
                inputs = processor(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
                input_values = inputs.input_values.cuda()  # FP32

                with torch.no_grad():
                    logits = model(input_values).logits  # FP32
                    probs = F.softmax(logits, dim=-1)
                    # Clamp to avoid log(0)
                    probs = probs.clamp(min=1e-10)
                    entropy = -(probs * probs.log2()).sum(dim=-1).mean().item()

                    if not math.isnan(entropy) and not math.isinf(entropy):
                        total_entropy += entropy * logits.size(1)
                        total_frames += logits.size(1)

        avg_entropy = total_entropy / total_frames if total_frames > 0 else float('nan')
        results.append({
            'method': 'wav2vec2_fp32',
            'audio_type': name,
            'entropy_bits_per_frame': round(avg_entropy, 4) if not math.isnan(avg_entropy) else 'NaN',
            'total_frames': total_frames
        })
        log(f"      {name}: entropy={avg_entropy:.4f} bits/frame")

    # --- Method 2: Mel-spectrogram entropy (no model needed) ---
    log("  === Method 2: Mel-spectrogram entropy ===")
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=256, n_mels=80
    )

    for name, waveform in audio_types:
        wt = torch.tensor(waveform).unsqueeze(0)
        mel = mel_transform(wt)  # [1, n_mels, time]
        mel_db = torchaudio.transforms.AmplitudeToDB()(mel)

        # Compute entropy of mel spectrogram distribution per frame
        # Normalize to probability distribution along mel axis
        mel_shifted = mel_db - mel_db.min()
        mel_norm = mel_shifted / (mel_shifted.sum(dim=1, keepdim=True) + 1e-10)
        frame_entropy = -(mel_norm * (mel_norm + 1e-10).log2()).sum(dim=1)  # [1, time]
        avg_entropy = frame_entropy.mean().item()

        results.append({
            'method': 'mel_spectrogram',
            'audio_type': name,
            'entropy_bits_per_frame': round(avg_entropy, 4),
            'total_frames': mel.size(2)
        })
        log(f"      {name}: mel entropy={avg_entropy:.4f} bits/frame ({mel.size(2)} frames)")

    # --- LJ Speech (real speech) ---
    lj_path = "/home/user1-gpu/LJSpeech-1.1/wavs"
    if os.path.exists(lj_path):
        log("  === LJ Speech (real audio) ===")
        import scipy.io.wavfile as wavfile
        wav_files = sorted([f for f in os.listdir(lj_path) if f.endswith('.wav')])[:200]

        # wav2vec2
        total_entropy, total_frames = 0.0, 0
        for wf in wav_files:
            try:
                rate, data = wavfile.read(os.path.join(lj_path, wf))
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                if rate != 16000:
                    from scipy.signal import resample
                    data = resample(data, int(len(data) * 16000 / rate)).astype(np.float32)

                inputs = processor(data, sampling_rate=16000, return_tensors="pt", padding=True)
                input_values = inputs.input_values.cuda()
                with torch.no_grad():
                    logits = model(input_values).logits
                    probs = F.softmax(logits, dim=-1).clamp(min=1e-10)
                    entropy = -(probs * probs.log2()).sum(dim=-1).mean().item()
                    if not math.isnan(entropy) and not math.isinf(entropy):
                        total_entropy += entropy * logits.size(1)
                        total_frames += logits.size(1)
            except:
                continue

        if total_frames > 0:
            avg = total_entropy / total_frames
            results.append({
                'method': 'wav2vec2_fp32',
                'audio_type': 'lj_speech_real',
                'entropy_bits_per_frame': round(avg, 4),
                'total_frames': total_frames
            })
            log(f"      LJ Speech wav2vec2: {avg:.4f} bits/frame ({total_frames} frames)")

        # Mel spectrogram on LJ Speech
        total_mel_entropy, total_mel_frames = 0.0, 0
        for wf in wav_files[:50]:
            try:
                rate, data = wavfile.read(os.path.join(lj_path, wf))
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                wt = torch.tensor(data).unsqueeze(0)
                mel = mel_transform(wt)
                mel_db = torchaudio.transforms.AmplitudeToDB()(mel)
                mel_shifted = mel_db - mel_db.min()
                mel_norm = mel_shifted / (mel_shifted.sum(dim=1, keepdim=True) + 1e-10)
                frame_entropy = -(mel_norm * (mel_norm + 1e-10).log2()).sum(dim=1)
                total_mel_entropy += frame_entropy.sum().item()
                total_mel_frames += mel.size(2)
            except:
                continue

        if total_mel_frames > 0:
            avg = total_mel_entropy / total_mel_frames
            results.append({
                'method': 'mel_spectrogram',
                'audio_type': 'lj_speech_real',
                'entropy_bits_per_frame': round(avg, 4),
                'total_frames': total_mel_frames
            })
            log(f"      LJ Speech mel: {avg:.4f} bits/frame ({total_mel_frames} frames)")

    gpu_mem_report()
    save_csv(results, f"{OUT}/results/audio_throughput_fixed.csv")

    del model
    torch.cuda.empty_cache()

    # Report
    report = "# Exp 4: Audio Throughput Across Complexity (Fixed)\n\n"
    report += "## wav2vec2 Output Entropy (FP32)\n\n"
    report += "| Audio Type | Entropy (bits/frame) | Frames |\n|---|---|---|\n"
    for r in results:
        if r['method'] == 'wav2vec2_fp32':
            report += f"| {r['audio_type']} | {r['entropy_bits_per_frame']} | {r['total_frames']} |\n"

    report += "\n## Mel-Spectrogram Entropy\n\n"
    report += "| Audio Type | Entropy (bits/frame) | Frames |\n|---|---|---|\n"
    for r in results:
        if r['method'] == 'mel_spectrogram':
            report += f"| {r['audio_type']} | {r['entropy_bits_per_frame']} | {r['total_frames']} |\n"

    report += "\n## Interpretation\n\n"
    report += "If entropy increases with audio complexity (silence < tone < chord < speech < noise), "
    report += "the audio throughput basin is data-driven — confirming Paper 8.\n"

    with open(f"{OUT}/results/EXP4_REPORT.md", 'w') as f:
        f.write(report)

    git_push("Decisive Exp 4: Audio throughput (FP32 fix)", ['weekend_experiments/decisive_round/exp4_maestro_audio/'])
    log("EXP 4 (FIXED) COMPLETE")


if __name__ == "__main__":
    main()
