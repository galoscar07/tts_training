#!/usr/bin/env python
"""Single-file objective evaluation for a trained VITS.

Resynthesizes each manifest utterance with its speaker id and compares the
output to the ground-truth recording: MCD (DTW), PESQ, STOI, F0 RMSE + corr.
Writes metrics.csv, summary.csv (overall + per emotion), and boxplot diagrams.

The manifest already carries the IPA phonemes VITS reads, so no text frontend is
needed here. Rows: `rel_wav|phonemes|speaker|emotion` (from
`tts_training.data.manifest`; for the emotion model use catalina_emotions.manifest).

Import order matters: Coqui-TTS loads before numpy/scipy/soundfile.
Metric libs: `pip install -e ".[eval]"` (pymcd, pesq, pystoi; librosa/matplotlib
come with training). PESQ/STOI/F0 assume aligned audio — TTS differs in length,
so they're truncated to the shorter signal (approximate); MCD's DTW is rigorous.

Usage:
    python scripts/vits_objective_eval.py \
        --run-dir out/training_runs/vits_catalina_emo/<run> \
        --manifest out/catalina_emotions.manifest --corpus-root datasets/CATALINA \
        --out-dir out/eval_vits_catalina [--checkpoint <ckpt.pth>] [--limit 20]
"""

from __future__ import annotations

# --- Coqui first (import-order matters) ------------------------------------
try:
    import torch
    import transformers.pytorch_utils as _pu

    if not hasattr(_pu, "isin_mps_friendly"):
        _pu.isin_mps_friendly = lambda e, t: torch.isin(e, t)
except Exception:
    pass
from TTS.utils.synthesizer import Synthesizer  # noqa: E402

import argparse  # noqa: E402
import csv  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402


def _load(path: str, sr=None):
    import librosa

    w, s = librosa.load(path, sr=sr, mono=True)
    return w.astype(np.float32), s


def _mcd(a: str, b: str) -> float:
    try:
        from pymcd.mcd import Calculate_MCD

        return float(Calculate_MCD(mode="dtw").calculate_mcd(a, b))
    except Exception:
        return float("nan")


def _pesq(a: str, b: str) -> float:
    try:
        from pesq import pesq

        r, _ = _load(a, 16000)
        d, _ = _load(b, 16000)
        n = min(len(r), len(d))
        return float(pesq(16000, r[:n], d[:n], "wb"))
    except Exception:
        return float("nan")


def _stoi(a: str, b: str) -> float:
    try:
        from pystoi import stoi

        r, s = _load(a, 16000)
        d, _ = _load(b, 16000)
        n = min(len(r), len(d))
        return float(stoi(r[:n], d[:n], s, extended=False))
    except Exception:
        return float("nan")


def _f0(a: str, b: str):
    try:
        import librosa

        r, s = _load(a, 22050)
        d, _ = _load(b, 22050)
        kw = dict(fmin=65.0, fmax=400.0, sr=s)
        f0r, _, _ = librosa.pyin(r, **kw)
        f0d, _, _ = librosa.pyin(d, **kw)
        n = min(len(f0r), len(f0d))
        f0r, f0d = f0r[:n], f0d[:n]
        m = ~np.isnan(f0r) & ~np.isnan(f0d)
        if m.sum() < 2:
            return float("nan"), float("nan")
        x, y = f0r[m], f0d[m]
        return float(np.sqrt(np.mean((x - y) ** 2))), float(np.corrcoef(x, y)[0, 1])
    except Exception:
        return float("nan"), float("nan")


def _emotion(rel_wav: str, speaker: str) -> str:
    mm = re.search(r"catalina_([a-z]+)_\d+", Path(rel_wav).name)
    if mm:
        return mm.group(1)
    if speaker.startswith("catalina_"):
        return speaker.split("_", 1)[1]
    return speaker or "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path, help="dir with config.json/speakers.pth/checkpoints")
    ap.add_argument("--checkpoint", type=Path, default=None, help="default: newest checkpoint_*.pth in run-dir")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--corpus-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ckpt = args.checkpoint
    if ckpt is None:
        chs = sorted(args.run_dir.glob("checkpoint_*.pth"), key=lambda p: p.stat().st_mtime)
        if not chs:
            raise SystemExit(f"no checkpoint_*.pth in {args.run_dir}")
        ckpt = chs[-1]
    print(f"checkpoint: {ckpt}")

    syn = Synthesizer(
        tts_checkpoint=str(ckpt),
        tts_config_path=str(args.run_dir / "config.json"),
        tts_speakers_file=str(args.run_dir / "speakers.pth"),
        use_cuda=not args.no_cuda,
    )

    rows = [ln for ln in args.manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.limit:
        rows = rows[: args.limit]

    synth_dir = args.out_dir / "synth"
    synth_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, line in enumerate(rows):
        parts = line.split("|")
        if len(parts) < 3:
            continue
        rel_wav, phonemes, speaker = parts[0], parts[1], parts[2]
        ref = str(args.corpus_root / rel_wav)
        if not Path(ref).exists() or not phonemes.strip():
            continue
        wav = np.asarray(syn.tts(phonemes, speaker_name=speaker), dtype=np.float32)
        syn_path = str(synth_dir / f"{i:05d}.wav")
        sf.write(syn_path, wav, syn.output_sample_rate)
        f0_rmse, f0_corr = _f0(ref, syn_path)
        results.append(dict(
            rel_wav=rel_wav, emotion=_emotion(rel_wav, speaker),
            mcd=_mcd(ref, syn_path), pesq=_pesq(ref, syn_path), stoi=_stoi(ref, syn_path),
            f0_rmse=f0_rmse, f0_corr=f0_corr,
        ))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["rel_wav", "emotion", "mcd", "pesq", "stoi", "f0_rmse", "f0_corr"]
    with (args.out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    _summary_and_plots(results, args.out_dir)
    print(f"done — {len(results)} utterances -> {args.out_dir}")


def _summary_and_plots(results, out_dir) -> None:
    metrics = ["mcd", "pesq", "stoi", "f0_rmse", "f0_corr"]

    def vals(rs, k):
        return [r[k] for r in rs if r[k] == r[k]]

    lines = ["metric,scope,n,mean,std"]
    for k in metrics:
        v = vals(results, k)
        if v:
            lines.append(f"{k},overall,{len(v)},{statistics.mean(v):.4f},{statistics.pstdev(v):.4f}")
    emotions = sorted({r["emotion"] for r in results})
    for e in emotions:
        rs = [r for r in results if r["emotion"] == e]
        for k in metrics:
            v = vals(rs, k)
            if v:
                lines.append(f"{k},{e},{len(v)},{statistics.mean(v):.4f},{statistics.pstdev(v):.4f}")
    (out_dir / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable — CSV only")
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    for ax, k in zip(axes, metrics):
        v = vals(results, k)
        ax.boxplot(v, showmeans=True)
        ax.set_title(f"{k}\n(n={len(v)})")
        ax.set_xticks([])
    fig.suptitle("VITS objective metrics — overall")
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_overall.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4.5))
    for ax, k in zip(axes, metrics):
        data = [vals([r for r in results if r["emotion"] == e], k) for e in emotions]
        ax.boxplot([d if d else [float("nan")] for d in data], labels=emotions, showmeans=True)
        ax.set_title(k)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("VITS objective metrics — by emotion")
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_by_emotion.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
