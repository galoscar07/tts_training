"""Objective TTS evaluation: resynthesize a dataset with F5 and compare each
output to its ground-truth recording — MCD, PESQ, STOI, and F0 error — then
write a per-utterance CSV and diagrams.

Resynthesis: for each manifest row (`rel_wav|phonemes|speaker|emotion`), F5 uses
the utterance's own audio as the reference and its phoneme string as both the
reference and target text, producing a synthetic copy comparable to the
original. The manifest already carries the IPA phonemes the model reads, so no
text frontend is needed here.

Metrics (per utterance):
  * MCD  — mel-cepstral distortion, DTW-aligned (pymcd). Lower is better.
  * PESQ — wideband perceptual quality (pesq). Higher is better.
  * STOI — short-time objective intelligibility (pystoi). Higher is better.
  * F0   — RMSE (Hz) and Pearson correlation of pitch (librosa.pyin).

PESQ/STOI/F0 assume time-aligned signals; TTS outputs differ in duration, so we
truncate to the shorter length — treat those three as approximate for TTS (MCD's
DTW is the rigorous one). Each metric is guarded; failures record NaN.

Deps: `pip install -e ".[eval]"` (pymcd, pesq, pystoi; librosa/matplotlib via
training). Run on a free GPU (or --device cpu).

Usage:
    python -m objective_evaluation.tts_objective \
        --ckpt <f5.pt> --vocab data/ro_catalina_char/vocab.txt \
        --manifest out/catalina.manifest --corpus-root datasets/CATALINA \
        --out-dir out/eval_catalina --device cuda
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf


# --- metric helpers --------------------------------------------------------


def _load_mono(path: str, target_sr: int | None = None):
    import librosa

    wav, sr = librosa.load(path, sr=target_sr, mono=True)
    return wav.astype(np.float32), sr


def _mcd(ref_path: str, syn_path: str) -> float:
    try:
        from pymcd.mcd import Calculate_MCD

        return float(Calculate_MCD(mode="dtw").calculate_mcd(ref_path, syn_path))
    except Exception:
        return float("nan")


def _pesq(ref_path: str, syn_path: str) -> float:
    try:
        from pesq import pesq

        ref, _ = _load_mono(ref_path, 16000)
        deg, _ = _load_mono(syn_path, 16000)
        n = min(len(ref), len(deg))
        return float(pesq(16000, ref[:n], deg[:n], "wb"))
    except Exception:
        return float("nan")


def _stoi(ref_path: str, syn_path: str) -> float:
    try:
        from pystoi import stoi

        ref, sr = _load_mono(ref_path, 16000)
        deg, _ = _load_mono(syn_path, 16000)
        n = min(len(ref), len(deg))
        return float(stoi(ref[:n], deg[:n], sr, extended=False))
    except Exception:
        return float("nan")


def _f0(ref_path: str, syn_path: str) -> tuple[float, float]:
    try:
        import librosa

        ref, sr = _load_mono(ref_path, 22050)
        deg, _ = _load_mono(syn_path, 22050)
        kw = dict(fmin=65.0, fmax=400.0, sr=sr)
        f0r, _, _ = librosa.pyin(ref, **kw)
        f0d, _, _ = librosa.pyin(deg, **kw)
        n = min(len(f0r), len(f0d))
        f0r, f0d = f0r[:n], f0d[:n]
        mask = ~np.isnan(f0r) & ~np.isnan(f0d)
        if mask.sum() < 2:
            return float("nan"), float("nan")
        a, b = f0r[mask], f0d[mask]
        rmse = float(np.sqrt(np.mean((a - b) ** 2)))
        corr = float(np.corrcoef(a, b)[0, 1])
        return rmse, corr
    except Exception:
        return float("nan"), float("nan")


def _emotion_of(rel_wav: str, meta_emotion: str) -> str:
    m = re.search(r"catalina_([a-z]+)_\d+", Path(rel_wav).name)
    if m:
        return m.group(1)
    return (meta_emotion or "unknown").strip().lower() or "unknown"


# --- main ------------------------------------------------------------------


def _read_manifest(path: Path):
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = raw.split("|")
        if len(parts) >= 3:
            emotion = parts[3] if len(parts) > 3 else ""
            yield parts[0], parts[1], emotion  # rel_wav, phonemes, emotion


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model", default="F5TTS_v1_Base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="cap utterances (smoke test)")
    args = parser.parse_args(argv)

    rows = list(_read_manifest(args.manifest))
    if args.limit:
        rows = rows[: args.limit]

    synth_dir = args.out_dir / "synth"
    synth_dir.mkdir(parents=True, exist_ok=True)

    from f5_tts.api import F5TTS

    tts = F5TTS(model=args.model, ckpt_file=args.ckpt, vocab_file=args.vocab, device=args.device)

    results = []
    for i, (rel_wav, phonemes, meta_emotion) in enumerate(rows):
        ref_path = str(args.corpus_root / rel_wav)
        if not Path(ref_path).exists() or not phonemes.strip():
            continue
        wav, sr, _ = tts.infer(ref_file=ref_path, ref_text=phonemes, gen_text=phonemes)
        syn_path = str(synth_dir / f"{i:05d}.wav")
        sf.write(syn_path, np.asarray(wav, dtype=np.float32), sr)

        mcd = _mcd(ref_path, syn_path)
        pesq_v = _pesq(ref_path, syn_path)
        stoi_v = _stoi(ref_path, syn_path)
        f0_rmse, f0_corr = _f0(ref_path, syn_path)
        emotion = _emotion_of(rel_wav, meta_emotion)
        results.append(dict(
            rel_wav=rel_wav, emotion=emotion, mcd=mcd, pesq=pesq_v,
            stoi=stoi_v, f0_rmse=f0_rmse, f0_corr=f0_corr,
        ))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)} evaluated", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "metrics.csv"
    fields = ["rel_wav", "emotion", "mcd", "pesq", "stoi", "f0_rmse", "f0_corr"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    _summary_and_plots(results, args.out_dir)
    print(f"done — {len(results)} utterances, metrics: {csv_path}, diagrams in {args.out_dir}")


def _summary_and_plots(results: list[dict], out_dir: Path) -> None:
    metrics = ["mcd", "pesq", "stoi", "f0_rmse", "f0_corr"]

    def vals(rs, key):
        return [r[key] for r in rs if r[key] == r[key]]  # drop NaN

    # text summary (overall + per emotion)
    lines = ["metric,scope,n,mean,std"]
    for key in metrics:
        v = vals(results, key)
        if v:
            lines.append(f"{key},overall,{len(v)},{statistics.mean(v):.4f},{statistics.pstdev(v):.4f}")
    emotions = sorted({r["emotion"] for r in results})
    for emo in emotions:
        rs = [r for r in results if r["emotion"] == emo]
        for key in metrics:
            v = vals(rs, key)
            if v:
                lines.append(f"{key},{emo},{len(v)},{statistics.mean(v):.4f},{statistics.pstdev(v):.4f}")
    (out_dir / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable — CSV written, skipping diagrams")
        return

    # one figure of overall distributions
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    for ax, key in zip(axes, metrics):
        v = vals(results, key)
        ax.boxplot(v, vert=True, showmeans=True)
        ax.set_title(f"{key}\n(n={len(v)})")
        ax.set_xticks([])
    fig.suptitle("Objective metrics — overall")
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_overall.png", dpi=120)
    plt.close(fig)

    # per-emotion boxplots, one panel per metric
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4.5))
    for ax, key in zip(axes, metrics):
        data = [vals([r for r in results if r["emotion"] == e], key) for e in emotions]
        ax.boxplot([d if d else [float("nan")] for d in data], labels=emotions, showmeans=True)
        ax.set_title(key)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Objective metrics — by emotion")
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_by_emotion.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
