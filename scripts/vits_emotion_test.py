#!/usr/bin/env python
"""Single-file test for a trained emotion-VITS.

Generates one clip for every (checkpoint × postprocess × emotion):
  * checkpoints: best_model.pth AND the newest checkpoint_*.pth in the run dir
  * postprocess: raw and post-processed
  * emotions:    the catalina_<emotion> speaker ids present in the model

Output: <out_dir>/<best|last>_<emotion>_<raw|pp>.wav

Import order matters: Coqui-TTS must load before numpy/scipy/soundfile, else it
segfaults — so the Coqui import is first. The frontend (Stanza/espeak) is run in
a SUBPROCESS (`tts_training.phonemize`) so it never shares a process with Coqui.

Usage:
    python scripts/vits_emotion_test.py \
        --run-dir out/training_runs/vits_catalina_emo/<run> \
        --text "Astăzi este o zi importantă pentru noi." \
        --out-dir out/vits_emo_test
"""

from __future__ import annotations

# --- Coqui first (import-order matters) ------------------------------------
try:  # restore a helper newer transformers dropped that coqui-tts still imports
    import torch
    import transformers.pytorch_utils as _pu

    if not hasattr(_pu, "isin_mps_friendly"):
        _pu.isin_mps_friendly = lambda e, t: torch.isin(e, t)
except Exception:
    pass
from TTS.utils.synthesizer import Synthesizer  # noqa: E402

import argparse  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from tts_training.postprocess import PostProcessConfig, postprocess  # noqa: E402

EMOTIONS = ["angry", "happy", "neutral", "calm"]


def phonemize(text: str) -> str:
    """Run the frontend in a separate process and return the IPA string."""
    with tempfile.TemporaryDirectory() as d:
        i = Path(d) / "in.txt"
        o = Path(d) / "out.txt"
        i.write_text(text + "\n", encoding="utf-8")
        subprocess.run(
            [sys.executable, "-m", "tts_training.phonemize", "--in", str(i), "--out", str(o)],
            check=True,
        )
        lines = [ln for ln in o.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return lines[0] if lines else ""


def list_speakers(synth) -> list[str]:
    m = getattr(synth.tts_model, "speaker_manager", None)
    if m is None:
        return []
    names = getattr(m, "speaker_names", None)
    return list(names) if names else list(getattr(m, "name_to_id", {}) or {})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path, help="dir with config.json/speakers.pth/checkpoints")
    ap.add_argument("--text", default="Astăzi este o zi importantă pentru noi.")
    ap.add_argument("--out-dir", type=Path, default=Path("out/vits_emo_test"))
    ap.add_argument("--emotions", default=",".join(EMOTIONS))
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    ph = phonemize(args.text)
    if not ph:
        sys.exit("frontend produced no phonemes")

    checkpoints: dict[str, Path] = {}
    best = args.run_dir / "best_model.pth"
    if best.exists():
        checkpoints["best"] = best
    chs = sorted(args.run_dir.glob("checkpoint_*.pth"), key=lambda p: p.stat().st_mtime)
    if chs:
        checkpoints["last"] = chs[-1]
    if not checkpoints:
        sys.exit(f"no best_model.pth / checkpoint_*.pth in {args.run_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    emotions = [e.strip() for e in args.emotions.split(",") if e.strip()]
    pp_cfg = PostProcessConfig()

    for cname, cpath in checkpoints.items():
        print(f"== checkpoint '{cname}': {cpath.name} ==")
        syn = Synthesizer(
            tts_checkpoint=str(cpath),
            tts_config_path=str(args.run_dir / "config.json"),
            tts_speakers_file=str(args.run_dir / "speakers.pth"),
            use_cuda=not args.no_cuda,
        )
        speakers = list_speakers(syn)
        for emo in emotions:
            spk = f"catalina_{emo}"
            if spk not in speakers:
                print(f"  skip {spk} — not in model ({speakers})")
                continue
            wav = np.asarray(syn.tts(ph, speaker_name=spk), dtype=np.float32)
            sr = syn.output_sample_rate
            sf.write(str(args.out_dir / f"{cname}_{emo}_raw.wav"), wav, sr)
            sf.write(str(args.out_dir / f"{cname}_{emo}_pp.wav"), postprocess(wav, sr, pp_cfg), sr)
            print(f"  {cname}/{emo}: raw + pp")

    print(f"done -> {args.out_dir}")


if __name__ == "__main__":
    main()
