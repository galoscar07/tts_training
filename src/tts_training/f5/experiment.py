"""Batch F5 generation for experiments: every (condition × sentence) pair.

A **condition** is a reference clip that fixes F5's voice *and* emotional style
(F5 is zero-shot — both come from the reference). Conditions are given as a TSV,
one per line:

    <label>\t<ref_wav>\t<ref_text>

Sentences are one target sentence per line. Output:
`<out_dir>/<label>/<NNN>.wav` for each sentence, plus a `synthesized.csv`
(`wav|condition|ref_wav|text`) for downstream metrics.

Reuses our phoneme frontend (F5 was fine-tuned on IPA) and the optional realism
postprocess. Loads the F5 model once and reuses it across all clips.

Example:
    python -m tts_training.f5.experiment \
        --ckpt <f5.pt> --vocab data/ro_catalina_char/vocab.txt \
        --conditions out/exp/emotions.tsv --sentences out/exp/mara20.txt \
        --out-dir out/exp/emotions_mara --device cuda --postprocess
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from expressive_tts.preprocess.pipeline import PreprocessPipeline
from tts_training.postprocess import PostProcessConfig, postprocess


def read_conditions(path: Path) -> list[tuple[str, str, str]]:
    conds: list[tuple[str, str, str]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = raw.split("\t")
        if len(parts) >= 3:
            conds.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return conds


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--conditions", required=True, type=Path, help="TSV: label<TAB>ref_wav<TAB>ref_text")
    parser.add_argument("--sentences", required=True, type=Path, help="one target sentence per line")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model", default="F5TTS_v1_Base")
    parser.add_argument("--device", default="cuda", help="cuda (free the GPU) or cpu")
    parser.add_argument("--postprocess", action="store_true")
    args = parser.parse_args(argv)

    sentences = [s.strip() for s in args.sentences.read_text(encoding="utf-8").splitlines() if s.strip()]
    conditions = read_conditions(args.conditions)
    if not sentences or not conditions:
        raise SystemExit("need at least one sentence and one condition")

    pipeline = PreprocessPipeline()
    gen_ph = [(pipeline.process(s, include={"phonemes"}).phoneme_text or "").strip() for s in sentences]

    print(f"{len(conditions)} condition(s) × {len(sentences)} sentence(s) = {len(conditions) * len(sentences)} clips")

    from f5_tts.api import F5TTS

    tts = F5TTS(model=args.model, ckpt_file=args.ckpt, vocab_file=args.vocab, device=args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, ref_wav, ref_text in conditions:
        ref_ph = (pipeline.process(ref_text, include={"phonemes"}).phoneme_text or "").strip()
        d = args.out_dir / label
        d.mkdir(parents=True, exist_ok=True)
        for i, (sentence, gp) in enumerate(zip(sentences, gen_ph)):
            if not gp:
                continue
            wav, sr, _ = tts.infer(ref_file=ref_wav, ref_text=ref_ph, gen_text=gp)
            wav = np.asarray(wav, dtype=np.float32)
            if args.postprocess:
                wav = postprocess(wav, sr, PostProcessConfig())
            wp = d / f"{i:03d}.wav"
            sf.write(str(wp), wav, sr)
            rows.append((str(wp), label, ref_wav, sentence))
        print(f"  [{label}] {len(sentences)} clips", flush=True)

    csv_path = args.out_dir / "synthesized.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|")
        w.writerow(["wav", "condition", "ref_wav", "text"])
        w.writerows(rows)
    print(f"done — {len(rows)} clips, manifest: {csv_path}")


if __name__ == "__main__":
    main()
