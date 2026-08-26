"""Batch-synthesize evaluation sets with a trained VITS model.

VITS is multi-speaker via speaker embeddings (no reference clip): you just pass
`speaker_name`. This synthesizes a set of target sentences for selected
speakers, writing `out_dir/<speaker>/<i>.wav` plus a `synthesized.csv`
(`wav|speaker|text`) for downstream stats. Reuses our phoneme frontend (VITS
trained with `use_phonemes=False` on IPA) and the optional realism postprocess.

Speaker selection: `--speakers a,b,c` (explicit), else all model speakers;
`--exclude mara` drops names; `--limit-speakers N` keeps the first N remaining.

Examples (run from the repo root):
  # 15 sentences for MARA
  python -m tts_training.vits.batch_synthesize \
      --checkpoint <run>/best_model.pth --config <run>/config.json --speakers-file <run>/speakers.pth \
      --sentences out/f5_eval/sentences.txt --num-sentences 15 \
      --speakers mara --out-dir out/vits_eval/mara --postprocess

  # 15 sentences for the first 10 SWARA speakers
  python -m tts_training.vits.batch_synthesize \
      --checkpoint <run>/best_model.pth --config <run>/config.json --speakers-file <run>/speakers.pth \
      --sentences out/f5_eval/sentences.txt --num-sentences 15 \
      --exclude mara --limit-speakers 10 --out-dir out/vits_eval/swara --postprocess
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from expressive_tts.preprocess.pipeline import PreprocessPipeline
from tts_training.postprocess import PostProcessConfig, postprocess
from tts_training.synthesize import list_model_speakers


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "speaker"


def _select_speakers(all_speakers, explicit, exclude, limit):
    if explicit:
        wanted = [s.strip() for s in explicit.split(",") if s.strip()]
        missing = [s for s in wanted if s not in all_speakers]
        if missing:
            raise SystemExit(f"speakers not in model: {missing}\nknown: {all_speakers}")
        speakers = wanted
    else:
        speakers = list(all_speakers)
    if exclude:
        drop = {s.strip() for s in exclude.split(",") if s.strip()}
        speakers = [s for s in speakers if s not in drop]
    if limit is not None:
        speakers = speakers[:limit]
    return speakers


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--speakers-file", type=Path, default=None, help="speakers.pth (else from config)")
    parser.add_argument("--sentences", required=True, type=Path, help="target sentences, one per line")
    parser.add_argument("--num-sentences", type=int, default=None, help="cap how many sentences to use")
    parser.add_argument("--speakers", default=None, help="explicit comma list; default all model speakers")
    parser.add_argument("--exclude", default=None, help="comma list of speakers to drop (e.g. mara)")
    parser.add_argument("--limit-speakers", type=int, default=None, help="keep first N remaining speakers")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--no-cuda", action="store_true", help="run on CPU (VITS is fast on CPU)")
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--list-speakers", action="store_true", help="print speakers and exit")
    args = parser.parse_args(argv)

    # --- Coqui (lazy) -----------------------------------------------------
    from TTS.utils.synthesizer import Synthesizer

    synthesizer = Synthesizer(
        tts_checkpoint=str(args.checkpoint),
        tts_config_path=str(args.config),
        tts_speakers_file=str(args.speakers_file) if args.speakers_file else None,
        use_cuda=not args.no_cuda,
    )
    all_speakers = list_model_speakers(synthesizer)
    if args.list_speakers:
        print(f"{len(all_speakers)} speakers:")
        for s in all_speakers:
            print(f"  {s}")
        return

    speakers = _select_speakers(all_speakers, args.speakers, args.exclude, args.limit_speakers)
    if not speakers:
        raise SystemExit("no speakers selected")

    sentences = [s.strip() for s in args.sentences.read_text(encoding="utf-8").splitlines() if s.strip()]
    if args.num_sentences is not None:
        sentences = sentences[: args.num_sentences]
    if not sentences:
        raise SystemExit("no sentences to synthesize")

    print(f"{len(speakers)} speaker(s) × {len(sentences)} sentence(s) = {len(speakers) * len(sentences)} clips")

    pipeline = PreprocessPipeline()
    gen_phonemes = [
        (pipeline.process(s, include={"phonemes"}).phoneme_text or "").strip() for s in sentences
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    total = 0
    for speaker in speakers:
        spk_dir = args.out_dir / _slug(speaker)
        spk_dir.mkdir(parents=True, exist_ok=True)
        for i, (sentence, phon) in enumerate(zip(sentences, gen_phonemes)):
            if not phon:
                continue
            wav = np.asarray(synthesizer.tts(phon, speaker_name=speaker), dtype=np.float32)
            sr = synthesizer.output_sample_rate
            if args.postprocess:
                wav = postprocess(wav, sr, PostProcessConfig())
            out_wav = spk_dir / f"{i:03d}.wav"
            sf.write(str(out_wav), wav, sr)
            rows.append((str(out_wav), speaker, sentence))
            total += 1
        print(f"  {speaker}: {len(sentences)} clips", flush=True)

    csv_path = args.out_dir / "synthesized.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["wav", "speaker", "text"])
        writer.writerows(rows)
    print(f"done — {total} clips, manifest: {csv_path}")


if __name__ == "__main__":
    main()
