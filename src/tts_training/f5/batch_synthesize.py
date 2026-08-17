"""Batch-synthesize evaluation sets with a trained F5 model, for objective stats.

For each speaker in a dataset, auto-pick a reference clip from the corpus, then
synthesize a set of target sentences in that speaker's voice. Reuses our phoneme
frontend (F5 was fine-tuned on IPA) and the optional realism postprocess. Writes
a results CSV (`wav|speaker|ref_wav|text`) for downstream metrics (WER, speaker
similarity, etc.).

GPU strongly recommended (`--device cuda`) for volume — free the GPU first
(stop/finish training; inference can't share VRAM with training). CPU works but
is ~1-2 min/clip.

Sentences come from `--sentences FILE` (one per line) or, if omitted,
`--num-from-corpus N` distinct transcripts pulled from the dataset itself. Every
sentence is rendered for every speaker in the dataset.

Examples:
  # 100 MARA sentences in MARA's voice
  python -m tts_training.f5.batch_synthesize --ckpt /tmp/f5_latest.pt \
     --vocab data/ro_mara_swara_char/vocab.txt \
     --dataset mara --corpus-root datasets/MARA \
     --num-from-corpus 100 --out-dir out/f5_eval/mara --device cuda --postprocess

  # a shared sentence set across every SWARA voice
  python -m tts_training.f5.batch_synthesize --ckpt /tmp/f5_latest.pt \
     --vocab data/ro_mara_swara_char/vocab.txt \
     --dataset swara_test --corpus-root datasets/SWARA \
     --sentences sentences.txt --out-dir out/f5_eval/swara --device cuda --postprocess
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from expressive_tts.preprocess.pipeline import PreprocessPipeline
from tts_training import paths
from tts_training.data.manifest import DATASETS
from tts_training.f5.prepare import _wav_duration
from tts_training.postprocess import PostProcessConfig, postprocess


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "speaker"


def _pick_references(utterances, corpus_root: Path, max_ref_sec: float):
    """First usable clip per speaker: existing wav, non-empty text, short enough
    to be a good F5 reference. Returns {speaker: (rel_wav, ref_text)}."""
    refs: dict[str, tuple[str, str]] = {}
    for utt in utterances:
        if utt.speaker in refs or not utt.text:
            continue
        wav = corpus_root / utt.rel_wav
        if not wav.exists():
            continue
        dur = _wav_duration(wav)
        if dur is not None and dur > max_ref_sec:
            continue
        refs[utt.speaker] = (utt.rel_wav, utt.text)
    return refs


def _sentences(args, utterances) -> list[str]:
    if args.sentences:
        lines = Path(args.sentences).read_text(encoding="utf-8").splitlines()
        return [s.strip() for s in lines if s.strip()]
    seen: list[str] = []
    unique: set[str] = set()
    for utt in utterances:
        t = utt.text.strip()
        if t and t not in unique:
            unique.add(t)
            seen.append(t)
        if len(seen) >= args.num_from_corpus:
            break
    return seen


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--corpus-root", type=Path, default=None)
    parser.add_argument("--sentences", type=Path, default=None, help="target sentences, one per line")
    parser.add_argument("--num-from-corpus", type=int, default=100, help="if --sentences omitted, take N transcripts")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="F5TTS_v1_Base")
    parser.add_argument("--device", default="cuda", help="cuda (fast; free the GPU first) or cpu")
    parser.add_argument("--max-ref-sec", type=float, default=12.0, help="max reference-clip length")
    parser.add_argument("--postprocess", action="store_true")
    args = parser.parse_args(argv)

    root = paths.corpus_root(args.dataset, corpus_root_override=args.corpus_root)
    utterances = list(DATASETS[args.dataset](root))
    refs = _pick_references(utterances, root, args.max_ref_sec)
    if not refs:
        raise SystemExit("no usable reference clips found (check --corpus-root / --max-ref-sec)")
    sentences = _sentences(args, utterances)
    if not sentences:
        raise SystemExit("no target sentences (pass --sentences or --num-from-corpus)")

    print(f"{len(refs)} speaker(s) × {len(sentences)} sentence(s) = {len(refs) * len(sentences)} clips")

    pipeline = PreprocessPipeline()
    gen_phonemes = [
        (pipeline.process(s, include={"phonemes"}).phoneme_text or "").strip() for s in sentences
    ]

    # --- F5 (lazy, loaded once) ------------------------------------------
    from f5_tts.api import F5TTS

    tts = F5TTS(model=args.model, ckpt_file=args.ckpt, vocab_file=args.vocab, device=args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    total = 0
    for speaker, (ref_rel, ref_text) in refs.items():
        ref_file = str(root / ref_rel)
        ref_ph = (pipeline.process(ref_text, include={"phonemes"}).phoneme_text or "").strip()
        spk_dir = args.out_dir / _slug(speaker)
        spk_dir.mkdir(parents=True, exist_ok=True)
        for i, (sentence, gen_ph) in enumerate(zip(sentences, gen_phonemes)):
            if not gen_ph:
                continue
            wav, sr, _ = tts.infer(ref_file=ref_file, ref_text=ref_ph, gen_text=gen_ph)
            wav = np.asarray(wav, dtype=np.float32)
            if args.postprocess:
                wav = postprocess(wav, sr, PostProcessConfig())
            out_wav = spk_dir / f"{i:03d}.wav"
            sf.write(str(out_wav), wav, sr)
            manifest_rows.append((str(out_wav), speaker, ref_file, sentence))
            total += 1
            if total % 10 == 0:
                print(f"  {total} clips written ...", flush=True)

    csv_path = args.out_dir / "synthesized.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["wav", "speaker", "ref_wav", "text"])
        writer.writerows(manifest_rows)
    print(f"done — {total} clips, manifest: {csv_path}")


if __name__ == "__main__":
    main()
