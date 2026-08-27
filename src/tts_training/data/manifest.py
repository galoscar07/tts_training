"""Build a Coqui-ready training manifest from a speech corpus, using the
`expressive_tts.preprocess` frontend to convert each transcript into an IPA
phoneme string (the VITS symbol sequence).

Runs locally — **no Coqui dependency here** — so dataset prep works on an
Apple M1 while training happens elsewhere. Corpora and the output manifest
live **outside** the module (see `tts_training.paths`): pass `--corpus-root`
or set `$TTS_DATASETS_DIR`, and `--out` wherever you want the manifest.

Output is a pipe-separated file, `audio_file` relative to the corpus root:

    audio_file|phonemes|speaker|emotion

`emotion` is empty unless `--with-emotion` (the neutral base model doesn't
need it; it's for the deferred emotion fine-tune).

Usage:
    python -m tts_training.data.manifest --dataset mara --out out/mara.manifest
    python -m tts_training.data.manifest --dataset swara --corpus-root /data/SWARA --out out/swara.manifest
    python -m tts_training.data.manifest --dataset common_voice --corpus-root /data/cv-ro --out out/cv.manifest --limit 500
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from expressive_tts.preprocess.pipeline import PreprocessPipeline
from expressive_tts.preprocess.phonemizer import phonetics_only
from tts_training import paths
from tts_training.data.readers import (
    DatasetReader,
    catalina_reader,
    common_voice_reader,
    ljspeech_reader,
    swara_metadata_reader,
    swara_reader,
)
from tts_training.frontend.symbols import symbol_set

# Registry of dataset key -> reader. MARA/HRIA are on disk in this repo;
# SWARA/Common Voice are wired but expect an external --corpus-root.
DATASETS: dict[str, DatasetReader] = {
    "mara": ljspeech_reader("metadata.csv", "wavs", "mara"),
    "hria": ljspeech_reader(
        "catalina/metadata_simple.txt", "catalina/data", "catalina", suffixes=(".WAV", ".wav")
    ),
    "swara": swara_reader,
    "swara_train": swara_metadata_reader("SWARA_ALL_training.csv"),
    "swara_test": swara_metadata_reader("SWARA_ALL_testing.csv"),
    "catalina": catalina_reader,
    # emotion as speaker id (catalina_angry/happy/neutral/calm) -> lets a
    # multi-speaker VITS select emotion via --speaker. Needs its own fine-tune.
    "catalina_emotions": partial(catalina_reader, per_emotion_speaker=True),
    "common_voice": common_voice_reader(),
}


@dataclass
class BuildStats:
    written: int = 0
    skipped_missing_wav: int = 0
    skipped_empty_text: int = 0
    skipped_empty_phonemes: int = 0
    unexpected_symbols: set = field(default_factory=set)


def build_manifest(
    dataset: str,
    out_path: Path,
    *,
    corpus_root: Path | None = None,
    datasets_dir: Path | None = None,
    limit: int | None = None,
    with_emotion: bool = False,
    pipeline: PreprocessPipeline | None = None,
    progress_every: int = 100,
    phonetics_only_mode: bool = False,
) -> BuildStats:
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; known: {sorted(DATASETS)}")

    root = paths.corpus_root(
        dataset, datasets_dir_override=datasets_dir, corpus_root_override=corpus_root
    )
    if not root.exists():
        raise FileNotFoundError(f"corpus root does not exist: {root}")

    reader = DATASETS[dataset]
    pipeline = pipeline or PreprocessPipeline()
    if phonetics_only_mode and with_emotion:
        raise ValueError("--phonetics-only cannot be combined with --with-emotion")
    include = (
        {"normalized"}
        if phonetics_only_mode
        else ({"phonemes", "emotion"} if with_emotion else {"phonemes"})
    )
    valid_symbols = symbol_set()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = BuildStats()
    seen = 0
    started = time.monotonic()

    with out_path.open("w", encoding="utf-8") as out:
        for utt in reader(root):
            seen += 1
            if limit is not None and stats.written >= limit:
                break
            if not utt.text:
                stats.skipped_empty_text += 1
                continue
            if not (root / utt.rel_wav).exists():
                stats.skipped_missing_wav += 1
                continue

            result = pipeline.process(utt.text, include=include)
            phonemes = (
                phonetics_only(result.normalized_text or result.clean_text or utt.text)
                if phonetics_only_mode
                else (result.phoneme_text or "").strip()
            )
            if not phonemes:
                stats.skipped_empty_phonemes += 1
                continue

            stats.unexpected_symbols |= set(phonemes) - valid_symbols
            emotion = _first_emotion(result) if with_emotion else ""
            out.write(f"{utt.rel_wav}|{phonemes}|{utt.speaker}|{emotion}\n")
            stats.written += 1
            if progress_every > 0 and (
                stats.written == 1 or stats.written % progress_every == 0
            ):
                elapsed = time.monotonic() - started
                print(
                    f"manifest progress: {stats.written} written / {seen} scanned "
                    f"({elapsed:.1f}s)",
                    file=sys.stderr,
                    flush=True,
                )

    return stats


def _first_emotion(result) -> str:
    for sentence in result.sentences:
        if sentence.emotion is not None and sentence.emotion.label != "unspecified":
            return sentence.emotion.label
    return "neutral"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--corpus-root", type=Path, default=None,
        help="corpus directory (overrides $TTS_DATASETS_DIR/<dataset>)",
    )
    parser.add_argument(
        "--datasets-dir", type=Path, default=None,
        help="parent dir holding <dataset>/ (else $TTS_DATASETS_DIR, else ./datasets)",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap rows (M1 smoke test)")
    parser.add_argument(
        "--progress-every", type=int, default=100,
        help="report progress every N written rows (0 disables; default: 100)",
    )
    parser.add_argument(
        "--phonetics-only", action="store_true",
        help=(
            "skip Stanza and generate stressed Romanian IPA directly with eSpeak; "
            "intended for neutral base-model manifests"
        ),
    )
    parser.add_argument(
        "--with-emotion", action="store_true",
        help="add per-utterance emotion label (loads the transformer; for the deferred fine-tune)",
    )
    args = parser.parse_args(argv)

    stats = build_manifest(
        args.dataset,
        args.out,
        corpus_root=args.corpus_root,
        datasets_dir=args.datasets_dir,
        limit=args.limit,
        with_emotion=args.with_emotion,
        progress_every=args.progress_every,
        phonetics_only_mode=args.phonetics_only,
    )

    print(f"manifest: {args.out}")
    print(f"  written:                {stats.written}")
    print(f"  skipped (missing wav):  {stats.skipped_missing_wav}")
    print(f"  skipped (empty text):   {stats.skipped_empty_text}")
    print(f"  skipped (no phonemes):  {stats.skipped_empty_phonemes}")
    if stats.unexpected_symbols:
        print(f"  WARNING unexpected symbols (outside inventory): {sorted(stats.unexpected_symbols)}")
    else:
        print("  all phoneme symbols within the canonical inventory ✓")


if __name__ == "__main__":
    main()
