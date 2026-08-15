"""Convert our VITS manifests into an F5-TTS dataset directory.

Input: one or more manifests (`audio|phonemes|speaker|emotion`) plus each
one's corpus root (the dir the `audio` paths are relative to).

Output: an F5 dataset dir with
    <out>/wavs/<unique>.wav        (symlinks to the real audio — no copying)
    <out>/metadata.csv             (`wavs/<unique>.wav|<phoneme_text>`)

The `text` column is our accented IPA phoneme string, unchanged — so F5's
prepare step will build a `vocab.txt` of IPA symbols (use the **char**
tokenizer, not pinyin, so nothing gets transliterated). Speaker/emotion
columns are dropped: F5 is zero-shot and has no speaker ids.

Wavs are symlinked (absolute targets), so MARA (2.6 GB) and SWARA aren't
duplicated. Filenames are prefixed per-manifest to avoid collisions when
several corpora share a stem.

Usage:
    python -m tts_training.f5.prepare \
        --manifest out/mara.manifest        --corpus-root datasets/MARA \
        --manifest out/swara_train.manifest --corpus-root dataset/SWARA \
        --out out/f5_dataset
"""

from __future__ import annotations

import argparse
import contextlib
import os
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PrepareStats:
    written: int = 0
    skipped_missing_wav: int = 0
    skipped_empty: int = 0
    skipped_too_long: int = 0


def _iter_manifest(manifest: Path):
    with manifest.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            yield parts[0], parts[1]  # rel_wav, phonemes


def _wav_duration(path: Path) -> float | None:
    """Seconds, read from the PCM WAV header (stdlib — no soundfile). Returns
    None if the file can't be parsed (then we keep it rather than drop it)."""
    try:
        with contextlib.closing(wave.open(str(path), "rb")) as w:
            rate = w.getframerate()
            return w.getnframes() / float(rate) if rate else None
    except Exception:
        return None


def to_f5_dataset(
    manifests: list[tuple[Path, Path]],
    out_dir: Path,
    *,
    copy: bool = False,
    abs_paths: bool = False,
    max_duration: float | None = None,
) -> PrepareStats:
    """Build the F5 dataset dir from (manifest, corpus_root) pairs.

    `copy=True` copies wavs instead of symlinking (use if the training reads
    from a filesystem that can't follow the symlinks). `abs_paths=True` writes
    absolute `audio_file` paths in metadata.csv — required by f5-tts versions
    whose `prepare_csv_wavs` rejects relative paths. `max_duration` (seconds)
    drops utterances longer than the cap — long clips blow up attention memory
    (O(seq_len²)) and cause single-sample CUDA OOMs on small GPUs."""
    out_dir = Path(out_dir).resolve() if abs_paths else Path(out_dir)
    wavs_dir = out_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    stats = PrepareStats()

    with (out_dir / "metadata.csv").open("w", encoding="utf-8") as meta:
        meta.write("audio_file|text\n")  # F5's read_audio_text_pairs requires this header
        for manifest, corpus_root in manifests:
            prefix = Path(manifest).stem  # e.g. "mara", "swara_train"
            for rel_wav, phonemes in _iter_manifest(Path(manifest)):
                phonemes = phonemes.strip()
                src = (Path(corpus_root) / rel_wav).resolve()
                if not phonemes:
                    stats.skipped_empty += 1
                    continue
                if not src.exists():
                    stats.skipped_missing_wav += 1
                    continue
                if max_duration is not None:
                    dur = _wav_duration(src)
                    if dur is not None and dur > max_duration:
                        stats.skipped_too_long += 1
                        continue

                dst_name = f"{prefix}__{Path(rel_wav).name}"
                dst = wavs_dir / dst_name
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                if copy:
                    import shutil

                    shutil.copy2(src, dst)
                else:
                    os.symlink(src, dst)

                audio_field = str(dst) if abs_paths else f"wavs/{dst_name}"
                meta.write(f"{audio_field}|{phonemes}\n")
                stats.written += 1

    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--corpus-root", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--copy", action="store_true", help="copy wavs instead of symlinking")
    parser.add_argument(
        "--abs-paths", action="store_true",
        help="write absolute audio paths (required by some f5-tts prepare_csv_wavs versions)",
    )
    parser.add_argument(
        "--max-duration", type=float, default=None,
        help="drop utterances longer than N seconds (avoids single-sample CUDA OOM)",
    )
    args = parser.parse_args(argv)

    if len(args.manifest) != len(args.corpus_root):
        parser.error("pass one --corpus-root per --manifest")

    stats = to_f5_dataset(
        list(zip(args.manifest, args.corpus_root)), args.out,
        copy=args.copy, abs_paths=args.abs_paths, max_duration=args.max_duration,
    )
    print(f"F5 dataset: {args.out}")
    print(f"  written:               {stats.written}")
    print(f"  skipped (missing wav): {stats.skipped_missing_wav}")
    print(f"  skipped (empty text):  {stats.skipped_empty}")
    print(f"  skipped (too long):    {stats.skipped_too_long}")
    print(f"\nNext: build arrow/vocab with F5, then finetune (use the CHAR tokenizer):")
    print(f"  python -m f5_tts.train.datasets.prepare_csv_wavs {args.out} {args.out}_prepared --tokenizer char")


if __name__ == "__main__":
    main()
