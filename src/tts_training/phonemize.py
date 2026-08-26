"""Phonemize a text file (one sentence per line) to IPA phoneme strings, one
per line. Frontend only — imports no Coqui.

Run as its own process so the frontend's native libs (Stanza/torch, espeak)
never share a process with Coqui-TTS: loading both in one interpreter can
segfault (OpenMP/native-lib clash). The VITS synth tools call this in a
subprocess for exactly that reason.

Blank input lines produce blank output lines, so line alignment is preserved.

Usage:
    python -m tts_training.phonemize --in sentences.txt --out phonemes.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from expressive_tts.preprocess.pipeline import PreprocessPipeline


def phonemize_lines(lines: list[str], pipeline: PreprocessPipeline | None = None) -> list[str]:
    pipeline = pipeline or PreprocessPipeline()
    out: list[str] = []
    for line in lines:
        text = line.strip()
        if not text:
            out.append("")
            continue
        out.append((pipeline.process(text, include={"phonemes"}).phoneme_text or "").strip())
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    lines = args.inp.read_text(encoding="utf-8").splitlines()
    phonemes = phonemize_lines(lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(phonemes) + "\n", encoding="utf-8")
    print(f"phonemized {len(lines)} lines -> {args.out}")


if __name__ == "__main__":
    main()
