"""Test a (possibly still-training) F5 model.

F5 is zero-shot / reference-conditioned: give it a short reference clip + that
clip's transcript, and it clones the voice. Because we fine-tuned on IPA
phonemes, this wrapper phonemizes BOTH the reference transcript and the target
text with our frontend before F5 sees them (F5's tokenizer then maps those IPA
characters through our custom vocab).

Meant to run **on CPU** (`--device cpu`, the default) so it doesn't fight the
training job for the 11 GB of VRAM. Point `--ckpt` at a COPY of the latest
checkpoint (copy it first — the live file is being rewritten).

Usage:
    # copy the current checkpoint so we don't read a half-written file
    cp out/f5_ckpts/F5TTS_v1_Base/model_last.pt /tmp/f5_test.pt

    python -m tts_training.f5.synthesize \
        --ckpt /tmp/f5_test.pt \
        --vocab data/ro_mara_swara_char/vocab.txt \
        --ref-audio datasets/MARA/wavs/mara_chp01_0002.wav \
        --ref-text "A rămas Mara, săraca, văduvă cu doi copii," \
        --text "Bună ziua, acesta este un test." \
        --out out/f5_test/sample.wav --postprocess
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from expressive_tts.preprocess.pipeline import PreprocessPipeline
from tts_training.postprocess import PostProcessConfig, postprocess


def _phonemes(text: str, pipeline: PreprocessPipeline) -> str:
    return (pipeline.process(text, include={"phonemes"}).phoneme_text or "").strip()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="checkpoint (.pt/.safetensors) — copy the live one first")
    parser.add_argument("--vocab", required=True, help="data/<dataset>_char/vocab.txt")
    parser.add_argument("--ref-audio", required=True, help="short reference clip (a training wav is fine)")
    parser.add_argument("--ref-text", required=True, help="Romanian transcript of the reference clip")
    parser.add_argument("--text", required=True, help="Romanian text to synthesize")
    parser.add_argument("--out", type=Path, default=Path("out/f5_test/sample.wav"))
    parser.add_argument("--model", default="F5TTS_v1_Base")
    parser.add_argument("--device", default="cpu", help="cpu (safe while training) or cuda")
    parser.add_argument("--postprocess", action="store_true", help="apply the realism filter chain")
    args = parser.parse_args(argv)

    pipeline = PreprocessPipeline()
    ref_ph = _phonemes(args.ref_text, pipeline)
    gen_ph = _phonemes(args.text, pipeline)
    if not gen_ph:
        raise SystemExit("frontend produced no phonemes for --text")

    # --- F5 (lazy) --------------------------------------------------------
    from f5_tts.api import F5TTS

    tts = F5TTS(
        model=args.model,
        ckpt_file=args.ckpt,
        vocab_file=args.vocab,
        device=args.device,
    )
    wav, sr, _ = tts.infer(ref_file=args.ref_audio, ref_text=ref_ph, gen_text=gen_ph)

    wav = np.asarray(wav, dtype=np.float32)
    if args.postprocess:
        wav = postprocess(wav, sr, PostProcessConfig())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.out), wav, sr)
    print(f"wrote {args.out}  ({len(wav) / sr:.2f}s, {sr} Hz, device={args.device})")


if __name__ == "__main__":
    main()
