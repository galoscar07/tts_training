"""Synthesize speech from Romanian text with a trained VITS model, with an
optional audio post-processing pass.

Pipeline: text -> `expressive_tts.preprocess` (IPA phonemes with accent) ->
VITS (Coqui) -> raw wav -> optional `postprocess` (realism filters) -> file.

Because the model was trained with `use_phonemes=False` over our phoneme
symbol set, the frontend must phonemize the text first and we feed the
*phoneme string* to Coqui (not raw graphemes).

Coqui is imported lazily; `postprocess` is not (it's pure numpy/scipy), so
`--postprocess` can also be applied to any existing wav via
`postprocess_file()` without a model.

Usage:
    python -m tts_training.synthesize \
        --checkpoint out/vits_ro_base/<run>/best_model.pth \
        --config     out/vits_ro_base/<run>/config.json \
        --text "Bună ziua, acesta este un test." \
        --speaker mara \
        --out sample.wav \
        --postprocess
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from expressive_tts.preprocess.pipeline import PreprocessPipeline
from tts_training.postprocess import PostProcessConfig, postprocess


def text_to_phonemes(text: str, pipeline: PreprocessPipeline | None = None) -> str:
    """Run the frontend and return the accented IPA phoneme string the model
    was trained on."""
    pipeline = pipeline or PreprocessPipeline()
    result = pipeline.process(text, include={"phonemes"})
    return (result.phoneme_text or "").strip()


def postprocess_file(in_path: str | Path, out_path: str | Path, config: PostProcessConfig | None = None) -> None:
    """Apply the realism chain to an existing wav — no model/Coqui needed."""
    wav, sr = sf.read(str(in_path))
    if wav.ndim > 1:  # mix to mono
        wav = wav.mean(axis=1)
    sf.write(str(out_path), postprocess(wav.astype(np.float32), sr, config), sr)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="trained VITS .pth")
    parser.add_argument("--config", required=True, type=Path, help="that run's config.json")
    parser.add_argument("--text", required=True)
    parser.add_argument("--speaker", default=None, help="speaker name (multi-speaker model)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--postprocess", action="store_true",
        help="apply the realism filter chain after synthesis (postprocess.PostProcessConfig)",
    )
    parser.add_argument("--no-cuda", action="store_true", help="run on CPU")
    args = parser.parse_args(argv)

    phonemes = text_to_phonemes(args.text)
    if not phonemes:
        raise SystemExit("frontend produced no phonemes for the given text")

    # --- Coqui (lazy) -----------------------------------------------------
    from TTS.utils.synthesizer import Synthesizer

    synthesizer = Synthesizer(
        tts_checkpoint=str(args.checkpoint),
        tts_config_path=str(args.config),
        use_cuda=not args.no_cuda,
    )
    # We already phonemized, so pass the phoneme string as the "text".
    wav = synthesizer.tts(phonemes, speaker_name=args.speaker)
    wav = np.asarray(wav, dtype=np.float32)
    sr = synthesizer.output_sample_rate

    if args.postprocess:
        wav = postprocess(wav, sr, PostProcessConfig())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.out), wav, sr)
    print(f"wrote {args.out}  ({len(wav) / sr:.2f}s, {sr} Hz, postprocess={args.postprocess})")


if __name__ == "__main__":
    main()
