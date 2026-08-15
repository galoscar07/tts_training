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

Single speaker → one file:
    python -m tts_training.synthesize --checkpoint ... --config ... \
        --text "Bună ziua." --speaker mara --out sample.wav --postprocess

All speakers → one file each into a folder:
    python -m tts_training.synthesize --checkpoint ... --config ... \
        --text "Bună ziua." --all-speakers --out-dir out/final --postprocess

List the model's speakers and exit:
    python -m tts_training.synthesize --checkpoint ... --config ... --list-speakers
"""

from __future__ import annotations

import argparse
import re
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


def list_model_speakers(synthesizer) -> list[str]:
    """Every speaker name the trained multi-speaker model knows."""
    manager = getattr(synthesizer.tts_model, "speaker_manager", None)
    if manager is None:
        return []
    names = getattr(manager, "speaker_names", None)
    if names:
        return list(names)
    mapping = getattr(manager, "name_to_id", None) or getattr(manager, "ids", None)
    return list(mapping) if mapping else []


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "speaker"


def _synth_one(synthesizer, phonemes: str, speaker: str | None, do_postprocess: bool):
    wav = np.asarray(synthesizer.tts(phonemes, speaker_name=speaker), dtype=np.float32)
    sr = synthesizer.output_sample_rate
    if do_postprocess:
        wav = postprocess(wav, sr, PostProcessConfig())
    return wav, sr


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="trained VITS .pth")
    parser.add_argument("--config", required=True, type=Path, help="that run's config.json")
    parser.add_argument("--speakers", type=Path, default=None, help="speakers.pth/json (else from config)")
    parser.add_argument("--text", default="Bună ziua, acesta este un test.")
    parser.add_argument("--speaker", default=None, help="one speaker name")
    parser.add_argument("--all-speakers", action="store_true", help="synthesize the text for every speaker")
    parser.add_argument("--out", type=Path, default=None, help="output file (single speaker)")
    parser.add_argument("--out-dir", type=Path, default=None, help="output folder (all speakers)")
    parser.add_argument("--list-speakers", action="store_true", help="print the model's speakers and exit")
    parser.add_argument(
        "--postprocess", action="store_true",
        help="apply the realism filter chain after synthesis (postprocess.PostProcessConfig)",
    )
    parser.add_argument("--no-cuda", action="store_true", help="run on CPU")
    args = parser.parse_args(argv)

    # --- Coqui (lazy) -----------------------------------------------------
    from TTS.utils.synthesizer import Synthesizer

    synthesizer = Synthesizer(
        tts_checkpoint=str(args.checkpoint),
        tts_config_path=str(args.config),
        tts_speakers_file=str(args.speakers) if args.speakers else None,
        use_cuda=not args.no_cuda,
    )

    speakers = list_model_speakers(synthesizer)
    if args.list_speakers:
        print(f"{len(speakers)} speakers:")
        for name in speakers:
            print(f"  {name}")
        return

    phonemes = text_to_phonemes(args.text)
    if not phonemes:
        raise SystemExit("frontend produced no phonemes for the given text")

    if args.all_speakers:
        if not speakers:
            raise SystemExit("model reports no speakers — is this a multi-speaker checkpoint?")
        out_dir = args.out_dir or Path("out/final")
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, speaker in enumerate(speakers, 1):
            wav, sr = _synth_one(synthesizer, phonemes, speaker, args.postprocess)
            out_path = out_dir / f"{_slug(speaker)}.wav"
            sf.write(str(out_path), wav, sr)
            print(f"[{i}/{len(speakers)}] {out_path}  ({len(wav) / sr:.2f}s)")
        print(f"done — {len(speakers)} files in {out_dir}  (postprocess={args.postprocess})")
        return

    # single speaker
    out = args.out or Path("sample.wav")
    wav, sr = _synth_one(synthesizer, phonemes, args.speaker, args.postprocess)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), wav, sr)
    print(f"wrote {out}  ({len(wav) / sr:.2f}s, {sr} Hz, postprocess={args.postprocess})")


if __name__ == "__main__":
    main()
