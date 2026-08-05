"""Coqui VITS configuration for the Romanian base (neutral, multi-speaker)
model. Coqui is imported lazily so this module loads without `coqui-tts`.

Key choices:
  * `use_phonemes=False` — the manifest's `text` column is already our IPA
    phoneme string (from `expressive_tts.preprocess`); Coqui's phonemizer and
    grapheme cleaners are bypassed.
  * `characters` comes from our phoneme inventory (`frontend.symbols`).
  * 22.05 kHz mono to match `datasets/mara`.
  * `use_speaker_embedding=True` so one model spans MARA + SWARA + Common
    Voice speakers (single-speaker corpora still train fine — they just have
    one speaker id).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tts_training import LANGUAGE, SAMPLE_RATE


def base_vits_config(
    manifest: str | Path,
    corpus_root: str | Path,
    output_path: str | Path,
    *,
    run_name: str = "vits_ro_base",
    batch_size: int = 16,
    eval_batch_size: int | None = None,
    num_loader_workers: int = 4,
    num_eval_loader_workers: int = 2,
    epochs: int = 1000,
    print_step: int = 50,
    save_step: int = 5000,
    save_n_checkpoints: int = 5,
    manifest_paths: Iterable[str | Path] | None = None,
    multi_speaker: bool = True,
):
    """Return a Coqui `VitsConfig` for base (neutral) training.

    `manifest` is a file from `tts_training.data.manifest`; `corpus_root` is
    the corpus directory its `audio_file` paths are relative to.
    """
    from TTS.config.shared_configs import BaseAudioConfig
    from TTS.tts.configs.shared_configs import BaseDatasetConfig
    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.models.vits import VitsArgs

    from tts_training.frontend.symbols import characters_config

    audio = BaseAudioConfig(
        sample_rate=SAMPLE_RATE,
        win_length=1024,
        hop_length=256,
        fft_size=1024,
        num_mels=80,
        mel_fmin=0.0,
        mel_fmax=None,
    )

    dataset = BaseDatasetConfig(
        # Custom formatter is passed to `load_tts_samples` in train.py; the
        # name here is only a label.
        formatter="",
        meta_file_train=str(manifest),
        path=str(corpus_root),
        language=LANGUAGE,
    )

    model_args = VitsArgs(
        use_speaker_embedding=multi_speaker,
        # Emotion conditioning is deferred (see finetune.py) — the base model
        # is neutral, so no emotion embedding here.
    )

    return VitsConfig(
        model_args=model_args,
        audio=audio,
        run_name=run_name,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size or batch_size,
        num_loader_workers=num_loader_workers,
        num_eval_loader_workers=num_eval_loader_workers,
        epochs=epochs,
        text_cleaner=None,          # our text is already normalized/phonemized
        use_phonemes=False,         # bring-your-own phonemizer
        characters=characters_config(manifest_paths or [manifest]),
        add_blank=True,
        print_step=print_step,
        save_step=save_step,
        save_n_checkpoints=save_n_checkpoints,
        save_best_after=10000,
        mixed_precision=True,
        output_path=str(output_path),
        datasets=[dataset],
        test_sentences=[],          # phoneme-string test inputs can be added later
        use_speaker_embedding=multi_speaker,
    )
