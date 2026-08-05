"""Train the base (neutral, multi-speaker) Romanian VITS with Coqui TTS.

Run this on a machine with a CUDA GPU and `coqui-tts` installed — it is not
expected to train on Apple M1 (see README.md). Everything Coqui is imported
lazily inside `main()` so the module imports without the training extra.

    python -m tts_training.train \
        --manifest out/mara.manifest \
        --corpus-root datasets/mara \
        --output out/vits_ro_base

Multiple corpora: build one manifest per corpus with `data.manifest`, then
pass several `--manifest`/`--corpus-root` pairs (repeatable). All speakers
share one multi-speaker model.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", action="append", required=True, type=Path,
        help="manifest from tts_training.data.manifest (repeatable)",
    )
    parser.add_argument(
        "--corpus-root", action="append", required=True, type=Path,
        help="corpus dir each manifest's audio paths are relative to (repeatable, paired with --manifest)",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-name", default="vits_ro_base")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--restore-path", type=Path, default=None, help="checkpoint to resume/continue from")
    args = parser.parse_args(argv)

    if len(args.manifest) != len(args.corpus_root):
        parser.error("pass one --corpus-root per --manifest")

    # --- Coqui imports (GPU box only) -------------------------------------
    from trainer import Trainer, TrainerArgs
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.models.vits import Vits
    from TTS.tts.utils.speakers import SpeakerManager
    from TTS.utils.audio import AudioProcessor

    from tts_training.data.formatter import coqui_formatter
    from tts_training.vits.config import base_vits_config

    # One config; attach every corpus as a dataset entry.
    from TTS.tts.configs.shared_configs import BaseDatasetConfig
    from tts_training import LANGUAGE

    config = base_vits_config(
        args.manifest[0], args.corpus_root[0], args.output,
        run_name=args.run_name, batch_size=args.batch_size,
    )
    config.datasets = [
        BaseDatasetConfig(
            formatter="", meta_file_train=str(m), path=str(root), language=LANGUAGE
        )
        for m, root in zip(args.manifest, args.corpus_root)
    ]

    train_samples, eval_samples = [], []
    for dataset in config.datasets:
        tr, ev = load_tts_samples(dataset, eval_split=True, formatter=coqui_formatter)
        train_samples += tr
        eval_samples += ev

    ap = AudioProcessor.init_from_config(config)
    speaker_manager = SpeakerManager()
    speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
    config.model_args.num_speakers = speaker_manager.num_speakers

    tokenizer, config = _init_tokenizer(config)
    model = Vits(config, ap, tokenizer, speaker_manager=speaker_manager)

    trainer = Trainer(
        TrainerArgs(restore_path=str(args.restore_path) if args.restore_path else None),
        config,
        str(args.output),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


def _init_tokenizer(config):
    from TTS.tts.utils.text.tokenizer import TTSTokenizer

    return TTSTokenizer.init_from_config(config)


if __name__ == "__main__":
    main()
