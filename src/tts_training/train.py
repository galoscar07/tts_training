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


def _bool_arg(value: str) -> bool:
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


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
    parser.add_argument("--batch-size", type=int, default=8, help="per-GPU batch size")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="per-GPU evaluation batch size")
    parser.add_argument("--num-loader-workers", type=int, default=4, help="workers per GPU")
    parser.add_argument("--num-eval-loader-workers", type=int, default=2, help="evaluation workers per GPU")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--print-step", type=int, default=25)
    parser.add_argument("--save-step", type=int, default=5000)
    parser.add_argument("--save-n-checkpoints", type=int, default=5)

    # trainer.distribute appends these arguments to every worker process.
    # Keep underscore spellings because that is the TrainerArgs CLI contract.
    parser.add_argument("--continue_path", type=str, default=None)
    parser.add_argument(
        "--restore-path", "--restore_path", dest="restore_path", type=str, default=None,
        help="checkpoint whose weights initialize a new run",
    )
    parser.add_argument("--best_path", type=str, default=None)
    parser.add_argument("--use_ddp", type=_bool_arg, default=False)
    parser.add_argument("--use_accelerate", type=_bool_arg, default=False)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--overfit_batch", type=_bool_arg, default=False)
    parser.add_argument("--skip_train_epoch", type=_bool_arg, default=False)
    parser.add_argument("--start_with_eval", type=_bool_arg, default=False)
    parser.add_argument("--small-run", "--small_run", dest="small_run", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--group_id", type=str, default="")
    args = parser.parse_args(argv)

    if len(args.manifest) != len(args.corpus_root):
        parser.error("pass one --corpus-root per --manifest")

    manifests = [path.resolve() for path in args.manifest]
    corpus_roots = [path.resolve() for path in args.corpus_root]
    for manifest in manifests:
        if not manifest.is_file():
            parser.error(f"manifest does not exist: {manifest}")
    for corpus_root in corpus_roots:
        if not corpus_root.is_dir():
            parser.error(f"corpus root does not exist: {corpus_root}")

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
        manifests[0], corpus_roots[0], args.output.resolve(),
        run_name=args.run_name,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_loader_workers=args.num_loader_workers,
        num_eval_loader_workers=args.num_eval_loader_workers,
        epochs=args.epochs,
        print_step=args.print_step,
        save_step=args.save_step,
        save_n_checkpoints=args.save_n_checkpoints,
        manifest_paths=manifests,
    )
    config.datasets = [
        BaseDatasetConfig(
            formatter="", meta_file_train=str(m), path=str(root), language=LANGUAGE
        )
        for m, root in zip(manifests, corpus_roots)
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
        TrainerArgs(
            continue_path=args.continue_path,
            restore_path=args.restore_path,
            best_path=args.best_path,
            use_ddp=args.use_ddp,
            use_accelerate=args.use_accelerate,
            grad_accum_steps=args.grad_accum_steps,
            overfit_batch=args.overfit_batch,
            skip_train_epoch=args.skip_train_epoch,
            start_with_eval=args.start_with_eval,
            small_run=args.small_run,
            gpu=args.gpu,
            rank=args.rank,
            group_id=args.group_id,
        ),
        config,
        str(args.output),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    try:
        trainer.fit()
    finally:
        # Trainer 0.3.3 leaves the DDP process group initialized at normal
        # interpreter shutdown, which PyTorch 2.4+ reports as a warning.
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _init_tokenizer(config):
    from TTS.tts.utils.text.tokenizer import TTSTokenizer

    return TTSTokenizer.init_from_config(config)


if __name__ == "__main__":
    main()
