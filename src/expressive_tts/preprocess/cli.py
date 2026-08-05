"""`tts-preprocess` CLI entrypoint. See readme.md section 1.2/1.3/12.

Implements the single-text, stdin, and plain-text-file input modes with
`--include`/`--exclude`/`--profile`/`--config` layer selection. CSV/TSV/
JSON/directory/manifest input modes are not implemented yet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from expressive_tts.preprocess.pipeline import PreprocessPipeline
from expressive_tts.preprocess.readers.text import read_input_text
from expressive_tts.preprocess.serializers import to_annotated_text, to_control_tokens, to_ssml_like


def _split_layers(value: str | None) -> set[str] | None:
    if value is None:
        return None
    return {layer.strip() for layer in value.split(",") if layer.strip()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tts-preprocess", description=__doc__)

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="Text to process, given directly on the command line.")
    input_group.add_argument("--stdin", action="store_true", help="Read text from standard input.")
    input_group.add_argument("--input-file", help="Read text from a plain-text file.")

    parser.add_argument("--include", help="Comma-separated output layers to include.")
    parser.add_argument("--exclude", help="Comma-separated output layers to exclude.")
    parser.add_argument("--profile", help="Named profile from configs/preprocess/<name>.yaml.")
    parser.add_argument("--config", help="Path to a preprocessing configuration YAML file.")
    parser.add_argument("--id", help="Record id to attach to the result.")
    parser.add_argument("--output-file", help="Write JSON output here instead of stdout.")
    parser.add_argument(
        "--serialize",
        choices=["control_tokens", "annotated_text", "ssml"],
        help="Print this serialized format instead of canonical JSON (preprocess/objectives.md Phase 12).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.config:
        pipeline = PreprocessPipeline.from_config(args.config)
    elif args.profile:
        pipeline = PreprocessPipeline.from_profile(args.profile)
    else:
        pipeline = PreprocessPipeline()

    text = read_input_text(text=args.text, stdin=args.stdin, input_file=args.input_file)

    result = pipeline.process(
        text,
        include=_split_layers(args.include),
        exclude=_split_layers(args.exclude),
        id=args.id,
    )

    if args.serialize == "control_tokens":
        output = to_control_tokens(result)
    elif args.serialize == "annotated_text":
        output = to_annotated_text(result)
    elif args.serialize == "ssml":
        output = to_ssml_like(result)
    else:
        output = result.model_dump_json(indent=2)

    if args.output_file:
        Path(args.output_file).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
