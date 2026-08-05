"""Evaluate the Phase 12 control-token serializer against
data/evaluation/{dev,test}.jsonl.

objectives.md Phase 12 acceptance criteria, checked directly:
- 100% serialization success on the test set (no exceptions).
- 0% unknown-control-token rate (guaranteed by construction via
  `..._UNSPECIFIED` tokens — this script verifies no *other* unrecognized
  bracket token slips through, e.g. from a malformed value).
- Identical input and configuration produce identical output (determinism).

Usage:
    ./.venv/bin/python -m objective_evaluation.evaluate_serialization --split dev
    ./.venv/bin/python -m objective_evaluation.evaluate_serialization --split test
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from expressive_tts.preprocess.pipeline import PreprocessPipeline
from expressive_tts.preprocess.serializers import to_control_tokens
from objective_evaluation.schemas import EvaluationExample

DATA_EVALUATION = Path(__file__).resolve().parents[2] / "data" / "evaluation"

_KNOWN_PREFIXES = {"SENT", "EMO", "INT", "RATE", "PITCH", "ENERGY", "BREAK", "FOCUS"}
_BRACKET_TOKEN_RE = re.compile(r"\[([A-Z]+)(?:_[^\]]+)?\]")


def load_examples(split: str) -> list[EvaluationExample]:
    return [
        EvaluationExample.model_validate_json(line)
        for line in (DATA_EVALUATION / f"{split}.jsonl").open(encoding="utf-8")
    ]


def unknown_tokens(serialized: str) -> set[str]:
    return {prefix for prefix in _BRACKET_TOKEN_RE.findall(serialized) if prefix not in _KNOWN_PREFIXES}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    args = parser.parse_args()

    examples = load_examples(args.split)
    pipeline = PreprocessPipeline()

    successes = 0
    failures: list[tuple[str, str]] = []
    unknown_hits: list[tuple[str, set[str]]] = []
    non_deterministic: list[str] = []

    for example in examples:
        try:
            result = pipeline.process(example.text, include={"emotion", "prosody", "focus", "phonemes"})
            output_a = to_control_tokens(result)
            output_b = to_control_tokens(pipeline.process(example.text, include={"emotion", "prosody", "focus", "phonemes"}))
        except Exception as exc:  # noqa: BLE001 — deliberately broad, this *is* the failure check
            failures.append((example.id, repr(exc)))
            continue

        successes += 1
        unknown = unknown_tokens(output_a)
        if unknown:
            unknown_hits.append((example.id, unknown))
        if output_a != output_b:
            non_deterministic.append(example.id)

    total = len(examples)
    print(f"Phase 12 serialization evaluation — {args.split}.jsonl ({total} examples)\n")
    print(f"Serialization success rate: {successes}/{total} ({successes / total:.1%})")
    for example_id, error in failures:
        print(f"  FAILURE {example_id}: {error}")
    print(f"Unknown-control-token rate: {len(unknown_hits)}/{total} ({len(unknown_hits) / total:.1%})")
    for example_id, tokens in unknown_hits:
        print(f"  UNKNOWN {example_id}: {sorted(tokens)}")
    print(f"Determinism violations: {len(non_deterministic)}/{total}")
    for example_id in non_deterministic:
        print(f"  NON-DETERMINISTIC {example_id}")


if __name__ == "__main__":
    main()
