"""Evaluate the Phase 10 interjection-suggestion gating against
data/evaluation/{dev,test}.jsonl.

Two checks:

1. **Formal-text violation rate** — a real acceptance-criterion check, not
   just a sanity check: for every example tagged `text_register == "formal"`
   or `"interjection_inappropriate"` in `phenomena`, the system must
   produce zero suggestions in "suggest" mode. objectives.md Phase 10:
   "Formal-text violation rate is 0% on the evaluation set."

2. **Sanity check against the draft `interjection_appropriate` field** —
   NOT gold data (my own quick Phase 1 judgment, same caveat as
   objective_evaluation.evaluate_focus), compared for interest only.

Usage:
    ./.venv/bin/python -m objective_evaluation.evaluate_interjections --split dev
    ./.venv/bin/python -m objective_evaluation.evaluate_interjections --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path

from expressive_tts.preprocess.pipeline import PreprocessPipeline
from objective_evaluation.schemas import EvaluationExample

DATA_EVALUATION = Path(__file__).resolve().parents[2] / "data" / "evaluation"


def load_examples(split: str) -> list[EvaluationExample]:
    rows = [
        EvaluationExample.model_validate_json(line)
        for line in (DATA_EVALUATION / f"{split}.jsonl").open(encoding="utf-8")
    ]
    return [row for row in rows if "multi_sentence" not in row.phenomena]


def would_suggest(pipeline: PreprocessPipeline, text: str) -> bool:
    result = pipeline.process(text, include={"interjections"}, interjection_mode="suggest")
    return any(s.interjection_suggestions for s in result.sentences)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    args = parser.parse_args()

    examples = load_examples(args.split)
    pipeline = PreprocessPipeline()

    formal_examples = [
        e for e in examples if e.text_register == "formal" or "interjection_inappropriate" in e.phenomena
    ]
    violations = [e for e in formal_examples if would_suggest(pipeline, e.text)]

    print(f"Formal-text violation check — {args.split}.jsonl")
    print(f"  {len(formal_examples)} formal/interjection-inappropriate examples")
    rate = len(violations) / len(formal_examples) if formal_examples else 0.0
    print(f"  Violation rate: {rate:.1%} ({len(violations)}/{len(formal_examples)})")
    for e in violations:
        print(f"    VIOLATION {e.id}: {e.text}")
    print()

    # Tier 2 ("bulk_added") rows carry a confidence=0.0 *placeholder*
    # interjection_appropriate=False, not a judgment — comparing against it
    # would just restate the placeholder, not measure agreement. Tier 1
    # only, same reasoning as objective_evaluation.evaluate_focus.
    reviewed = [e for e in examples if "bulk_added" not in e.phenomena]
    print("Sanity check vs. draft 'interjection_appropriate' (NOT gold — my own Phase 1 judgment; Tier 1 only)")
    agree = 0
    for e in reviewed:
        predicted = would_suggest(pipeline, e.text)
        draft = e.interjection_appropriate.value
        match = predicted == draft
        agree += match
        if not match:
            print(f"    DISAGREE {e.id}: draft={draft} predicted={predicted} | {e.text}")
    print(f"  Agreement: {agree}/{len(reviewed)} ({agree / len(reviewed):.1%})")


if __name__ == "__main__":
    main()
