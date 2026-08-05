"""Sanity-check the Phase 8 rule-based focus detector against
data/evaluation/{dev,test}.jsonl's `focus_words` field.

IMPORTANT — this is NOT a gold-standard evaluation, unlike
objective_evaluation.evaluate_emotion: `focus_words` in the evaluation set is
`provenance="predicted"`, my own draft judgment from the Phase 1 pilot,
not an independent human annotation. A disagreement here doesn't tell you
the rule engine is wrong — it may just as easily mean the Phase 1 draft
was the imprecise one. Treat the numbers below as a sanity check that the
two independent guesses roughly agree, not as accuracy.

Usage:
    ./.venv/bin/python -m objective_evaluation.evaluate_focus --split dev
    ./.venv/bin/python -m objective_evaluation.evaluate_focus --split test
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from expressive_tts.preprocess.pipeline import PreprocessPipeline
from objective_evaluation.schemas import EvaluationExample

DATA_EVALUATION = Path(__file__).resolve().parents[2] / "data" / "evaluation"
_WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)


def load_examples(split: str) -> list[EvaluationExample]:
    rows = [
        EvaluationExample.model_validate_json(line)
        for line in (DATA_EVALUATION / f"{split}.jsonl").open(encoding="utf-8")
    ]
    # "bulk_added" (Tier 2, eval-set scaling) rows carry a confidence=0.0
    # *placeholder* empty focus_words list, not a judgment that there is no
    # focus — comparing against it would look like disagreement for every
    # single Tier 2 row and make the rule engine look far worse than this
    # sanity check can actually support. Tier 1 only, same as before.
    return [
        row for row in rows if "multi_sentence" not in row.phenomena and "bulk_added" not in row.phenomena
    ]


def draft_focus_words(example: EvaluationExample) -> set[str]:
    """Split any multi-word draft focus phrases into individual lowercased
    words, so they're comparable to per-token predictions."""
    words: set[str] = set()
    for phrase in example.focus_words.value:
        words.update(w.lower() for w in _WORD_PATTERN.findall(phrase))
    return words


def predicted_focus_words(pipeline: PreprocessPipeline, text: str) -> set[str]:
    result = pipeline.process(text, include={"focus"})
    return {
        token.text.lower()
        for sentence in result.sentences
        for token in sentence.tokens
        if token.is_focus
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    args = parser.parse_args()

    examples = load_examples(args.split)
    pipeline = PreprocessPipeline()

    total_tp = total_fp = total_fn = 0
    print(f"Sanity check: rule-based focus vs. draft (non-gold) focus_words — {args.split}.jsonl")
    print(f"({len(examples)} single-sentence examples)\n")

    for example in examples:
        gold = draft_focus_words(example)
        predicted = predicted_focus_words(pipeline, example.text)
        tp = len(gold & predicted)
        fp = len(predicted - gold)
        fn = len(gold - predicted)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        if predicted != gold:
            print(f"  {example.id}: draft={sorted(gold)} predicted={sorted(predicted)} | {example.text}")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print()
    print(f"Token-level precision: {precision:.3f}")
    print(f"Token-level recall:    {recall:.3f}")
    print(f"Token-level F1:        {f1:.3f}")
    print()
    print("Reminder: 'draft' is my own Phase 1 judgment, not gold data.")


if __name__ == "__main__":
    main()
