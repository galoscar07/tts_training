"""Run the pipeline over the edge-case corpus + data/evaluation/{dev,test}
and group any exception by the layer being requested when it happened
(preprocess/objectives.md Phase 13: "Export an error report grouped by
component"). Writes real output — if nothing fails, the report says so
because the run actually completed cleanly, not because failures were
assumed away.

Usage:
    ./.venv/bin/python -m objective_evaluation.error_report
"""

from __future__ import annotations

import traceback
from pathlib import Path

from expressive_tts.preprocess.pipeline import PreprocessPipeline
from objective_evaluation.schemas import EvaluationExample

DATA_EVALUATION = Path(__file__).resolve().parents[2] / "data" / "evaluation"
REPORT_PATH = DATA_EVALUATION / "error_report.md"

LAYERS = [
    "clean",
    "sentences",
    "normalized",
    "linguistic",
    "phonemes",
    "syllables",
    "emotion",
    "focus",
    "prosody",
    "interjections",
]

EDGE_CASES = {
    "malformed_unicode_lone_surrogate": "Bun� venit! Ce mai faci�?",
    "malformed_unicode_combining": "Ta̦re bine, mult̃umesc!",
    "code_switching": "Am avut un meeting foarte productiv, apoi am mers la coffee break cu echipa.",
    "unsupported_symbols_emoji_control": "Sunt fericit azi \U0001f600\U0001f389! \x0bText cu tab\x09si control char.",
    "repeated_punctuation": "Ce???!!! Chiar nu știi???",
    "empty_string": "",
    "whitespace_only": "   \n\n",
    "very_long_paragraph": ("Astăzi este o zi frumoasă și senină. " * 40),
}


def load_eval_texts() -> dict[str, str]:
    texts = {}
    for split in ("dev", "test"):
        for line in (DATA_EVALUATION / f"{split}.jsonl").open(encoding="utf-8"):
            example = EvaluationExample.model_validate_json(line)
            texts[example.id] = example.text
    return texts


def main() -> None:
    pipeline = PreprocessPipeline()
    corpus = {**EDGE_CASES, **load_eval_texts()}

    errors_by_layer: dict[str, list[tuple[str, str]]] = {layer: [] for layer in LAYERS}
    total_runs = 0

    for layer in LAYERS:
        for case_id, text in corpus.items():
            total_runs += 1
            try:
                pipeline.process(text, include={layer})
            except Exception:  # noqa: BLE001 — deliberately broad, this *is* the error catcher
                errors_by_layer[layer].append((case_id, traceback.format_exc(limit=3)))

    total_errors = sum(len(v) for v in errors_by_layer.values())
    lines = [
        "# Pipeline error report",
        "",
        f"{len(corpus)} inputs ({len(EDGE_CASES)} hand-picked edge cases + "
        f"{len(corpus) - len(EDGE_CASES)} from data/evaluation/dev+test.jsonl) "
        f"x {len(LAYERS)} layers = {total_runs} runs.",
        "",
        f"**{total_errors} exception(s) raised.**",
        "",
    ]
    for layer in LAYERS:
        failures = errors_by_layer[layer]
        lines.append(f"## `{layer}` — {len(failures)} error(s)")
        for case_id, tb in failures:
            lines.append(f"- `{case_id}`:")
            lines.append("  ```")
            lines.extend(f"  {ln}" for ln in tb.strip().splitlines())
            lines.append("  ```")
        lines.append("")

    lines.append(
        "Caveat: this covers a finite, hand-picked edge-case set plus the "
        "38-sentence evaluation pilot — not exhaustive fuzzing. A clean "
        "report here means these specific inputs didn't crash, not that no "
        "input can."
    )

    report = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"{total_errors} exception(s) across {total_runs} runs. Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
