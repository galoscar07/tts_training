"""Measure pipeline runtime and peak memory (preprocess/objectives.md
Phase 13: "Record runtime and memory on M1"). Reports the actual detected
host CPU rather than assuming Apple M1 — this machine may or may not be
one, and asserting a specific chip without checking would be exactly the
kind of unverified claim this project's docs try to avoid.

Usage:
    ./.venv/bin/python -m objective_evaluation.benchmark
"""

from __future__ import annotations

import platform
import time
import tracemalloc
from pathlib import Path
from statistics import median

from expressive_tts.preprocess.pipeline import PreprocessPipeline
from objective_evaluation.schemas import EvaluationExample

DATA_EVALUATION = Path(__file__).resolve().parents[2] / "data" / "evaluation"
REPORT_PATH = DATA_EVALUATION / "benchmark_report.md"

ALL_LAYERS = {
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
}


def load_texts() -> list[str]:
    texts = []
    for split in ("dev", "test"):
        for line in (DATA_EVALUATION / f"{split}.jsonl").open(encoding="utf-8"):
            texts.append(EvaluationExample.model_validate_json(line).text)
    return texts


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(round(p * (len(sorted_values) - 1))))
    return sorted_values[index]


def host_description() -> str:
    return f"{platform.system()} {platform.release()}, {platform.machine()} ({platform.processor() or 'unknown processor'})"


def main() -> None:
    texts = load_texts()
    pipeline = PreprocessPipeline()

    # warm up: model loading / lazy singleton construction shouldn't count
    # against per-sentence latency.
    pipeline.process(texts[0], include=ALL_LAYERS)

    durations_ms: list[float] = []
    tracemalloc.start()
    for text in texts:
        start = time.perf_counter()
        pipeline.process(text, include=ALL_LAYERS)
        durations_ms.append((time.perf_counter() - start) * 1000)
    _current, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    sorted_durations = sorted(durations_ms)
    p50 = percentile(sorted_durations, 0.50)
    p95 = percentile(sorted_durations, 0.95)
    p99 = percentile(sorted_durations, 0.99)

    lines = [
        "# Pipeline runtime/memory benchmark",
        "",
        f"Host: {host_description()}",
        f"Sentences: {len(texts)} (data/evaluation/dev.jsonl + test.jsonl), all layers requested",
        "",
        f"- p50 latency: {p50:.1f} ms",
        f"- p95 latency: {p95:.1f} ms",
        f"- p99 latency: {p99:.1f} ms",
        f"- median latency: {median(durations_ms):.1f} ms",
        f"- peak traced memory during the run (tracemalloc, Python-object "
        f"allocations only — does not include the Stanza model's own C/torch "
        f"buffers): {peak_bytes / (1024 * 1024):.1f} MB",
        "",
        "Caveats: single-process, single-run measurement on whatever machine "
        "this was generated on (see Host above) — not averaged across "
        "multiple runs or isolated from other system load. `tracemalloc` "
        "tracks Python heap allocations; native memory used by Stanza's "
        "underlying torch model is not captured by it, so peak memory here "
        "understates true process RSS.",
    ]
    report = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
