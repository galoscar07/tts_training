"""Fetch and cache the external sentence sources used to build the Phase 1
evaluation set (preprocess/objectives.md). Network access required; re-run
to refresh the cache. `objective_evaluation.build_dataset` reads only the cached files
under data/external/, so it works offline once this has been run.

Sources (see data/external/SOURCES.md for full attribution):
- REDv2: Alexandra Ciobotaru's Romanian Emotions Dataset v2 (MIT), GitHub.
- RONEC: Romanian Named Entity Corpus, via the Hugging Face datasets-server
  REST API (no `datasets`/`huggingface_hub` dependency needed).
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

DATA_EXTERNAL = Path(__file__).resolve().parents[2] / "data" / "external"

REDV2_URL = (
    "https://raw.githubusercontent.com/Alegzandra/"
    "RED-Romanian-Emotion-Datasets/main/REDv2/data/test.json"
)
REDV2_LABEL_ORDER = ["sad", "surprise", "fear", "angry", "neutral", "trust", "happy"]

RONEC_ROWS_URL_TEMPLATE = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=community-datasets%2Fronec&config=ronec&split=train&offset={offset}&length=100"
)
RONEC_PAGES = 15  # datasets-server caps `length` at 100 per request; raised from 3 for eval-set scaling

_NOISY_PATTERN = re.compile(r"<\|[A-Z]+\|>|#\w+|https?://|www\.")


def _fetch_json(url: str) -> dict | list:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_redv2() -> list[dict]:
    rows = _fetch_json(REDV2_URL)
    selected = []
    for row in rows:
        text = row["text"].strip()
        if _NOISY_PATTERN.search(text):
            continue
        if not (10 <= len(text) <= 200):
            continue
        agreed = row["agreed_labels"]
        if sum(agreed) != 1:
            continue  # keep only clear, single-emotion agreement
        label = REDV2_LABEL_ORDER[agreed.index(1)]
        if label == "trust":
            continue  # not in objectives.md's emotion label set
        selected.append(
            {
                "text_id": row["text_id"],
                "text": text,
                "emotion": label,
                "procentual_labels": dict(zip(REDV2_LABEL_ORDER, row["procentual_labels"])),
            }
        )
    return selected


def _detokenize(tokens: list[str], space_after: list[bool]) -> str:
    chunks = []
    for token, sep in zip(tokens, space_after):
        chunks.append(token)
        if sep:
            chunks.append(" ")
    return "".join(chunks).strip()


def fetch_ronec() -> list[dict]:
    selected = []
    for page in range(RONEC_PAGES):
        payload = _fetch_json(RONEC_ROWS_URL_TEMPLATE.format(offset=page * 100))
        for entry in payload["rows"]:
            row = entry["row"]
            text = _detokenize(row["tokens"], row["space_after"])
            if not (20 <= len(text) <= 220):
                continue
            selected.append({"row_idx": entry["row_idx"], "id": row["id"], "text": text})
    return selected


def main() -> None:
    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)

    redv2 = fetch_redv2()
    (DATA_EXTERNAL / "redv2_sample.json").write_text(
        json.dumps(redv2, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"REDv2: cached {len(redv2)} candidate sentences")

    ronec = fetch_ronec()
    (DATA_EXTERNAL / "ronec_sample.json").write_text(
        json.dumps(ronec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"RONEC: cached {len(ronec)} candidate sentences")


if __name__ == "__main__":
    main()
