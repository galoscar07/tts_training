# Phase 1 evaluation set

`preprocess/objectives.md` Phase 1 requires a fixed, 300+ sentence /
30+ paragraph evaluation set built before further tuning the pipeline. The
set is built by `objective_evaluation.build_dataset` in **two tiers**:

- **Tier 1 — hand-curated pilot (38 sentences, `dev-001`..`dev-021` /
  `test-001`..`test-017`, 3 `context.jsonl` paragraphs `ctx-001`..`ctx-003`)**:
  every field individually considered, including the subjective ones
  (`intensity`, `focus_words`, `interjection_appropriate`). Built first, to
  validate the format/sourcing/annotation approach before scaling.
- **Tier 2 — programmatically sampled bulk items (266 sentences, appended
  after Tier 1's IDs; 40 more `context.jsonl` paragraphs, `ctx-mara-*` /
  `ctx-hria-*`)**: added to reach the 300+/30+ target. Hand-judging 266
  more sentences with the same individual care as Tier 1 isn't achievable
  in one pass, and silently generating that many "quick judgments" would
  be a materially different — and worse — kind of data presented as
  equivalent. So Tier 2 is explicit about what's real and what isn't:
  - **Real**: the text itself (sampled from the same cached REDv2/RONEC
    pools Tier 1 draws from); REDv2's own human-annotated gold emotion
    label, kept with `provenance="source"` exactly like Tier 1;
    `sentence_type` (from terminal punctuation) and `pause_locations`
    (from punctuation offsets) — the same mechanical derivation Tier 1
    already uses, not weaker for being bulk.
  - **Explicitly unreviewed**: `intensity`, `focus_words`, and
    `interjection_appropriate` are placeholder values
    (`confidence=0.0`, `note="bulk-added, unreviewed..."`) — a *stronger*
    caveat than even Tier 1's "my own quick draft," not a downgrade of the
    same thing. Don't score any component against these three fields on
    Tier 2 rows.
  - `context.jsonl`'s Tier 2 paragraphs are reconstructed from real
    continuous text (`datasets/mara/metadata.csv` narrative fragments,
    `datasets/hria/catalina/metadata_simple.txt` conversational
    utterances) — real adjacency, no fabricated paragraphs — but chunk
    boundaries are a fixed-size grouping, not necessarily the source's own
    original paragraph breaks (see `build_tier2_context_paragraphs` in
    `objective_evaluation.build_dataset`).

The gold-labeled (`provenance="source"`) subset — now 193 sentences, up
from 15 in the Tier-1-only pilot — was used to re-measure the Phase 6
rule-based emotion baseline at a meaningful sample size — see
`emotion_baseline_report.md`. The `focus_words` field (Tier 1 only — Tier
2's is an unreviewed placeholder) was used to sanity-check the Phase 8
focus detector — see `focus_sanity_check.md`. The `text_register`/
`phenomena` tags and the draft `interjection_appropriate` field (Tier 1
only, same reason) were used for the Phase 10 interjection-suggestion
gating — see `interjection_evaluation_report.md`, which also includes one
genuine, verified acceptance-criterion result (0% formal-text violation
rate). `context_evaluation_report.md` covers the Phase 11 context
processor's behavior sanity check. `emotion_classifier_report.md` covers
the optional Phase 7 trained classifier experiment (real result: +0.238
macro F1 over the rule baseline on a held-out split, but not adopted —
poorly calibrated).

## What's gold vs. draft vs. unreviewed

Every field in `EvaluationExample`
(`src/objective_evaluation/schemas.py`) is wrapped in an `Annotation`
carrying `provenance`:

- **`provenance: "source"`** — real human annotations pulled directly from
  a labeled dataset. Currently only `emotion`, when the sentence comes from
  REDv2 (3-annotator-agreed label) or `hria/catalina/metadata_final.txt`
  (its own recorded label). **This is the only gold-standard field**, in
  both Tier 1 and Tier 2.
- **`provenance: "rule"`** — deterministic, computed by actually running
  `expressive_tts.preprocess.PreprocessPipeline` (`expected_normalized_text`,
  `sentence_boundaries`).
- **`provenance: "predicted"`** — subjective judgment, further split by
  `confidence`: Tier 1's is my own individually-considered draft
  (confidence 0.5-0.7 depending on field) — not gold, needs human review,
  but a real per-sentence judgment. Tier 2's `intensity`/`focus_words`/
  `interjection_appropriate` are `confidence=0.0` placeholders, not
  judgments at all — see the Tier 2 section above. There is no second
  annotator on this project yet (objectives.md's inter-annotator-agreement
  step is out of reach until one is available).

## Sources

See `data/external/SOURCES.md` for full dataset attribution/licenses
(REDv2, RONEC — both MIT) and `objective_evaluation.sources` /
`objective_evaluation.build_dataset` for how the cache and the curated set
were produced. Every `source` field in the JSONL traces back to an exact
row in a real, licensed dataset or an in-repo corpus — nothing here is
synthetic except the 4 hand-authored lexical-stress minimal-pair sentences
(`"copii"` copies-vs-children, `"veselă"` cheerful-vs-dishes), added because
none of the sampled sources reliably exercise that phenomenon.

## Coverage (Tier 1 + Tier 2 combined, `dev.jsonl` + `test.jsonl`, 304 sentences)

- Registers: conversational (201), news (88), formal (10), narrative (3),
  technical (2) — all 5 required registers represented. Skewed toward
  conversational/news because that's what REDv2/RONEC's own composition
  is — Tier 2 samples from those pools rather than commissioning new
  register-balanced text.
- Emotion: unspecified (96), neutral (46), happy (35), angry (34), fear
  (32), sad (32), surprise (29) — all 6 objectives.md labels plus
  `unspecified`, reasonably balanced (29-46 per real label; `unspecified`
  is high because Tier 2's RONEC rows have no emotion label and are
  deliberately marked `unspecified` rather than guessed — see Tier 2
  above). 193 of the 304 have `provenance="source"` gold emotion (up from
  15 in the Tier-1-only pilot).
- `context.jsonl`: 43 paragraphs (3 Tier 1 + 40 Tier 2 — 21 narrative from
  `mara`, 22 conversational from `hria`), exceeding the 30+ target.
- Phenomena tags (Tier 1) span numbers/dates/times/currency/percentage,
  common abbreviations (via the normalizer itself), a Roman-numeral
  ordinal, questions/exclamations/commands/negation/incomplete sentences,
  existing interjections, a case where an interjection would be clearly
  inappropriate (formal legal text), and hard lexical stress. Tier 2 rows
  are tagged `["bulk_added"]` only — no per-phenomenon tagging was done at
  that volume.
- `expected_normalized_text` was validated against real corpus text, not
  just the objectives.md worked example — e.g. RONEC's "secolului al
  XIV-lea" → "secolului al paisprezecelea", "80%" → "optzeci la sută",
  "2100 de lei" → "două mii o sută de lei".

## Known gaps

- Tier 2's `intensity`/`focus_words`/`interjection_appropriate` are
  unreviewed placeholders, not judgments — don't score against them (see
  Tier 2 above). Tier 1's own versions of those fields are unreviewed
  drafts, one step better but still not gold.
- `pause_locations` specifically carries no *duration* information and is
  mechanically derived from punctuation offsets by `objective_evaluation.build_dataset`
  itself, not an independent judgment of where a human would actually
  pause — it can't meaningfully evaluate Phase 9's prosody/pause
  predictions (`preprocess/prosody.py`), which is why that phase has no
  `evaluate_prosody` counterpart to `objective_evaluation.evaluate_emotion`/
  `objective_evaluation.evaluate_focus`. A real evaluation needs either
  aligned audio or dedicated human pause/contour annotation.
- No inter-annotator agreement (single "annotator": me).
- `datasets/hria` and `datasets/mara` license status is unclarified (noted
  in `data/external/SOURCES.md`) — fine for internal evaluation use, not
  yet cleared for redistribution.

## Regenerating

```bash
./.venv/bin/python -m objective_evaluation.sources   # refresh data/external/ cache (needs network)
./.venv/bin/python -m objective_evaluation.build_dataset        # rebuild data/evaluation/*.jsonl from the cache
```
