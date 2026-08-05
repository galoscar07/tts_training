# Preprocessing pipeline

The Romanian text-preprocessing frontend described in the [root
readme](../../../readme.md) and [`preprocess/objectives.md`](../../../preprocess/objectives.md).
This document covers only what's implemented in `src/expressive_tts/preprocess/`
today, not the full aspirational design.

## Evaluation set

A pilot Phase 1 evaluation set (real, licensed Romanian sentences — REDv2,
RONEC, plus in-repo corpora — with draft annotations pending human review)
lives in [`data/evaluation/`](../../../data/evaluation/README.md).

## Status

Implemented (Phase 0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12 of `objectives.md`;
Phase 7 was attempted and evaluated but not adopted — see below):

| Layer | Processor | Notes |
|---|---|---|
| `clean` | `cleaner.CleanerProcessor` | Unicode NFC, legacy `ş/ţ` → `ș/ț`, whitespace, quote/apostrophe normalization |
| `sentences` | `sentence_segmenter.SentenceSegmenterProcessor` | Rule-based v0 splitter — a placeholder until Trankit-based segmentation is used for this too |
| `normalized` | `normalizer.NormalizerProcessor` | Dates, times, percentages, currencies, measurement units, cardinal numbers, decimals, abbreviations, `al X-lea`/`a X-a` Roman-numeral ordinals |
| `tokens`, `linguistic` | `linguistic.LinguisticProcessor` | Tokens, lemmas, UPOS, morphology, dependencies, negation, sentence type (root-verb mood + punctuation), interjection detection. Two backends: [Trankit](https://github.com/nlp-uoregon/trankit) (transformer, XLM-RoBERTa; preferred) with automatic fallback to [Stanza](https://stanfordnlp.github.io/stanza/) (neural) when Trankit can't load — e.g. Python 3.13. Disk-cached by backend + text hash |
| `phonemes` | `phonemizer.PhonemizerProcessor` | Project overrides → `espeak` (subprocess, IPA) → grapheme fallback cascade, validated against a canonical phoneme inventory |
| `syllables`, `lexical_stress` | `stress.StressProcessor` | Rule-based Romanian syllabifier + stress derived from espeak's stress mark (or a documented fallback heuristic) |
| `emotion`, `intensity` | `emotion.EmotionProcessor` | Transformer classifier: multilingual XLM-RoBERTa emotion model ([`tabularisai/multilingual-emotion-classification`](https://huggingface.co/tabularisai/multilingual-emotion-classification)) run locally, aggregated to the project's 6-label set; valence/arousal from per-emotion VAD prototypes; intensity nudged by punctuation/caps; low-confidence/tied predictions abstain to `unspecified` |
| `focus` | `focus.FocusProcessor` | Sentence-level emphasis: capitalization, intensifier targets, repetition, emotion-bearing words, contrastive "nu X, ci Y" corrections, main predicate — plus explicit caller-provided emphasis with full priority |
| `prosody` | `prosody.ProsodyProcessor` | Model-independent prosodic control values: punctuation/clause-boundary pauses, terminal contour from sentence type, arousal-driven speaking rate/pitch/energy, focus applied locally per-token — all clamped to documented safe ranges |
| `interjections` | `interjections.InterjectionProcessor` | Document-style detection (formal/conversational) + emotion-matched interjection suggestions, disabled by default; `"suggest"`/`"insert"` modes, original text always preserved |
| `context` | `context.ContextProcessor` | Paragraph-aware emotion smoothing: blends the local prediction's category distribution with the previous sentence's, gated on local confidence/discourse markers/paragraph boundaries; local prediction always preserved separately |

Also implemented, not a registered layer (operates on the finished
`PreprocessResult`, not `SentenceSpan`): `serializers.py` — canonical
JSON, human-readable annotated text, TTS control tokens (+ a round-trip
parser), and a minimal illustrative SSML-like format. See "Serialization"
below.

Not implemented yet (deferred, not stubbed): the HTTP service, RoLEX
lexicon integration (licensing unclear — see `phonemizer.py`),
CSV/TSV/JSON/directory/manifest input readers (only
`--text`/`--stdin`/`--input-file` plain text work today), and the VITS/
Matcha-TTS serializer adapters (objectives.md marks these "Future"
explicitly). See `preprocess/objectives.md` for the full phase plan and
`readme.md` section 13 for the overall checklist.

**Phase 7 (optional trained emotion classifier)**: trained and evaluated
(`objective_evaluation.train_emotion_classifier` / `objective_evaluation.evaluate_emotion_classifier`)
— a scikit-learn logistic regression beat the rule baseline by +0.238
macro F1 on a held-out split, but its predicted confidences are poorly
calibrated (Brier score 0.664), so per objectives.md's own acceptance
criteria it was **not adopted** into the pipeline. Real, run, honestly
reported — see `data/evaluation/emotion_classifier_report.md`.

## Install

From the repo root, into the existing `.venv`:

```bash
./.venv/bin/pip install -e ".[dev]"
# add linguistic analysis / phonemes / stress / emotion (heavy — pulls torch + transformers):
./.venv/bin/pip install -e ".[dev,linguistic,ai]"
# POS/lemma/deps backend — pick the one matching your Python:
./.venv/bin/python scripts/download_trankit_model.py  # transformer (XLM-R), Python <=3.12; ~1.1GB
./.venv/bin/python scripts/download_stanza_model.py   # neural fallback, works on Python 3.13; ~217MB
./.venv/bin/python scripts/download_emotion_model.py  # one-time, ~1.1GB, needs network (transformer emotion)
./.venv/bin/python scripts/fetch_emotion_lexicon.py   # one-time, ~109MB, needs network (NRC lexicon, focus layer)
```

On Python 3.13 the linguistic layer uses Stanza automatically (Trankit 1.1.1
crashes on import there); on Python <=3.12 it uses the Trankit transformer.
Either way the emotion layer is the transformer.

`phonemes`/`lexical_stress` also need the `espeak` binary on `PATH` (not a
pip package — e.g. `brew install espeak` on macOS). This project
deliberately uses classic `espeak`, not `espeak-ng`: this machine already
had `espeak` installed, and `espeak -v ro --ipa` already gives IPA phonemes
with stress marks, so no system package changes were needed. If `espeak`
isn't available, phonemization falls back to a low-confidence grapheme
identity mapping (flagged in `warnings`) rather than failing.

`emotion`/`intensity` need the NRC lexicon cache from
`fetch_emotion_lexicon.py` — **not committed to this repo** (the license
prohibits redistribution, see `data/external/SOURCES.md`), fetched fresh
into `.cache/lexicons/` (gitignored) instead. Re-run that script any time
the cache is missing or cleared.

## Run

```bash
# Single text
tts-preprocess --text "Dr. Popescu a trimis 25 kg pe 12.07.2026, la ora 14:30." --include normalized

# stdin
echo "Nu pot să cred! Am reușit." | tts-preprocess --stdin --include normalized

# Full pronunciation profile: normalized text, phonemes, syllables, lexical stress
tts-preprocess --text "Nu pot să cred! Am reușit." --profile pronunciation

# Emotion + intensity + focus + prosody + interjections (suggestions still
# off by default — interjection_mode isn't exposed as a CLI flag yet)
tts-preprocess --text "Uau! Chiar am reușit?" --profile expressive

# Plain-text file, using a named profile
tts-preprocess --input-file examples/input.txt --profile normalize_only --output-file outputs/input.processed.json
```

Python API (readme.md section 1.8):

```python
from expressive_tts.preprocess import PreprocessPipeline

pipeline = PreprocessPipeline.from_profile("default")
result = pipeline.process("Uau! Chiar ai terminat?", include={"normalized"})
print(result.normalized_text)
```

Explicit user-provided emphasis (Python API only — no CLI flag yet)
overrides `focus.FocusProcessor`'s rule-based scoring with full priority:

```python
result = pipeline.process("Vreau cafea.", include={"focus"}, user_focus_words={"cafea"})
```

Preserving specific user-provided prosody values (Python API only) works
the same way, keyed by 0-based sentence index:

```python
result = pipeline.process(
    "Plouă.", include={"prosody"}, user_prosody_overrides={0: {"terminal_contour": "rising"}}
)
```

Interjection suggestions (Python API only, `interjection_mode` defaults to
`"disabled"`):

```python
result = pipeline.process(
    "Sunt extrem de fericit!", include={"interjections"}, interjection_mode="insert"
)
print(result.document_style)                       # "conversational"
print(result.sentences[0].text)                     # unmodified: "Sunt extrem de fericit!"
print(result.sentences[0].text_with_interjections)   # "Bravo, sunt extrem de fericit!"
```

Serialization (Phase 12) — CLI:

```bash
tts-preprocess --text "Uau! Chiar am reușit?" --profile expressive --serialize control_tokens
tts-preprocess --text "Uau! Chiar am reușit?" --profile expressive --serialize annotated_text
tts-preprocess --text "Uau! Chiar am reușit?" --profile expressive --serialize ssml
```

or via the Python API, which also lets you populate the (otherwise always
`None`) `PreprocessResult.tts_token_text` field directly on the result:

```python
from expressive_tts.preprocess.serializers import to_control_tokens

result = pipeline.process("Uau! Chiar am reușit?", include={"emotion", "prosody"}, serialize_control_tokens=True)
print(result.tts_token_text)          # same as to_control_tokens(result)
print(to_control_tokens(result))
```

See `examples/preprocess_text.py` for a runnable version.

## Output shape

Every call returns a `PreprocessResult` (`preprocess/schemas.py`) — the
Phase 0 annotation contract. `tts_token_text` stays `None` unless you pass
`serialize_control_tokens=True` to `.process()` (Phase 12 — see
"Serialization" above); `original_text` (and `Sentence.text`) are never
mutated — see `interjections.InterjectionProcessor`, whose `"insert"` mode
writes to the separate `Sentence.text_with_interjections` field instead.
Every applied transformation is recorded in `trace` with its stage,
operation, original/replacement spans, and producer id. Each `Token` in
`Sentence.tokens` carries its own provenance for pronunciation
(`pronunciation_provenance`/`_producer`/`_confidence`) and stress
(`stress_provenance`/`_producer`/`_confidence`) — e.g. an override-dictionary
hit is `Provenance.LEXICON` with confidence 1.0, an espeak-derived value is
`Provenance.PREDICTED`, and a fallback (grapheme identity, or the default
stress heuristic) is `Provenance.FALLBACK` with low confidence and a
matching entry in `warnings`.

## Layers and profiles

`--include`/`--exclude` accept a comma-separated subset of `{clean,
sentences, normalized, tokens, linguistic, phonemes, syllables,
lexical_stress, emotion, intensity, focus, prosody, interjections,
context}`. The `ProcessorRegistry` (`registry.py`) resolves prerequisites
automatically and skips any processor not on the dependency path to what
you requested — `--include clean` alone will not run the sentence
segmenter, normalizer, linguistic analysis, phonemizer, stress, emotion,
focus, prosody, interjections, or context processor. `emotion`/
`intensity`/`focus`/`prosody` only need `linguistic`
(tokens/lemmas/negation/UPOS/dependencies), not `phonemes`/
`lexical_stress` — e.g. `--include prosody` alone never touches `espeak`
or the NRC lexicon cache. Note that `prosody` doesn't *require* `emotion`/
`focus` either (only `linguistic`) — request them explicitly alongside it
if you want arousal-driven rate/pitch/energy or local focus boosts; on
their own they degrade gracefully to the neutral baseline (`1.0`),
documented via the `"no_emotion_data"` rule. `interjections` and `context`
*do* require `emotion` (`context` also reads `emotion.distribution`), so
requesting either pulls in the NRC lexicon cache, even though
`interjection_mode` defaults to `"disabled"` and context only ever adds a
separate `Sentence.context_emotion` field alongside the untouched local one.
Serialization (`serializers.py`, Phase 12) isn't a registry layer — it's a
post-hoc transform of the finished `PreprocessResult`, invoked via
`--serialize` or the Python API, not `--include`.

Profiles under `configs/preprocess/`: `default.yaml` (normalized +
linguistic + lexical_stress + emotion + focus + prosody + interjections +
context — everything currently implemented; safe by construction, since
`interjection_mode` still defaults to `"disabled"` and `context` never
overwrites the local prediction), `normalize_only.yaml` (just normalized
text), `pronunciation.yaml` (readme.md §1.3: normalized text, phonemes,
syllables, lexical stress), and `expressive.yaml` (readme.md §1.3: emotion
+ intensity + focus + prosody + interjections + context — everything
expressive-related implemented so far). More profiles (`tts-training`,
...) will be added as later phases land.

## Linguistic analysis cache

`linguistic.LinguisticProcessor` caches Trankit's raw output on disk under
`.cache/linguistic/` (gitignored), keyed by `sha256` of the normalized
sentence text, per objectives.md Phase 3 ("cache annotations by
normalized-text hash"). Delete that directory to force re-analysis (e.g.
after a Trankit model upgrade).

## Normalization dictionaries

`configs/preprocess/{units,currencies,abbreviations}.yaml` are plain
dictionaries the normalizer loads by default — edit them directly to add
vocabulary rather than changing code. Unit/currency entries carry a
`gender` (`masculine`/`feminine`/`neuter`) used to pick the correct Romanian
numeral form (`numbers_ro.count_phrase`): `neuter` nouns like *kilogram*
take the masculine article in the singular ("un kilogram") but feminine
numeral agreement in the plural ("două kilograme").

## Phonemization and stress: methodology notes

- **Phoneme inventory** (`configs/preprocess/phoneme_inventory.yaml`) was
  built empirically: `espeak -v ro --ipa` was run over the ~9,590 distinct
  words in `datasets/mara` and `datasets/hria/catalina`, and every distinct
  IPA symbol actually produced was collected — not guessed. Anything the
  phonemizer emits outside that set is flagged as an unknown phoneme.
- **Syllabification** (`stress.syllabify`) is a deterministic
  grapheme-based approximation (vowel-nucleus runs + Romanian
  consonant-cluster rules). It's verified against objectives.md's own
  worked example (`"fericire"` → `["fe","ri","ci","re"]`, stress index 2)
  plus several others — see `tests/unit/test_stress.py`. Known limitation:
  it can't always distinguish a true diphthong from a hiatus at a morpheme
  boundary (e.g. "reuși" is treated as one diphthong-bearing syllable "reu"
  plus "șit", which happens to be correct here, but the general rule isn't
  exception-free).
- **Stress alignment**: espeak's `ˈ` mark is matched to a grapheme syllable
  by counting IPA vowel-nucleus groups, not by character position — see
  `stress.stressed_group_index`. Known unresolved case: `"copii"` is a
  homograph ("children", co-**PII**, vs "copies", **CO**-pii); espeak has no
  sentence context so it always returns one reading, and no blanket
  override is applied since that would just break the other sense (see
  `configs/preprocess/stress_overrides.yaml`).
- **Pronunciation overrides**
  (`configs/preprocess/pronunciation_overrides.yaml`) currently has one
  verified real entry: `TVA` (the VAT acronym) is read by espeak as one
  made-up word instead of spelled out letter-by-letter ("te-ve-a").

## Emotion classifier: methodology notes

The emotion layer is a transformer classifier (it replaced the earlier
rule-based NRC-lexicon baseline; the shared NRC lexicon loaders it used to
own now live in `lexicons.py`, still consumed by the focus layer).

- **Model**: `tabularisai/multilingual-emotion-classification`, a
  multilingual XLM-RoBERTa emotion classifier, run locally through
  HuggingFace `transformers` — free and offline after a one-time download
  (`scripts/download_emotion_model.py`, cached under
  `.cache/models/emotion_transformer/`). Chosen because its label set maps
  almost 1:1 onto this project's, and XLM-R covers Romanian.
- **Label aggregation**: the model's fine-grained classes are summed into
  this project's 6 labels (`joy/gratitude/love→happy`,
  `anger/frustration/contempt/disgust→angry`, `sadness→sad`, `fear→fear`,
  `surprise→surprise`, `neutral→neutral`) and re-normalized; `argmax` picks
  the label.
- **Abstention**: a prediction resolves to `unspecified` when the top
  label's aggregated probability is below `MIN_CONFIDENCE` (0.40) or ties
  another label — low-confidence predictions are marked rather than
  silently accepted (objectives.md Phase 6). `neutral` (a positive model
  prediction) stays distinct from `unspecified` (our abstention).
- **Valence/arousal**: distribution-weighted average over per-emotion VAD
  prototypes (`VAD_PROTOTYPES`, NRC 0-1 scale) — coarse category anchors,
  not per-word lexicon values, so V/A vary smoothly with model uncertainty.
- **Punctuation** (repeated `!`/`?`, ALL-CAPS) only nudges the *intensity*
  computation, never which category wins — objectives.md Phase 6's
  "supporting, not decisive" instruction, preserved from the baseline.
- **Provenance/evidence**: predictions are `Provenance.PREDICTED` (a model
  output, not a rule), producer `emotion_xlmr_v1`; evidence records the
  model id rather than per-token spans, since the transformer isn't
  span-explainable.
- **Testing**: `EmotionProcessor` takes an injectable `predictor` (config
  key `emotion_predictor`, a `str -> dict[label, prob]` callable) so unit
  tests exercise aggregation/thresholding/VAD with a deterministic fake and
  never download or run the ~1.1GB model.
- **Evaluation**: `objective_evaluation.evaluate_emotion` measures macro
  F1/per-class F1/confusion matrix/coverage against the gold-labeled
  (`provenance="source"`) subset of `data/evaluation/`. (The earlier
  rule-baseline numbers in `data/evaluation/emotion_baseline_report.md`
  predate this switch; re-run the evaluator with the models installed to
  get the transformer's figures.)

## Focus detection: methodology notes

- **Distinct from lexical stress**: `stress.py` marks *which syllable
  within a word* is stressed (Phase 5); `focus.py` marks *which words in
  the sentence* carry emphasis (Phase 8) — separate `Token` fields
  (`stressed_syllable_index` vs. `focus_score`/`is_focus`), per
  objectives.md's explicit requirement that the two not be conflated.
- **Signals are additive and self-documenting**: every contribution (ALL
  CAPS, intensifier target, repetition, emotion-bearing word, contrastive
  correction, main predicate) appends its own rule name to
  `Token.focus_rules`, so a decision is always traceable to specific
  evidence.
- **Contrastive corrections** ("Nu e roșu, ci albastru." — "It's not red,
  but blue.") de-emphasize the corrected span and boost the corrective
  one — verified directly against a constructed example (see
  `tests/integration/test_pipeline_focus.py`).
- **Function words need strong evidence**: tokens tagged
  `DET`/`ADP`/`CCONJ`/`SCONJ`/`AUX`/`PART`/`PUNCT` are zeroed out unless
  their accumulated score already clears a high bar — objectives.md:
  "Prevent function words from receiving focus without strong evidence."
- **User-provided emphasis has unconditional priority**: pass
  `user_focus_words` to `PreprocessPipeline.process()` (Python API only,
  no CLI flag yet) to force specific words to `is_focus=True`,
  `Provenance.USER`, bypassing every rule above — objectives.md: "100%
  priority over predictions."
- **The NRC lexicon is optional here**, unlike `emotion.py`: if
  `.cache/lexicons/` hasn't been fetched, the emotion-bearing-word signal
  is silently skipped rather than raising — focus detection shouldn't
  hard-require a 109MB download for the other signals to work.
- **Sanity-checked, not gold-evaluated**: `objective_evaluation.evaluate_focus`
  compares predictions against `data/evaluation/*.jsonl`'s `focus_words`
  field, but that field is my own quick Phase 1 draft judgment, not an
  independent gold annotation. Measured token-level F1 was low (0.11 dev,
  0.08 test) — investigated and traced to a real conceptual mismatch, not
  a bug: the rule engine targets *emphasis* (objectives.md's actual
  definition) while the Phase 1 drafts were closer to *topical salience*
  ("what's this sentence about"), a related but different notion. See
  `data/evaluation/focus_sanity_check.md` for the full writeup — not
  tuned to chase these numbers, since that would mean fitting to my own
  possibly-mistaken quick annotations instead of the actual target
  concept.

## Prosody: methodology notes

- **Model-independent control values, not acoustic targets**: `speaking_rate`/
  `relative_pitch`/`relative_energy` are multipliers around a `1.0` baseline,
  clamped to documented safe ranges (`0.80-1.20`, `0.85-1.20`, `0.80-1.20`;
  pauses `0-1000ms`) — objectives.md is explicit that these are
  "intermediate control values, not guaranteed acoustic outputs". Verified
  directly: 229 real sentences sampled from `datasets/hria`/`datasets/mara`
  produced zero range violations.
- **Terminal contour is a direct, not inferred, mapping** from
  `sentence_type` (Phase 3): `interrogative→rising`,
  `declarative`/`exclamative`/`imperative`→`falling`,
  `incomplete→continuation`. No new model or heuristic needed — the
  sentence-type classification already *is* the rule.
- **Clause boundaries use real dependency structure**, not just
  punctuation (objectives.md's explicit instruction): tokens with
  `deprel` in `{"mark","advcl","ccomp"}` (subordinating conjunctions,
  adverbial/complement clauses — e.g. "că", "dacă", "când") get a
  `pause_before_ms`, independent of the punctuation-driven pauses on comma/
  semicolon/colon/period/`!`/`?`/ellipsis tokens.
- **Arousal drives rate/pitch/energy with one symmetric formula**:
  `value = 1.0 + (arousal - 0.5) * 0.4`, then clamped — implements
  "increase energy/pitch for high-arousal, reduce rate/energy for
  low-arousal" directly. Falls back to the `1.0` baseline (tagged
  `"no_emotion_data"`) when `sentence.emotion`/`arousal` isn't available —
  `prosody` doesn't require `emotion` to run, so this is the normal case
  unless both are requested together.
- **Focus is applied locally, per token, not to the whole sentence**
  (objectives.md's specific instruction for this phase): `is_focus=True`
  tokens get `local_relative_pitch`/`local_relative_energy` — the sentence
  baseline further boosted by `focus_score`, independently clamped;
  non-focused tokens leave these `None` rather than repeating the baseline.
- **User-provided values are preserved verbatim**: `user_prosody_overrides`
  (keyed by 0-based sentence index) bypasses the rule engine field-by-field
  — a partial override (e.g. just `terminal_contour`) leaves the other
  fields rule-computed, each still traceable via `ProsodyAnnotation.rules`
  (`"user_override_<field>"` vs. the rule-derived ones).
- **No dedicated evaluation script**: unlike Phase 6/8,
  `data/evaluation/*.jsonl`'s `pause_locations` field is mechanically
  derived from punctuation offsets by the Phase 1 build script itself, not
  an independent annotation of where a human would actually pause, and
  carries no duration information — comparing against it would measure
  "does our punctuation list match our punctuation list," not real
  accuracy. Tracked as follow-up (needs aligned audio or dedicated human
  annotation) rather than faked with a script over noise — see
  `data/evaluation/README.md`.

## Interjection suggestions: methodology notes

- **Disabled by default, by construction**: `interjection_mode` defaults
  to `"disabled"` on `PreprocessPipeline.process()` — objectives.md's
  acceptance criterion is satisfied structurally, not by a runtime check
  that could regress. Requesting the `interjections` layer alone (e.g. via
  `default.yaml`) never produces suggestions unless the caller explicitly
  passes `interjection_mode="suggest"`/`"insert"`.
- **`Sentence.text` is never modified**: `"insert"` mode writes the
  enriched variant to the separate `Sentence.text_with_interjections`
  field — objectives.md: "Preserve the unmodified text."
- **Document style is a deliberately coarse binary**
  (`"formal"`/`"conversational"`), not the full academic/legal/technical/
  formal taxonomy objectives.md's prose mentions — without labeled
  formality data, a finer split would be unfounded guessing, and the
  actual feature requirement only needs a binary gate. A hit against
  `configs/preprocess/formal_markers.yaml` (grounded in real phrases from
  `data/evaluation/`'s formal/legal examples) is treated as authoritative
  on its own; three weaker signals (long average sentence length, no 1st/
  2nd-person markers, no existing interjections) only count as "formal" if
  *all three* agree. **Found and fixed a real bug this way**: a flat
  equal-weighted average let one fragile signal (Romanian is pro-drop, and
  the parser mis-tags the syncretic verb form "sunt" as 3rd-person-plural
  instead of 1st-person-singular with no explicit subject to
  disambiguate) combine with a weak, almost-always-true one ("no existing
  interjection") to misclassify "Sunt extrem de fericit!" as formal.
- **This also fixed a pre-existing, unrelated bug**: `document_style` has
  been a real field on `PreprocessResult`/`PipelineDocument` since Phase 0
  but `pipeline.py`'s `_to_result` never actually mapped it through — it
  was always `None` regardless of what any processor set. Fixed as part
  of wiring this phase in, since otherwise `document_style` would still be
  unobservable.
- **Gating, in order**: no suggestions if `interjection_mode == "disabled"`,
  if `document_style == "formal"` (0% formal-text violations — see
  `data/evaluation/interjection_evaluation_report.md`), if the sentence
  already contains an existing interjection (reuses Phase 6's
  dictionary-based detection — "prevent repeated interjections"), or if
  `sentence.emotion` is missing or below a confidence threshold (0.6,
  documented, not gold-tuned).
- **Candidates**: reverse lookup of
  `configs/preprocess/interjection_emotions.yaml` for entries matching
  `sentence.emotion.label`, ranked by `weight * emotion.confidence`,
  capped at 2 — "at most a small number of ranked candidates."
- **Evaluated against real acceptance criteria and a non-gold sanity
  check**: `objective_evaluation.evaluate_interjections` — the formal-text violation
  rate (0% on both dev/test, a genuine objectives.md acceptance criterion,
  not just a sanity check) plus a softer comparison against the draft
  `interjection_appropriate` field (same non-gold caveat as
  `focus_sanity_check.md`). See
  `data/evaluation/interjection_evaluation_report.md` for the full
  writeup, including two explainable disagreement patterns (existing-
  interjection cases where the draft answered a different question than
  "should the system suggest *another* one"; and the Phase 6 emotion
  tie-breaking conservatism cascading into fewer suggestions here too).

## Context smoothing: methodology notes

- **The local prediction is never discarded**: `context.ContextProcessor`
  only ever writes a separate `Sentence.context_emotion` field;
  `Sentence.emotion` (Phase 6) is untouched, always — objectives.md:
  "Local predictions are never discarded."
- **Blends `EmotionAnnotation.distribution`, not a single scalar**: the
  objectives.md baseline formula
  (`context_score = α·current + β·previous`, α>β) is implemented over
  Phase 6's already-computed per-category evidence weights, so the blend
  can still produce a real label via argmax rather than an opaque score.
  α=0.7/β=0.3 by default — selected by inspection on `dev.jsonl`/
  `context.jsonl`, not gold-tuned.
- **"Prevent a neutral sentence from erasing a strong explicit emotion"**:
  implemented as skipping blending entirely once the *local* confidence is
  already high (≥0.75) — a confident sentence keeps its own reading
  regardless of a weaker neighbor.
- **Discourse markers** (`configs/preprocess/discourse_markers.yaml`,
  checked against the sentence's leading content token only): a contrast
  marker (dar/însă/totuși/ci) zeroes the previous-sentence weight entirely
  — a deliberate tonal break shouldn't be smoothed over; a consequence
  marker (deci/așadar/prin urmare/astfel) slightly raises it.
- **Chains off each sentence's local prediction, not the previous
  sentence's smoothed one** — otherwise smoothing error would compound
  across a long paragraph instead of each sentence being pulled toward
  what was actually observed next to it (verified via a regression test
  that constructs a case where the two chaining strategies diverge:
  `tests/unit/test_context.py::test_processor_chains_off_local_not_smoothed_prediction`).
- **Paragraph boundaries are a real, if approximate, signal**: `SentenceSpan`
  offsets are relative to `document.clean_text`, which has already
  collapsed all whitespace (including blank lines) to single spaces by the
  time this runs — so a blank line can't be located by character offset
  anymore. Instead, the *original* source text is split on blank lines and
  each chunk's sentence count is estimated from terminal punctuation, then
  walked cumulatively to find which span starts each paragraph. Verified
  directly: a paragraph break correctly stops a strong "happy" sentence
  from pulling a following low-confidence sentence toward it
  (`tests/integration/test_pipeline_context.py::test_paragraph_boundary_resets_context_in_real_pipeline`).
- **Not evaluated as macro F1**: `context.jsonl` carries no per-sentence
  gold emotion labels, so `objective_evaluation.evaluate_context` is a behavior
  sanity check (local always preserved, changes always traceable, real
  change rate reported: 25.0% of 212 sentences across 43 paragraphs) —
  objectives.md's "must improve or preserve macro F1" acceptance criterion
  is left honestly unchecked for this reason. See
  `data/evaluation/context_evaluation_report.md`, including one flagged
  case where a change traces to an inherited Phase 6 imprecision rather
  than a new defect.

## Serialization: methodology notes

- **Not a registry layer**: `serializers.py`'s functions take an
  already-built `PreprocessResult` and transform it — they don't add a
  field to `SentenceSpan`/`PipelineDocument` the way every other processor
  does, so they don't fit the `Processor` protocol and aren't registered.
- **Every unresolvable value maps to an explicit `..._UNSPECIFIED` token**
  rather than being silently dropped (`to_control_tokens`) — this is what
  keeps the unknown-control-token rate at 0% *by construction*, verified
  at scale (`objective_evaluation.evaluate_serialization`: 100% serialization
  success, 0% unknown tokens, 0 determinism violations across all 304
  `dev`+`test` examples).
- **Continuous values are quantized** (rounded to the nearest 0.05) before
  formatting — objectives.md: "continuous values should initially be
  quantized into bins." `parse_control_tokens` round-trips to that
  *quantized* representation, not the original unrounded float — documented
  as a real limitation, not glossed over, since objectives.md only asks for
  reversibility "where possible."
- **Prefers `token.phonemes` over surface text** when the `phonemes` layer
  ran, for the control-token format specifically (TTS-facing); the
  human-readable annotated-text format always uses surface text.
- **`to_control_tokens` prefers `Sentence.context_emotion` over
  `Sentence.emotion`** when both are present, since it represents the
  paragraph-smoothed final prediction — falls back to the local one via
  `getattr(..., None)`, so this works whether or not the `context` layer
  ran (and worked correctly before `context_emotion` existed on the schema
  at all, since Phase 12 was built before Phase 11 in this pass).
- **SSML output is explicitly illustrative**, not validated against the
  real SSML schema — objectives.md marks it optional.

## Tests

```bash
./.venv/bin/pytest tests/
```

`tests/unit/` covers each processor in isolation with pure-function tests
that need neither Trankit nor `espeak`; tests that exercise the real model
or binary are marked `skipif` and skip cleanly if either isn't installed.
`tests/integration/` exercises the full pipeline, including the worked
examples from `objectives.md` Phase 2 and 5, a 0%-unknown-phoneme check,
and a schema round-trip through JSON.
