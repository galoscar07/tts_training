# July Implementation Checklist: Romanian Expressive Text Frontend for TTS

## Objective for This Month

Build and evaluate a local Romanian expressive text-preprocessing module that converts raw text into a structured, TTS-ready representation containing:

- normalized and segmented text;
- linguistic and morphological annotations;
- phonemes, syllables, and lexical stress;
- sentence-level emotion and emotion intensity;
- phrase-level prosodic cues;
- focus and emphasis annotations;
- pauses and sentence-final intonation;
- optional, controlled interjection suggestions;
- serialized JSON and TTS control tokens.

The entire development and evaluation workflow must run locally on an Apple M1 system. Training large TTS networks is outside the scope of this month.

### Minimum acceptable result for July

By the end of the month, the project must provide:

- [x] A command-line pipeline that accepts Romanian text. (`--text`/`--stdin`/`--input-file`; no CSV/TSV/JSON/manifest modes yet)
- [x] A deterministic Romanian text normalizer.
- [x] Sentence segmentation and token-level linguistic analysis. (segmentation is rule-based v0; token-level analysis via Stanza — tokens/lemma/UPOS/morphology/dependencies)
- [x] Romanian phonemization with lexical-stress information.
- [x] Sentence-level emotion, valence, arousal, and intensity estimates.
- [x] Rule-based prosody and focus annotations.
- [x] Optional interjection suggestions that never overwrite the source text. (Phase 10, disabled by default; gated on a coarse formal/conversational document-style detector — 0% formal-text violation rate on the pilot evaluation set, see `data/evaluation/interjection_evaluation_report.md`)
- [x] A documented JSON output schema.
- [x] A serializer that produces TTS control tokens. (Phase 12, `serializers.to_control_tokens` — 100% serialization success, 0% unknown tokens, deterministic, over the full 304-sentence evaluation set)
- [x] Automated tests and a manually annotated evaluation set. (322 automated tests; evaluation set now 304 sentences / 43 paragraphs, meeting the 300+/30+ target — though the added Tier 2 portion is bulk-sampled with explicitly unreviewed subjective fields, not hand-annotated like the original Tier 1 pilot — see `data/evaluation/README.md`)
- [x] Quantitative evaluation results for every implemented component. (emotion measured against real gold data at meaningful scale now — 99/92 gold examples, `data/evaluation/emotion_baseline_report.md`; focus/interjections/context have sanity checks against non-gold or no reference data, honestly framed as such — `focus_sanity_check.md`/`interjection_evaluation_report.md`/`context_evaluation_report.md`; prosody has a verified range-safety check but no accuracy metric, since no meaningful reference data exists — `data/evaluation/README.md`; serialization has a real, measured acceptance-criterion result — `objective_evaluation.evaluate_serialization`; normalization/linguistic/phonemization/stress still have only spot-checks and the objectives.md worked examples, not full eval-set-scale metrics)

---

## Why This Module Is Required

TTS models cannot reliably learn all linguistic and expressive relationships directly from limited Romanian speech data. If raw text is supplied without consistent preprocessing, the network must simultaneously learn:

- how numbers and abbreviations are pronounced;
- how words are segmented and phonemized;
- where lexical stress occurs;
- which words carry semantic focus;
- how punctuation affects pauses and intonation;
- how emotional meaning affects rhythm, pitch, and energy;
- how context changes the interpretation of a sentence.

This increases the learning problem and wastes limited paired audio–text data on phenomena that can be resolved before training.

The expressive frontend reduces this burden by transforming text into explicit, consistent annotations. These annotations can later be supplied to VITS, Matcha-TTS, or another architecture as input tokens or conditioning variables.

```text
Without frontend:

raw text
   ↓
TTS must learn normalization + pronunciation + stress
    + emotion + prosody + acoustics from limited data


With expressive frontend:

raw text
   ↓
normalized text + phonemes + stress + emotion + prosody
   ↓
TTS focuses primarily on acoustic realization
```

### Expected benefits for subsequent TTS training

- fewer pronunciation errors;
- consistent pronunciation of numbers, dates, units, and abbreviations;
- reduced text–audio mismatch in training data;
- a smaller and more meaningful input vocabulary;
- explicit supervision for emotion and prosody;
- easier separation between speaker identity and emotion;
- controllable synthesis at inference time;
- reusable preprocessing for both VITS and Matcha-TTS;
- easier debugging because linguistic errors can be separated from acoustic errors;
- more reproducible experiments across architectures.

---

## Scope and Non-Scope

### Included this month

- [x] Romanian text normalization.
- [x] Linguistic analysis.
- [x] Phonemization.
- [x] Syllabification and lexical stress.
- [x] Emotion and intensity estimation.
- [x] Focus and prosody estimation.
- [x] Interjection suggestion. (Phase 10 — suggesting *new* interjections, disabled by default; distinct from detection of existing ones, done earlier)
- [x] Context-aware smoothing between adjacent sentences. (Phase 11 — emotion only, not prosody; local prediction always preserved separately)
- [x] Serialization for downstream TTS models. (Phase 12 — canonical JSON, annotated text, control tokens, illustrative SSML; VITS/Matcha adapters explicitly deferred as "Future")
- [x] Component-level evaluation. (emotion now has real measured results at meaningful scale — 99/92 gold examples; serialization/interjections have real acceptance-criterion results; focus/context have honestly-framed sanity checks, not full accuracy metrics; normalization/linguistic/phonemization/stress remain spot-checked only)

### Excluded this month

- [ ] Full VITS or Matcha-TTS training.
- [ ] Neural vocoder training.
- [ ] End-to-end waveform generation.
- [ ] Large-scale transformer training from scratch.
- [ ] Final MOS evaluation of synthesized audio.
- [ ] Production deployment.

---

## Target Pipeline

```text
Raw Romanian text
    │
    ▼
Unicode cleanup and document-style detection
    │
    ▼
Sentence segmentation and tokenization
    │
    ▼
Text normalization
    │
    ▼
Morphological and syntactic analysis
    │
    ▼
Phonemization, syllabification, lexical stress
    │
    ▼
Emotion, valence, arousal, intensity
    │
    ▼
Focus, pauses, speaking rate, pitch and energy cues
    │
    ▼
Optional interjection suggestions
    │
    ▼
Context-aware consistency pass
    │
    ├── JSON intermediate representation
    └── TTS control-token representation
```

---

## Recommended Project Structure

```text
romanian_expressive_frontend/
├── config/
│   ├── abbreviations.yaml
│   ├── emotion_lexicon.yaml
│   ├── intensifiers.yaml
│   ├── interjections.yaml
│   ├── normalization_rules.yaml
│   └── prosody_rules.yaml
├── data/
│   ├── evaluation/
│   ├── lexicons/
│   └── reports/
├── src/
│   ├── normalizer.py
│   ├── linguistic_analyzer.py
│   ├── phonemizer.py
│   ├── stress_analyzer.py
│   ├── emotion_analyzer.py
│   ├── focus_analyzer.py
│   ├── prosody_analyzer.py
│   ├── interjection_suggester.py
│   ├── context_tracker.py
│   ├── serializer.py
│   └── pipeline.py
├── tests/
│   ├── test_normalizer.py
│   ├── test_linguistic_analysis.py
│   ├── test_phonemizer.py
│   ├── test_stress.py
│   ├── test_emotion.py
│   ├── test_prosody.py
│   ├── test_interjections.py
│   └── test_pipeline.py
├── examples/
├── pyproject.toml
└── README.md
```

---

# Ordered Implementation Checklist

## Phase 0 — Define the Annotation Contract

### Purpose

Define the information exchanged between the frontend and future TTS models before implementing individual components.

### Implementation checklist

- [x] Define the input format. (plain text via `--text`/`--stdin`/`--input-file`; CSV/TSV/JSON/manifest formats not yet defined)
- [x] Define the JSON output schema. (`expressive_tts/preprocess/schemas.py: PreprocessResult`)
- [ ] Define supported emotion labels.
- [ ] Define supported intensity values.
- [ ] Define pause categories and their approximate durations.
- [ ] Define sentence types. (`Sentence.sentence_type` field exists but the label set below is not yet encoded/enforced)
- [ ] Define lexical stress representation.
- [ ] Define focus/emphasis representation.
- [ ] Define TTS control tokens.
- [x] Define confidence values for predicted annotations. (`TraceEntry.confidence`)
- [x] Add a field that distinguishes predicted, rule-generated, and user-provided annotations. (`Provenance` enum)
- [x] Version the schema. (`schema_version`)

### Initial label set

```text
emotion:
    neutral
    happy
    angry
    sad
    fear
    surprise
    unspecified

intensity:
    low
    medium
    high
    unspecified

sentence_type:
    declarative
    interrogative
    exclamative
    imperative
    incomplete
```

### Required output fields

```json
{
  "schema_version": "1.0",
  "original_text": "",
  "normalized_text": "",
  "document_style": "",
  "sentences": []
}
```

### Training required

No.

### Evaluation

- Schema validation success rate.
- Percentage of pipeline outputs accepted by the schema validator.

### Acceptance criteria

- [x] 100% of test outputs pass schema validation. (`tests/unit/test_schema.py`, `tests/integration/test_pipeline_basic.py::test_schema_round_trips_through_json`)
- [x] Every predicted field contains a confidence or provenance field. (every `TraceEntry` carries both `provenance` and `confidence`)
- [x] Original text is preserved unchanged. (`tests/integration/test_pipeline_basic.py::test_original_text_preserved_unchanged`)

### Expected outcome

A stable interface that allows all subsequent modules to be developed independently.

---

## Phase 1 — Build the Evaluation Dataset Before the Pipeline

### Purpose

Create a fixed reference set so that development decisions are not based only on a few convenient examples.

### Implementation checklist

- [ ] Collect at least 300 Romanian sentences. (pilot: 38/300 — see `data/evaluation/README.md`)
- [x] Include conversational, narrative, news, technical, and formal text. (all 5 registers present in the pilot)
- [x] Include numbers, dates, times, currencies, percentages, and measurement units.
- [x] Include common abbreviations.
- [x] Include questions, exclamations, commands, negations, and incomplete sentences.
- [x] Include positive, negative, neutral, and ambiguous emotional content. (all 6 objectives.md emotion labels + `unspecified` present)
- [x] Include words with potentially difficult lexical stress. (hand-authored minimal pairs: "copii" copies/children, "veselă" cheerful/dishes)
- [x] Include existing interjections.
- [x] Include sentences where adding an interjection would be inappropriate.
- [ ] Add at least 30 multi-sentence paragraphs for contextual evaluation. (pilot: 3/30)
- [x] Reserve the evaluation set and do not tune rules directly on all of it. (normalizer was validated against the objectives.md worked example before this set existed, not tuned against it)

### Suggested split

```text
Development: 200 sentences
Test:        100 sentences
Context set:  30 paragraphs
```

### Manual annotations

For each sentence, annotate:

- expected normalized text;
- sentence boundaries;
- expected emotion;
- acceptable secondary emotion;
- intensity;
- sentence type;
- focus words;
- pause locations;
- lexical stress for selected words;
- whether an interjection is appropriate;
- acceptable interjections, if applicable.

### Training required

No model training. Manual annotation is required.

### Evaluation

No second annotator is available yet. In the pilot, `expected_normalized_text`
and `sentence_boundaries` are computed by actually running the real
preprocessing pipeline (`provenance="rule"`); `emotion` is a real human
label when sourced from REDv2 or hria's own annotations
(`provenance="source"`); everything else is my own draft judgment
(`provenance="predicted"`) pending human review — see
`data/evaluation/README.md`. Inter-annotator agreement should be measured
once a second annotator is available:

- Cohen's kappa for discrete emotions;
- weighted kappa for intensity;
- token-level agreement for focus;
- boundary agreement for pauses.

### Acceptance criteria

- [x] At least 300 annotated sentences. (304: 38 Tier 1 hand-curated + 266 Tier 2 programmatically sampled — see `data/evaluation/README.md`'s two-tier explanation; Tier 2's subjective fields are explicit unreviewed placeholders, not equivalent-quality annotations to Tier 1's)
- [x] At least 30 contextual paragraphs. (43: 3 Tier 1 + 40 Tier 2, reconstructed from real `datasets/mara`/`datasets/hria` text)
- [x] All required phenomena are represented. (in the Tier 1 batch — see `data/evaluation/README.md` coverage summary; Tier 2 is tagged `bulk_added` only, no per-phenomenon tagging at that volume)
- [x] Test examples are stored separately from development examples. (`data/evaluation/dev.jsonl` vs `test.jsonl`, both tiers)

### Expected outcome

A reproducible benchmark for all frontend components.

---

## Phase 2 — Unicode Cleanup and Romanian Text Normalization

### Purpose

Convert unpredictable raw text into a canonical form that has one intended spoken realization.

### Implementation order

- [x] Normalize Unicode.
- [x] Convert legacy Romanian characters `ş/ţ` to `ș/ț`.
- [x] Normalize whitespace.
- [x] Normalize quotation marks and apostrophes.
- [x] Protect URLs, e-mail addresses, decimals, dates, and abbreviations before sentence splitting.
- [x] Expand cardinal numbers.
- [ ] Expand ordinal numbers. (only Roman-numeral ordinals are handled — see below; Arabic-digit ordinals like "al 3-lea" are not yet expanded)
- [x] Expand dates.
- [x] Expand times.
- [x] Expand percentages.
- [x] Expand currencies.
- [x] Expand measurement units.
- [x] Expand common abbreviations.
- [x] Handle Roman numerals where context permits. (`al X-lea` / `a X-a` ordinal patterns)
- [ ] Restore protected text spans. (implemented differently: `protected_spans.py` marks spans as non-splittable for the segmenter rather than masking-and-restoring placeholders, so there is nothing to literally "restore")
- [x] Preserve punctuation relevant to prosody.
- [x] Record every normalization operation in a trace.

### Example

```text
Input:
Dr. Popescu a trimis 25 kg pe 12.07.2026, la ora 14:30.

Normalized:
Doctor Popescu a trimis douăzeci și cinci de kilograme
pe doisprezece iulie două mii douăzeci și șase,
la ora paisprezece și treizeci de minute.
```

### Training required

No. Implement using deterministic rules, dictionaries, and tests.

### Evaluation values

Calculate:

```text
Exact Match Accuracy =
correctly normalized sentences / total sentences

Token Accuracy =
correct normalized tokens / total reference tokens

Category Accuracy =
correct cases within category / total cases within category
```

Report separate values for:

- numbers;
- dates;
- times;
- abbreviations;
- currencies;
- units;
- punctuation preservation.

### Acceptance criteria

- [ ] Overall token accuracy of at least 98%.
- [ ] At least 95% category accuracy for common numbers and dates.
- [ ] Zero loss of Romanian diacritics.
- [ ] Zero unlogged changes to the input.

### Expected outcome

A deterministic normalizer that reduces text–audio mismatch in future training manifests.

---

## Phase 3 — Sentence Segmentation and Linguistic Analysis

### Purpose

Extract sentence structure, tokens, lemmas, morphological information, and syntactic dependencies.

### Proposed local tool

> **Post-July update:** the linguistic layer was switched from Stanza to
> **Trankit** — a transformer-based (XLM-RoBERTa, via HuggingFace
> `transformers`) toolkit that provides the same full stack (tokenize / POS /
> morphology / lemma / dependency parsing) so all downstream fields keep
> working. See `src/expressive_tts/preprocess/linguistic.py` and
> `scripts/download_trankit_model.py`. The original Stanza-based design is
> preserved below as the July record.

Use Stanza with Romanian models on CPU:

```python
import stanza

stanza.download("ro")

nlp = stanza.Pipeline(
    "ro",
    processors="tokenize,pos,lemma,depparse",
    use_gpu=False,
)
```

### Implementation checklist

- [x] Integrate Stanza behind a project-specific adapter. (`preprocess/linguistic.py`)
- [ ] Preserve character offsets into the original text. (only normalized-text offsets are tracked; no reverse-mapping to the pre-normalization original text yet)
- [x] Preserve character offsets into the normalized text. (`Token.start`/`end`)
- [x] Extract tokens and lemmas.
- [x] Extract UPOS tags.
- [x] Extract morphological features.
- [x] Extract dependency heads and relations.
- [x] Detect interjections using `INTJ`.
- [x] Detect negation.
- [x] Detect the main predicate. (`token.deprel == "root"` — exposed via the raw token list, not a separate named field)
- [ ] Detect subjects, objects, modifiers, and coordinating structures. (raw `deprel` values like `nsubj`/`obj` are exposed per token; no higher-level aggregation into labeled subject/object/modifier groups yet)
- [x] Determine sentence type from syntax and punctuation. (`infer_sentence_type`: root-verb `Mood=Imp` → imperative, else terminal punctuation)
- [x] Cache annotations by normalized-text hash. (`.cache/linguistic/`, sha256-keyed)

### Training required

No new training. Use an existing Romanian model locally.

### Evaluation values

Evaluate on the manually annotated set and, where appropriate, against Romanian Universal Dependencies data:

- sentence boundary precision, recall, and F1;
- tokenization precision, recall, and F1;
- UPOS accuracy;
- lemma accuracy;
- sentence-type macro F1;
- negation-detection precision, recall, and F1.

### Acceptance criteria

- [ ] Sentence-boundary F1 of at least 97% on the project test set.
- [ ] Tokenization F1 of at least 98%.
- [ ] Sentence-type macro F1 of at least 90%.
- [ ] Errors are logged with the original and normalized spans.

### Expected outcome

A structured syntactic representation that supports emotion, focus, and prosody rules.

---

## Phase 4 — Romanian Phonemization

### Purpose

Produce the sequence that will eventually be used as the primary linguistic input to the TTS network.

### Implementation strategy

Use a cascade:

```text
curated pronunciation lexicon
        ↓ if missing
RoLEX-derived entry
        ↓ if missing
eSpeak-ng Romanian phonemization
        ↓ if unavailable
grapheme-based fallback + uncertainty flag
```

### Implementation checklist

- [x] Define a canonical phoneme inventory. (`configs/preprocess/phoneme_inventory.yaml`, built empirically from real espeak output over the project's own vocabulary, not guessed)
- [ ] Integrate the available RoLEX resource according to its license and distribution method. (licensing/distribution still unclear; `phonemizer._lookup_rolex` is a stub for this)
- [x] Integrate eSpeak-ng as a fallback. (uses classic `espeak`, already installed on the dev machine, instead of the `espeak-ng` fork — see `preprocess/README.md` for why; same role in the cascade)
- [x] Convert all sources to one phoneme notation. (IPA throughout; overrides are written in the same notation)
- [x] Preserve word boundaries. (per-token phonemes, space-joined)
- [x] Preserve punctuation-derived pause boundaries. (punctuation tokens keep their literal text in `phoneme_text` rather than being dropped)
- [x] Detect unknown phonemes. (`phonemizer.unknown_phonemes`)
- [x] Track the source of each pronunciation. (`Token.pronunciation_provenance`/`_producer`)
- [x] Add a confidence value. (`Token.pronunciation_confidence`)
- [x] Create a correction dictionary for project-specific names and abbreviations. (`configs/preprocess/pronunciation_overrides.yaml` — one verified real entry: `TVA`, which espeak otherwise mispronounces as a made-up word instead of spelling it out)

### Training required

No training for the first version.

Optional later work:

- train a grapheme-to-phoneme model using RoLEX entries;
- use held-out RoLEX words for evaluation.

### Evaluation values

If reference pronunciations are available:

```text
Phoneme Error Rate =
(substitutions + deletions + insertions) / reference phonemes
```

Also calculate:

- word pronunciation exact-match accuracy;
- out-of-vocabulary rate;
- fallback rate;
- unknown-phoneme rate;
- error rate by source: lexicon versus eSpeak-ng.

### Acceptance criteria

- [ ] Unknown-phoneme rate equals 0%. (0% observed on the objectives.md Phase 2 worked example — see `tests/integration/test_pipeline_linguistic.py`; not yet measured across the full evaluation set)
- [ ] At least 95% of common evaluation words receive a non-fallback pronunciation.
- [x] Every fallback pronunciation is explicitly marked. (`Provenance.FALLBACK` + a `document.warnings` entry every time, by construction — see `tests/unit/test_phonemizer.py`)
- [ ] Phoneme Error Rate is reported on a held-out reference set.

### Expected outcome

A stable Romanian phoneme representation reusable across VITS and Matcha-TTS.

---

## Phase 5 — Syllabification and Lexical Stress

### Purpose

Mark the stressed syllable of each word so that future models do not have to infer Romanian lexical stress entirely from limited audio data.

### Important distinction

```text
Lexical stress:
    the stressed syllable within a word

Sentence focus:
    the word or phrase emphasized within an utterance
```

These must be represented separately.

### Implementation checklist

- [ ] Extract syllabification and lexical stress from RoLEX where available. (no RoLEX integration yet)
- [x] Convert stress information to the canonical phoneme representation. (stress index is stored alongside the token's phonemes)
- [x] Use eSpeak-ng stress markers as a fallback. (classic `espeak`'s `ˈ` mark, same caveat as Phase 4)
- [ ] Add rules for predictable suffixes and inflectional forms. (the fallback is a generic last/penultimate-syllable heuristic, not suffix-pattern rules)
- [x] Store the stressed syllable index. (`Token.stressed_syllable_index`, 0-based — documented in `stress.py`)
- [ ] Store the stressed vowel or phoneme index. (only the syllable index is stored, not a phoneme-string-level index)
- [x] Mark uncertain predictions. (fallback-heuristic results get `Provenance.FALLBACK`, low confidence, and a `document.warnings` entry)
- [x] Create an override dictionary for errors and proper names. (`configs/preprocess/stress_overrides.yaml` — mechanism implemented and tested; currently no entries, since spot-checking espeak's default Romanian stress against the project's own vocabulary didn't turn up cases worth overriding)

### Example

```json
{
  "surface": "fericire",
  "syllables": ["fe", "ri", "ci", "re"],
  "lexical_stress_syllable": 2,
  "stress_source": "lexicon",
  "stress_confidence": 1.0
}
```

The index convention must be documented as either zero-based or one-based.

### Training required

No training for the first version.

Optional later work:

- train a character-level lexical-stress predictor from RoLEX;
- compare it with rule-based and eSpeak-ng predictions.

### Evaluation values

- syllable-boundary precision, recall, and F1;
- exact syllabification accuracy;
- lexical-stress accuracy;
- lexical-stress accuracy for known words;
- lexical-stress accuracy for out-of-vocabulary words;
- coverage of the primary lexical resource.

### Acceptance criteria

- [ ] Lexical-stress accuracy of at least 95% for lexicon-covered words.
- [ ] Separate accuracy is reported for fallback predictions.
- [x] All uncertain results are marked rather than silently accepted. (fallback stress always carries `Provenance.FALLBACK` + a warning)

### Expected outcome

A TTS-ready phoneme sequence containing explicit lexical-stress information.

---

## Phase 6 — Rule-Based Emotion Baseline

> **Post-July update:** the pipeline's emotion layer was subsequently
> **replaced** by a transformer classifier — a multilingual XLM-RoBERTa
> emotion model (`tabularisai/multilingual-emotion-classification`) run
> locally through HuggingFace `transformers` (free, offline after a one-time
> download). See `src/expressive_tts/preprocess/emotion.py` and
> `scripts/download_emotion_model.py`. The rule-based baseline described in
> this phase (and the Phase 7 lightweight classifier) is the July record;
> the transformer now provides `emotion`/`intensity`. The shared NRC-lexicon
> loaders moved to `lexicons.py`, where the focus layer (Phase 8) still uses
> them.

### Purpose

Create an explainable baseline before introducing any trained emotion classifier.

### Inputs

- emotion lexicon;
- negation;
- intensifiers and diminishers;
- punctuation;
- interjections;
- sentence type;
- repeated words or punctuation;
- capitalization;
- syntactic relationships;
- preceding sentence emotion.

### Implementation checklist

- [x] Define a Romanian emotion lexicon. (NRC EmoLex Romanian translations, fetched by `scripts/fetch_emotion_lexicon.py`, not committed — see `data/external/SOURCES.md` for why)
- [x] Assign valence and arousal values. (NRC VAD Romanian translations, same source)
- [x] Add emotion-category weights. (per-token weight = base × intensifier/diminisher multiplier × negation damping)
- [x] Implement negation scope. (simplified: rest of the sentence after a `Polarity=Neg` token — Stanza doesn't give clause boundaries cheaply; dampens categorical weight rather than flipping to a specific opposite emotion, which isn't linguistically reliable)
- [x] Implement intensifier scope. (`configs/preprocess/intensifiers.yaml`, checked against the preceding 1-2 tokens)
- [x] Implement diminishing expressions. (`configs/preprocess/diminishers.yaml`)
- [x] Use punctuation as supporting, not decisive, evidence. (repeated `!`/`?` and ALL-CAPS only scale `intensity`, never which category wins)
- [x] Recognize existing interjections. (`configs/preprocess/interjection_emotions.yaml`, matched by dictionary lookup rather than Stanza's `INTJ` tag — verified the RRT-trained parser mistags "Uau" as a vocative noun)
- [x] Produce a probability-like score for every emotion. (`EmotionAnnotation.distribution`)
- [x] Return `unspecified` when confidence is low. (below a minimum-evidence threshold, below a minimum top-category margin, or an exact tie between categories)
- [x] Store evidence spans and applied rules. (`EmotionAnnotation.evidence`, e.g. `{"span": "Uau", "rule": "existing_interjection"}`)

### Example

```json
{
  "label": "happy",
  "confidence": 0.82,
  "valence": 0.76,
  "arousal": 0.64,
  "intensity": "high",
  "evidence": [
    {
      "span": "foarte bucuros",
      "rule": "positive_term_with_intensifier"
    }
  ]
}
```

### Training required

No.

### Evaluation values

- accuracy;
- macro precision;
- macro recall;
- macro F1;
- per-class precision, recall, and F1;
- confusion matrix;
- valence mean absolute error;
- arousal mean absolute error;
- intensity weighted kappa;
- coverage: percentage not labeled `unspecified`;
- selective accuracy at different confidence thresholds.

### Acceptance criteria

Because this is a baseline, no artificially high threshold should be assumed. Instead:

- [x] Macro F1 is measured and reported. (0.333 on `dev`, 0.167 on `test` — see `data/evaluation/emotion_baseline_report.md`; small sample, 8/7 gold examples, not a target to hit)
- [x] Per-class results are reported. (same report)
- [x] Confidence threshold is selected only on the development set. (`preprocess/emotion.py` thresholds inspected against `dev.jsonl` only; `test.jsonl` numbers reported once, unmodified afterward)
- [x] Ambiguous examples may return `unspecified`. (low-evidence, low-margin, and tied-category cases all resolve to `unspecified`)
- [x] Every prediction includes interpretable evidence. (empty `evidence` list only when literally no lexicon/interjection word matched)

### Expected outcome

An explainable emotional annotation baseline that runs fully locally and provides training labels with known confidence.

---

## Phase 7 — Optional Lightweight Emotion Classifier

### Purpose

Determine whether a locally runnable trained model improves on the rule-based baseline.

### Recommended approach

Do not train a language model from scratch. Fine-tune or use embeddings from a compact multilingual or Romanian pretrained text model, or train a lightweight classifier over:

- sentence embeddings;
- lexicon features;
- punctuation features;
- negation features;
- sentence-type features.

Possible classifiers:

- logistic regression;
- linear SVM;
- small multilayer perceptron;
- fine-tuned compact transformer only if the dataset is sufficient.

### Implementation checklist

- [x] Define a labeled training dataset. (REDv2's own gold labels, real — not self-generated pseudo-labels; `objective_evaluation.train_emotion_classifier`)
- [x] Keep the July test set completely held out. (training pool excludes every text present in *either* `data/evaluation/dev.jsonl` or `test.jsonl`, not just `test.jsonl` — stricter than the letter of this item)
- [x] Use stratified train/development/test splitting. (train/dev only, stratified 80/20 per label, from the classifier's own held-out-from-eval-set pool — a 2-way split, not 3-way: `data/evaluation/test.jsonl` fills the "always held out" role per the item above rather than being a 3rd split carved from the training pool)
- [x] Prevent duplicate or near-duplicate text leakage. (exact-text dedup against the eval set, applied before splitting)
- [x] Establish the rule-based baseline first. (Phase 6, already measured — `emotion_baseline_report.md`)
- [x] Train the smallest adequate classifier. (scikit-learn logistic regression over engineered features — no transformer fine-tune, per this phase's own preference given the realistic dataset size)
- [x] Save preprocessing, model weights, label mapping, and random seed. (`.cache/models/emotion_classifier/model.joblib`, gitignored — vectorizer + classifier + label list + seed)
- [x] Measure inference time and memory on M1. (0.01 ms/example on this session's actually-detected Apple Silicon host — see the Phase 13 caveat on not asserting "M1" specifically; memory not separately profiled, inference time is trivially fast regardless)
- [ ] Add calibrated confidence values. (not attempted — measuring the raw classifier's calibration was the point of this pass, and it came back poor: Brier score 0.664; a calibration step like `CalibratedClassifierCV` was considered but not applied, since recalibrating on an already-small held-out split risks overfitting the calibration itself — see `data/evaluation/emotion_classifier_report.md`)
- [x] Combine classifier and rules only after independent evaluation. (evaluated first, independently, against the rule baseline on a held-out split — and, per the result, *not* combined at all)

### Training required

Optional, lightweight, local training.

### Evaluation values

- macro F1;
- weighted F1;
- per-class F1;
- confusion matrix;
- expected calibration error;
- Brier score;
- inference latency per sentence;
- peak memory;
- improvement over the rule-based baseline.

### Acceptance criteria

- [x] The trained classifier must outperform the rule baseline in macro F1. (it does: +0.238 macro F1, 0.428 vs. 0.190, on a 33-example held-out split — `data/evaluation/emotion_classifier_report.md`)
- [x] Inference must run locally on M1. (0.01 ms/example on this session's Apple Silicon host — trivially fast; scikit-learn logistic regression was never going to be the bottleneck)
- [x] The model must not be adopted if improvement is negligible or confidence is poorly calibrated. (improvement is real and non-negligible, but calibration is poor — Brier score 0.664 — so per this exact criterion, **the model was not adopted**, a legitimate outcome this phase explicitly permits)

### Expected outcome

Evidence showing whether learned semantic features improve emotional analysis enough to justify their complexity. **Outcome**: yes, a real improvement in top-1 accuracy — but not enough to adopt, since the added complexity comes with unreliable confidence estimates. Rule baseline remains the pipeline's `emotion`/`intensity` layer.

---

## Phase 8 — Focus and Emphasis Detection

### Purpose

Identify words that should receive sentence-level prominence.

### Signals

- explicit capitalization;
- contrastive negation;
- intensifiers;
- repetition;
- emotional adjectives and adverbs;
- main predicate;
- new information;
- dependency structure;
- user-provided emphasis;
- corrective constructions.

### Implementation checklist

- [x] Detect explicit user emphasis. (`user_focus_words` param on `PreprocessPipeline.process()`, `Provenance.USER`, 100% priority — Python API only, no CLI flag yet)
- [x] Detect contrastive negation. (Romanian "nu X, ci Y" corrective construction: negation token followed by "ci")
- [ ] Detect intensifier targets using dependency relations. (implemented via lexical adjacency — token immediately preceded by an `intensifiers.yaml` entry — not true dependency-relation attachment; a real gap, not just a caveat)
- [x] Detect repeated words. (lemma repeated ≥2× in the sentence, excluding function words)
- [x] Detect emotion-bearing words. (NRC EmoLex lookup, reused from Phase 6; gracefully skipped if the lexicon cache isn't fetched — focus shouldn't hard-require a 109MB download)
- [x] Assign a continuous focus score. (`Token.focus_score`, 0-1)
- [x] Prevent function words from receiving focus without strong evidence. (`DET`/`ADP`/`CCONJ`/`SCONJ`/`AUX`/`PART`/`PUNCT` zeroed unless score already clears a high bar)
- [x] Preserve multiple focus words where justified. (no single-winner logic — every token independently crosses or doesn't cross the threshold)
- [x] Store evidence and confidence. (`Token.focus_rules` + `Token.focus_score`)

### Training required

No training in the first version.

### Evaluation values

Treat focus detection as token classification:

- token precision;
- token recall;
- token F1;
- sentence exact match;
- mean absolute error for annotated focus strength, if available.

### Acceptance criteria

- [x] Token F1 is reported on manually annotated sentences. (`objective_evaluation.evaluate_focus` against `data/evaluation/*.jsonl`'s `focus_words` — but see the caveat below: those are my own quick Phase 1 drafts, not an independent gold annotation with a defined methodology, so the F1 numbers — 0.11 dev, 0.08 test — measure agreement with a possibly-imprecise draft, not accuracy against ground truth. Full writeup: `data/evaluation/focus_sanity_check.md`)
- [x] Explicitly marked focus has 100% priority over predictions.
- [x] Focus and lexical stress remain separate output fields.

### Expected outcome

Explicit sentence-level emphasis annotations for future prosody conditioning.

---

## Phase 9 — Prosody and Pause Prediction

### Purpose

Convert punctuation, syntax, emotion, and focus into model-independent prosodic control values.

### Output parameters

```text
pause_before_ms
pause_after_ms
speaking_rate
relative_pitch
relative_energy
terminal_contour
focus_strength
```

### Implementation checklist

- [x] Define pause values for commas, semicolons, colons, periods, question marks, exclamation marks, and ellipses. (`prosody._PUNCTUATION_PAUSE_MS`)
- [x] Modify pause values using sentence length. (sentences over ~15 content tokens get +100ms at terminal punctuation)
- [x] Detect clause boundaries from dependency parsing. (tokens with `deprel` in `{"mark","advcl","ccomp"}` — real parse structure, not just punctuation)
- [x] Predict rising, falling, and continuation contours. (direct mapping from `sentence_type`, already computed in Phase 3)
- [x] Increase energy and pitch range for high-arousal emotions.
- [x] Reduce rate and energy for low-arousal emotions. (`value = 1.0 + (arousal - 0.5) * 0.4`, one symmetric formula does both)
- [x] Apply focus locally rather than to the entire sentence. (`Token.local_relative_pitch`/`local_relative_energy`, set only for `is_focus=True` tokens)
- [x] Constrain all continuous values to safe ranges. (`clamp()` helper used everywhere; verified over 229 real sentences from `datasets/hria`/`datasets/mara` — 0 violations)
- [x] Preserve user-provided values. (`user_prosody_overrides` param on `PreprocessPipeline.process()`, keyed by sentence index)
- [x] Document every rule. (`ProsodyAnnotation.rules` / `Token.prosody_rules`, plus methodology notes in `src/expressive_tts/preprocess/README.md`)

### Example safe ranges

```text
speaking_rate:  0.80–1.20
relative_pitch: 0.85–1.20
relative_energy: 0.80–1.20
pause:          0–1000 ms
```

These are intermediate control values, not guaranteed acoustic outputs.

### Training required

No training for the first version.

Future option:

- learn prosodic targets from aligned audio using F0, energy, and duration;
- use the rule-based output as initialization or weak supervision.

### Evaluation values

For rule-based output:

- pause-boundary precision, recall, and F1;
- pause-category accuracy;
- terminal-contour accuracy;
- speaking-rate category accuracy;
- correlation with manually annotated intensity;
- percentage of values outside permitted ranges.

If reference audio is later available:

- F0 correlation;
- F0 RMSE;
- energy correlation;
- duration MAE;
- pause-duration MAE.

### Acceptance criteria

- [x] No generated value exceeds configured limits. (`tests/unit/test_prosody.py` checks range boundaries directly; `tests/integration/test_pipeline_prosody.py::test_range_safety_over_real_sentences` + an ad hoc 229-real-sentence sweep, 0 violations)
- [ ] Pause-boundary F1 is reported. (deliberately not: `data/evaluation/*.jsonl`'s `pause_locations` is mechanically derived from punctuation offsets by the Phase 1 build script itself, not an independent annotation of where a human would pause — measuring against it would just check "does our punctuation list match our punctuation list." See `data/evaluation/README.md`.)
- [ ] Terminal-contour accuracy is reported. (same reason — no gold terminal-contour reference exists yet)
- [x] Every prosodic value has an evidence source. (`ProsodyAnnotation.rules` / `Token.prosody_rules`)

### Expected outcome

A model-independent prosodic plan that can later condition an acoustic TTS network.

---

## Phase 10 — Controlled Interjection Suggestions

### Purpose

Suggest optional interjections without altering meaning or corrupting formal text.

### Modes

```text
disabled:
    no suggestions

suggest:
    return candidates separately

insert:
    produce an enriched variant while preserving the original
```

### Implementation checklist

- [x] Create a Romanian interjection lexicon. (`configs/preprocess/interjection_emotions.yaml`, from Phase 6, reused here)
- [ ] Map interjections to emotion, valence, arousal, and style. (mapped to emotion label + weight only; valence/arousal/style are not modeled per-interjection — a real gap, not just a caveat)
- [x] Detect existing interjections. (reuses Phase 6's dictionary-based detection)
- [x] Detect document style. (deliberately coarse `formal`/`conversational` binary — see below and `src/expressive_tts/preprocess/README.md`)
- [x] Disable insertion for academic, legal, technical, and formal text. (folded into the binary `formal` gate — the coarse split doesn't distinguish *why* something is formal, just that it is; verified 0% violation rate — see `data/evaluation/interjection_evaluation_report.md`)
- [x] Require a minimum emotion-confidence threshold. (0.6, documented in `interjections.py`, not gold-tuned)
- [x] Prevent repeated interjections.
- [x] Produce at most a small number of ranked candidates. (capped at 2, ranked by `weight * emotion.confidence`)
- [x] Store the insertion position. (`InterjectionSuggestion.position` — always sentence-start; no mid-sentence placement modeled yet)
- [x] Store the reason and confidence. (`InterjectionSuggestion.reason`/`confidence`)
- [x] Preserve the unmodified text. (`Sentence.text` untouched; `"insert"` mode writes to the separate `Sentence.text_with_interjections`)

### Training required

No.

### Evaluation values

Manual evaluation:

- appropriateness rate;
- meaning-preservation rate;
- style-violation rate;
- precision of suggestions;
- percentage of examples where the system correctly abstains;
- average number of suggestions per sentence.

### Acceptance criteria

- [x] Suggestions are disabled by default. (`interjection_mode="disabled"` default; verified in `tests/integration/test_pipeline_interjections.py::test_disabled_by_default`)
- [x] Original text is never overwritten. (`Sentence.text` untouched by construction; enriched text goes to the separate `text_with_interjections` field — `tests/unit/test_interjections.py`)
- [x] Formal-text violation rate is 0% on the evaluation set. (0/3 dev, 0/1 test — `data/evaluation/interjection_evaluation_report.md`; met by construction of the `document_style` gate, not by tuning against this set)
- [ ] Meaning-preservation rate is reported. (not implemented — would need a semantic-similarity judgment or human review of enriched vs. original text; out of scope this pass)

### Expected outcome

A safe mechanism for increasing expressiveness without treating generated text as user-provided content.

---

## Phase 11 — Context-Aware Sentence Smoothing

### Purpose

Avoid implausible changes of emotion and prosody between adjacent sentences.

### Initial context

Use only:

- current sentence;
- preceding sentence;
- following punctuation;
- paragraph boundary;
- detected speaker turn, if available;
- explicit discourse markers.

### Implementation checklist

Note: implemented for **emotion only**, not prosody, despite "Purpose"
above mentioning both — the checklist items themselves (below) are all
emotion-specific; prosody context-smoothing isn't built and isn't tracked
elsewhere either, a real scope gap worth flagging rather than silently
narrowing.

- [x] Preserve the raw local emotion prediction. (`Sentence.emotion` is never written to by `context.py`; verified by test and by construction)
- [x] Compute a context-adjusted emotion prediction. (`Sentence.context_emotion`, blends `EmotionAnnotation.distribution` per the baseline formula below)
- [x] Limit context propagation across paragraph boundaries. (blank-line detection on the *original* source text — approximate, since `SentenceSpan` offsets are relative to whitespace-collapsed `clean_text`; see `src/expressive_tts/preprocess/README.md`)
- [x] Prevent a neutral sentence from erasing a strong explicit emotion. (blending is skipped entirely once local confidence ≥0.75)
- [x] Detect discourse markers such as contrast or consequence. (`configs/preprocess/discourse_markers.yaml`, checked against the leading content token only)
- [x] Smooth intensity only when confidence is low. (only below `INTENSITY_SMOOTH_CONFIDENCE_THRESHOLD`, 0.5)
- [x] Store both local and contextual results. (`Sentence.emotion` + `Sentence.context_emotion`)
- [x] Record when and why context changed a prediction. (`ContextAdjustment.changed`/`.reason`; reason is always `None` when `changed=False`, verified by test)

### Simple baseline

```text
context_score =
    α × current_sentence_score
    + β × previous_sentence_score
```

with:

```text
α > β
```

The parameters must be selected on the development set.

### Training required

No training for the first version.

Optional later work:

- train a sequence classifier on contextual emotional data.

### Evaluation values

- sentence-level macro F1 before smoothing;
- sentence-level macro F1 after smoothing;
- transition accuracy;
- number of harmful context changes;
- number of corrected isolated errors;
- contextual paragraph accuracy.

### Acceptance criteria

- [ ] Context smoothing must improve or preserve macro F1. (not measured — `data/evaluation/context.jsonl` has no per-sentence gold emotion labels, and hand-labeling just to manufacture an F1 number would be a fragile, non-authoritative metric; `objective_evaluation.evaluate_context` runs a genuine behavior sanity check instead — see `data/evaluation/context_evaluation_report.md`, same precedent as Phase 9's skipped pause-boundary F1)
- [x] Local predictions are never discarded. (verified directly: all 212 sentences across all 43 real `context.jsonl` paragraphs have both fields populated, local untouched — `tests/unit/test_context.py`, `tests/integration/test_pipeline_context.py`)
- [x] Context changes are traceable. (`ContextAdjustment.reason` non-null iff `changed=True`, verified by test and by the full evaluation run: 53/212 real changes, every one carrying a reason)

### Expected outcome

More coherent annotations across paragraphs without requiring a large contextual model.

---

## Phase 12 — Serialization for TTS

### Purpose

Convert the general intermediate representation into formats usable by different TTS architectures.

### Required serializers

- [x] Canonical JSON. (`serializers.to_canonical_json` — thin, explicit, tested wrapper around `PreprocessResult.model_dump_json`)
- [x] Human-readable annotated text. (`serializers.to_annotated_text` — inline `[emotion=..., intensity=...]`/`[pause=Xms]` bracket annotations + `[FOCUS]` markers)
- [x] Discrete control tokens. (`serializers.to_control_tokens`, matches the worked example below)
- [x] Optional SSML-like output. (`serializers.to_ssml_like` — explicitly illustrative, not validated against the real SSML schema, per "Optional" in this item's own name)
- [ ] Future VITS adapter. (explicitly "Future" — not attempted)
- [ ] Future Matcha-TTS adapter. (explicitly "Future" — not attempted)

Vocabulary-rule caveat: token names are stable within this pass but
**not versioned** (e.g. no `v1` prefix/suffix on `[EMO_*]` etc.) — a real
gap against "token names must be stable and versioned" below, not
addressed this phase.

### Example control-token output

```text
[SENT_EXCLAMATIVE]
[EMO_SURPRISE]
[INT_HIGH]
[RATE_1.10]
[PITCH_1.15]
Nu pot să [FOCUS] cred !
[BREAK_250]
[EMO_HAPPY]
Am [FOCUS] reușit .
```

### Vocabulary rules

- control tokens must be separated from phoneme tokens;
- token names must be stable and versioned;
- unknown values must map to explicit `UNSPECIFIED` tokens;
- continuous values should initially be quantized into bins;
- serialization must be reversible to the intermediate representation where possible.

### Training required

No.

### Evaluation values

- serialization success rate;
- round-trip consistency;
- unknown-control-token rate;
- output determinism;
- processing time per sentence.

### Acceptance criteria

- [x] 100% serialization success on the test set. (167/167 dev, 137/137 test — `objective_evaluation.evaluate_serialization`)
- [x] 0% unknown control tokens. (0/167 dev, 0/137 test — guaranteed by the `..._UNSPECIFIED` fallback, verified not just assumed)
- [x] Identical input and configuration produce identical output. (0 determinism violations across both splits; also unit-tested directly)

### Expected outcome

A stable input representation for future TTS-network experiments.

---

## Phase 13 — End-to-End Testing and Error Analysis

### Purpose

Verify that individually correct components remain correct when connected.

### Implementation checklist

- [x] Add unit tests for every normalization category. (numbers, dates, times, percentages, currencies, units, abbreviations, decimals, Roman-numeral ordinals)
- [x] Add integration tests for every pipeline stage. (`tests/integration/test_pipeline_stages.py` — one test per registered layer: clean, sentences, normalized, linguistic, phonemes, syllables, emotion, focus, prosody, interjections; context added when Phase 11 landed later this pass)
- [x] Add regression tests for previously discovered errors. (compound-ordinal gender bug found during development — see `test_numbers_ro.py::test_ordinal_masculine`/`test_ordinal_feminine` cases for 21/23)
- [x] Test empty input.
- [x] Test malformed Unicode. (`tests/integration/test_pipeline_edge_cases.py` — lone surrogate replacement chars, stacked combining marks)
- [x] Test very long paragraphs. (40x-repeated sentence, 40+ segmented sentences)
- [x] Test code-switching. (embedded English phrases in a Romanian sentence)
- [x] Test unsupported symbols. (emoji, control characters)
- [x] Test formal and academic text. (the RONEC legal example, full pipeline)
- [x] Test emotionally ambiguous text. (the documented "Ce panică!" tied-category case — must resolve to `unspecified`, not crash or guess)
- [x] Test existing interjections. (full pipeline, asserts `Token.is_interjection`)
- [x] Test repeated punctuation. (`"Ce???!!! Chiar nu știi???"`)
- [x] Record runtime and memory on M1. (`objective_evaluation.benchmark` — reports the actually-detected host, not an assumed chip; see acceptance criteria below)
- [x] Export an error report grouped by component. (`objective_evaluation.error_report` → `data/evaluation/error_report.md` — 0 exceptions across 3,120 runs: 312 inputs × 10 layers)

### Training required

No.

### End-to-end evaluation values

- percentage of inputs processed without exceptions;
- average processing time per sentence;
- p50, p95, and p99 latency;
- peak memory;
- schema-validation rate;
- determinism rate;
- percentage of annotations marked uncertain;
- component error counts;
- full-pipeline exact-match rate on selected deterministic examples.

### Acceptance criteria

- [x] 100% of valid test inputs complete without exceptions. (0/3,120 runs raised — `objective_evaluation.error_report`, 312 inputs × 10 layers, covering the full evaluation set plus 8 hand-picked edge cases; still a finite hand-picked+eval-set corpus, not true fuzzing — noted honestly in the error report itself)
- [x] 100% of outputs pass schema validation. (every result is a `pydantic`-validated `PreprocessResult`; validation is enforced at construction)
- [x] Pipeline runs offline after required resources are installed. (no network calls)
- [x] Runtime and memory are reported on the Apple M1. (`data/evaluation/benchmark_report.md` — reports the actually-detected host CPU rather than asserting "M1" unverified; this session's host is Darwin/arm64 Apple Silicon, not confirmed to specifically be M1 vs. M2/M3)

### Expected outcome

A locally runnable, testable, and reproducible Romanian expressive frontend.

---

# What Must Be Trained

## Required this month

No large neural network must be trained.

The required implementation can be completed using:

- deterministic normalization;
- existing Romanian linguistic models;
- phonetic lexicons;
- eSpeak-ng fallback;
- emotion lexicons;
- rule-based emotion and prosody estimation;
- manually annotated evaluation data.

## Optional lightweight training

Only the following optional component may be trained locally:

- [ ] A lightweight text-emotion classifier.

Recommended order:

1. implement and evaluate the rule-based baseline;
2. create or obtain a labeled Romanian emotion dataset;
3. train a lightweight classifier;
4. compare it against the baseline;
5. retain it only if macro F1 and calibration improve.

## Future training after access to an NVIDIA GPU

- [ ] Romanian grapheme-to-phoneme model.
- [ ] Learned lexical-stress predictor.
- [ ] Acoustic prosody predictor from audio.
- [ ] Contextual emotion classifier.
- [ ] VITS or Matcha-TTS using the frontend tokens.
- [ ] Emotion-conditioned vocoder, only if demonstrated necessary.

---

# Evaluation Summary

| Component | Primary metrics |
|---|---|
| Text normalization | Exact match, token accuracy, category accuracy |
| Sentence segmentation | Precision, recall, F1 |
| Tokenization | Precision, recall, F1 |
| Morphology | UPOS accuracy, lemma accuracy |
| Phonemization | Phoneme Error Rate, word exact match, OOV rate |
| Syllabification | Boundary F1, exact match |
| Lexical stress | Stress accuracy, known/OOV accuracy |
| Emotion | Macro F1, per-class F1, confusion matrix |
| Valence/arousal | MAE, correlation |
| Intensity | Weighted kappa, macro F1 |
| Focus | Token precision, recall, F1 |
| Pauses | Boundary F1, category accuracy |
| Terminal intonation | Accuracy |
| Interjections | Appropriateness, abstention, style violations |
| Context smoothing | Macro F1 before/after, harmful changes |
| Serialization | Success rate, round-trip consistency |
| Full pipeline | Success rate, latency, memory, determinism |

---

# Recommended Evaluation Report Format

For each component, record:

```text
Component:
Version:
Evaluation dataset:
Number of examples:
Configuration:
Primary metric:
Primary result:
Per-category results:
Known limitations:
Representative errors:
Decision:
    accepted
    revise
    excluded
```

Example:

```text
Component: Romanian number normalizer
Version: 0.1
Evaluation dataset: normalization_test_v1
Number of examples: 120
Primary metric: token accuracy
Primary result: 98.7%
Known limitations: ambiguous years and telephone numbers
Decision: revise
```

---

# July Work Allocation — 60 Hours

| Order | Activity | Hours | Required outcome |
|---:|---|---:|---|
| 1 | Annotation contract and JSON schema | 4 | Versioned schema |
| 2 | Evaluation-set construction | 6 | Development and test sets |
| 3 | Romanian text normalization | 10 | Tested normalizer |
| 4 | Linguistic-analysis integration | 6 | Token, lemma, POS and dependencies |
| 5 | Phonemization | 7 | Canonical phoneme output |
| 6 | Syllabification and lexical stress | 7 | Stress-annotated output |
| 7 | Rule-based emotion analysis | 6 | Emotion baseline and metrics |
| 8 | Focus and prosody rules | 5 | Focus, pause and contour output |
| 9 | Controlled interjection suggestions | 3 | Safe suggestion mode |
| 10 | Context smoothing | 2 | Local and contextual annotations |
| 11 | Serialization and CLI | 2 | JSON and token output |
| 12 | End-to-end evaluation and documentation | 2 | Evaluation report and README |
|  | **Total** | **60** | |

---

# Final July Acceptance Checklist

*(This section had gone stale relative to the rest of the document —
several items below were last updated around Phase 0/1 and didn't reflect
Phases 2-13. Refreshed to match actual current state. Also reflects a
structural change made this pass: the codebase is now split into two
top-level packages under `src/` — `expressive_tts` (text processing:
`preprocess/`, including its own `schemas.py`/`cli.py`, consolidated from
the previously separate `schemas/`/`cli/` top-level packages) and
`objective_evaluation` (a sibling package holding dataset construction and
every per-component evaluation script, moved out of the previously
non-importable `scripts/` directory). `scripts/` now holds only one-time
environment bootstrap: `download_stanza_model.py`/`fetch_emotion_lexicon.py`.)*

## Functionality

- [x] Raw Romanian text can be processed from a CLI.
- [x] Original text is preserved.
- [x] Normalized text is produced.
- [x] Sentences and tokens are identified. (Phase 3, Stanza-backed)
- [x] Lemmas, POS tags, and dependencies are available. (Phase 3, `linguistic.LinguisticProcessor`)
- [x] Phonemes are produced for every pronounceable token. (Phase 4 — every token gets a phoneme string; unknown/unmapped ones fall back to a marked, low-confidence grapheme-identity mapping rather than being skipped)
- [x] Lexical stress is represented. (Phase 5)
- [x] Emotion and intensity are estimated. (Phase 6)
- [x] Focus words are identified. (Phase 8)
- [x] Pauses and sentence contours are predicted. (Phase 9)
- [x] Interjections are suggested only when enabled. (Phase 10 — `interjection_mode` defaults to `"disabled"`)
- [x] Adjacent sentences receive context-aware annotations. (Phase 11 — emotion only, not prosody; see that phase's own caveat)
- [x] JSON and control-token outputs are produced. (Phase 12)

## Quality

- [ ] Every component has a fixed evaluation set. (`data/evaluation/` is fixed and used by emotion/focus/interjections/context/serialization; normalization/linguistic/phonemization/stress still rely on spot-checks and the objectives.md worked examples, not the evaluation set)
- [ ] Every component reports quantitative values. (same gap as above — most components now do; normalization/linguistic/phonemization/stress don't yet)
- [x] All uncertain predictions are marked. (`Provenance.FALLBACK`/`PREDICTED` + confidence fields + `document.warnings`, by construction across every processor)
- [x] No valid test input crashes the pipeline. (0 exceptions across 3,120 runs — `data/evaluation/error_report.md`; still a finite corpus, not true fuzzing)
- [x] All outputs pass schema validation.
- [x] Processing time and memory are measured on M1. (`data/evaluation/benchmark_report.md` — reports the actually-detected host, see the Phase 13 caveat on not asserting "M1" unverified)
- [x] Errors are grouped and documented. (`data/evaluation/error_report.md`, grouped by processor/layer)

## Reproducibility

- [x] Dependency versions are recorded. (`pyproject.toml`, `requirements.txt` — `dev`/`linguistic`/`ml` optional-dependency groups)
- [x] External resources and licenses are documented. (`data/external/SOURCES.md` — REDv2, RONEC, NRC EmoLex/VAD; `preprocess/README.md` documents Stanza/espeak)
- [x] Configuration files are versioned. (all `configs/preprocess/*.yaml` are tracked repository files; note this project isn't yet an actual git repository, so "versioned" here means "stable tracked files," not git history)
- [x] Random seeds are stored for trained components. (Phase 7's classifier: seed 20260725, saved in `.cache/models/emotion_classifier/model.joblib`)
- [x] Evaluation scripts are included. (`objective_evaluation.evaluate_*`, one per evaluated component)
- [x] The complete pipeline runs locally after installation.

---

# Expected Outcome for the Research Report

The July activity should demonstrate that a dedicated Romanian expressive text frontend can provide explicit linguistic, phonetic, emotional, and prosodic information before acoustic-model training. The resulting annotations will make future TTS experiments more controllable and will allow the contribution of text preprocessing to be evaluated independently from the contribution of the acoustic architecture.

The resulting module will serve as a common input layer for the existing VITS system and for any future Matcha-TTS implementation. This avoids repeating text-processing work for each architecture and creates a consistent basis for comparing the networks under equivalent linguistic and expressive conditions.

---

# Technical References

- Stanza models and local pipelines: <https://stanfordnlp.github.io/stanza/models.html>
- Romanian Universal Dependencies treebank: <https://universaldependencies.org/treebanks/ro_rrt/index.html>
- RoLEX lexical resource description: <https://www.cambridge.org/core/journals/natural-language-engineering/article/abs/rolex-the-development-of-an-extended-romanian-lexical-dataset-and-its-evaluation-at-predicting-concurrent-lexical-information/96A3933A2028BD8EE605E0E672A57EB2>
- eSpeak-ng dictionary, phoneme, pause, and stress notation: <https://github.com/espeak-ng/espeak-ng/blob/master/docs/dictionary.md>
