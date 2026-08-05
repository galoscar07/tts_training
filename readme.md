# Romanian Expressive TTS

Modular tools for building an expressive, context-aware Text-to-Speech system for Romanian.

The repository is organized around independent pipelines rather than a single monolithic model:

1. **Preprocessing pipeline** — transforms raw text into normalized, linguistic, phonetic, emotional, and prosodic representations.
2. **Training pipeline** — prepares datasets and trains configurable acoustic and vocoder models.
3. **Inference pipeline** — converts annotated text into acoustic representations and audio.
4. **Postprocessing pipeline** — validates, joins, normalizes, and exports generated audio.
5. **Evaluation pipeline** — evaluates text processing, pronunciation, acoustic quality, emotion, and speaker similarity.

The preprocessing pipeline is designed as a standalone application. It can be developed, tested, and used without training or loading a TTS model.

> **Project status:** architecture and interfaces are being defined. Components should be considered planned until their implementation and tests are available.

---

## Project Goals

The long-term goal is to build a Romanian TTS system that can:

- normalize and interpret Romanian text;
- generate consistent phonetic representations;
- identify lexical stress and sentence-level emphasis;
- extract emotion, emotional intensity, and contextual information;
- represent pauses and prosodic intent;
- optionally suggest interjections;
- condition multiple TTS architectures using the same frontend;
- synthesize multiple speakers and emotional styles;
- evaluate text, acoustic, speaker, and emotional quality independently.

The preprocessing pipeline reduces the amount of linguistic information that the acoustic network must infer from limited Romanian audio data. Instead of learning number expansion, abbreviation pronunciation, lexical stress, emotion, and prosody implicitly, a TTS model can receive explicit annotations.

```text
Raw text
    │
    ▼
Preprocessing pipeline
    │
    ├── normalized text
    ├── linguistic annotations
    ├── phonemes
    ├── syllables and lexical stress
    ├── emotion and intensity
    ├── focus and prosody
    └── model-specific control tokens
    │
    ▼
Training or inference pipeline
    │
    ▼
Acoustic output
    │
    ▼
Postprocessing pipeline
    │
    ▼
Final audio
```

---

# 1. Preprocessing Pipeline

## 1.1 Responsibilities

The preprocessing application accepts text and returns one or more requested representations.

Supported information layers are intended to include:

| Layer | Description |
|---|---|
| `clean` | Unicode cleanup and whitespace normalization |
| `normalized` | Spoken-form expansion of numbers, dates, units, abbreviations, and symbols |
| `sentences` | Sentence boundaries and sentence types |
| `tokens` | Tokens and character offsets |
| `linguistic` | Lemmas, POS tags, morphology, and dependency relations |
| `phonemes` | Canonical Romanian phoneme sequence |
| `syllables` | Word syllabification |
| `lexical_stress` | Stressed syllable or phoneme |
| `emotion` | Emotion label, distribution, valence, arousal, and confidence |
| `intensity` | Emotional-intensity estimate |
| `focus` | Sentence-level emphasis candidates |
| `prosody` | Pauses, speaking rate, relative pitch, relative energy, and terminal contour |
| `interjections` | Existing and optionally suggested interjections |
| `context` | Context-adjusted sentence annotations |
| `tts_tokens` | Discrete control-token representation for a TTS model |
| `all` | All available layers |

The caller selects only the information required for a particular task. Expensive components should not run when they are not requested.

## 1.2 Input modes

The standalone application should support:

- a text argument;
- standard input;
- a plain-text file;
- a directory of files;
- CSV;
- TSV;
- JSON;
- JSON Lines;
- dataset manifest files;
- a Python API.

### Single text

```bash
tts-preprocess \
  --text "Am cumpărat 25 kg pe 12.07.2026." \
  --include normalized,phonemes,lexical_stress
```

### Standard input

```bash
echo "Nu pot să cred! Am reușit." | \
  tts-preprocess \
    --stdin \
    --include normalized,emotion,focus,prosody
```

### Plain-text file

```bash
tts-preprocess \
  --input-file examples/input.txt \
  --include all \
  --output-file outputs/input.processed.json
```

### Directory

```bash
tts-preprocess \
  --input-dir data/texts \
  --glob "**/*.txt" \
  --recursive \
  --include normalized,phonemes \
  --output-dir outputs/texts
```

### CSV

```bash
tts-preprocess \
  --input-file data/metadata.csv \
  --format csv \
  --text-column transcript \
  --id-column utterance_id \
  --include normalized,phonemes,emotion \
  --output-file outputs/metadata.processed.csv
```

### Dataset manifest

```bash
tts-preprocess \
  --input-file data/train.txt \
  --format delimited \
  --delimiter "|" \
  --columns audio_path,speaker_id,emotion,text \
  --text-column text \
  --include normalized,phonemes,lexical_stress \
  --output-file outputs/train.processed.jsonl
```

## 1.3 Selecting output information

### Include specific layers

```bash
tts-preprocess \
  --text "Uau! Chiar am reușit?" \
  --include normalized,phonemes,emotion,prosody
```

### Exclude layers

```bash
tts-preprocess \
  --input-file data/texts.jsonl \
  --include all \
  --exclude interjections,context
```

### Use a named profile

```bash
tts-preprocess \
  --input-file data/train.csv \
  --profile tts-training \
  --output-file outputs/train.jsonl
```

Suggested profiles:

| Profile | Included information |
|---|---|
| `normalize-only` | cleaned and normalized text |
| `linguistic` | sentences, tokens, lemmas, POS, morphology, dependencies |
| `pronunciation` | normalized text, phonemes, syllables, lexical stress |
| `expressive` | emotion, intensity, focus, prosody, interjections, context |
| `tts-training` | all information required for dataset preparation |
| `tts-inference` | normalized text, phonemes, stress, emotion, focus, prosody, control tokens |
| `debug` | every layer plus traces and confidence values |

## 1.4 Input file specification

File parsing must be configuration-driven rather than hard-coded for one corpus.

Example:

```yaml
input:
  format: csv
  path: data/metadata.csv
  encoding: utf-8
  delimiter: ","
  header: true
  id_column: utterance_id
  text_column: transcript
  passthrough_columns:
    - audio_path
    - speaker_id
    - emotion

output:
  format: jsonl
  path: outputs/metadata.processed.jsonl

processing:
  include:
    - normalized
    - linguistic
    - phonemes
    - lexical_stress
    - emotion
    - prosody
```

### Parsing requirements

The parser should:

- preserve unprocessed columns;
- preserve row order unless explicitly configured otherwise;
- support UTF-8 Romanian diacritics;
- report malformed rows;
- support quoted delimiters;
- support missing values;
- optionally skip or fail on invalid rows;
- produce a machine-readable error report;
- record input file, row number, and record ID;
- support configurable field mappings;
- support configurable output columns;
- avoid modifying the original input file.

### Invalid-record policies

```text
fail:
    stop on the first invalid record

skip:
    skip invalid records and write an error report

keep:
    keep the record and attach validation errors
```

Example:

```bash
tts-preprocess \
  --config configs/preprocess/dataset.yaml \
  --on-error keep \
  --error-report outputs/preprocess_errors.jsonl
```

## 1.5 Multiple output forms

Given one input text, the application may return several synchronized representations:

```json
{
  "schema_version": "1.0",
  "id": "example-001",
  "original_text": "Nu pot să cred! Am reușit.",
  "clean_text": "Nu pot să cred! Am reușit.",
  "normalized_text": "Nu pot să cred! Am reușit.",
  "phoneme_text": "...",
  "tts_token_text": "[EMO_SURPRISE] Nu pot să [FOCUS] cred ! [BREAK_250] [EMO_HAPPY] Am reușit .",
  "sentences": [
    {
      "text": "Nu pot să cred!",
      "sentence_type": "exclamative",
      "emotion": {
        "label": "surprise",
        "confidence": 0.87,
        "valence": 0.35,
        "arousal": 0.91,
        "intensity": "high"
      },
      "prosody": {
        "speaking_rate": 1.1,
        "relative_pitch": 1.18,
        "relative_energy": 1.2,
        "pause_after_ms": 250,
        "terminal_contour": "falling"
      },
      "tokens": []
    }
  ],
  "trace": []
}
```

Every predicted or generated annotation should contain:

- its value;
- confidence, when meaningful;
- provenance;
- evidence;
- model or rule version.

Example provenance:

```json
{
  "label": "happy",
  "provenance": "predicted",
  "producer": "emotion_xlmr_v1",
  "confidence": 0.81
}
```

## 1.6 Output formats

The preprocessing pipeline should support:

- JSON;
- JSON Lines;
- CSV with selected flattened fields;
- TSV;
- annotated text;
- phoneme text;
- TTS control-token text;
- dataset manifests.

Example:

```bash
tts-preprocess \
  --input-file data/metadata.csv \
  --text-column text \
  --include normalized,phonemes,emotion \
  --output-format jsonl \
  --output-file outputs/metadata.jsonl
```

## 1.7 Configuration precedence

Configuration values should be resolved in this order:

```text
command-line argument
    overrides
experiment configuration
    overrides
profile configuration
    overrides
application defaults
```

The resolved configuration should be written to the output directory for reproducibility.

## 1.8 Python API

```python
from expressive_tts.preprocess import PreprocessPipeline

pipeline = PreprocessPipeline.from_config(
    "configs/preprocess/expressive.yaml"
)

result = pipeline.process(
    text="Uau! Chiar ai terminat?",
    include={
        "normalized",
        "phonemes",
        "lexical_stress",
        "emotion",
        "prosody",
    },
)

print(result.normalized_text)
print(result.sentences[0].emotion.label)
```

Batch processing:

```python
results = pipeline.process_file(
    path="data/metadata.csv",
    format="csv",
    text_column="transcript",
    id_column="utterance_id",
    include={"normalized", "phonemes", "emotion"},
)
```

## 1.9 Standalone service

An optional local HTTP service may expose the same application:

```http
POST /v1/preprocess
Content-Type: application/json
```

```json
{
  "text": "Nu pot să cred! Am reușit.",
  "include": [
    "normalized",
    "phonemes",
    "emotion",
    "prosody"
  ],
  "options": {
    "interjection_mode": "suggest"
  }
}
```

The CLI, Python API, and HTTP service must use the same underlying pipeline implementation.

---

# 2. Preprocessing Stages

The initial execution order is:

```text
1. input validation
2. Unicode cleanup
3. protected-span detection
4. sentence segmentation
5. text normalization
6. tokenization and linguistic analysis
7. phonemization
8. syllabification and lexical stress
9. local emotion analysis
10. focus detection
11. prosody planning
12. interjection suggestion
13. contextual adjustment
14. serialization
15. output validation
```

Dependencies between requested layers should be resolved automatically.

Example:

```text
Request: lexical_stress

Automatically required:
clean → normalized → tokens → phonemes → syllables → lexical_stress
```

The user should not need to list internal prerequisites.

## 2.1 Text normalization

Intended coverage:

- Romanian Unicode normalization;
- numbers and ordinals;
- dates and times;
- currencies;
- percentages;
- measurement units;
- abbreviations;
- acronyms;
- Roman numerals;
- punctuation;
- special symbols;
- optional handling of URLs and e-mail addresses.

## 2.2 Linguistic analysis

Intended annotations:

- sentence boundaries;
- tokens;
- lemmas;
- universal POS tags;
- morphological features;
- syntactic dependencies;
- negation;
- sentence type;
- existing interjections.

## 2.3 Pronunciation

Intended pronunciation cascade:

```text
project overrides
    ↓
Romanian pronunciation lexicon
    ↓
RoLEX-derived resource
    ↓
eSpeak-ng Romanian
    ↓
grapheme fallback with low confidence
```

Output:

- phonemes;
- syllables;
- lexical stress;
- pronunciation source;
- pronunciation confidence.

## 2.4 Expressive analysis

Intended output:

- categorical emotion;
- emotion distribution;
- valence;
- arousal;
- intensity;
- focus words;
- pause plan;
- speaking-rate factor;
- relative pitch;
- relative energy;
- terminal contour;
- optional interjection suggestions.

`neutral` and `unspecified` must remain different:

```text
neutral:
    evidence supports a neutral realization

unspecified:
    insufficient evidence for a reliable emotional label
```

## 2.5 Context processing

Context processing may use:

- preceding sentence;
- following sentence, when batch input permits it;
- paragraph boundaries;
- speaker turns;
- discourse markers;
- explicit user controls.

Local and context-adjusted predictions must both be retained.

---

# 3. Postprocessing Pipeline

## 3.1 Responsibilities

The postprocessing pipeline receives generated acoustic or waveform outputs and prepares final audio.

Planned operations:

- denormalize generated mel-spectrograms;
- run the vocoder;
- remove excessive leading and trailing silence;
- preserve intentional expressive pauses;
- normalize loudness;
- detect clipping;
- apply safe peak limiting;
- concatenate sentence-level audio;
- insert configured pauses;
- apply short crossfades where appropriate;
- resample to export formats;
- write WAV and optional compressed formats;
- preserve generation metadata;
- create an output manifest;
- run audio validation.

## 3.2 Postprocessing CLI

```bash
tts-postprocess \
  --input-dir outputs/generated_segments \
  --manifest outputs/generation_manifest.jsonl \
  --config configs/postprocess/default.yaml \
  --output-dir outputs/final_audio
```

## 3.3 Configuration example

```yaml
audio:
  sample_rate: 24000
  channels: 1
  peak_limit_dbfs: -1.0
  target_lufs: -20.0

silence:
  trim_leading: true
  trim_trailing: true
  preserve_internal_pauses: true
  maximum_trim_ms: 500

concatenation:
  use_manifest_pauses: true
  crossfade_ms: 10

export:
  formats:
    - wav
  write_metadata: true
```

## 3.4 Required output

```text
outputs/final_audio/
├── audio/
├── metadata.jsonl
├── validation_report.json
└── resolved_postprocess_config.yaml
```

---

# 4. Training Pipeline

## 4.1 Responsibilities

The training pipeline should be architecture-independent at the configuration level while allowing model-specific parameters.

It should support:

- dataset selection;
- multiple datasets;
- sampling weights;
- train/validation/test manifests;
- speaker mappings;
- language mappings;
- emotion mappings;
- input representation selection;
- acoustic-feature configuration;
- model architecture;
- optimizer and scheduler;
- batch size and gradient accumulation;
- precision;
- checkpointing;
- resuming;
- logging;
- evaluation during training;
- experiment metadata;
- reproducible random seeds.

## 4.2 Intended architectures

Initial adapters may include:

- VITS;
- Matcha-TTS;
- standalone vocoders.

The preprocessing representation must remain independent of the selected acoustic architecture.

## 4.3 Training CLI

```bash
tts-train \
  --config configs/training/vits_romanian_emotional.yaml
```

Override selected parameters:

```bash
tts-train \
  --config configs/training/vits_romanian_emotional.yaml \
  --set trainer.batch_size=8 \
  --set trainer.accumulate_grad_batches=4 \
  --set trainer.max_steps=100000 \
  --set model.emotion_conditioning=true
```

Resume:

```bash
tts-train \
  --config outputs/runs/vits-ro-001/resolved_config.yaml \
  --resume outputs/runs/vits-ro-001/checkpoints/last.ckpt
```

## 4.4 Training configuration example

```yaml
experiment:
  name: vits-ro-emotional
  seed: 1234
  output_dir: outputs/runs/vits-ro-emotional

data:
  datasets:
    - name: mara
      manifest: data/manifests/mara_train.jsonl
      sampling_weight: 0.35

    - name: swara
      manifest: data/manifests/swara_train.jsonl
      sampling_weight: 0.45

    - name: hria_emotional
      manifest: data/manifests/hria_train.jsonl
      sampling_weight: 0.20

  validation_manifest: data/manifests/validation.jsonl
  test_manifest: data/manifests/test.jsonl

  input_representation:
    type: phonemes
    lexical_stress: true
    emotion_tokens: true
    focus_tokens: true
    prosody_tokens: true

audio:
  sample_rate: 22050
  n_fft: 1024
  n_mels: 80
  hop_length: 256
  win_length: 1024
  f_min: 0
  f_max: 8000

model:
  architecture: vits
  multi_speaker: true
  speaker_conditioning: true
  emotion_conditioning: true
  intensity_conditioning: true
  language_conditioning: false

trainer:
  accelerator: auto
  devices: 1
  precision: 16-mixed
  batch_size: 8
  accumulate_grad_batches: 4
  max_steps: 100000
  validation_interval: 2000
  log_interval: 100

optimizer:
  name: adamw
  learning_rate: 0.0001
  weight_decay: 0.01

checkpoint:
  save_last: true
  save_top_k: 5
  monitor: validation_loss
  mode: min
  every_n_steps: 2000

evaluation:
  fixed_sentences: data/evaluation/fixed_sentences.jsonl
  generate_every_n_steps: 2000
  metrics:
    - mcd
    - f0_rmse
    - f0_correlation
    - stoi
    - pesq
```

## 4.5 Experiment outputs

Every training run should produce:

```text
outputs/runs/<run-id>/
├── resolved_config.yaml
├── environment.json
├── git_state.json
├── mappings/
├── checkpoints/
├── logs/
├── validation_audio/
├── validation_metrics/
└── run_summary.json
```

## 4.6 Local and remote execution

The same configuration should work on:

- CPU for smoke tests;
- Apple MPS where supported;
- NVIDIA CUDA for full training;
- remote GPU infrastructure.

Hardware-specific settings should be overrides rather than separate experiment definitions.

---

# 5. Inference Pipeline

## 5.1 Responsibilities

The inference pipeline connects preprocessing, model inference, vocoding, and postprocessing.

```text
text
  ↓
preprocess
  ↓
model adapter
  ↓
mel or waveform
  ↓
vocoder
  ↓
postprocess
  ↓
audio
```

## 5.2 CLI

```bash
tts-synthesize \
  --text "Sunt foarte bucuros să te revăd!" \
  --model outputs/runs/vits-ro/checkpoints/best.ckpt \
  --speaker spk_01 \
  --emotion happy \
  --intensity high \
  --output outputs/example.wav
```

From a file:

```bash
tts-synthesize \
  --input-file examples/story.txt \
  --model outputs/runs/vits-ro/checkpoints/best.ckpt \
  --speaker spk_01 \
  --emotion-mode automatic \
  --output outputs/story.wav
```

## 5.3 Manual and automatic control

The inference pipeline should support:

```text
manual:
    caller provides emotion and intensity

automatic:
    preprocessing pipeline predicts emotion and intensity

hybrid:
    caller overrides selected predictions
```

Manual values always take priority and their provenance must be recorded as `user`.

---

# 6. Evaluation Pipeline

## 6.1 Text-frontend evaluation

Metrics may include:

- normalization exact match;
- normalization token accuracy;
- sentence-boundary F1;
- tokenization F1;
- phoneme error rate;
- syllabification F1;
- lexical-stress accuracy;
- emotion macro F1;
- valence/arousal error;
- intensity weighted kappa;
- focus token F1;
- pause-boundary F1;
- context improvement;
- serialization success rate;
- latency and peak memory.

## 6.2 Acoustic evaluation

Metrics may include:

- Mel Cepstral Distortion with DTW;
- F0 RMSE;
- F0 correlation;
- STOI;
- PESQ;
- duration error;
- speaker-embedding similarity;
- emotion-classification accuracy;
- real-time factor.

## 6.3 Perceptual evaluation

Planned listening tests:

- Mean Opinion Score for naturalness;
- intelligibility;
- speaker similarity;
- emotion appropriateness;
- emotion intensity;
- AB or ABX model comparisons.

## 6.4 Evaluation CLI

```bash
tts-evaluate frontend \
  --predictions outputs/preprocess/test.jsonl \
  --references data/evaluation/frontend_test.jsonl \
  --output-dir outputs/evaluation/frontend
```

```bash
tts-evaluate acoustic \
  --generated outputs/generated \
  --reference data/test \
  --metrics mcd,f0_rmse,f0_correlation,stoi,pesq \
  --output-dir outputs/evaluation/acoustic
```

---

# 7. Repository Layout

This is the target layout. Pieces that exist today are marked `[done]`
below; everything else is still planned. The `preprocess` pipeline is the
only one with code so far — see
[`src/expressive_tts/preprocess/README.md`](src/expressive_tts/preprocess/README.md)
for what it currently does, how to run it, and what's intentionally
deferred. `LICENSE` and `Makefile` have not been added yet.

```text
romanian-expressive-tts/
├── README.md
├── LICENSE
├── pyproject.toml                     [done]
├── requirements.txt                   [done, not in original plan]
├── Makefile
│
├── configs/
│   ├── preprocess/
│   │   ├── default.yaml                       [done]
│   │   ├── normalize_only.yaml                [done]
│   │   ├── units.yaml                         [done, not in original plan]
│   │   ├── currencies.yaml                    [done, not in original plan]
│   │   ├── abbreviations.yaml                 [done, not in original plan]
│   │   ├── phoneme_inventory.yaml              [done, not in original plan]
│   │   ├── pronunciation_overrides.yaml        [done, not in original plan]
│   │   ├── stress_overrides.yaml               [done, not in original plan]
│   │   ├── intensifiers.yaml                   [done, not in original plan]
│   │   ├── diminishers.yaml                    [done, not in original plan]
│   │   ├── interjection_emotions.yaml          [done, not in original plan]
│   │   ├── formal_markers.yaml                 [done, not in original plan]
│   │   ├── discourse_markers.yaml              [done, not in original plan]
│   │   ├── pronunciation.yaml                 [done]
│   │   ├── expressive.yaml                    [done, partial — emotion+intensity+focus+prosody+interjections (interjections included but disabled by default)]
│   │   └── dataset.yaml
│   ├── postprocess/
│   │   └── default.yaml
│   ├── training/
│   │   ├── vits_romanian.yaml
│   │   ├── vits_romanian_emotional.yaml
│   │   └── matcha_romanian.yaml
│   ├── inference/
│   │   └── default.yaml
│   └── evaluation/
│       ├── frontend.yaml
│       └── acoustic.yaml
│
├── src/
│   ├── expressive_tts/
│       ├── preprocess/
│       │   ├── README.md                      [done, not in original plan]
│       │   ├── pipeline.py                    [done]
│       │   ├── registry.py                    [done]
│       │   ├── cleaner.py                     [done, not in original plan]
│       │   ├── protected_spans.py             [done, not in original plan]
│       │   ├── sentence_segmenter.py          [done, not in original plan — rule-based v0]
│       │   ├── numbers_ro.py                  [done, not in original plan]
│       │   ├── trace_utils.py                 [done, not in original plan]
│       │   ├── normalizer.py                  [done]
│       │   ├── linguistic.py                  [done]
│       │   ├── phonemizer.py                  [done]
│       │   ├── stress.py                      [done]
│       │   ├── emotion.py                  [done]
│       │   ├── focus.py                    [done]
│       │   ├── prosody.py                  [done]
│       │   ├── interjections.py           [done]
│       │   ├── context.py                 [done]
│       │   ├── serializers.py              [done]
│       │   ├── schemas.py                  [done, moved from src/expressive_tts/schemas/preprocess.py — text-processing module consolidation]
│       │   ├── cli.py                      [done, moved from src/expressive_tts/cli/preprocess.py — same consolidation]
│       │   └── readers/
│       │       ├── text.py                    [done — text/stdin/file only, no base.py yet]
│       │       ├── tabular.py
│       │       ├── json_reader.py
│       │       └── manifest.py
│       │
│       ├── postprocess/
│       │   ├── pipeline.py
│       │   ├── silence.py
│       │   ├── loudness.py
│       │   ├── concatenation.py
│       │   ├── export.py
│       │   └── validation.py
│       │
│       ├── training/
│       │   ├── pipeline.py
│       │   ├── datasets.py
│       │   ├── samplers.py
│       │   ├── callbacks.py
│       │   ├── checkpoints.py
│       │   └── adapters/
│       │       ├── base.py
│       │       ├── vits.py
│       │       └── matcha.py
│       │
│       ├── inference/
│       │   ├── pipeline.py
│       │   ├── controls.py
│       │   └── adapters/
│       │
│       ├── evaluation/                        (deviated — see objective_evaluation/ below, a sibling top-level package, not nested here. Frontend-component evaluation is built and lives there today; acoustic.py/perceptual.py/reports.py-equivalents for postprocess/training/inference remain aspirational once those modules exist)
│       │
│       ├── schemas/                           (removed — preprocess.py moved into preprocess/schemas.py; training.py/output.py remain aspirational, for when those modules exist)
│       │
│       └── cli/                               (removed — preprocess.py moved into preprocess/cli.py; postprocess.py/train.py/synthesize.py/evaluate.py remain aspirational stubs for modules not built yet)
│   │
│   └── objective_evaluation/                  [done, not in original plan — sibling package to expressive_tts under src/, not nested inside it; the app is moving toward multiple independent pipelines]
│       ├── schemas.py                         [done, moved from src/expressive_tts/schemas/evaluation.py]
│       ├── sources.py                         [done, moved from scripts/fetch_evaluation_sources.py]
│       ├── build_dataset.py                   [done, moved from scripts/build_evaluation_set.py]
│       ├── evaluate_emotion.py                [done, moved from scripts/evaluate_emotion.py]
│       ├── evaluate_focus.py                  [done, moved from scripts/evaluate_focus.py]
│       ├── evaluate_interjections.py          [done, moved from scripts/evaluate_interjections.py]
│       ├── evaluate_serialization.py          [done, moved from scripts/evaluate_serialization.py]
│       ├── evaluate_context.py                [done, moved from scripts/evaluate_context.py]
│       ├── train_emotion_classifier.py        [done, moved from scripts/train_emotion_classifier.py]
│       ├── evaluate_emotion_classifier.py     [done, moved from scripts/evaluate_emotion_classifier.py]
│       ├── benchmark.py                       [done, moved from scripts/benchmark_pipeline.py]
│       └── error_report.py                    [done, moved from scripts/error_report.py]
│   └── tts_training/                          [done, not in original plan — standalone VITS training module; sibling package, depends only on expressive_tts.preprocess; data/manifests kept external. See src/tts_training/README.md]
│       ├── paths.py                           [external corpus/manifest path resolution ($TTS_DATASETS_DIR / --corpus-root)]
│       ├── frontend/symbols.py               [Coqui `characters` vocab from the phoneme inventory; use_phonemes=False]
│       ├── data/readers.py                    [per-corpus readers: MARA, HRIA, SWARA (multi-speaker), Common Voice (tsv+clips)]
│       ├── data/manifest.py                   [build Coqui manifest via the preprocess pipeline — runs locally, no Coqui]
│       ├── data/formatter.py                  [Coqui formatter for the manifest]
│       ├── vits/config.py                     [base multi-speaker VITS config (22.05 kHz)]
│       ├── train.py                           [base (neutral) training entry — CUDA GPU box, lazy Coqui import]
│       └── finetune.py                        [emotion fine-tune scaffold — deferred until emotional RO speech exists]
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── manifests/
│   ├── mappings/
│   ├── lexicons/                              (not used yet — NRC emotion lexicons live in .cache/lexicons/, gitignored, not here: their license prohibits redistribution, see data/external/SOURCES.md)
│   ├── external/                              [done, not in original plan — cached REDv2/RONEC source rows + SOURCES.md]
│   └── evaluation/                            [done — 304/300+ sentences, 43/30+ paragraphs; see data/evaluation/README.md's two-tier (hand-curated + bulk-sampled) explanation]
│
├── scripts/                                   (only one-time environment bootstrap now — evaluation logic lives in objective_evaluation/ above)
│   ├── download_trankit_model.py               [done — transformer POS/lemma/deps (XLM-R via Trankit)]
│   ├── download_emotion_model.py               [done — transformer emotion model (XLM-R)]
│   ├── fetch_emotion_lexicon.py                [done, not in original plan — NRC lexicon, focus layer]
│   ├── audit_audio.py
│   ├── build_manifest.py
│   ├── validate_manifest.py
│   └── export_experiment.py
│
├── tests/
│   ├── unit/                                  [done, partial — preprocess only]
│   ├── integration/                           [done, partial — preprocess only]
│   ├── regression/
│   └── fixtures/
│
├── examples/
│   ├── preprocess_text.py                     [done]
│   ├── preprocess_csv.py
│   ├── train_model.py
│   └── synthesize.py
│
├── notebooks/
│   └── analysis/
│
└── outputs/
    ├── preprocess/
    ├── runs/
    ├── synthesis/
    └── evaluation/
```

---

# 8. Component Interfaces

Each pipeline component should implement a small interface.

```python
from typing import Protocol


class Processor(Protocol):
    name: str
    version: str
    provides: set[str]
    requires: set[str]

    def process(self, document, config):
        ...
```

The pipeline registry should:

- resolve requested layers;
- determine dependencies;
- construct the execution graph;
- reject cycles;
- skip unnecessary processors;
- cache deterministic results;
- record processor versions;
- collect warnings and errors.

Example:

```python
pipeline.process(
    text="Am 25 de exemple.",
    include={"normalized", "phonemes"},
)
```

Resolved execution:

```text
cleaner → normalizer → tokenizer → phonemizer
```

The emotion, context, interjection, and prosody processors are not executed.

---

# 9. Data and Schema Principles

## 9.1 Preserve source data

- raw files are read-only;
- generated data is written to a separate directory;
- original text is never overwritten;
- generated interjections are stored separately;
- every transformation is traceable.

## 9.2 Provenance

Every annotation should identify its origin:

```text
user
source
rule
lexicon
predicted
generated
fallback
```

## 9.3 Confidence and uncertainty

- uncertain values are explicitly marked;
- low-confidence emotion may become `unspecified`;
- fallback pronunciation must be visible;
- no component should invent a high-confidence value when evidence is missing.

## 9.4 Schema versioning

Serialized results must contain:

```json
{
  "schema_version": "1.0"
}
```

Breaking schema changes require a new major version.

---

# 10. Logging and Reproducibility

Every command should record:

- command and arguments;
- resolved configuration;
- software version;
- processor and model versions;
- input-file checksum;
- start and finish time;
- record counts;
- warning and error counts;
- output-file checksums.

Example summary:

```json
{
  "records_read": 1000,
  "records_processed": 992,
  "records_with_warnings": 17,
  "records_failed": 8,
  "duration_seconds": 41.3
}
```

---

# 11. Testing Strategy

## Unit tests

Test isolated behavior:

- number normalization;
- date normalization;
- abbreviations;
- phoneme conversion;
- stress lookup;
- emotion rules;
- pause rules;
- interjection restrictions;
- serializers;
- CSV parsing.

## Integration tests

Test:

- raw text to JSON;
- CSV to JSON Lines;
- manifest to processed manifest;
- preprocessing to inference;
- inference to postprocessing;
- resolved configuration persistence.

## Regression tests

Every corrected error should receive a permanent regression test.

## Required edge cases

- empty input;
- malformed Unicode;
- missing CSV columns;
- quoted delimiters;
- multiline CSV fields;
- duplicate IDs;
- unsupported symbols;
- code-switching;
- long paragraphs;
- ambiguous numbers;
- formal text with emotional vocabulary;
- existing interjections;
- repeated punctuation.

---

# 12. Planned Command Summary

```bash
# Process one text
tts-preprocess --text "Text..." --include normalized,phonemes

# Process a CSV
tts-preprocess --input-file data.csv --format csv \
  --text-column text --include all

# Postprocess generated audio
tts-postprocess --config configs/postprocess/default.yaml

# Train a model
tts-train --config configs/training/vits_romanian.yaml

# Synthesize audio
tts-synthesize --text "Text..." --model checkpoint.ckpt

# Evaluate the frontend
tts-evaluate frontend --predictions predictions.jsonl \
  --references references.jsonl

# Evaluate generated audio
tts-evaluate acoustic --generated outputs/generated \
  --reference data/test
```

---

# 13. Initial Development Order

- [x] Define preprocessing schemas.
- [ ] Implement file readers and writers. (partial: plain text/stdin/file reader only; no CSV/TSV/JSON/manifest readers or a dedicated writer abstraction yet)
- [x] Implement component registry and dependency resolution.
- [x] Implement Unicode cleanup.
- [x] Implement Romanian normalization.
- [x] Add linguistic analysis. (transformer Trankit backend, XLM-RoBERTa, with automatic Stanza fallback on Python 3.13; see `src/expressive_tts/preprocess/linguistic.py`)
- [x] Add phonemization. (espeak-backed cascade; see `src/expressive_tts/preprocess/phonemizer.py`)
- [x] Add syllabification and lexical stress. (see `src/expressive_tts/preprocess/stress.py`)
- [x] Add transformer emotion analysis. (multilingual XLM-RoBERTa classifier; see `src/expressive_tts/preprocess/emotion.py`)
- [x] Add focus and prosody. (see `src/expressive_tts/preprocess/focus.py`, `src/expressive_tts/preprocess/prosody.py`)
- [x] Add controlled interjection suggestions. (see `src/expressive_tts/preprocess/interjections.py`; disabled by default, gated on a coarse formal/conversational document-style detector)
- [x] Add context smoothing. (see `src/expressive_tts/preprocess/context.py`; emotion only, not prosody; local prediction always preserved separately)
- [x] Add serializers. (see `src/expressive_tts/preprocess/serializers.py`; canonical JSON, annotated text, control tokens, illustrative SSML — VITS/Matcha adapters still deferred)
- [x] Add preprocessing CLI. (text/stdin/file input modes; JSON output plus `--serialize {control_tokens,annotated_text,ssml}`)
- [x] Add frontend evaluation. (`objective_evaluation.evaluate_*` per component; `data/evaluation/` scaled to 304 sentences / 43 paragraphs)
- [ ] Add postprocessing interfaces.
- [ ] Add training interfaces.
- [ ] Add VITS adapter.
- [ ] Add Matcha-TTS adapter when GPU resources are available.
- [ ] Add inference orchestration.
- [ ] Add acoustic and perceptual evaluation.

---

# 14. Technical References

- Trankit (transformer-based multilingual NLP toolkit): <https://github.com/nlp-uoregon/trankit>
- Multilingual emotion model: <https://huggingface.co/tabularisai/multilingual-emotion-classification>
- Romanian Universal Dependencies treebank: <https://universaldependencies.org/treebanks/ro_rrt/index.html>
- RoLEX lexical resource: <https://www.cambridge.org/core/journals/natural-language-engineering/article/abs/rolex-the-development-of-an-extended-romanian-lexical-dataset-and-its-evaluation-at-predicting-concurrent-lexical-information/96A3933A2028BD8EE605E0E672A57EB2>
- eSpeak-ng pronunciation and stress notation: <https://github.com/espeak-ng/espeak-ng/blob/master/docs/dictionary.md>
- Matcha-TTS: <https://github.com/shivammehta25/Matcha-TTS>

