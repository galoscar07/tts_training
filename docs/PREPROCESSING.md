# Romanian Expressive Text Preprocessing

`expressive_tts.preprocess` turns raw Romanian text into a structured,
TTS-ready representation: normalized text, linguistic analysis, IPA phonemes
with lexical stress, sentence-level emotion, prosody control values, and
serialized control tokens. It runs fully locally.

All examples below are **real output** from the pipeline, not illustrations.

---

## 1. Quickstart

**Python API**

```python
from expressive_tts.preprocess.pipeline import PreprocessPipeline

pipe = PreprocessPipeline()
result = pipe.process(
    "Sunt foarte fericit că am reușit!",
    include={"normalized", "linguistic", "phonemes", "lexical_stress",
             "emotion", "focus", "prosody"},
)
print(result.sentences[0].emotion.label)      # -> happy
print(result.phoneme_text)                    # -> sˈunt fˈɔarte fˌeɾitʃˈit kˈə ˈam reuʃˈit !
```

**CLI** (installed as `tts-preprocess`)

```bash
tts-preprocess --text "Sunt foarte fericit!" --profile expressive
tts-preprocess --input-file in.txt --serialize control_tokens
echo "Ce panică!" | tts-preprocess --stdin
```

**Only compute what you ask for.** `include={...}` (or a profile) selects
output layers; the pipeline resolves dependencies and skips everything else.
Requesting only `normalized` never loads the phonemizer or the emotion model.

---

## 2. Pipeline architecture

Raw text flows through independent processors, each adding one layer:

```
clean → sentences → normalized → linguistic → phonemes → lexical_stress
                                     ├→ emotion → context
                                     ├→ focus
                                     ├→ prosody
                                     └→ interjections
```

| Layer | Processor | Adds |
|---|---|---|
| `clean` | `cleaner` | Unicode NFC, `ş/ţ→ș/ț`, whitespace/quote normalization, document-style detection |
| `sentences` | `sentence_segmenter` | sentence spans (rule-based v0) |
| `normalized` | `normalizer` | numbers, dates, times, %, currencies, units, abbreviations spelled out |
| `tokens`,`linguistic` | `linguistic` | tokens, lemma, UPOS/XPOS, morphology, dependencies, negation, sentence type |
| `phonemes` | `phonemizer` | IPA per token + `phoneme_text` |
| `syllables`,`lexical_stress` | `stress` | syllables + stressed-syllable index |
| `emotion`,`intensity` | `emotion` | label + confidence + valence/arousal + intensity |
| `focus` | `focus` | per-token emphasis score |
| `prosody` | `prosody` | pauses, rate, pitch, energy, terminal contour |
| `interjections` | `interjections` | optional interjection suggestions (off by default) |
| `context` | `context` | context-smoothed emotion across sentences |

Every predicted field carries a **provenance** (`rule` / `predicted` /
`fallback` / `user` / `source`) and a **confidence**, so downstream consumers
know what to trust.

---

## 3. Components

### 3.1 Unicode cleaning
Normalizes to NFC, converts legacy Romanian `ş/ţ` to the cedilla-correct
`ș/ț`, collapses whitespace, standardizes quotes/apostrophes, and preserves
prosody-relevant punctuation. Also classifies the document as
`formal` / `conversational` (gates interjection insertion).

### 3.2 Sentence segmentation
Splits into sentences while protecting abbreviations, decimals, dates, URLs
and e-mails from false boundaries.

### 3.3 Normalization
Deterministic, fully traced expansion of non-lexical tokens into their spoken
form.

**Input**
```
Dr. Popescu a trimis 25 kg pe 12.07.2026, la ora 14:30.
```
**Normalized**
```
Doctor Popescu a trimis douăzeci și cinci de kilograme pe doisprezece iulie
două mii douăzeci și șase, la ora paisprezece și treizeci de minute.
```

Handled categories:

| Category | Example in → out |
|---|---|
| Abbreviations | `Dr.` → `Doctor` |
| Cardinals | `25` → `douăzeci și cinci` |
| Units | `kg` → `de kilograme` |
| Dates | `12.07.2026` → `doisprezece iulie două mii douăzeci și șase` |
| Times | `14:30` → `paisprezece și treizeci de minute` |
| Currencies, percentages, decimals, Roman-numeral ordinals | supported |

Every change is recorded in `result.trace` (operation, original, replacement),
and diacritics are never lost.

### 3.4 Linguistic analysis
Transformer-based (Trankit / XLM-RoBERTa, with automatic Stanza fallback):
tokens, lemma, UPOS, morphological features, dependency relations, negation,
and sentence type.

For `"Nu pot să cred, chiar s-a întâmplat?"`:

| token | lemma | UPOS | key feats |
|---|---|---|---|
| Nu | nu | PART | `Polarity=Neg` |
| pot | putea | VERB | `Mood=Ind` |
| să | să | PART | `Mood=Sub` |
| cred | crede | VERB | `Mood=Ind` |
| s- | sine | PRON | `Case=Acc` |
| a | avea | AUX | |
| întâmplat | întâmpla | VERB | |

Derived: `sentence_type = interrogative`, `is_negated = true` (from the
`Polarity=Neg` on *Nu*).

Sentence type is inferred from the root verb's mood + terminal punctuation:
`declarative` / `interrogative` / `exclamative` / `imperative` / `incomplete`.

### 3.5 Phonemization
Cascade: project overrides → eSpeak IPA → grapheme fallback (flagged). Output
is IPA with primary/secondary stress marks, per token and joined as
`phoneme_text`.

```
Sunt foarte fericit că am reușit!
→ sˈunt fˈɔarte fˌeɾitʃˈit kˈə ˈam reuʃˈit !
```
Every phoneme symbol is validated against the canonical inventory
(`configs/preprocess/phoneme_inventory.yaml`); anything outside it is reported.

### 3.6 Syllabification & lexical stress
Splits each token into syllables and marks the **stressed** one (0-based index).

| token | syllables | stressed index |
|---|---|---|
| kilograme | `ki · lo · gra · me` | 2 → **gra** |
| fericit | `fe · ri · cit` | 2 → **cit** |
| întâmplat | `în · tâm · plat` | 2 → **plat** |
| Popescu | `Po · pes · cu` | 1 → **pes** |

Lexical stress (which syllable in a word) is kept **separate** from focus
(which word in the sentence) — see 3.8.

### 3.7 Emotion & intensity
A multilingual XLM-RoBERTa classifier (`emotion_xlmr_v1`) run locally.
Labels: `happy · angry · sad · fear · surprise · neutral · unspecified`.
Also produces valence, arousal (0–1), an intensity bucket (`low/medium/high`),
and the full probability distribution. Low-confidence/tied predictions abstain
to `unspecified`.

| sentence | label | conf | valence | arousal | intensity |
|---|---|---|---|---|---|
| Sunt foarte fericit că am reușit! | happy | 0.98 | 0.84 | 0.65 | high |
| Nu pot să cred, chiar s-a întâmplat? | surprise | 0.79 | 0.63 | 0.73 | high |
| Doctor Popescu a trimis … | neutral | 0.96 | 0.49 | 0.41 | low |

Distribution for the happy example:
```json
{"happy": 0.979, "surprise": 0.009, "neutral": 0.007,
 "angry": 0.003, "sad": 0.001, "fear": 0.001}
```

### 3.8 Focus / emphasis
Per-token continuous score (0–1) for sentence-level prominence, from
capitalization, contrastive negation, intensifier targets, repetition,
emotion-bearing words, and dependency structure. Function words are suppressed
unless evidence is strong.

In `"Sunt foarte fericit că am reușit!"`, **fericit** scores **0.9** (the
emotional adjective under the intensifier *foarte*), the rest ~0. User-supplied
emphasis (`user_focus_words=`) overrides with 100% priority.

### 3.9 Prosody
Converts punctuation, syntax, emotion and focus into model-independent control
values, all clamped to safe ranges.

| sentence | rate | pitch | energy | pause_after | terminal_contour |
|---|---|---|---|---|---|
| happy (exclamative, high arousal) | 1.06 | 1.06 | 1.06 | 400 ms | falling |
| interrogative | 1.09 | 1.09 | 1.09 | 400 ms | **rising** |
| neutral declarative | 0.97 | 0.97 | 0.97 | 500 ms | falling |

Safe ranges: `speaking_rate 0.80–1.20`, `relative_pitch 0.85–1.20`,
`relative_energy 0.80–1.20`, `pause 0–1000 ms`. Higher arousal → higher
rate/pitch/energy; questions get a rising terminal contour. Focus is applied
locally (per-token pitch/energy on focused words), not to the whole sentence.

### 3.10 Interjection suggestions (optional, off by default)
Modes: `disabled` (default) / `suggest` (candidates returned separately) /
`insert` (writes to a *separate* `text_with_interjections` field, never the
original). Gated on a minimum emotion confidence and the formal/conversational
document style, so formal text is never altered. Ranked, capped at 2 candidates.

### 3.11 Context smoothing
Blends each sentence's emotion with the previous one's
(`context_score = α·current + β·previous`, α>β), without ever overwriting the
raw local prediction (`sentence.emotion` stays; `sentence.context_emotion` is
added). Skipped across paragraph boundaries and when local confidence is high;
records when and why it changed a prediction.

---

## 4. Serialization

Four output formats from one intermediate representation:

**Canonical JSON** — the full `PreprocessResult` (schema-versioned, pydantic-validated).

**Annotated text** — human-readable inline annotations:
```
Sunt foarte [FOCUS] fericit că am reușit ! [emotion=happy, intensity=high; pause=400ms]
```

**Control tokens** — discrete tokens for a TTS model (continuous values binned,
unknowns → `*_UNSPECIFIED`):
```
[SENT_EXCLAMATIVE]
[EMO_HAPPY]
[INT_HIGH]
[RATE_1.05]
[PITCH_1.05]
[ENERGY_1.05]
sˈunt fˈɔarte [FOCUS] fˌeɾitʃˈit kˈə ˈam reuʃˈit !
[BREAK_400]
```

**SSML-like** — illustrative SSML output (not schema-validated).

---

## 5. Provenance & confidence

Every predicted annotation is tagged so consumers can filter by trust level:

| provenance | meaning |
|---|---|
| `rule` | deterministic rule output |
| `predicted` | model prediction (e.g. emotion) |
| `fallback` | low-confidence fallback (e.g. grapheme phonemes), also logged to `warnings` |
| `user` | caller-provided (focus words, prosody overrides) |
| `source` | gold label from a dataset |

Uncertain results are marked, never silently accepted (e.g. an ambiguous
`"Ce panică!"`-type case can resolve to `unspecified`).

---

## 6. Profiles & configuration

Profiles (`configs/preprocess/*.yaml`) preset the `include` set:

| profile | layers |
|---|---|
| `normalize_only` | clean → normalized |
| `pronunciation` | + linguistic, phonemes, lexical_stress |
| `expressive` | emotion, intensity, focus, prosody, interjections, context |
| `default` | everything implemented |

```python
PreprocessPipeline.from_profile("expressive")
PreprocessPipeline.from_config("configs/preprocess/default.yaml")
```

Behavior is tuned by versioned YAML config files: `abbreviations`, `units`,
`currencies`, `intensifiers`, `diminishers`, `interjection_emotions`,
`discourse_markers`, `phoneme_inventory`, `pronunciation_overrides`,
`stress_overrides`, and more, all under `configs/preprocess/`.

---

## 7. Capabilities at a glance

- ✅ Deterministic, fully-traced Romanian text normalization (numbers, dates, times, currencies, %, units, abbreviations) with zero diacritic loss
- ✅ Sentence segmentation + transformer linguistic analysis (tokens, lemma, POS, morphology, dependencies, negation, sentence type)
- ✅ IPA phonemization with primary/secondary lexical stress and syllabification
- ✅ Transformer emotion (6 labels + `unspecified`), valence/arousal, intensity, full distribution
- ✅ Focus/emphasis detection (kept separate from lexical stress)
- ✅ Rule-based prosody: pauses, speaking rate, relative pitch/energy, terminal contour — range-safe
- ✅ Optional, safe interjection suggestions (never corrupts the source or formal text)
- ✅ Context-aware emotion smoothing across sentences
- ✅ Four serialization formats incl. TTS control tokens
- ✅ Provenance + confidence on every predicted field; on-demand layer computation; runs offline

---

## 8. Models & setup

- **Linguistic:** Trankit (XLM-RoBERTa) on Python ≤3.12, else Stanza — `scripts/download_trankit_model.py` / `download_stanza_model.py`
- **Phonemes:** the `espeak` binary on `PATH`
- **Emotion:** `tabularisai/multilingual-emotion-classification` — `scripts/download_emotion_model.py`
- **Lexicons (focus):** NRC EmoLex/VAD — `scripts/fetch_emotion_lexicon.py`

Install: `pip install -e ".[dev,linguistic,ai]"`. See
`src/expressive_tts/preprocess/README.md` for methodology details and
`preprocess/objectives.md` for the full component-by-component design +
evaluation notes.
