# External sources

Raw, read-only inputs (readme.md section 9.1). Regenerate the cached JSON
files with `python -m objective_evaluation.sources`; the curated
evaluation set in `data/evaluation/` is built from these caches by
`objective_evaluation.build_dataset`.

## NRC Emotion Lexicon (EmoLex) and NRC VAD Lexicon — Phase 6

- **What**: EmoLex — 14,182 English words × 105 language translations × 8
  Plutchik emotions (anger/anticipation/disgust/fear/joy/sadness/surprise/
  trust) + 2 sentiments, binary word-emotion association; 4,463 words have
  ≥1 association. VAD — continuous valence/arousal/dominance (0-1) per
  word. Both include Romanian translations (Google Translate-derived).
- **Source**: <https://saifmohammad.com/WebPages/AccessResource.htm>
  (non-commercial research direct download,
  `NRC-Suite-of-Sentiment-Emotion-Lexicons.zip`, ~109MB), retrieved
  2026-07-25.
- **License**: "Free for non-commercial research and educational purposes."
  **Redistribution is explicitly prohibited** ("do not redistribute the
  data") — unlike REDv2/RONEC above (MIT), this data is **not** cached into
  this directory or committed. `scripts/fetch_emotion_lexicon.py`
  downloads it fresh into `.cache/lexicons/` (gitignored) on demand; run
  that script yourself to populate it, and re-run it if the cache is ever
  cleared.
- **Citation** (required by the license terms):
  Saif Mohammad and Peter Turney. "Crowdsourcing a Word-Emotion Association
  Lexicon." Computational Intelligence, 29(3), 2013.
  Saif Mohammad. "Obtaining Reliable Human Ratings of Valence, Arousal, and
  Dominance for 20,000 English Words." ACL 2018.
- **Used for**: `preprocess/emotion.py`'s rule-based emotion baseline
  (lexicon lookup by lemma + VAD-derived valence/arousal). NRC's 8
  categories are mapped onto this project's label set where there's a
  direct equivalent (`joy→happy`, `anger→angry`, `sadness→sad`, `fear→fear`,
  `surprise→surprise`); `anticipation`/`disgust`/`trust` have none and are
  dropped from the categorical vote, same treatment as REDv2's dropped
  "Trust" category above.
- **Cache**: `.cache/lexicons/nrc_emolex_ro.json`,
  `.cache/lexicons/nrc_vad_ro.json` (both gitignored, not here).

## REDv2 — Romanian Emotions Dataset v2

- **What**: 5,449 Romanian tweets, multi-label human-annotated (3
  annotators) across 7 emotions.
- **Source**: <https://github.com/Alegzandra/RED-Romanian-Emotion-Datasets>
  (`REDv2/data/test.json`), retrieved 2026-07-25.
- **License**: MIT (Alexandra Ciobotaru, 2021/2022).
- **Citation**: Ciobotaru, A., et al. "RED: A Novel Dataset for Romanian
  Emotion Detection from Tweets." RANLP 2021; "RED v2: Enhancing RED Dataset
  for Multi-Label Emotion Detection." LREC 2022.
- **Used for**: primary emotion ground truth (`Provenance.SOURCE`). Only
  single-agreed-label rows are kept, excluding the "Trust" category (no
  equivalent in our label set) and rows containing tweet artifacts
  (`<|PERSON|>`/`<|URL|>`/hashtags). Label order:
  `['Sadness','Surprise','Fear','Anger','Neutral','Trust','Joy']` mapped to
  `sad/surprise/fear/angry/neutral/(dropped)/happy`.
- **Cache**: `redv2_sample.json` — a filtered candidate subset, not the full
  dataset.

## RONEC — Romanian Named Entity Corpus

- **What**: 12,300 Romanian sentences (SETimes news, Wikipedia, CommonCrawl)
  with NER annotations; pre-tokenized.
- **Source**: <https://huggingface.co/datasets/community-datasets/ronec>,
  fetched via the `datasets-server` REST API (`train` split), retrieved
  2026-07-25.
- **License**: MIT.
- **Citation**: Dumitrescu, S.D. & Avram, A-M. "Introducing RONEC - the
  Romanian Named Entity Corpus." arXiv:1909.01247.
- **Used for**: formal/news-register sentences. Tokens are detokenized back
  into sentence strings using the dataset's `space_after` flags. Initially
  fetched 3 pages (~150 candidates after filtering) for the Tier 1 pilot;
  raised to 15 pages (~820 candidates) for the Tier 2 eval-set-scaling
  pass — see `data/evaluation/README.md`.
- **Cache**: `ronec_sample.json`.

## In-repo datasets (already present before this pass)

- `datasets/hria/catalina/metadata_final.txt` — conversational,
  emotion-labeled (neutru/furios/fericit) speech transcripts.
- `datasets/hria/catalina/metadata_simple.txt` — conversational speech
  transcripts, unlabeled, rich in numbers/dates/prices/percentages and
  existing interjections.
- `datasets/mara/metadata.csv` — literary narration (Ioan Slavici, *Mara*),
  used for the narrative register.

No separate license file was found for these alongside the datasets; treat
them as internal project data pending clarification, not for redistribution
outside this project.
