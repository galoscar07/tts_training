"""Build the Phase 1 pilot evaluation set (preprocess/objectives.md).

Reads cached external sources from data/external/ (run
objective_evaluation.sources first) plus the in-repo datasets/ corpora,
curates a hand-picked pilot batch spanning every required register and
phenomenon, runs each sentence through the real PreprocessPipeline for its
deterministic fields, and writes data/evaluation/{dev,test,context}.jsonl.

Every subjective annotation (emotion when not sourced from a labeled
dataset, intensity, focus words, pause locations, lexical stress,
interjection judgments) is my own draft judgment, marked
provenance="predicted" — see data/evaluation/README.md and
src/expressive_tts/schemas/evaluation.py for why. Nothing here is gold
data until a human reviews it.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import yaml

from expressive_tts.preprocess import PreprocessPipeline
from expressive_tts.preprocess.sentence_segmenter import segment as segment_sentences
from objective_evaluation.schemas import (
    Annotation,
    ContextParagraph,
    EvaluationExample,
    LexicalStressAnnotation,
)
from expressive_tts.preprocess.schemas import Provenance

ROOT = Path(__file__).resolve().parents[2]
DATA_EXTERNAL = ROOT / "data" / "external"
DATA_EVALUATION = ROOT / "data" / "evaluation"
DATASETS = ROOT / "datasets"

REDV2_SOURCE = "REDv2 (Alegzandra/RED-Romanian-Emotion-Datasets), MIT"
RONEC_SOURCE = "RONEC (community-datasets/ronec on Hugging Face), MIT"
HRIA_LICENSE = "in-repo project data, see data/external/SOURCES.md"
MARA_LICENSE = "in-repo project data, see data/external/SOURCES.md"
HAND_AUTHORED = "hand-authored (gap-filling: lexical-stress minimal pairs)"

TIER2_REDV2_PER_LABEL = 30  # up to ~30 x 6 labels = ~180 additional real-gold-emotion examples
TIER2_RONEC_COUNT = 90
TIER2_SEED = 20260725  # fixed seed: reproducible builds, not "random" each run

TIER2_CONTEXT_MARA_TARGET = 20
TIER2_CONTEXT_HRIA_TARGET = 20

_PAUSE_CATEGORY = {",": "comma", ";": "semicolon", ":": "colon", ".": "period", "!": "exclamation", "?": "question"}
_ELLIPSIS_PATTERN = re.compile(r"\.\.\.|…")


def derive_pause_locations(text: str) -> list[dict]:
    """Draft pause candidates from punctuation offsets (provenance=predicted
    — which punctuation marks are true prosodic pauses is a judgment call,
    this just proposes candidates for review)."""
    pauses = []
    for match in _ELLIPSIS_PATTERN.finditer(text):
        pauses.append({"offset": match.start(), "category": "ellipsis"})
    ellipsis_spans = {(m.start(), m.end()) for m in _ELLIPSIS_PATTERN.finditer(text)}
    for i, char in enumerate(text):
        if char in _PAUSE_CATEGORY and not any(start <= i < end for start, end in ellipsis_spans):
            pauses.append({"offset": i, "category": _PAUSE_CATEGORY[char]})
    return sorted(pauses, key=lambda p: p["offset"])


def run_pipeline(text: str) -> tuple[str, list[tuple[int, int]]]:
    pipeline = PreprocessPipeline()
    result = pipeline.process(text, include={"normalized"})
    boundaries = [(s.start, s.end) for s in result.sentences]
    return result.normalized_text, boundaries


def load_json(name: str) -> list[dict]:
    return json.loads((DATA_EXTERNAL / name).read_text(encoding="utf-8"))


def find_by_text(rows: list[dict], text: str) -> dict:
    for row in rows:
        if row["text"] == text:
            return row
    raise KeyError(f"text not found in cache: {text!r}")


def grep_line(path: Path, needle: str, column: int) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        if needle in line:
            return parts[column]
    raise KeyError(f"line not found in {path}: {needle!r}")


# ---------------------------------------------------------------------------
# Curated pilot items
# ---------------------------------------------------------------------------


def build_items() -> list[dict]:
    redv2 = load_json("redv2_sample.json")
    ronec = load_json("ronec_sample.json")
    hria_final = DATASETS / "hria" / "catalina" / "metadata_final.txt"
    hria_simple = DATASETS / "hria" / "catalina" / "metadata_simple.txt"
    mara = DATASETS / "mara" / "metadata.csv"

    items: list[dict] = []

    def redv2_item(text: str, **kw) -> None:
        row = find_by_text(redv2, text)
        items.append(
            dict(
                text=text,
                source=f"REDv2/test#{row['text_id']}",
                license=REDV2_SOURCE,
                text_register="conversational",
                emotion=row["emotion"],
                emotion_provenance=Provenance.SOURCE,
                emotion_confidence=max(row["procentual_labels"].values()),
                **kw,
            )
        )

    def ronec_item(text: str, *, register: str = "formal", **kw) -> None:
        row = find_by_text(ronec, text)
        items.append(
            dict(
                text=text,
                source=f"ronec/train#{row['id']}",
                license=RONEC_SOURCE,
                text_register=register,
                emotion="neutral",
                emotion_provenance=Provenance.PREDICTED,
                emotion_confidence=0.7,
                **kw,
            )
        )

    def hria_final_item(text: str, **kw) -> None:
        emotion_ro = grep_line(hria_final, text, column=1)
        mapping = {"neutru": "neutral", "furios": "angry", "fericit": "happy"}
        items.append(
            dict(
                text=text,
                source=f"hria/catalina/metadata_final#{text[:24]!r}",
                license=HRIA_LICENSE,
                text_register="conversational",
                emotion=mapping[emotion_ro],
                emotion_provenance=Provenance.SOURCE,
                emotion_confidence=1.0,
                **kw,
            )
        )

    def hria_simple_item(text: str, *, emotion: str, **kw) -> None:
        items.append(
            dict(
                text=text,
                source="hria/catalina/metadata_simple",
                license=HRIA_LICENSE,
                text_register="conversational",
                emotion=emotion,
                emotion_provenance=Provenance.PREDICTED,
                emotion_confidence=0.6,
                **kw,
            )
        )

    def mara_item(text: str, **kw) -> None:
        items.append(
            dict(
                text=text,
                source="datasets/mara/metadata.csv",
                license=MARA_LICENSE,
                text_register="narrative",
                emotion="unspecified",
                emotion_provenance=Provenance.PREDICTED,
                emotion_confidence=0.4,
                **kw,
            )
        )

    def hand_item(text: str, *, register: str, emotion: str = "unspecified", **kw) -> None:
        items.append(
            dict(
                text=text,
                source=HAND_AUTHORED,
                license="N/A (hand-authored)",
                text_register=register,
                emotion=emotion,
                emotion_provenance=Provenance.PREDICTED,
                emotion_confidence=0.5,
                **kw,
            )
        )

    # --- REDv2: primary emotion ground truth, spanning all 6 labels --------
    redv2_item(
        "Ce panică!",
        phenomena=["exclamation", "existing_interjection"],
        intensity="high",
        sentence_type=["exclamative"],
        focus_words=["panică"],
        interjection_appropriate=True,
        acceptable_interjections=["Ce", "Vai"],
    )
    redv2_item(
        "O întâmplare înspăimântătoare",
        phenomena=["incomplete"],
        intensity="medium",
        sentence_type=["incomplete"],
        focus_words=["înspăimântătoare"],
        interjection_appropriate=False,
    )
    redv2_item(
        "Antrenamentele cognitive sunt extrem de utile, dar pot fi și foarte distractive.",
        phenomena=["declarative"],
        intensity="medium",
        sentence_type=["declarative"],
        focus_words=["utile", "distractive"],
        interjection_appropriate=False,
    )
    redv2_item(
        "La mulți ani! Sănătate, veselie și mult noroc! Chef mare să fie!",
        phenomena=["exclamation", "multi_sentence"],
        intensity="high",
        sentence_type=["exclamative", "exclamative", "exclamative"],
        focus_words=["mulți ani", "sănătate", "chef mare"],
        interjection_appropriate=True,
        acceptable_interjections=["Ura", "Noroc"],
    )
    redv2_item(
        "Este trist ce se întâmplă-n lume.",
        phenomena=["declarative"],
        intensity="medium",
        sentence_type=["declarative"],
        focus_words=["trist"],
        interjection_appropriate=True,
        acceptable_interjections=["Vai", "Ah"],
    )
    redv2_item(
        "Când mai vorbiți de secetă și de foamete uitați-vă la aceste imagini.",
        phenomena=["imperative"],
        intensity="medium",
        sentence_type=["imperative"],
        focus_words=["uitați-vă"],
        interjection_appropriate=False,
    )
    redv2_item(
        "Chiar dacă avem de respectat distanțarea fizică, atingerile ne definesc umanitatea",
        phenomena=["incomplete", "negation"],
        intensity="low",
        sentence_type=["incomplete"],
        focus_words=["atingerile", "umanitatea"],
        interjection_appropriate=False,
    )
    redv2_item(
        "Protest al agricultorilor dobrogeni. Seceta le-a distrus mai mult de jumătate din culturi",
        phenomena=["multi_sentence", "news"],
        intensity="medium",
        sentence_type=["declarative", "incomplete"],
        focus_words=["protest", "distrus"],
        interjection_appropriate=False,
    )
    redv2_item(
        "Oau, așa arată ploaia?",
        phenomena=["question", "existing_interjection"],
        intensity="medium",
        sentence_type=["interrogative"],
        focus_words=["ploaia"],
        interjection_appropriate=True,
        note="already contains 'Oau'",
    )
    redv2_item(
        "Triplul proiector cu laser de la Samsung este uimitor",
        phenomena=["incomplete", "technical"],
        intensity="medium",
        sentence_type=["incomplete"],
        focus_words=["uimitor"],
        interjection_appropriate=False,
    )

    # --- RONEC: formal/news register + normalizer-exercising numbers -------
    ronec_item(
        "Vechiul oraș Visoki a fost un faimos castel regal medieval în timpul secolului al "
        "XIV-lea, situat în Visoko , Bosnia și Herțegovina.",
        phenomena=["roman_numeral", "news"],
        intensity="unspecified",
        sentence_type=["declarative"],
        focus_words=["castel", "medieval"],
        interjection_appropriate=False,
    )
    ronec_item(
        "Premiera operei a avut loc la Milano, la „Teatro Cannobiana”, în ziua de 2 mai 1832.",
        register="news",
        phenomena=["dates", "news"],
        intensity="unspecified",
        sentence_type=["declarative"],
        focus_words=["premiera"],
        interjection_appropriate=False,
    )
    ronec_item(
        "Tarifele la gaze naturale pentru populație se majorează cu 5%.",
        register="news",
        phenomena=["percentage", "news"],
        intensity="unspecified",
        sentence_type=["declarative"],
        focus_words=["majorează"],
        interjection_appropriate=False,
    )
    ronec_item(
        "Până la urmă, intrusul a găsit în sertarul unui dulap 2100 de lei și a plecat cu banii.",
        register="news",
        phenomena=["currency", "news"],
        intensity="unspecified",
        sentence_type=["declarative"],
        focus_words=["intrusul", "2100 de lei"],
        interjection_appropriate=False,
    )
    ronec_item(
        "Mașina a ars în proporție de 80%, valoarea distrugerilor fiind de circa 1500 de lei.",
        register="news",
        phenomena=["percentage", "currency", "news"],
        intensity="unspecified",
        sentence_type=["declarative"],
        focus_words=["ars"],
        interjection_appropriate=False,
    )
    ronec_item(
        "Termenul de livrare a standardelor va fi de 15 aprilie 2006.",
        register="technical",
        phenomena=["dates", "technical"],
        intensity="unspecified",
        sentence_type=["declarative"],
        focus_words=["termenul"],
        interjection_appropriate=False,
    )
    ronec_item(
        "Astfel, valoarea punctului de pensie se va determina prin actualizarea valorii din "
        "luna decembrie a fiecărui an cu cel puțin rata inflației, prognozată pentru anul "
        "bugetar următor.",
        phenomena=["technical", "formal"],
        intensity="unspecified",
        sentence_type=["declarative"],
        focus_words=["punctului de pensie"],
        interjection_appropriate=False,
    )
    ronec_item(
        "Joi, 22 iunie, ora 17, primarul Timișoarei, dl Gheorghe Ciuhandu, și reprezentanți ai "
        "Primăriei vor avea o întâlnire cu cetățenii din cartierul Calea Girocului, la Școala "
        "cu clasele I-VIII nr. 25 de pe str. Cosminului.",
        register="news",
        phenomena=["abbreviation", "times", "dates", "news"],
        intensity="unspecified",
        sentence_type=["declarative"],
        focus_words=["întâlnire"],
        interjection_appropriate=False,
    )
    ronec_item(
        "(1) Consiliul Superior al Magistraturii propune Președintelui României numirea în "
        "funcție a judecătorilor și a procurorilor, cu excepția celor stagiari, în condițiile "
        "legii.",
        phenomena=["formal", "legal", "interjection_inappropriate"],
        intensity="unspecified",
        sentence_type=["declarative"],
        focus_words=["Consiliul Superior al Magistraturii"],
        interjection_appropriate=False,
        note="formal legal register — interjections would clearly be inappropriate",
    )

    # --- hria/catalina metadata_simple: numbers/prices/existing interjections
    hria_simple_item(
        "Produsul costă 149 de lei și include TVA 19%.",
        phenomena=["currency", "percentage"],
        intensity="unspecified",
        sentence_type=["declarative"],
        focus_words=["149 de lei"],
        interjection_appropriate=False,
        emotion="neutral",
    )
    hria_simple_item(
        "Am fost suprataxat cu 20 de dolari la supermarket.",
        phenomena=["currency"],
        intensity="medium",
        sentence_type=["declarative"],
        focus_words=["suprataxat"],
        interjection_appropriate=True,
        acceptable_interjections=["Vai", "Of"],
        emotion="angry",
    )
    hria_simple_item(
        "Zborul de la București la Londra durează aproximativ 3 ore și 20 de minute.",
        phenomena=["measurement_unit"],
        intensity="unspecified",
        sentence_type=["declarative"],
        focus_words=["durează"],
        interjection_appropriate=False,
        emotion="neutral",
    )
    hria_simple_item(
        "Uau! Pare un salariu frumos și potențial suficient pentru trai. Îmi imaginez că ar "
        "face bine moralului, dar e trist să aud că corporațiilor nu le place.",
        phenomena=["existing_interjection", "multi_sentence"],
        intensity="medium",
        sentence_type=["exclamative", "declarative", "declarative"],
        focus_words=["salariu", "trist"],
        interjection_appropriate=True,
        note="already contains 'Uau'",
        emotion="surprise",
    )
    hria_simple_item(
        "Bine. Bine. Ascultă-mă.",
        phenomena=["repeated_word", "imperative", "incomplete"],
        intensity="low",
        sentence_type=["incomplete", "incomplete", "imperative"],
        focus_words=["ascultă-mă"],
        interjection_appropriate=False,
        emotion="neutral",
    )

    # --- hria/catalina metadata_final: real emotion labels, conversational -
    hria_final_item(
        "Da, este acolo jos, la capătul drumului, lângă supermarket.",
        phenomena=["declarative"],
        intensity="low",
        sentence_type=["declarative"],
        focus_words=["capătul drumului"],
        interjection_appropriate=False,
    )
    hria_final_item(
        "Este un expert în arte sau științe, un profesor de rang înalt.",
        phenomena=["declarative", "formal"],
        intensity="low",
        sentence_type=["declarative"],
        focus_words=["expert"],
        interjection_appropriate=False,
    )
    hria_final_item(
        "Îmi pare foarte rău pentru orice neplăcere v-am cauzat.",
        phenomena=["formal", "apology"],
        intensity="medium",
        sentence_type=["declarative"],
        focus_words=["rău"],
        interjection_appropriate=False,
    )
    hria_final_item(
        "Am făcut 3 reclamații și încă nimeni nu m-a contactat!",
        phenomena=["numbers", "negation", "exclamation"],
        intensity="high",
        sentence_type=["exclamative"],
        focus_words=["nimeni", "3 reclamații"],
        interjection_appropriate=True,
        acceptable_interjections=["Of", "Vai"],
    )
    hria_final_item(
        "Când cineva îmi spune o minciună și o descopăr, ajung la culmea furiei.",
        phenomena=["declarative"],
        intensity="high",
        sentence_type=["declarative"],
        focus_words=["culmea furiei"],
        interjection_appropriate=False,
    )
    hria_final_item(
        "Când am născut un băiețel sănătos, am fost cea mai fericită!",
        phenomena=["exclamation"],
        intensity="high",
        sentence_type=["exclamative"],
        focus_words=["fericită"],
        interjection_appropriate=True,
        acceptable_interjections=["Vai", "Ce bine"],
    )
    hria_final_item(
        "Vai, Chris, sunt gata de mult timp!",
        phenomena=["existing_interjection", "exclamation"],
        intensity="medium",
        sentence_type=["exclamative"],
        focus_words=["gata"],
        interjection_appropriate=True,
        note="already contains 'Vai'",
    )

    # --- mara: narrative register (archaic literary Romanian) --------------
    mara_item(
        "dar era tânără și voinică și harnică și Dumnezeu a mai lăsat să aibă și noroc.",
        phenomena=["narrative", "declarative"],
        intensity="unspecified",
        sentence_type=["declarative"],
        focus_words=["tânără", "voinică", "harnică"],
        interjection_appropriate=False,
    )
    mara_item(
        "Numai în zilele de Sântă Marie se întoarce Mara cu coșurile deșerte la casa ei.",
        phenomena=["narrative", "declarative"],
        intensity="unspecified",
        sentence_type=["declarative"],
        focus_words=["Sântă Marie"],
        interjection_appropriate=False,
    )
    mara_item(
        "E însă altceva la mijloc.",
        phenomena=["narrative", "incomplete"],
        intensity="unspecified",
        sentence_type=["declarative"],
        focus_words=["altceva"],
        interjection_appropriate=False,
    )

    # --- hand-authored: lexical-stress minimal pairs (no sampled source
    # reliably provides these) -----------------------------------------------
    hand_item(
        "Am făcut cinci copii ale documentului.",
        register="technical",
        phenomena=["numbers", "hard_lexical_stress", "interjection_inappropriate"],
        intensity="unspecified",
        sentence_type=["declarative"],
        focus_words=["copii"],
        interjection_appropriate=False,
        lexical_stress=[("copii", ["co", "pii"], 0, 0.85)],  # "copii" = copies: CO-pii
        note="'copii' = copies (CO-pii), contrast with the children sense below",
    )
    hand_item(
        "Are trei copii veseli și jucăuși.",
        register="conversational",
        phenomena=["numbers", "hard_lexical_stress"],
        intensity="medium",
        sentence_type=["declarative"],
        focus_words=["copii"],
        interjection_appropriate=False,
        lexical_stress=[("copii", ["co", "pii"], 1, 0.85)],  # "copii" = children: co-PII
        note="'copii' = children (co-PII), contrast with the copies sense above",
    )
    hand_item(
        "Mama a spălat toată vesela după masă.",
        register="conversational",
        phenomena=["hard_lexical_stress"],
        intensity="unspecified",
        sentence_type=["declarative"],
        focus_words=["vesela"],
        interjection_appropriate=False,
        lexical_stress=[("vesela", ["ve", "se", "la"], 1, 0.8)],  # "vesela" = the dishes
        note="'vesela' = the dishes (ve-SE-la), contrast with the adjective below",
    )
    hand_item(
        "Fetița era veselă și zâmbea.",
        register="conversational",
        phenomena=["hard_lexical_stress"],
        intensity="medium",
        sentence_type=["declarative"],
        focus_words=["veselă"],
        interjection_appropriate=False,
        lexical_stress=[("veselă", ["ve", "se", "lă"], 0, 0.8)],  # "veselă" = cheerful
        note="'veselă' = cheerful (VE-se-lă), contrast with the dishes sense above",
        emotion="happy",
    )

    return items


# ---------------------------------------------------------------------------
# Tier 2: programmatically sampled bulk items (eval-set scaling)
#
# Tier 1 above (38 items) is hand-curated: every subjective field is my own
# individually-considered draft judgment. Scaling to 300+ sentences by hand
# with the same care isn't achievable in one pass, and silently generating
# 260+ "quick judgments" at that volume would be a materially different
# (and worse) kind of data dressed up as equivalent — so Tier 2 is built
# programmatically instead, with a stronger, more explicit caveat than even
# Tier 1's "draft": `intensity`/`focus_words`/`interjection_appropriate` are
# flagged confidence=0.0 with an "unreviewed" note rather than guessed.
# What *is* real: the text itself, REDv2's own human-annotated gold emotion
# label (kept, `provenance=SOURCE`, exactly as Tier 1 does), and mechanical
# fields (sentence_type from terminal punctuation, pause_locations from
# punctuation offsets — the same method Tier 1 already uses). See
# data/evaluation/README.md for the full two-tier explanation.
# ---------------------------------------------------------------------------


def _normalize_for_dedup(text: str) -> str:
    return " ".join(text.split()).strip().casefold()


def _sentence_type_from_punctuation(text: str) -> str:
    stripped = text.strip()
    if stripped.endswith("?"):
        return "interrogative"
    if stripped.endswith("!"):
        return "exclamative"
    return "declarative"


def _bulk_item(
    *, text: str, source: str, license: str, text_register: str, emotion: str, emotion_provenance, emotion_confidence: float
) -> dict:
    return dict(
        text=text,
        source=source,
        license=license,
        text_register=text_register,
        emotion=emotion,
        emotion_provenance=emotion_provenance,
        emotion_confidence=emotion_confidence,
        phenomena=["bulk_added"],
        intensity="unspecified",
        sentence_type=[_sentence_type_from_punctuation(text)],
        focus_words=[],
        interjection_appropriate=False,
        note="bulk-added, unreviewed (Tier 2 eval-set scaling — see data/evaluation/README.md)",
        bulk=True,
    )


def build_tier2_items(used_texts: set[str]) -> list[dict]:
    redv2 = load_json("redv2_sample.json")
    ronec = load_json("ronec_sample.json")
    formal_markers = [
        m.lower()
        for m in yaml.safe_load((ROOT / "configs" / "preprocess" / "formal_markers.yaml").read_text(encoding="utf-8"))
    ]

    seen = {_normalize_for_dedup(t) for t in used_texts}
    rng = random.Random(TIER2_SEED)
    items: list[dict] = []

    by_label: dict[str, list[dict]] = {}
    for row in redv2:
        by_label.setdefault(row["emotion"], []).append(row)

    for _label, rows in sorted(by_label.items()):
        rng.shuffle(rows)
        added = 0
        for row in rows:
            if added >= TIER2_REDV2_PER_LABEL:
                break
            key = _normalize_for_dedup(row["text"])
            if key in seen:
                continue
            seen.add(key)
            added += 1
            items.append(
                _bulk_item(
                    text=row["text"],
                    source=f"REDv2/test#{row['text_id']}",
                    license=REDV2_SOURCE,
                    text_register="conversational",
                    emotion=row["emotion"],
                    emotion_provenance=Provenance.SOURCE,
                    emotion_confidence=max(row["procentual_labels"].values()),
                )
            )

    ronec_shuffled = list(ronec)
    rng.shuffle(ronec_shuffled)
    added = 0
    for row in ronec_shuffled:
        if added >= TIER2_RONEC_COUNT:
            break
        key = _normalize_for_dedup(row["text"])
        if key in seen:
            continue
        seen.add(key)
        added += 1
        lowered = row["text"].lower()
        register = "formal" if any(marker in lowered for marker in formal_markers) else "news"
        items.append(
            _bulk_item(
                text=row["text"],
                source=f"ronec/train#{row['id']}",
                license=RONEC_SOURCE,
                text_register=register,
                emotion="unspecified",
                emotion_provenance=Provenance.PREDICTED,
                emotion_confidence=0.0,
            )
        )

    rng.shuffle(items)  # so a positional dev/test split isn't skewed by label/source grouping order
    return items


def build_context_paragraphs() -> list[ContextParagraph]:
    return [
        ContextParagraph(
            id="ctx-001",
            sentences=[
                "A rămas Mara, săraca, văduvă cu doi copii, sărăcuții de ei, dar era tânără "
                "și voinică și harnică și Dumnezeu a mai lăsat să aibă și noroc.",
                "E, nu-i vorbă, Bârzovanu, răposatul, era, când a fost, mai mult cârpaci decât "
                "cizmar și ședea mai bucuros la birt decât acasă; tot le-au mai rămas însă "
                "copiilor vreo două sute de pruni pe lunca Murășului, viuța din dealul despre "
                "Păuliș și casa, pe care muma lor o căpătase de zestre.",
            ],
            source="datasets/mara/metadata.csv (rows 2-7)",
            license=MARA_LICENSE,
            text_register="narrative",
        ),
        ContextParagraph(
            id="ctx-002",
            sentences=[
                "Astăzi m-am simțit foarte fericit că am reușit să fac niște treabă!",
                "Au fost niște săptămâni foarte ocupate.",
                "Am și un job nou la care lucrez.",
            ],
            source="hria/catalina/metadata_final#fericit-51",
            license=HRIA_LICENSE,
            text_register="conversational",
        ),
        ContextParagraph(
            id="ctx-003",
            sentences=[
                "Când fratele meu a luat note foarte mici la examene, am fost foarte furios.",
                "Mi-am lăsat studiile deoparte ca să-l ajut, pentru că de fiecare dată când "
                "fratele meu se descurcă prost la școală, familia mă învinovățește și pe mine.",
            ],
            source="hria/catalina/metadata_final#furios-51",
            license=HRIA_LICENSE,
            text_register="conversational",
        ),
    ]


def _read_pipe_delimited(path: Path) -> list[str]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("|", 1)
        if len(parts) == 2 and parts[1].strip():
            rows.append(parts[1].strip())
    return rows


def _reconstitute_sentences_from_fragments(fragments: list[str]) -> list[str]:
    """mara/metadata.csv rows are short TTS-recording fragments, not one
    sentence per row (verified: rows 2-3 together form a single sentence in
    the Tier 1 ctx-001 paragraph already in context.jsonl). Accumulate
    fragments until one ends in terminal punctuation to reconstruct real
    sentences before grouping them into paragraphs."""
    sentences: list[str] = []
    buffer: list[str] = []
    for fragment in fragments:
        buffer.append(fragment)
        if fragment.rstrip().endswith((".", "!", "?", "…")):
            sentences.append(" ".join(buffer))
            buffer = []
    if buffer:
        sentences.append(" ".join(buffer))
    return sentences


def build_tier2_context_paragraphs() -> list[ContextParagraph]:
    """Scale context.jsonl 3 -> 30+ real multi-sentence paragraphs
    (preprocess/objectives.md Phase 1's "at least 30 contextual
    paragraphs"). No per-sentence emotion labels are attached — the
    `ContextParagraph` schema doesn't require them, and Phase 11's own
    evaluation (`objective_evaluation.evaluate_context`) is a behavior sanity check,
    not a metric that would need them (see that script's docstring).

    Grouping into fixed-size chunks doesn't necessarily respect the
    source's own actual paragraph breaks (mara/metadata.csv doesn't
    preserve them; hria's rows are independent utterances, not a single
    continuous narrative) — documented here rather than implied as
    perfectly faithful paragraphing.
    """
    def _evenly_spaced(candidates: list, target: int) -> list:
        if len(candidates) <= target:
            return candidates
        step = len(candidates) / target
        return [candidates[int(i * step)] for i in range(target)]

    mara_rows = _read_pipe_delimited(DATASETS / "mara" / "metadata.csv")
    mara_sentences = _reconstitute_sentences_from_fragments(mara_rows[1:])  # skip title row
    chunk_size = 5
    mara_chunks = [
        (start, mara_sentences[start : start + chunk_size])
        for start in range(0, len(mara_sentences) - chunk_size + 1, chunk_size)
    ]

    hria_rows = _read_pipe_delimited(DATASETS / "hria" / "catalina" / "metadata_simple.txt")
    group_size = 3
    hria_chunks = []
    for start in range(0, len(hria_rows) - group_size + 1, group_size):
        group_text = " ".join(hria_rows[start : start + group_size])
        sentences = [s.text for s in segment_sentences(group_text) if s.text.strip()]
        if len(sentences) >= 2:  # not genuinely multi-sentence otherwise — skip rather than pad
            hria_chunks.append((start, sentences))

    paragraphs: list[ContextParagraph] = []
    for idx, (start, chunk) in enumerate(_evenly_spaced(mara_chunks, TIER2_CONTEXT_MARA_TARGET), start=1):
        paragraphs.append(
            ContextParagraph(
                id=f"ctx-mara-{idx:03d}",
                sentences=chunk,
                source=f"datasets/mara/metadata.csv (reconstructed sentences {start + 1}-{start + chunk_size})",
                license=MARA_LICENSE,
                text_register="narrative",
            )
        )
    for idx, (start, sentences) in enumerate(_evenly_spaced(hria_chunks, TIER2_CONTEXT_HRIA_TARGET), start=1):
        paragraphs.append(
            ContextParagraph(
                id=f"ctx-hria-{idx:03d}",
                sentences=sentences,
                source=f"hria/catalina/metadata_simple (rows {start + 1}-{start + group_size})",
                license=HRIA_LICENSE,
                text_register="conversational",
            )
        )

    return paragraphs


def to_example(index: int, item: dict, split: str) -> EvaluationExample:
    normalized_text, boundaries = run_pipeline(item["text"])
    is_bulk = item.get("bulk", False)
    bulk_note = item.get("note") if is_bulk else None

    lexical_stress = [
        LexicalStressAnnotation(
            word=word,
            syllables=syllables,
            stressed_syllable_index=stress_idx,
            provenance=Provenance.PREDICTED,
            confidence=confidence,
        )
        for word, syllables, stress_idx, confidence in item.get("lexical_stress", [])
    ]

    acceptable_interjections = item.get("acceptable_interjections")

    return EvaluationExample(
        id=f"{split}-{index:03d}",
        text=item["text"],
        source=item["source"],
        license=item["license"],
        text_register=item["text_register"],
        phenomena=item["phenomena"],
        split=split,
        expected_normalized_text=normalized_text,
        sentence_boundaries=boundaries,
        emotion=Annotation(
            value=item["emotion"],
            provenance=item["emotion_provenance"],
            confidence=item["emotion_confidence"],
        ),
        intensity=Annotation(
            value=item["intensity"],
            provenance=Provenance.PREDICTED,
            confidence=0.0 if is_bulk else 0.6,
            note=bulk_note,
        ),
        sentence_type=Annotation(value=item["sentence_type"], provenance=Provenance.PREDICTED, confidence=0.75),
        focus_words=Annotation(
            value=item["focus_words"],
            provenance=Provenance.PREDICTED,
            confidence=0.0 if is_bulk else 0.6,
            note=bulk_note,
        ),
        pause_locations=Annotation(
            value=derive_pause_locations(item["text"]), provenance=Provenance.PREDICTED, confidence=0.7
        ),
        lexical_stress=lexical_stress,
        interjection_appropriate=Annotation(
            value=item["interjection_appropriate"],
            provenance=Provenance.PREDICTED,
            confidence=0.0 if is_bulk else 0.65,
            note=bulk_note or item.get("note"),
        ),
        acceptable_interjections=(
            Annotation(value=acceptable_interjections, provenance=Provenance.PREDICTED, confidence=0.5)
            if acceptable_interjections
            else None
        ),
    )


def _split(items: list[dict], ratio: float = 0.55) -> tuple[list[dict], list[dict]]:
    point = round(len(items) * ratio)
    return items[:point], items[point:]


def main() -> None:
    DATA_EVALUATION.mkdir(parents=True, exist_ok=True)

    tier1_items = build_items()
    tier2_items = build_tier2_items(used_texts={item["text"] for item in tier1_items})

    # split each tier independently, then concatenate — keeps Tier 1's
    # existing dev-001..dev-021/test-001..test-017 IDs stable (several
    # reports in data/evaluation/ already cite specific IDs) instead of
    # reshuffling everything once Tier 2 is appended.
    tier1_dev, tier1_test = _split(tier1_items)
    tier2_dev, tier2_test = _split(tier2_items)
    dev_items = tier1_dev + tier2_dev
    test_items = tier1_test + tier2_test

    dev = [to_example(i + 1, item, "dev") for i, item in enumerate(dev_items)]
    test = [to_example(i + 1, item, "test") for i, item in enumerate(test_items)]
    context = build_context_paragraphs() + build_tier2_context_paragraphs()

    for path, examples in [
        (DATA_EVALUATION / "dev.jsonl", dev),
        (DATA_EVALUATION / "test.jsonl", test),
    ]:
        with path.open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(example.model_dump_json() + "\n")

    with (DATA_EVALUATION / "context.jsonl").open("w", encoding="utf-8") as handle:
        for paragraph in context:
            handle.write(paragraph.model_dump_json() + "\n")

    print(f"Tier 1 (hand-curated): {len(tier1_items)} items")
    print(f"Tier 2 (bulk, unreviewed subjective fields): {len(tier2_items)} items")
    print(f"dev.jsonl: {len(dev)} examples")
    print(f"test.jsonl: {len(test)} examples")
    print(f"context.jsonl: {len(context)} paragraphs")


if __name__ == "__main__":
    main()
