from expressive_tts.preprocess.cleaner import CleanerProcessor
from expressive_tts.preprocess.registry import PipelineDocument


def clean(text: str) -> str:
    document = PipelineDocument(original_text=text)
    CleanerProcessor().process(document, {})
    return document.clean_text


def test_legacy_diacritics_converted():
    # ş/ţ (cedilla forms) -> ș/ț (correct comma-below forms).
    assert clean("Ştiu că e ţeapă") == "Știu că e țeapă"


def test_quotes_normalized():
    assert clean("„Bună” ziua ’n’") == '"Bună" ziua \'n\''


def test_whitespace_collapsed_and_stripped():
    assert clean("  Bună   ziua  \n\t din nou  ") == "Bună ziua din nou"


def test_original_text_untouched():
    document = PipelineDocument(original_text="  ş  ")
    CleanerProcessor().process(document, {})
    assert document.original_text == "  ş  "


def test_produces_trace_entries():
    document = PipelineDocument(original_text="  ş  ")
    CleanerProcessor().process(document, {})
    operations = {entry.operation for entry in document.trace}
    assert "legacy_diacritics" in operations
    assert "whitespace" in operations
