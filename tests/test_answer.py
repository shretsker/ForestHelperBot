import os

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test")
os.environ.setdefault("GROQ_API_KEY", "test")

from bot.answer import (  # noqa: E402
    format_source, merge_clarification, needs_clarification, place_note, resolve_place,
)
from bot.retrieval import Hit  # noqa: E402


def hit(doc_id, glove=None, part=None, score=0.5):
    return Hit("t", score, 3, doc_id, "Док", glove, part, "Кабан", "2026-10-14T09:00:00+00:00")


def test_source_formats():
    s = format_source("Членські внески", "Під дубом", None, "Кабан", "2026-10-14T09:00:00+00:00")
    assert s == '— з документа "Членські внески" рукавички "Під дубом", надіслав(ла) Кабан, 14.10.2026'
    assert "частина лісу: хвойний ліс" in format_source("П", None, "хвойний ліс", "Сова", "2026-10-14T09:00:00+00:00")
    assert "для всього лісу" in format_source("П", None, None, "Сова", "2026-10-14T09:00:00+00:00")


def test_clarify_only_when_place_unknown_and_ambiguous():
    hits = [hit(1, glove="Під дубом"), hit(2, glove="Під ялинами")]
    assert len(needs_clarification(hits, None, None)) == 2
    assert needs_clarification(hits, "Під дубом", None) == []
    assert needs_clarification([hit(1, glove="Під дубом"), hit(2)], None, None) == []
    assert needs_clarification([hit(1), hit(2)], None, None) == []


def test_clarify_ignores_much_weaker_places():
    strong = hit(1, glove="Під дубом", score=0.5)
    weak = hit(2, glove="Під ялинами", score=0.2)
    assert needs_clarification([strong, weak], None, None) == []


def test_resolve_place_matches_known_only():
    known = ["Під дубом", "Під ялинами"]
    assert resolve_place("під дубом", known) == "Під дубом"
    assert resolve_place("Під вербою", known) is None
    assert resolve_place(None, known) is None


def test_place_note_only_for_single_specific_place():
    assert place_note([hit(1, glove="Під ялинами")]) == 'рукавичка "Під ялинами"'
    assert place_note([hit(1, part="хвойний ліс"), hit(2)]) == "частина лісу: хвойний ліс"
    assert place_note([hit(1, glove="Під дубом"), hit(2, glove="Під ялинами")]) is None
    assert place_note([hit(1)]) is None


def test_merge_clarification_only_for_short_answer_to_clarifying_question():
    history = [
        {"role": "user", "content": "Скільки внесок?"},
        {"role": "assistant", "content": "Для якого місця питаєш? У архіві є поради для: ..."},
    ]
    assert merge_clarification(history, "під ялинами") == "Скільки внесок? (під ялинами)"
    assert merge_clarification(history, "Де шукати гриби біля верби?") == "Де шукати гриби біля верби?"
    assert merge_clarification([], "під ялинами") == "під ялинами"
    normal = [
        {"role": "user", "content": "Скільки внесок під дубом?"},
        {"role": "assistant", "content": "4 жолуді"},
    ]
    assert merge_clarification(normal, "а до якого числа?") == "а до якого числа?"