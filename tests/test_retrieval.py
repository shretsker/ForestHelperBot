import asyncio
import os

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test")
os.environ.setdefault("GROQ_API_KEY", "test")

import pytest  # noqa: E402

from bot import db  # noqa: E402
from bot.ingest import embed_texts  # noqa: E402
from bot.retrieval import scope_level, search  # noqa: E402


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
    run(db.init_db())


def add(title, glove, part, text):
    emb = run(embed_texts([text]))[0]
    return run(db.add_document(title, "f.txt", glove, part, "тема",
                               1, "Кабан", [(text, emb)]))


def test_scope_levels():
    assert scope_level("Під дубом", None, "під дубом", None) == 0
    assert scope_level(None, "Хвойний ліс", None, "хвойний ліс") == 1
    assert scope_level(None, None, None, "хвойний ліс") == 2
    assert scope_level("Під дубом", None, None, "хвойний ліс") is None
    assert scope_level("Під ялинами", None, "Під дубом", None) is None
    assert scope_level("Під дубом", None, None, None) == 3


def test_specific_advice_goes_first_and_other_places_excluded():
    add("Грибний путівник хвойного лісу", None, "хвойний ліс",
        "У хвойному лісі гриби з'являються на 2–3 день після дощу.")
    add("Їстівне й отруйне", None, None,
        "Гриби шукай після дощу, коли земля волога, але не хлюпає.")
    add("Членські внески", "Під дубом", None,
        "Внесок під дубом: 4 жолуді з дорослого звіра на місяць.")
    hits = run(search("Через скільки днів після дощу йти по гриби?",
                      part="хвойний ліс"))
    titles = [h.title for h in hits]
    assert titles[0] == "Грибний путівник хвойного лісу"
    assert "Їстівне й отруйне" in titles
    assert "Членські внески" not in titles


def test_glove_filter_and_relevance():
    add("Членські внески", "Під дубом", None,
        "Внесок під дубом: 4 жолуді з дорослого звіра на місяць.")
    add("Внески під ялинами", "Під ялинами", None,
        "Внесок під ялинами: 6 горіхів на місяць.")
    hits = run(search("Скільки внесок?", glove="Під дубом"))
    assert hits and hits[0].title == "Членські внески"
    assert all(h.glove == "Під дубом" for h in hits)


def test_empty_archive_returns_nothing():
    assert run(search("Де гриби?")) == []