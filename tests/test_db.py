import asyncio

from bot import db


def run(coro):
    return asyncio.run(coro)


def test_user_name_roundtrip():
    run(db.init_db())
    assert run(db.get_user_name(1)) is None
    run(db.set_user_name(1, "Кабан"))
    run(db.set_user_name(1, "Кабан Старий"))
    assert run(db.get_user_name(1)) == "Кабан Старий"


def test_new_version_replaces_old():
    run(db.init_db())
    old = run(db.add_document("Внески 2025", "a.pdf", "Під дубом", "дубовий гай",
                              "внески", 1, "Кабан", [("4 жолуді", b"\x00" * 8)]))
    same = run(db.find_active_same_document("під дубом", "Дубовий гай", "внески"))
    assert [r[0] for r in same] == [old]
    run(db.deactivate_document(old))
    new = run(db.add_document("Внески 2026", "b.pdf", "Під дубом", "дубовий гай",
                              "внески", 1, "Кабан", [("5 жолудів", b"\x00" * 8)]))
    texts = [r[0] for r in run(db.get_active_chunks())]
    assert "5 жолудів" in texts and "4 жолуді" not in texts
    assert new != old


def test_history_limit_and_clear():
    run(db.init_db())
    for i in range(10):
        run(db.add_message(7, "user", f"m{i}"))
    hist = run(db.get_history(7, 6))
    assert [h["content"] for h in hist] == [f"m{i}" for i in range(4, 10)]
    run(db.clear_history(7))
    assert run(db.get_history(7, 6)) == []