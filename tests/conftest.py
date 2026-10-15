import os

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test")
os.environ.setdefault("GROQ_API_KEY", "test")

import pytest  # noqa: E402

from bot import db  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Every test gets its own empty SQLite file."""
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))