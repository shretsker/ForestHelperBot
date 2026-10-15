import os
from datetime import datetime, timezone

import aiosqlite

from bot.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    tg_id INTEGER PRIMARY KEY,
    real_name TEXT
);
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    filename TEXT,
    glove TEXT,
    forest_part TEXT,
    topic TEXT,
    uploader_tg_id INTEGER NOT NULL,
    uploader_name TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES documents(id),
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tg_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(tg_id, id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


async def init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(SCHEMA)
        await db.commit()


async def get_user_name(tg_id: int) -> str | None:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT real_name FROM users WHERE tg_id=?", (tg_id,))
        row = await cur.fetchone()
        return row[0] if row else None


async def set_user_name(tg_id: int, name: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO users(tg_id, real_name) VALUES(?, ?) "
            "ON CONFLICT(tg_id) DO UPDATE SET real_name=excluded.real_name",
            (tg_id, name),
        )
        await db.commit()


def _norm(value: str | None) -> str:
    return " ".join((value or "").casefold().split())


async def find_active_same_document(glove, forest_part, topic):
    """Active documents with the same glove + forest part + topic (case-insensitive, Unicode-safe)."""
    key = (_norm(glove), _norm(forest_part), _norm(topic))
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT id, title, uploader_name, uploaded_at, glove, forest_part, topic "
            "FROM documents WHERE active=1"
        )
        rows = await cur.fetchall()
    return [
        r[:4] for r in rows
        if (_norm(r[4]), _norm(r[5]), _norm(r[6])) == key
    ]


async def deactivate_document(doc_id: int) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE documents SET active=0 WHERE id=?", (doc_id,))
        await db.commit()


async def add_document(title, filename, glove, forest_part, topic,
                       uploader_tg_id, uploader_name, chunks) -> int:
    """chunks: list of (text, embedding_bytes). Returns new document id."""
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "INSERT INTO documents(title, filename, glove, forest_part, topic, "
            "uploader_tg_id, uploader_name, uploaded_at) VALUES(?,?,?,?,?,?,?,?)",
            (title, filename, glove, forest_part, topic,
             uploader_tg_id, uploader_name, now),
        )
        doc_id = cur.lastrowid
        await db.executemany(
            "INSERT INTO chunks(doc_id, text, embedding) VALUES(?,?,?)",
            [(doc_id, t, e) for t, e in chunks],
        )
        await db.commit()
        return doc_id


async def get_active_chunks():
    """All chunks of active documents with their metadata."""
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT c.text, c.embedding, d.id, d.title, d.glove, d.forest_part, "
            "d.topic, d.uploader_name, d.uploaded_at "
            "FROM chunks c JOIN documents d ON d.id=c.doc_id WHERE d.active=1"
        )
        return await cur.fetchall()


async def add_message(tg_id: int, role: str, content: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO messages(tg_id, role, content) VALUES(?,?,?)",
            (tg_id, role, content),
        )
        await db.commit()


async def get_history(tg_id: int, limit: int):
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT role, content FROM messages WHERE tg_id=? "
            "ORDER BY id DESC LIMIT ?",
            (tg_id, limit),
        )
        rows = await cur.fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


async def clear_history(tg_id: int) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM messages WHERE tg_id=?", (tg_id,))
        await db.commit()


async def get_known_places():
    """Gloves and forest parts that exist in active documents."""
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT DISTINCT glove, forest_part FROM documents WHERE active=1"
        )
        rows = await cur.fetchall()
    gloves = sorted({g for g, _ in rows if g})
    parts = sorted({p for _, p in rows if p})
    return gloves, parts