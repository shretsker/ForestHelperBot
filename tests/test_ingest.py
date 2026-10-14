import asyncio

import numpy as np

from bot.ingest import chunk_text, embed_texts, extract_text


def test_txt_utf8_and_cp1251():
    assert extract_text("Привіт, ліс".encode("utf-8"), "a.txt") == "Привіт, ліс"
    assert "ліс" in extract_text("ліс".encode("cp1251"), "b.txt")


def test_chunking_keeps_all_text():
    text = "\n".join(f"Порада {i}: " + "слово " * 40 for i in range(20))
    chunks = chunk_text(text, size=400, overlap=50)
    assert len(chunks) > 1 and all(len(c) <= 400 for c in chunks)
    assert "Порада 19" in "".join(chunks)


def test_embeddings_are_semantic():
    q, a, b = asyncio.run(embed_texts([
        "Скільки внесок під дубом?",
        "Внесок під дубом: 4 жолуді на місяць.",
        "Гриби з'являються після дощу в сосновому лісі.",
    ]))
    v = [np.frombuffer(x, dtype=np.float32) for x in (q, a, b)]
    cos = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    assert cos(v[0], v[1]) > cos(v[0], v[2])