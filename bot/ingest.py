import asyncio
import io
import re

import numpy as np
from docx import Document
from fastembed import TextEmbedding
from pypdf import PdfReader

from bot.config import EMBED_MODEL

SUPPORTED = (".pdf", ".docx", ".txt")
_embedder: TextEmbedding | None = None


def extract_text(data: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        text = "\n".join(p.text for p in doc.paragraphs)
    elif name.endswith(".txt"):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("cp1251", errors="replace")
    else:
        raise ValueError("unsupported format")
    text = text.strip()
    if not text:
        raise ValueError("no text found in document")
    return text


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) + 1 <= size:
            current = f"{current}\n{p}".strip()
            continue
        if current:
            chunks.append(current)
        while len(p) > size:
            chunks.append(p[:size])
            p = p[size - overlap:]
        current = p
    if current:
        chunks.append(current)
    return chunks


def _get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(EMBED_MODEL)
    return _embedder


def _embed_sync(texts: list[str]) -> list[bytes]:
    vecs = list(_get_embedder().embed(texts))
    return [np.asarray(v, dtype=np.float32).tobytes() for v in vecs]


async def embed_texts(texts: list[str]) -> list[bytes]:
    return await asyncio.to_thread(_embed_sync, texts)