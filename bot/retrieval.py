from dataclasses import dataclass

import numpy as np

from bot import db
from bot.config import TOP_K
from bot.db import _norm as norm
from bot.ingest import embed_texts


MIN_SCORE_BY_LEVEL = {0: 0.15, 1: 0.20, 2: 0.30, 3: 0.15}


@dataclass
class Hit:
    text: str
    score: float
    level: int
    doc_id: int
    title: str
    glove: str | None
    forest_part: str | None
    uploader_name: str
    uploaded_at: str


def scope_level(doc_glove, doc_part, glove, part) -> int | None:
    """0 = same glove, 1 = same forest part, 2 = general, 3 = no place in query.
    None = document belongs to another place and must be excluded."""
    g, p = norm(doc_glove), norm(doc_part)
    q_g, q_p = norm(glove), norm(part)
    if not q_g and not q_p:
        return 3
    if q_g and g == q_g:
        return 0
    if q_p and p == q_p and (not g or not q_g):
        return 1
    if not g and not p:
        return 2
    return None


async def search(question: str, glove: str | None = None,
                 part: str | None = None, top_k: int = TOP_K) -> list[Hit]:
    rows = await db.get_active_chunks()
    if not rows:
        return []

    query_text = " ".join(x for x in (glove, part, question) if x)
    q = np.frombuffer((await embed_texts([query_text]))[0], dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-9)

    hits: list[Hit] = []
    for text, emb, doc_id, title, d_glove, d_part, _topic, uploader, uploaded_at in rows:
        level = scope_level(d_glove, d_part, glove, part)
        if level is None:
            continue
        v = np.frombuffer(emb, dtype=np.float32)
        score = float(q @ v / (np.linalg.norm(v) + 1e-9))
        if score < MIN_SCORE_BY_LEVEL[level]:
            continue
        hits.append(Hit(text, score, level, doc_id, title, d_glove, d_part,
                        uploader, uploaded_at))

    hits.sort(key=lambda h: -h.score)
    top = hits[:top_k]
    top.sort(key=lambda h: h.level)  # stable: most specific first, then by score
    return top