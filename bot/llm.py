import asyncio
import json
import logging
import os
import re
import time

import groq
from groq import AsyncGroq

from bot.config import GROQ_API_KEY, GROQ_MODELS

log = logging.getLogger(__name__)

# max_retries=0: ми самі перемикаємось між моделями, не чекаючи на повтори SDK
client = AsyncGroq(api_key=GROQ_API_KEY, max_retries=0)

FAILOVER_ERRORS = (
    groq.RateLimitError,
    groq.NotFoundError,
    groq.InternalServerError,
    groq.APIConnectionError,
)
ROUNDS = 3
DEFAULT_COOLDOWN_S = 30.0
MAX_SLEEP_S = 10.0
_cooldown: dict[str, float] = {}  # model -> monotonic time when it may be used again

METADATA_PROMPT = """Ти розбираєш документ з порадами лісових звірів. Українською.
Поверни ТІЛЬКИ JSON з полями:
- "title": назва документа. Бери з заголовка або першого речення-назви в тексті (наприклад «Членські внески»). Назву файлу використовуй лише якщо в тексті немає жодної назви
- "glove": назва рукавички (спільноти), для якої документ. Лише якщо вона ЯВНО названа (наприклад «Під дубом»). Фрази на кшталт «біля нашої рукавички» без назви означають null
- "forest_part": конкретна частина лісу, для якої документ: «хвойний ліс», «листяний ліс», «біля греблі» тощо. Якщо в тексті названо будь-яку таку частину (навіть у формі «всього хвойного лісу», «у соснах»), запиши її в називному відмінку («хвойний ліс»). Назва рукавички НЕ є частиною лісу: якщо документ для рукавички, а частина лісу не названа окремо, forest_part = null. null, якщо частина не названа
- "forest_wide": true ЛИШЕ якщо текст прямо каже, що поради для ВСЬОГО лісу або для всіх, і при цьому ніяка конкретна частина лісу не названа. Якщо forest_part не null, forest_wide завжди false
- "topic": тема одним-трьома словами (внески, гриби, мисливці, пожежі, сезонні справи тощо)
Правила:
- Якщо вказано «Відомі рукавички» чи «Відомі частини лісу», а документ (або уточнення автора) стосується однієї з них, запиши назву РІВНО так, як у списку. Збіг із відомою рукавичкою означає glove, а не forest_part. Не вигадуй нових назв: нову назву бери лише з тексту чи уточнення автора, як вона там написана.
- Якщо вказано «Існуючі теми» і документ про ту саму тему, що одна з них, запиши topic РІВНО в тій самій формі. Не вигадуй синонімів.
- Якщо є уточнення автора документа («Автор документа сам уточнив…»), воно має пріоритет над текстом: «для всього лісу» означає forest_wide=true, назва рукавички чи частини лісу береться з нього.
Нічого не вигадуй: якщо невпевнений, став null."""


def _retry_after(exc: Exception) -> float | None:
    try:
        return float(exc.response.headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return None


async def _complete(system: str, user: str) -> str:
    """Chat completion in JSON mode with automatic fallback across GROQ_MODELS."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last_exc: Exception | None = None

    for _ in range(ROUNDS):
        now = time.monotonic()
        candidates = [m for m in GROQ_MODELS if _cooldown.get(m, 0.0) <= now] or list(GROQ_MODELS)
        shortest_wait: float | None = None

        for model in candidates:
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=messages,
                )
                log.debug("LLM model used: %s", model)
                return resp.choices[0].message.content
            except FAILOVER_ERRORS as exc:
                last_exc = exc
                if isinstance(exc, groq.NotFoundError):
                    pause = 3600.0  # model is not available for this key
                else:
                    pause = min(_retry_after(exc) or DEFAULT_COOLDOWN_S, 3600.0)
                _cooldown[model] = time.monotonic() + pause
                shortest_wait = pause if shortest_wait is None else min(shortest_wait, pause)
                log.warning("model %s failed (%s), trying next", model, type(exc).__name__)

        await asyncio.sleep(min(shortest_wait or 1.0, MAX_SLEEP_S))

    assert last_exc is not None
    raise last_exc


def _fallback_title(text: str, filename: str) -> str:
    """First line / first sentence of the text, if it looks like a title."""
    first_line = next((l.strip() for l in text.splitlines() if l.strip()), "")
    candidate = re.split(r"(?<=[.!?:])\s", first_line, maxsplit=1)[0].strip(" .:")
    if 3 <= len(candidate) <= 80:
        return candidate
    return os.path.splitext(filename)[0]


def _looks_like_filename(title: str, filename: str) -> bool:
    stem = os.path.splitext(filename)[0].lower()
    t = title.strip().lower()
    return t in (stem, filename.lower()) or t.endswith((".pdf", ".docx", ".txt"))


async def extract_metadata(text: str, filename: str, hint: str | None = None,
                           gloves: list[str] | None = None,
                           parts: list[str] | None = None,
                           topics: list[str] | None = None) -> dict:
    extra = ""
    if gloves:
        extra += f"\nВідомі рукавички: {', '.join(gloves)}"
    if parts:
        extra += f"\nВідомі частини лісу: {', '.join(parts)}"
    if topics:
        extra += f"\nІснуючі теми: {', '.join(topics)}"
    if hint:
        extra += f"\nАвтор документа сам уточнив, для кого він: {hint}"
    raw = await _complete(
        METADATA_PROMPT,
        f"Назва файлу: {filename}{extra}\n\nТекст:\n{text[:6000]}",
    )
    data = json.loads(raw)

    title = (data.get("title") or "").strip()
    if not title or _looks_like_filename(title, filename):
        title = _fallback_title(text, filename)

    glove = data.get("glove") or None
    forest_part = data.get("forest_part") or None
    if glove and forest_part and forest_part.strip().casefold() == glove.strip().casefold():
        forest_part = None
    forest_wide = bool(data.get("forest_wide")) and not forest_part

    return {
        "title": title,
        "glove": glove,
        "forest_part": forest_part,
        "forest_wide": forest_wide,
        "topic": data.get("topic") or None,
    }


async def chat_json(system: str, user: str) -> dict:
    return json.loads(await _complete(system, user))