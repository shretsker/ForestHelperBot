import json
import os
import re

from groq import AsyncGroq

from bot.config import GROQ_API_KEY, GROQ_MODEL

client = AsyncGroq(api_key=GROQ_API_KEY)

METADATA_PROMPT = """Ти розбираєш документ з порадами лісових звірів. Українською.
Поверни ТІЛЬКИ JSON з полями:
- "title": назва документа. Бери з заголовка або першого речення-назви в тексті (наприклад «Членські внески»). Назву файлу використовуй лише якщо в тексті немає жодної назви
- "glove": назва рукавички (спільноти), для якої документ. Лише якщо вона ЯВНО названа (наприклад «Під дубом»). Фрази на кшталт «біля нашої рукавички» без назви означають null
- "forest_part": конкретна частина лісу, для якої документ: «хвойний ліс», «листяний ліс», «біля греблі», «під ялинами» тощо. Якщо в тексті названо будь-яку таку частину (навіть у формі «всього хвойного лісу», «у соснах»), запиши її в називному відмінку («хвойний ліс»). null, якщо частина не названа
- "forest_wide": true ЛИШЕ якщо текст прямо каже, що поради для ВСЬОГО лісу або для всіх, і при цьому ніяка конкретна частина лісу не названа. Якщо forest_part не null, forest_wide завжди false
- "topic": тема одним-трьома словами (внески, гриби, мисливці, пожежі, сезонні справи тощо)
Нічого не вигадуй: якщо невпевнений, став null."""


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


async def extract_metadata(text: str, filename: str) -> dict:
    resp = await client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": METADATA_PROMPT},
            {"role": "user", "content": f"Назва файлу: {filename}\n\nТекст:\n{text[:6000]}"},
        ],
    )
    data = json.loads(resp.choices[0].message.content)

    title = (data.get("title") or "").strip()
    if not title or _looks_like_filename(title, filename):
        title = _fallback_title(text, filename)

    forest_part = data.get("forest_part") or None
    forest_wide = bool(data.get("forest_wide")) and not forest_part

    return {
        "title": title,
        "glove": data.get("glove") or None,
        "forest_part": forest_part,
        "forest_wide": forest_wide,
        "topic": data.get("topic") or None,
    }