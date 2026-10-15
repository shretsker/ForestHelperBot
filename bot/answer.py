import logging
from datetime import datetime

import groq

from bot import db
from bot.config import HISTORY_LIMIT
from bot.db import _norm as norm
from bot.llm import chat_json
from bot.retrieval import Hit, search

log = logging.getLogger(__name__)

UNCLEAR_TEXT = (
    "Не зовсім зрозумів, про що ти питаєш. Спробуй так:\n"
    "• «Скільки внесок під дубом?»\n"
    "• «Де шукати гриби біля верби?»\n"
    "Щоб додати пораду, надішли документ (PDF, Word або txt)."
)
NOT_FOUND_TEXT = "У документах цього немає."
ERROR_TEXT = "Щось пішло не так, спробуй ще раз трохи згодом."
RATE_LIMIT_TEXT = "Зараз забагато запитів, спробуй ще раз за хвилину."
CLARIFY_PREFIX = "Для якого місця питаєш?"

ANALYZE_PROMPT = """Ти аналізуєш повідомлення звіра до бота з порадами лісового архіву. Українською.
Відомі рукавички: {gloves}
Відомі частини лісу: {parts}

Історія розмови (може бути порожньою). Після відповідей бота може стояти тег [місце відповіді: ...]:
{history}

Поверни ТІЛЬКИ JSON:
- "intent": "question", якщо це питання чи прохання про пораду (включно з уточненням до попередньої відповіді); "unclear" для привітань, випадкових слів, повідомлень без сенсу
- "standalone_question": питання, переписане так, щоб воно було зрозуміле без історії
- "glove": рукавичка ЗІ СПИСКУ відомих, якщо питання (або уточнення) явно про неї, інакше null
- "forest_part": частина лісу ЗІ СПИСКУ відомих, якщо питання явно про неї, інакше null

Правила:
- Якщо повідомлення це лише назва місця або короткий фрагмент («під ялинами», «а до якого числа?»), а в історії є питання звіра або перепитування бота, то це уточнення. intent = "question", а standalone_question поєднує останнє питання з історії з новим місцем чи деталлю.
- Якщо звір не називає місця, але це явне продовження теми, і в останній відповіді бота є тег [місце відповіді: ...], використай це місце: заповни glove або forest_part і додай місце у standalone_question.
- Місце для уточнення бери з ОСТАННЬОГО питання звіра, останнього перепитування бота або останнього тега місця. Ігноруй старіші, якщо вони суперечать.
- Якщо звір назвав місце, якого немає у списках (наприклад «біля греблі»), став glove і forest_part у null, але залиш це місце у standalone_question.
- Не вигадуй місць, яких немає в списках."""

ANSWER_PROMPT = """Ти бот лісового архіву порад. Відповідай українською, коротко і по суті.
Тобі дано фрагменти документів. Відповідай ВИКЛЮЧНО з них. Нічого не додавай від себе.
Правила:
- Місце, яке назвав звір (наприклад «біля греблі»), може не згадуватися у фрагментах. Фрагмент, позначений як «загальна порада для всього лісу», підходить для будь-якого місця: використовуй його.
- Якщо фрагменти суперечать, перевага в такому порядку: порада для рукавички, потім для частини лісу, потім загальна.
- Загальну пораду можна додати до конкретної, лише якщо вона доповнює, а не суперечить.
- Якщо у фрагментах немає відповіді на питання, поверни answerable=false.
- У used_doc_ids перелічи ВСІ фрагменти, які містять відповідь на питання, навіть якщо вони для різних рукавичок чи частин лісу і відповіді різні. Фрагменти без відповіді не перелічуй.
Поверни ТІЛЬКИ JSON: {{"answerable": true|false, "answer": "текст відповіді без згадок про джерела", "used_doc_ids": [id документів]}}

Фрагменти:
{context}"""


def resolve_place(value: str | None, known: list[str]) -> str | None:
    key = norm(value)
    if not key:
        return None
    for name in known:
        n = norm(name)
        if key == n or key in n or n in key:
            return name
    return None


def format_date(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%d.%m.%Y")


def format_source(title, glove, forest_part, uploader, uploaded_at) -> str:
    if glove:
        scope = f' рукавички "{glove}"'
    elif forest_part:
        scope = f" (частина лісу: {forest_part})"
    else:
        scope = " (загальна порада для всього лісу)"
    return f'— з документа "{title}"{scope}, надіслав(ла) {uploader}, {format_date(uploaded_at)}'


def place_options(hits: list[Hit]) -> list[str]:
    """Distinct specific places among hits, as labels."""
    seen, out = set(), []
    for h in hits:
        if not (h.glove or h.forest_part):
            continue
        key = (norm(h.glove), norm(h.forest_part))
        if key in seen:
            continue
        seen.add(key)
        out.append(f'рукавичка "{h.glove}"' if h.glove else f"частина лісу: {h.forest_part}")
    return out


def needs_clarification(hits: list[Hit], glove, part, margin: float = 0.10) -> list[str]:
    """Options to ask about, if the place is unknown and equally relevant hits span several places."""
    if glove or part or not hits:
        return []
    best = max(h.score for h in hits)
    close = [h for h in hits if h.score >= best - margin]
    options = place_options(close)
    return options if len(options) > 1 else []


def place_note(hits: list[Hit]) -> str | None:
    """Single specific place the answer relied on (for the conversation context tag)."""
    options = place_options(hits)
    return options[0] if len(options) == 1 else None


def merge_clarification(history: list[dict], question: str) -> str:
    """If the bot just asked for the place and the user answers briefly,
    combine the answer with the original question."""
    if (
        len(history) >= 2
        and history[-1]["role"] == "assistant"
        and history[-1]["content"].startswith(CLARIFY_PREFIX)
        and history[-2]["role"] == "user"
        and "?" not in question
        and len(question.split()) <= 5
    ):
        return f'{history[-2]["content"]} ({question})'
    return question


def build_context(hits: list[Hit]) -> str:
    lines = []
    for h in hits:
        if h.glove:
            label = f'для рукавички "{h.glove}"'
        elif h.forest_part:
            label = f"для частини лісу: {h.forest_part}"
        else:
            label = "загальна порада для всього лісу"
        lines.append(f'[id={h.doc_id}] документ "{h.title}" ({label}):\n{h.text}')
    return "\n\n".join(lines)


def _history_text(history: list[dict]) -> str:
    names = {"user": "Звір", "assistant": "Бот"}
    return "\n".join(f"{names.get(m['role'], m['role'])}: {m['content']}" for m in history) or "(порожньо)"


def _as_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


async def _reply(question: str, history: list[dict]) -> tuple[str, str | None]:
    """Returns (reply text, place label for conversation context or None)."""
    gloves, parts = await db.get_known_places()
    analysis = await chat_json(
        ANALYZE_PROMPT.format(
            gloves=", ".join(gloves) or "(немає)",
            parts=", ".join(parts) or "(немає)",
            history=_history_text(history),
        ),
        question,
    )
    log.info("analysis: %s", analysis)
    if analysis.get("intent") != "question":
        return UNCLEAR_TEXT, None

    standalone = (analysis.get("standalone_question") or question).strip()
    glove = resolve_place(analysis.get("glove"), gloves)
    part = resolve_place(analysis.get("forest_part"), parts)

    hits = await search(standalone, glove=glove, part=part)
    if not hits:
        return NOT_FOUND_TEXT, None

    result = await chat_json(ANSWER_PROMPT.format(context=build_context(hits)), standalone)
    answer = (result.get("answer") or "").strip()
    if not result.get("answerable") or not answer:
        return NOT_FOUND_TEXT, None

    by_id = {h.doc_id: h for h in hits}
    used_ids: list[int] = []
    for raw in result.get("used_doc_ids", []):
        i = _as_int(raw)
        if i in by_id and i not in used_ids:
            used_ids.append(i)
    if not used_ids:
        used_ids = [hits[0].doc_id]
    used_hits = [by_id[i] for i in used_ids]

    options = needs_clarification(used_hits, glove, part, margin=1.0)
    if options:
        return f"{CLARIFY_PREFIX} У архіві є поради для: " + "; ".join(options) + ".", None

    sources = [
        format_source(h.title, h.glove, h.forest_part, h.uploader_name, h.uploaded_at)
        for h in used_hits
    ]
    return answer + "\n\n" + "\n".join(sources), place_note(used_hits)


async def answer_question(tg_id: int, question: str) -> str:
    history = await db.get_history(tg_id, HISTORY_LIMIT)
    effective = merge_clarification(history, question)
    try:
        reply, note = await _reply(effective, history)
    except groq.RateLimitError:
        log.warning("all models are rate limited")
        return RATE_LIMIT_TEXT  # failed turns are not stored in history
    except Exception:
        log.exception("answer failed")
        return ERROR_TEXT
    await db.add_message(tg_id, "user", question)
    stored = reply + (f"\n[місце відповіді: {note}]" if note else "")
    await db.add_message(tg_id, "assistant", stored)
    return reply