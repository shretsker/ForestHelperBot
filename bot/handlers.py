import asyncio
import io
import logging

import groq
from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message

from bot import db
from bot.answer import ERROR_TEXT, RATE_LIMIT_TEXT, answer_question, resolve_place
from bot.config import MAX_FILE_BYTES
from bot.ingest import SUPPORTED, chunk_text, embed_texts, extract_text
from bot.llm import extract_metadata

log = logging.getLogger(__name__)

router = Router()
router.message.filter(F.chat.type == "private")

START_TEXT = (
    "Привіт! Я лісовий архів порад.\n\n"
    "• Постав мені питання, наприклад: «Скільки внесок під дубом?» або "
    "«Де шукати гриби біля верби?». Я відповім і скажу, з якого документа взяв пораду.\n"
    "• Щоб додати пораду, надішли документ (PDF, Word або txt). Я спитаю твоє справжнє ім'я, "
    "щоб звірі знали, кому дякувати, і для якої рукавички чи частини лісу документ, якщо це не видно з тексту.\n\n"
    "Команди: /name змінити ім'я, /reset почати розмову заново, /cancel скасувати завантаження."
)
MAX_REPLY_CHARS = 4000


class Upload(StatesGroup):
    waiting_name = State()
    waiting_place = State()
    changing_name = State()


def valid_name(text: str | None) -> str | None:
    name = " ".join((text or "").split())
    return name if 2 <= len(name) <= 50 else None


def scope_label(glove, part, wide) -> str:
    if glove:
        return f'рукавичка "{glove}"'
    if part:
        return f"частина лісу: {part}"
    return "весь ліс" if wide else "без прив'язки"


def apply_known_places(meta: dict, hint: str | None, gloves: list[str], parts: list[str]) -> dict:
    """Use canonical names of places that already exist in the archive.
    The author's own answer (hint) wins: if it names a known glove or forest part,
    the document belongs to exactly that place."""
    meta = dict(meta)
    if meta.get("glove"):
        meta["glove"] = resolve_place(meta["glove"], gloves) or meta["glove"]
    if meta.get("forest_part"):
        meta["forest_part"] = resolve_place(meta["forest_part"], parts) or meta["forest_part"]
    if hint:
        known_glove = resolve_place(hint, gloves)
        known_part = resolve_place(hint, parts)
        if known_glove:
            meta.update(glove=known_glove, forest_part=None, forest_wide=False)
        elif known_part:
            meta.update(glove=None, forest_part=known_part, forest_wide=False)
    return meta


async def process_document(message: Message, state: FSMContext, text: str, filename: str,
                           name: str, hint: str | None = None, attempts: int = 0) -> None:
    try:
        gloves, parts = await db.get_known_places()
        topics = await db.get_known_topics()
        meta = await extract_metadata(text, filename, hint, gloves, parts, topics)
        meta = apply_known_places(meta, hint, gloves, parts)
    except groq.RateLimitError:
        await message.answer(RATE_LIMIT_TEXT)
        return
    except Exception:
        log.exception("metadata extraction failed")
        await message.answer(ERROR_TEXT)
        return
    log.info("document metadata: %s", meta)

    if not (meta["glove"] or meta["forest_part"] or meta["forest_wide"]):
        if attempts >= 2:
            await state.clear()
            await message.answer(
                "Так і не зміг зрозуміти, для кого цей документ. Надішли його ще раз, "
                "вказавши в тексті рукавичку або частину лісу."
            )
            return
        await state.set_state(Upload.waiting_place)
        await state.update_data(text=text, filename=filename, attempts=attempts + 1)
        await message.answer(
            "Не зрозумів, для якої рукавички чи частини лісу цей документ. Напиши назву рукавички "
            "(наприклад «Під дубом») або частини лісу, або «для всього лісу». /cancel щоб скасувати."
        )
        return

    try:
        same = await db.find_active_same_document(meta["glove"], meta["forest_part"], meta["topic"])
        chunks = chunk_text(text)
        embeddings = await embed_texts(chunks)
        await db.add_document(
            meta["title"], filename, meta["glove"], meta["forest_part"], meta["topic"],
            message.from_user.id, name, list(zip(chunks, embeddings)),
        )
        for row in same:
            await db.deactivate_document(row[0])
    except Exception:
        log.exception("saving document failed")
        await message.answer(ERROR_TEXT)
        return

    await state.clear()
    reply = (
        f'Дякую, {name}! Додав документ "{meta["title"]}" '
        f'({scope_label(meta["glove"], meta["forest_part"], meta["forest_wide"])}).'
    )
    if same:
        old = same[0]
        reply += f'\nСтару версію "{old[1]}" ({old[2]}) замінено: звірі бачитимуть лише чинну пораду.'
    await message.answer(reply)


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(START_TEXT)


@router.message(Command("reset"))
async def cmd_reset(message: Message, state: FSMContext):
    await state.clear()
    await db.clear_history(message.from_user.id)
    await message.answer("Почнемо розмову заново.")


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("Скасовано.")


@router.message(Command("name"))
async def cmd_name(message: Message, state: FSMContext):
    await state.set_state(Upload.changing_name)
    current = await db.get_user_name(message.from_user.id)
    prefix = f"Зараз я знаю тебе як {current}. " if current else ""
    await message.answer(prefix + "Напиши нове справжнє ім'я.")


@router.message(Upload.changing_name, F.text, ~F.text.startswith("/"))
async def on_new_name(message: Message, state: FSMContext):
    name = valid_name(message.text)
    if not name:
        await message.answer("Ім'я має бути від 2 до 50 символів. Спробуй ще раз.")
        return
    await db.set_user_name(message.from_user.id, name)
    await state.clear()
    await message.answer(f"Гаразд, {name}!")


@router.message(F.document)
async def on_document(message: Message, state: FSMContext, bot: Bot):
    await state.clear()
    doc = message.document
    filename = doc.file_name or "document"
    if not filename.lower().endswith(SUPPORTED):
        await message.answer("Приймаю лише PDF, Word (.docx) та текстові файли (.txt).")
        return
    if doc.file_size and doc.file_size > MAX_FILE_BYTES:
        await message.answer("Файл завеликий: можу прийняти до 20 МБ.")
        return

    await message.answer("Читаю документ…")
    buf = io.BytesIO()
    await bot.download(doc, destination=buf)
    try:
        text = await asyncio.to_thread(extract_text, buf.getvalue(), filename)
    except ValueError:
        await message.answer("Не знайшов у файлі тексту (можливо, це скан). Надішли текстовий документ.")
        return
    except Exception:
        log.exception("text extraction failed")
        await message.answer("Не вдалося прочитати файл. Перевір, що він не пошкоджений.")
        return

    name = await db.get_user_name(message.from_user.id)
    if not name:
        await state.set_state(Upload.waiting_name)
        await state.update_data(text=text, filename=filename)
        await message.answer(
            "Як тебе звати? Напиши справжнє ім'я: його побачать звірі, щоб знати, кому дякувати."
        )
        return
    await process_document(message, state, text, filename, name)


@router.message(Upload.waiting_name, F.text, ~F.text.startswith("/"))
async def on_upload_name(message: Message, state: FSMContext):
    name = valid_name(message.text)
    if not name:
        await message.answer("Ім'я має бути від 2 до 50 символів. Спробуй ще раз.")
        return
    await db.set_user_name(message.from_user.id, name)
    data = await state.get_data()
    await state.clear()
    await process_document(message, state, data["text"], data["filename"], name)


@router.message(Upload.waiting_place, F.text, ~F.text.startswith("/"))
async def on_upload_place(message: Message, state: FSMContext):
    data = await state.get_data()
    name = await db.get_user_name(message.from_user.id) or "Невідомий звір"
    await state.clear()
    await process_document(message, state, data["text"], data["filename"], name,
                           hint=message.text.strip(), attempts=data.get("attempts", 1))


@router.message(StateFilter(None), F.text, ~F.text.startswith("/"))
async def on_question(message: Message, bot: Bot):
    await bot.send_chat_action(message.chat.id, "typing")
    reply = await answer_question(message.from_user.id, message.text)
    await message.answer(reply[:MAX_REPLY_CHARS])


@router.message()
async def on_other(message: Message):
    await message.answer(
        "Напиши питання текстом або надішли документ з порадою (PDF, Word, txt). Фото я не читаю."
    )