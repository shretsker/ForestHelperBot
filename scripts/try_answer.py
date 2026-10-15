import asyncio
import logging
import os
import tempfile

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "demo.db")

from bot import db  # noqa: E402
from bot.answer import answer_question  # noqa: E402
from bot.ingest import chunk_text, embed_texts  # noqa: E402
from bot.llm import extract_metadata  # noqa: E402

DOCS = [
    ("vnesky.txt", "Членські внески. Рукавичка Під дубом. 4 жолуді з дорослого звіра на місяць, здати до 10 числа Кабанові в кошик біля входу. Замість 4 жолудів можна віддати 2 лісові горіхи."),
    ("vnesky2.txt", "Внески рукавички Під ялинами. 6 горіхів на місяць, здати до 5 числа Їжакові."),
    ("hunters.txt", "Як кричати, коли бачиш мисливців. Крикни тричі коротко, як Сорока. Якщо почув постріл, не біжи на відкрите, ховайся в гущавину."),
    ("pine.txt", "Грибний путівник хвойного лісу. У хвойному лісі гриби з'являються на 2–3 день після дощу."),
    ("edible.txt", "Їстівне й отруйне. Загальна порада для всього лісу: шукай гриби після дощу, коли земля волога, але не хлюпає."),
]


async def main():
    await db.init_db()
    for filename, text in DOCS:
        meta = await extract_metadata(text, filename)
        embs = await embed_texts(chunk_text(text))
        await db.add_document(meta["title"], filename, meta["glove"], meta["forest_part"],
                              meta["topic"], 1, "Кабан", list(zip(chunk_text(text), embs)))
        print("додано:", meta)

    singles = [
        "Скільки внесок під дубом і до якого числа?",
        "Біля греблі мисливці! Як кричати?",
        "Через скільки днів після дощу йти по гриби в соснах?",
        "Скільки внесок?",
        "Де знайти малину біля струмка?",
        "ааааа",
    ]
    for i, q in enumerate(singles, start=100):
        print("\nЗвір:", q)
        print("Бот:", await answer_question(i, q))

    print("\n--- розмова з уточненням ---")
    for q in ["Скільки внесок?", "під ялинами", "а до якого числа?"]:
        print("\nЗвір:", q)
        print("Бот:", await answer_question(500, q))


asyncio.run(main())