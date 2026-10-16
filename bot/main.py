import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand

from bot import db
from bot.config import TELEGRAM_BOT_TOKEN
from bot.handlers import router
from bot.ingest import embed_texts


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    await db.init_db()
    await embed_texts(["прогрів моделі"])  # завантажити модель ембедінгів до першого запиту

    bot = Bot(TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    await bot.set_my_commands([
        BotCommand(command="start", description="Що вміє бот"),
        BotCommand(command="name", description="Змінити ім'я"),
        BotCommand(command="reset", description="Почати розмову заново"),
        BotCommand(command="cancel", description="Скасувати завантаження"),
    ])
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())