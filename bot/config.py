import os

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DB_PATH = os.getenv("DB_PATH", "/data/archive.db")
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HISTORY_LIMIT = 6
MAX_FILE_BYTES = 20 * 1024 * 1024
TOP_K = 5