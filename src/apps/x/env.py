import os

from dotenv import load_dotenv

load_dotenv()

def fill_env():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise OSError("OPENAI_API_KEY not found")
    os.environ["OPENAI_API_KEY"] = openai_api_key

    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    if not langsmith_api_key:
        raise OSError("LANGSMITH_API_KEY not found")
    os.environ["LANGSMITH_API_KEY"] = langsmith_api_key

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise OSError("GOOGLE_API_KEY not found")
    os.environ["GOOGLE_API_KEY"] = google_api_key

    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not telegram_bot_token:
        raise OSError("TELEGRAM_BOT_TOKEN not found")
    os.environ["TELEGRAM_BOT_TOKEN"] = telegram_bot_token