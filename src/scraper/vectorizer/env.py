import os

def fill_env():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise EnvironmentError("OPENAI_API_KEY not found")
    os.environ["OPENAI_API_KEY"] = openai_api_key

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise EnvironmentError("GOOGLE_API_KEY not found")
    os.environ["GOOGLE_API_KEY"] = google_api_key