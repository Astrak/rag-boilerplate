import os

def load_env_variable(var_name: str) -> None:
    """Load and set an environment variable, raise error if not found."""
    value = os.getenv(var_name)
    if not value:
        raise EnvironmentError(f"{var_name} not found")
    os.environ[var_name] = value

def fill_env():
    env_vars = [
        "OPENAI_API_KEY",
        "LANGSMITH_API_KEY",
        "GOOGLE_API_KEY",
        "TELEGRAM_BOT_TOKEN",
    ]
    for var in env_vars:
        load_env_variable(var)