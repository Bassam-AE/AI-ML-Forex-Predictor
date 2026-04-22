from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()

DB_PATH = "app/data/forex.db"

SUPPORTED_PAIRS = [
    {"pair": "EURUSD", "base": "EUR", "quote": "USD"},
    {"pair": "GBPUSD", "base": "GBP", "quote": "USD"},
    {"pair": "USDINR", "base": "USD", "quote": "INR"},
]

SUPPORTED_PAIR_CODES = {p["pair"] for p in SUPPORTED_PAIRS}
