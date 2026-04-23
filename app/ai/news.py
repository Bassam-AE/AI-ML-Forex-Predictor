import sys
from datetime import datetime, timedelta, timezone

import httpx

from app.ai.schemas import RawArticle
from app.config import settings

NEWSAPI_URL = "https://newsapi.org/v2/everything"

# Targeted pair-specific queries — require both sides to appear so off-topic articles are filtered out
_PAIR_QUERIES: dict[str, str] = {
    "EURUSD": (
        '("EUR/USD" OR EURUSD OR "euro dollar") AND '
        '("Federal Reserve" OR ECB OR "interest rate" OR inflation OR "central bank" OR "trade war")'
    ),
    "GBPUSD": (
        '("GBP/USD" OR GBPUSD OR "pound dollar" OR cable) AND '
        '("Bank of England" OR "Federal Reserve" OR "interest rate" OR inflation OR "UK economy")'
    ),
    "USDINR": (
        '("USD/INR" OR USDINR OR "dollar rupee") AND '
        '(RBI OR "Reserve Bank of India" OR "Indian rupee" OR "India economy" OR inflation)'
    ),
    "GBPINR": (
        '("GBP/INR" OR GBPINR OR "pound rupee") AND '
        '("Bank of England" OR RBI OR "Indian rupee" OR "British pound" OR economy)'
    ),
}

_CURRENCY_KEYWORDS: dict[str, str] = {
    "USD": '"US dollar" OR "Federal Reserve" OR "dollar index"',
    "EUR": '"euro" OR "ECB" OR "eurozone" OR "euro area"',
    "GBP": '"British pound" OR "sterling" OR "Bank of England"',
    "INR": '"Indian rupee" OR "RBI" OR "Reserve Bank of India"',
}


def _build_query(base: str, quote: str) -> str:
    pair = f"{base}{quote}".upper()
    if pair in _PAIR_QUERIES:
        return _PAIR_QUERIES[pair]
    base_q = _CURRENCY_KEYWORDS.get(base.upper(), f'"{base}"')
    quote_q = _CURRENCY_KEYWORDS.get(quote.upper(), f'"{quote}"')
    return f"({base_q}) AND ({quote_q})"


# Cache: pair_code → (fetched_at, articles)
_cache: dict[str, tuple[datetime, list[RawArticle]]] = {}
_CACHE_TTL = timedelta(minutes=30)


async def fetch_news(
    base: str,
    quote: str,
    hours: int = 72,
    max_articles: int = 20,
) -> list[RawArticle]:
    cache_key = f"{base}{quote}"
    now = datetime.now(timezone.utc)

    cached = _cache.get(cache_key)
    if cached and (now - cached[0]) < _CACHE_TTL:
        print(f"[news] cache hit for {cache_key}", file=sys.stderr)
        return cached[1]

    query = _build_query(base, quote)
    from_dt = (now - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "q": query,
        "from": from_dt,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": max_articles,
        "apiKey": settings.news_api_key,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(NEWSAPI_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        print(f"[news] fetch error for {cache_key}: {exc}", file=sys.stderr)
        return []

    articles: list[RawArticle] = []
    for item in data.get("articles", []):
        try:
            articles.append(
                RawArticle(
                    title=item.get("title") or "",
                    description=item.get("description"),
                    source=item.get("source", {}).get("name") or "",
                    url=item.get("url") or "",
                    published_at=item.get("publishedAt") or "",
                )
            )
        except Exception:
            continue

    _cache[cache_key] = (now, articles)
    return articles
