import sys
from datetime import datetime, timedelta, timezone

import httpx

from app.ai.schemas import RawArticle
from app.config import settings

NEWSAPI_URL = "https://newsapi.org/v2/everything"

CURRENCY_KEYWORDS: dict[str, str] = {
    "USD": '"US dollar" OR "Federal Reserve" OR "US economy"',
    "EUR": '"euro" OR "ECB" OR "eurozone"',
    "GBP": '"British pound" OR "Bank of England" OR "UK economy"',
    "INR": '"Indian rupee" OR "RBI" OR "India economy"',
}

# Cache: pair_code → (fetched_at, articles)
_cache: dict[str, tuple[datetime, list[RawArticle]]] = {}
_CACHE_TTL = timedelta(minutes=30)


async def fetch_news(
    base: str,
    quote: str,
    hours: int = 48,
    max_articles: int = 8,
) -> list[RawArticle]:
    cache_key = f"{base}{quote}"
    now = datetime.now(timezone.utc)

    cached = _cache.get(cache_key)
    if cached and (now - cached[0]) < _CACHE_TTL:
        print(f"[news] cache hit for {cache_key}", file=sys.stderr)
        return cached[1]

    base_q = CURRENCY_KEYWORDS.get(base.upper(), base)
    quote_q = CURRENCY_KEYWORDS.get(quote.upper(), quote)
    query = f"({base_q}) OR ({quote_q})"
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
