import asyncio
import time

from app.ai.news import fetch_news
from app.ai.sentiment import score_articles
from app.ai.aggregate import aggregate_sentiment


async def main():
    pairs = [("USD", "INR"), ("EUR", "USD"), ("GBP", "USD")]

    for base, quote in pairs:
        print(f"\n=== {base}/{quote} ===")
        t0 = time.perf_counter()

        articles = await fetch_news(base, quote)
        print(f"Fetched {len(articles)} articles")

        scored = await score_articles(articles, base, quote)
        print(f"Scored  {len(scored)} articles")

        for s in scored[:3]:
            print(f"  [{s.impact_score:+.2f} conf={s.confidence:.2f}] {s.title[:80]}")

        sentiment = aggregate_sentiment(scored)
        print(f"Aggregate sentiment: {sentiment:+.3f}")
        print(f"Elapsed: {time.perf_counter() - t0:.1f}s")

    print("\n--- Running again to verify cache ---")
    t0 = time.perf_counter()
    await fetch_news("EUR", "USD")
    print(f"EUR/USD re-fetch: {time.perf_counter() - t0:.3f}s (should be near 0)")


if __name__ == "__main__":
    asyncio.run(main())
