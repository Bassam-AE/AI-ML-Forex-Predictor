import sys

from google import genai
from google.genai import types

from app.ai.schemas import GeminiBatchOutput, RawArticle, ScoredArticle
from app.config import settings

_MODEL = "gemini-2.5-flash"


def _get_client() -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)


async def score_articles(
    articles: list[RawArticle],
    base: str,
    quote: str,
) -> list[ScoredArticle]:
    if not articles:
        return []

    numbered = "\n\n".join(
        f"[{i + 1}] Title: {a.title}\n"
        f"    Description: {a.description or 'N/A'}\n"
        f"    Published: {a.published_at}"
        for i, a in enumerate(articles)
    )

    prompt = (
        f"You are a forex analyst. Score how each news article affects the {base}/{quote} pair.\n\n"
        f"For each article score:\n"
        f"- impact_score: -1.0 (strong negative for {base}) to +1.0 (strong positive for {base})\n"
        f"- confidence: 0.0 (unrelated/unclear) to 1.0 (very clear impact)\n"
        f"- summary: one sentence summary\n"
        f"- reasoning: one sentence explanation\n\n"
        f"Return exactly {len(articles)} scores in the same order as the articles.\n\n"
        f"Articles:\n{numbered}"
    )

    try:
        client = _get_client()
        response = await client.aio.models.generate_content(
            model=_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GeminiBatchOutput,
            ),
        )
        batch = GeminiBatchOutput.model_validate_json(response.text)
    except Exception as exc:
        print(f"[sentiment] batch score error for {base}/{quote}: {exc}", file=sys.stderr)
        return []

    results: list[ScoredArticle] = []
    for article, score in zip(articles, batch.scores):
        results.append(
            ScoredArticle(
                title=article.title,
                description=article.description,
                source=article.source,
                url=article.url,
                published_at=article.published_at,
                summary=score.summary,
                impact_score=max(-1.0, min(1.0, score.impact_score)),
                confidence=max(0.0, min(1.0, score.confidence)),
                reasoning=score.reasoning,
            )
        )
    return results
