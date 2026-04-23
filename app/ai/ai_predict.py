import sys

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from app.config import settings

_MODEL = "gemini-2.5-flash"


class _GeminiDirectional(BaseModel):
    prob_up: float = Field(ge=0.0, le=1.0)


async def get_ai_prob(
    pair: str,
    sentiment_score: float,
    top_titles: list[str],
) -> float:
    """Ask Gemini for a 0-1 directional probability. Falls back to 0.5 on any error."""
    if not settings.gemini_api_key:
        return 0.5

    titles_block = "\n".join(f"- {t}" for t in top_titles[:5]) if top_titles else "No recent headlines."

    prompt = (
        f"You are a concise forex market analyst.\n"
        f"Currency pair: {pair}\n"
        f"News sentiment score: {sentiment_score:.3f} (range -1 bearish to +1 bullish)\n"
        f"Recent headlines:\n{titles_block}\n\n"
        "Based solely on the above, estimate the probability (0.0 to 1.0) that the base currency "
        f"will be HIGHER vs the quote currency 24 hours from now. "
        "Return only valid JSON matching the schema — no explanation."
    )

    try:
        client = genai.Client(api_key=settings.gemini_api_key)
        response = await client.aio.models.generate_content(
            model=_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_GeminiDirectional,
            ),
        )
        result = _GeminiDirectional.model_validate_json(response.text)
        return round(float(result.prob_up), 4)
    except Exception as exc:
        print(f"[ai_predict] error for {pair}: {exc}", file=sys.stderr)
        return 0.5
