import sys

from google import genai

from app.config import settings

_MODEL = "gemini-2.5-flash"


async def generate_overview(
    pair: str,
    verdict: str,
    model_probs: dict,
    sentiment_score: float,
    top_titles: list[str],
) -> str:
    fallback = (
        f"The ensemble model gives a {verdict} verdict for {pair}. "
        "Review the individual model probabilities and news below before drawing conclusions."
    )
    if not settings.gemini_api_key:
        return fallback

    titles_block = "\n".join(f"- {t}" for t in top_titles[:3]) if top_titles else "No recent articles."

    prompt = (
        f"Currency pair: {pair}\n"
        f"Ensemble verdict: {verdict}\n"
        f"XGBoost prob_up: {model_probs.get('xgb_prob', 0):.3f}\n"
        f"LSTM prob_up: {model_probs.get('lstm_prob', 0):.3f}\n"
        f"Meta-learner prob_up: {model_probs.get('meta_prob', 0):.3f}\n"
        f"News sentiment score: {sentiment_score:.3f} (range -1 to +1)\n"
        f"Recent headlines:\n{titles_block}\n\n"
        "Write a 2-3 sentence plain-English summary of the outlook for this pair. "
        "Be concise, factual, and reference the models and news. No markdown."
    )

    try:
        client = genai.Client(api_key=settings.gemini_api_key)
        response = await client.aio.models.generate_content(
            model=_MODEL,
            contents=prompt,
        )
        return response.text.strip()
    except Exception as exc:
        print(f"[overview] error: {exc}", file=sys.stderr)
        return fallback
