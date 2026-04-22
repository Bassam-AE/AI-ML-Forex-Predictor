from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    Composite,
    ModelPrediction,
    NewsArticle,
    PredictRequest,
    PredictResponse,
    PredictionsBlock,
    Sentiment,
)
from app.config import SUPPORTED_PAIR_CODES
from app.db import get_connection

router = APIRouter()

_STUB_ARTICLES = [
    NewsArticle(
        title="Central bank signals cautious stance amid inflation data",
        source="Reuters",
        url="https://example.com/article-1",
        published_at="2024-01-01T09:00:00Z",
        summary="Policymakers reiterated a data-dependent approach as core inflation remains above target.",
        impact_score=0.72,
        reasoning="Hawkish rhetoric typically supports the base currency.",
    ),
    NewsArticle(
        title="PMI figures beat expectations, boosting risk appetite",
        source="Bloomberg",
        url="https://example.com/article-2",
        published_at="2024-01-01T07:30:00Z",
        summary="Manufacturing activity expanded for the third consecutive month, signalling resilience.",
        impact_score=0.58,
        reasoning="Stronger economic data increases demand for the quote currency.",
    ),
]


def _det(pair: str, lo: float, hi: float, salt: int = 0) -> float:
    """Deterministic float in [lo, hi] derived from pair hash."""
    raw = (hash(pair + str(salt)) % 1000) / 1000.0
    return lo + raw * (hi - lo)


def _latest_close(pair: str) -> float:
    conn = get_connection()
    try:
        row = conn.execute(
            f'SELECT Close FROM "{pair}" ORDER BY Datetime DESC LIMIT 1'
        ).fetchone()
    finally:
        conn.close()
    return float(row["Close"]) if row else 1.0


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    pair = (body.base + body.quote).upper()
    if pair not in SUPPORTED_PAIR_CODES:
        raise HTTPException(status_code=400, detail=f"Pair '{pair}' is not supported.")

    xgb_prob = _det(pair, 0.45, 0.65, salt=1)
    lstm_prob = _det(pair, 0.45, 0.65, salt=2)
    meta_prob = (xgb_prob + lstm_prob) / 2

    sent_score = _det(pair, -0.5, 0.5, salt=3)
    composite_prob = round(0.7 * meta_prob + 0.3 * ((sent_score + 1) / 2), 4)

    if composite_prob > 0.58:
        verdict = "bullish"
        overview = (
            f"The composite signal for {pair} is bullish, driven by positive momentum across "
            "technical indicators and supportive sentiment. Models agree on an elevated probability "
            "of an upward move in the next hour."
        )
    elif composite_prob < 0.42:
        verdict = "bearish"
        overview = (
            f"The composite signal for {pair} is bearish, with technical models and sentiment "
            "both tilting negative. The probability of a downward move in the next hour is elevated."
        )
    else:
        verdict = "neutral"
        overview = (
            f"The composite signal for {pair} is neutral. Technical models show mixed signals and "
            "sentiment is close to zero, suggesting no strong directional bias for the next hour."
        )

    return PredictResponse(
        pair=pair,
        timestamp=datetime.now(timezone.utc).isoformat(),
        current_price=_latest_close(pair),
        predictions=PredictionsBlock(
            xgboost=ModelPrediction(prob_up=round(xgb_prob, 4)),
            lstm=ModelPrediction(prob_up=round(lstm_prob, 4)),
            meta_learner=ModelPrediction(prob_up=round(meta_prob, 4)),
        ),
        sentiment=Sentiment(score=round(sent_score, 4), articles=_STUB_ARTICLES),
        composite=Composite(prob_up=composite_prob, verdict=verdict, ai_overview=overview),
        disclaimer="Educational use only. Not financial advice.",
        stub=True,
    )
