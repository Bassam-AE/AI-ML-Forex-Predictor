import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request

from app.ai.aggregate import aggregate_sentiment
from app.ai.ai_predict import get_ai_prob
from app.ai.news import fetch_news
from app.ai.overview import generate_overview
from app.ai.sentiment import score_articles
from app.api.schemas import (
    Composite,
    ModelPrediction,
    NewsArticle,
    PredictRequest,
    PredictResponse,
    PredictionsBlock,
    Sentiment,
)
from app.config import SUPPORTED_PAIR_CODES, SUPPORTED_PAIRS
from app.serving.model_loader import predict_for_pair

router = APIRouter()


def _verdict(prob: float) -> str:
    if prob > 0.58:
        return "bullish"
    if prob < 0.42:
        return "bearish"
    return "neutral"


def _pair_info(pair: str) -> dict:
    for p in SUPPORTED_PAIRS:
        if p["pair"] == pair:
            return p
    return {}


@router.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest, request: Request):
    pair = (body.base + body.quote).upper()
    if pair not in SUPPORTED_PAIR_CODES:
        raise HTTPException(status_code=400, detail=f"Pair '{pair}' is not supported.")

    bundle = request.app.state.models.get(pair)
    if bundle is None:
        raise HTTPException(status_code=503, detail=f"Models for {pair} are not loaded.")

    info = _pair_info(pair)
    base, quote = info["base"], info["quote"]

    # --- Model inference (CPU-bound — run in thread pool) ---
    try:
        model_result = await asyncio.to_thread(predict_for_pair, pair, bundle)
    except Exception as exc:
        model_result = {
            "xgb_prob": 0.5,
            "lstm_prob": 0.5,
            "meta_prob": 0.5,
            "current_price": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        print(f"[predict] model error for {pair}: {exc}")

    xgb_prob = model_result["xgb_prob"]
    lstm_prob = model_result["lstm_prob"]
    meta_prob = model_result["meta_prob"]

    # --- News fetch ---
    try:
        raw_articles = await fetch_news(base=base, quote=quote)
    except Exception:
        raw_articles = []

    # --- Sentiment scoring ---
    scored = []
    if raw_articles:
        try:
            scored = await score_articles(raw_articles, base=base, quote=quote)
        except Exception:
            scored = []

    sentiment_score = aggregate_sentiment(scored)

    # --- Composite ---
    composite_prob = round(0.7 * meta_prob + 0.3 * ((sentiment_score + 1) / 2), 4)
    verdict = _verdict(composite_prob)

    # --- AI overview + Gemini directional (run concurrently) ---
    top_titles = [a.title for a in raw_articles[:5]]
    fallback_overview = (
        f"The ensemble model gives a {verdict} verdict for {pair}. "
        "Review the individual model probabilities and news below before drawing conclusions."
    )
    try:
        ai_overview, gemini_prob = await asyncio.gather(
            generate_overview(
                pair=pair,
                verdict=verdict,
                model_probs=model_result,
                sentiment_score=sentiment_score,
                top_titles=top_titles,
            ),
            get_ai_prob(
                pair=pair,
                sentiment_score=sentiment_score,
                top_titles=top_titles,
            ),
        )
    except Exception:
        ai_overview = fallback_overview
        gemini_prob = 0.5

    # --- Build response ---
    news_articles = [
        NewsArticle(
            title=a.title,
            source=a.source,
            url=a.url,
            published_at=a.published_at,
            summary=a.summary,
            impact_score=a.impact_score,
            reasoning=a.reasoning,
        )
        for a in scored
    ]

    return PredictResponse(
        pair=pair,
        timestamp=model_result["timestamp"],
        current_price=model_result["current_price"],
        predictions=PredictionsBlock(
            xgboost=ModelPrediction(prob_up=round(xgb_prob, 4)),
            lstm=ModelPrediction(prob_up=round(lstm_prob, 4)),
            meta_learner=ModelPrediction(prob_up=round(meta_prob, 4)),
            gemini=ModelPrediction(prob_up=gemini_prob),
        ),
        sentiment=Sentiment(score=round(sentiment_score, 4), articles=news_articles),
        composite=Composite(prob_up=composite_prob, verdict=verdict, ai_overview=ai_overview),
        disclaimer="Educational use only. Not financial advice.",
        stub=False,
    )
