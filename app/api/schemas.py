from typing import Literal

from pydantic import BaseModel


class OHLCBar(BaseModel):
    ts: str
    open: float
    high: float
    low: float
    close: float
    volume: float | None


class PredictRequest(BaseModel):
    base: str
    quote: str


class ModelPrediction(BaseModel):
    prob_up: float


class NewsArticle(BaseModel):
    title: str
    source: str
    url: str
    published_at: str
    summary: str
    impact_score: float
    reasoning: str


class Sentiment(BaseModel):
    score: float
    articles: list[NewsArticle]


class Composite(BaseModel):
    prob_up: float
    verdict: Literal["bullish", "bearish", "neutral"]
    ai_overview: str


class PredictionsBlock(BaseModel):
    xgboost: ModelPrediction
    lstm: ModelPrediction
    meta_learner: ModelPrediction
    gemini: ModelPrediction


class PredictResponse(BaseModel):
    pair: str
    timestamp: str
    current_price: float
    predictions: PredictionsBlock
    sentiment: Sentiment
    composite: Composite
    disclaimer: str
    stub: bool = True
