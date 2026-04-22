from pydantic import BaseModel


class RawArticle(BaseModel):
    title: str
    description: str | None
    source: str
    url: str
    published_at: str


class ScoredArticle(BaseModel):
    title: str
    description: str | None
    source: str
    url: str
    published_at: str
    summary: str
    impact_score: float   # -1.0 to +1.0
    confidence: float     # 0.0 to 1.0
    reasoning: str


class GeminiScoreOutput(BaseModel):
    summary: str
    impact_score: float
    confidence: float
    reasoning: str


class GeminiBatchOutput(BaseModel):
    scores: list[GeminiScoreOutput]
