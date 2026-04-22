from app.ai.schemas import ScoredArticle


def aggregate_sentiment(scored: list[ScoredArticle]) -> float:
    if not scored:
        return 0.0

    total_weight = sum(a.confidence for a in scored)
    if total_weight == 0.0:
        return 0.0

    weighted_sum = sum(a.impact_score * a.confidence for a in scored)
    result = weighted_sum / total_weight
    return max(-1.0, min(1.0, result))
