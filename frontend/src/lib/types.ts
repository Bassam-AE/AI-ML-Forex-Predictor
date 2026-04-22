export type OHLCBar = {
  ts: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number | null;
};

export type Pair = {
  pair: string;
  base: string;
  quote: string;
};

export type PredictResponse = {
  pair: string;
  timestamp: string;
  current_price: number;
  predictions: {
    xgboost: { prob_up: number };
    lstm: { prob_up: number };
    meta_learner: { prob_up: number };
  };
  sentiment: {
    score: number;
    articles: Array<{
      title: string;
      source: string;
      url: string;
      published_at: string;
      summary: string;
      impact_score: number;
      reasoning: string;
    }>;
  };
  composite: {
    prob_up: number;
    verdict: "bullish" | "bearish" | "neutral";
    ai_overview: string;
  };
  disclaimer: string;
  stub: boolean;
};
