import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.config import SUPPORTED_PAIR_CODES

router = APIRouter()


@router.get("/metrics/{pair}")
def get_metrics(pair: str):
    pair = pair.upper()
    if pair not in SUPPORTED_PAIR_CODES:
        raise HTTPException(status_code=404, detail=f"Pair '{pair}' not supported.")

    path = Path(f"models/{pair}/metrics.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No metrics found for {pair}.")

    return json.loads(path.read_text())
