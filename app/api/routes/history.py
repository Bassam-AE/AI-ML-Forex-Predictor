from fastapi import APIRouter, HTTPException, Query

from app.api.schemas import OHLCBar
from app.config import SUPPORTED_PAIR_CODES
from app.db import get_connection

router = APIRouter()


@router.get("/history/{pair}", response_model=list[OHLCBar])
def get_history(pair: str, hours: int = Query(default=168, ge=1, le=1000)):
    pair = pair.upper()
    if pair not in SUPPORTED_PAIR_CODES:
        raise HTTPException(status_code=404, detail=f"Pair '{pair}' not supported.")

    hours = max(1, min(hours, 1000))

    conn = get_connection()
    try:
        rows = conn.execute(
            f'SELECT Datetime, Open, High, Low, Close, Volume FROM "{pair}" ORDER BY Datetime DESC LIMIT ?',
            (hours,),
        ).fetchall()
    finally:
        conn.close()

    return [
        OHLCBar(
            ts=row["Datetime"],
            open=row["Open"],
            high=row["High"],
            low=row["Low"],
            close=row["Close"],
            volume=row["Volume"],
        )
        for row in reversed(rows)
    ]
