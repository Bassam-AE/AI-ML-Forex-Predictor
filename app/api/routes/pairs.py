from fastapi import APIRouter

from app.config import SUPPORTED_PAIRS

router = APIRouter()


@router.get("/pairs")
def get_pairs():
    return SUPPORTED_PAIRS
