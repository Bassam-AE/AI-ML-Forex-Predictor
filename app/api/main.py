from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, history, pairs, predict
from app.api.routes import metrics
from app.serving.model_loader import load_all_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = load_all_models()
    yield


app = FastAPI(title="ForexOracle API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(pairs.router)
app.include_router(history.router)
app.include_router(predict.router)
app.include_router(metrics.router)
