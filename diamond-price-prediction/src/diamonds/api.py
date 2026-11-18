"""REST API serving price predictions with 90 % prediction intervals.

Run:  uvicorn diamonds.api:app --reload      (docs at /docs)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .features import CLARITY, COLOR, CUT

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")
MAX_BATCH = 1000

CutT = Literal[tuple(CUT)]
ColorT = Literal[tuple(COLOR)]
ClarityT = Literal[tuple(CLARITY)]


class Diamond(BaseModel):
    carat: float = Field(gt=0, le=6, description="Weight in carats")
    cut: CutT
    color: ColorT = Field(description="D (best) to J (worst)")
    clarity: ClarityT
    depth: float = Field(ge=40, le=80, description="Total depth percentage")
    table: float = Field(ge=40, le=100, description="Table width percentage")
    x: float = Field(gt=0, le=20, description="Length (mm)")
    y: float = Field(gt=0, le=20, description="Width (mm)")
    z: float = Field(gt=0, le=20, description="Depth (mm)")

    model_config = {
        "json_schema_extra": {
            "example": {"carat": 1.0, "cut": "Ideal", "color": "G", "clarity": "VS2",
                        "depth": 61.8, "table": 57.0, "x": 6.4, "y": 6.4, "z": 3.96}
        }
    }


class Prediction(BaseModel):
    price_usd: float
    lower_usd: float
    upper_usd: float
    confidence_level: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = joblib.load(MODEL_PATH)
    yield


app = FastAPI(title="Diamond Price API", version="1.0.0", lifespan=lifespan)


def _predict(items: list[Diamond]) -> list[Prediction]:
    bundle = app.state.bundle
    raw = pd.DataFrame([d.model_dump() for d in items])
    try:
        res = bundle.predict(raw)
    except ValueError as exc:  # unknown category etc.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    level = 1 - bundle.alpha
    return [
        Prediction(price_usd=round(r.price, 2), lower_usd=round(r.lower, 2),
                   upper_usd=round(r.upper, 2), confidence_level=level)
        for r in res.itertuples()
    ]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict:
    return app.state.bundle.metadata


@app.post("/predict", response_model=Prediction)
def predict(diamond: Diamond) -> Prediction:
    return _predict([diamond])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(diamonds: list[Diamond]) -> list[Prediction]:
    if not 0 < len(diamonds) <= MAX_BATCH:
        raise HTTPException(status_code=422, detail=f"Send between 1 and {MAX_BATCH} diamonds")
    return _predict(diamonds)
