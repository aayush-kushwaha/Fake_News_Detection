import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.model_service import FakeNewsModelService
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    FactCheckResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from groq_utils import (
    explain_fake_news_risk,
    fact_check_with_external_references,
    summarize_article,
)

DATASET_CSV = os.environ.get("FAKE_NEWS_DATASET", "data/fake_real_combined.csv")
model_service = FakeNewsModelService(dataset_csv=DATASET_CSV)


def _label_name(label: int) -> str:
    return "fake" if label == 1 else "real"


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        model_service.load_or_train()
        print("Model trained and ready")
    except Exception as exc:
        print(f"Model initialization warning: {exc}")
    yield


app = FastAPI(
    title="Fake News Detection API",
    version="1.0.0",
    description="Backend API for fake news classification and optional LLM analysis",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_ready=model_service.is_ready,
        dataset_path=DATASET_CSV,
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if not model_service.is_ready:
        raise HTTPException(status_code=503, detail="Model is not ready")

    try:
        label, confidence = model_service.predict(payload.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return PredictResponse(label=label, label_name=_label_name(label), confidence=confidence)


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    if not model_service.is_ready:
        raise HTTPException(status_code=503, detail="Model is not ready")

    label, confidence = model_service.predict(payload.text)
    prediction = PredictResponse(label=label, label_name=_label_name(label), confidence=confidence)

    if not payload.use_llm:
        return AnalyzeResponse(prediction=prediction)

    warnings: list[str] = []
    summary = None
    risk_explanation = None
    fact_check = None

    try:
        summary = summarize_article(payload.text)
    except Exception as exc:
        warnings.append(f"Summary unavailable: {exc}")

    try:
        risk_explanation = explain_fake_news_risk(payload.text)
    except Exception as exc:
        warnings.append(f"Risk analysis unavailable: {exc}")

    try:
        fc_raw: dict[str, Any] = fact_check_with_external_references(payload.text, max_results=5)
        fact_check = FactCheckResponse(**fc_raw)
    except Exception as exc:
        warnings.append(f"Fact-check unavailable: {exc}")

    return AnalyzeResponse(
        prediction=prediction,
        summary=summary,
        risk_explanation=risk_explanation,
        fact_check=fact_check,
        warnings=warnings,
    )
