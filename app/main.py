import os
import threading
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
    FinalAssessment,
    FactCheckResponse,
    HealthResponse,
    LlmAssessment,
    MlAssessment,
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


def _warm_model_in_background() -> None:
    try:
        model_service.load_or_train()
        print("Model artifact loaded and ready")
    except Exception as exc:
        print(f"Model initialization warning: {exc}")


def _label_name(label: int) -> str:
    if label == -1:
        return "review"
    return "fake" if label == 1 else "real"


def _ml_fake_score(prediction: PredictResponse) -> float:
    if prediction.label_name == "fake":
        return float(prediction.confidence)
    if prediction.label_name == "real":
        return 1.0 - float(prediction.confidence)
    return 0.5


def _llm_fake_score(verdict: str) -> float:
    mapping = {
        "contradicted": 0.9,
        "mixed": 0.65,
        "insufficient_evidence": 0.5,
        "supported": 0.1,
    }
    return mapping.get(verdict, 0.5)


def _fuse_assessments(
    prediction: PredictResponse,
    fact_check: FactCheckResponse | None,
) -> FinalAssessment:
    ml_label = prediction.label_name
    ml_score = _ml_fake_score(prediction)

    if fact_check is None:
        verdict = "fake" if ml_score >= 0.78 else "real"
        confidence = int(round(max(0.0, abs(ml_score - 0.5) * 2.0) * 100))
        return FinalAssessment(
            verdict=verdict,
            confidence=confidence,
            agreement_status="ml_only",
            explanation="LLM verification was unavailable, so final recommendation follows the ML pattern model.",
        )

    llm_verdict = fact_check.verdict
    llm_score = _llm_fake_score(llm_verdict)
    llm_supports_fake = llm_verdict in {"contradicted", "mixed"}
    llm_supports_real = llm_verdict == "supported"
    ml_supports_fake = ml_label == "fake"
    ml_supports_real = ml_label == "real"

    disagreement = (ml_supports_fake and llm_supports_real) or (ml_supports_real and llm_supports_fake)
    agreement_status = "disagree" if disagreement else "agree"

    llm_weight = 0.40 if fact_check.has_trusted_references else 0.25
    ml_weight = 1.0 - llm_weight
    fused_risk = (ml_weight * ml_score) + (llm_weight * llm_score)

    if disagreement:
        return FinalAssessment(
            verdict="review",
            confidence=55,
            agreement_status=agreement_status,
            explanation="The pattern model and evidence-check model disagree, so this article is flagged for manual review.",
        )

    if fused_risk >= 0.72:
        verdict = "fake"
    elif fused_risk <= 0.35:
        verdict = "real"
    else:
        verdict = "review"

    confidence = int(round(max(0.0, abs(fused_risk - 0.5) * 2.0) * 100))
    return FinalAssessment(
        verdict=verdict,
        confidence=confidence,
        agreement_status=agreement_status,
        explanation="Final recommendation combines ML writing-pattern risk with LLM evidence verification.",
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_warm_model_in_background, daemon=True).start()
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
    ml_assessment = MlAssessment(
        label=prediction.label,
        label_name=prediction.label_name,
        confidence=prediction.confidence,
        summary="Pattern-based classification from the trained text model.",
    )

    if not payload.use_llm:
        final_assessment = _fuse_assessments(prediction, None)
        return AnalyzeResponse(
            prediction=prediction,
            ml_assessment=ml_assessment,
            final_assessment=final_assessment,
        )

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

    llm_assessment = None
    if fact_check is not None:
        llm_assessment = LlmAssessment(
            verdict=fact_check.verdict,
            confidence=fact_check.confidence,
            explanation=fact_check.explanation,
            has_trusted_references=bool(fact_check.has_trusted_references),
            summary="Evidence-oriented claim verification using retrieved references.",
        )

    final_assessment = _fuse_assessments(prediction, fact_check)
    if final_assessment.agreement_status == "disagree":
        warnings.insert(0, "ML and LLM disagree on this article. Final recommendation is REVIEW.")

    return AnalyzeResponse(
        prediction=prediction,
        ml_assessment=ml_assessment,
        llm_assessment=llm_assessment,
        final_assessment=final_assessment,
        summary=summary,
        risk_explanation=risk_explanation,
        fact_check=fact_check,
        warnings=warnings,
    )
