from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=20, description="News article text to classify")


class PredictResponse(BaseModel):
    label: int
    label_name: str
    confidence: float


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=20, description="News article text to analyze")
    use_llm: bool = True


class ReferenceItem(BaseModel):
    title: str
    url: str
    snippet: str
    source_type: Optional[str] = "fallback"


class FactCheckResponse(BaseModel):
    verdict: str
    confidence: int
    explanation: str
    references: List[ReferenceItem]
    has_trusted_references: Optional[bool] = False


class MlAssessment(BaseModel):
    label: int
    label_name: str
    confidence: float
    summary: str


class LlmAssessment(BaseModel):
    verdict: str
    confidence: int
    explanation: str
    has_trusted_references: bool
    summary: str


class FinalAssessment(BaseModel):
    verdict: str
    confidence: int
    agreement_status: str
    explanation: str


class AnalyzeResponse(BaseModel):
    prediction: PredictResponse
    ml_assessment: Optional[MlAssessment] = None
    llm_assessment: Optional[LlmAssessment] = None
    final_assessment: Optional[FinalAssessment] = None
    summary: Optional[str] = None
    risk_explanation: Optional[str] = None
    fact_check: Optional[FactCheckResponse] = None
    warnings: List[str] = []


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    dataset_path: str
