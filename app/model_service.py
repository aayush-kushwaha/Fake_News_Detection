import os
import re
from dataclasses import dataclass

from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


@dataclass
class ModelArtifacts:
    vectorizer: TfidfVectorizer
    model: LogisticRegression


class FakeNewsModelService:
    FAKE_THRESHOLD = 0.78

    def __init__(self, dataset_csv: str) -> None:
        self.dataset_csv = dataset_csv
        self.artifact_path = os.getenv("FAKE_NEWS_ARTIFACT_PATH", "data/model_artifacts.joblib")
        self.artifacts: ModelArtifacts | None = None

    @staticmethod
    def _remove_publisher_tags(text: str) -> str:
        cleaned_text = re.sub(r'^.*?(?:reuters).*?-\s*', '', text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'^[\s\w]*?\([^\)]*\)\s*-\s*', '', cleaned_text)
        cleaned_text = re.sub(r'^\s*\d+\.\s*', '', cleaned_text)
        cleaned_text = re.sub(r'^["“”\']+|["“”\']+$', '', cleaned_text)
        return cleaned_text.strip()

    @classmethod
    def _clean_text(cls, text: str) -> str:
        text = str(text)
        text = cls._remove_publisher_tags(text)
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def load_or_train(self) -> None:
        if not os.path.exists(self.artifact_path):
            raise FileNotFoundError(
                f"Model artifact not found at {self.artifact_path}. "
                "Server mode is artifact-only; train locally and copy model_artifacts.joblib to the server."
            )

        cached = load(self.artifact_path)
        if not isinstance(cached, dict) or "vectorizer" not in cached or "model" not in cached:
            raise ValueError("Invalid model artifact format. Expected keys: vectorizer, model")

        self.artifacts = ModelArtifacts(vectorizer=cached["vectorizer"], model=cached["model"])

    @property
    def is_ready(self) -> bool:
        return self.artifacts is not None

    def predict(self, text: str) -> tuple[int, float]:
        if self.artifacts is None:
            raise RuntimeError("Model is not initialized")

        cleaned_text = self._clean_text(text)
        X = self.artifacts.vectorizer.transform([cleaned_text])
        proba = self.artifacts.model.predict_proba(X)[0]
        real_prob = float(proba[0])
        fake_prob = float(proba[1])

        if fake_prob >= self.FAKE_THRESHOLD:
            return 1, fake_prob

        # Relaxed mode: prefer real unless fake evidence is strong.
        if real_prob >= fake_prob:
            return 0, real_prob

        return 0, real_prob
