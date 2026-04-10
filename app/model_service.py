import os
import re
from dataclasses import dataclass

import pandas as pd
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
        if not os.path.exists(self.dataset_csv):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_csv}")

        df = pd.read_csv(self.dataset_csv)
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("Dataset must contain 'text' and 'label' columns")

        df = df[["text", "label"]].dropna().copy()
        df["text"] = df["text"].astype(str).apply(self._clean_text)
        df["label"] = df["label"].astype(int)

        vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)
        X = vectorizer.fit_transform(df["text"])

        model = LogisticRegression(max_iter=1200, C=1.0, class_weight="balanced", random_state=42)
        model.fit(X, df["label"])

        self.artifacts = ModelArtifacts(vectorizer=vectorizer, model=model)

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
