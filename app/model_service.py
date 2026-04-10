import os
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


@dataclass
class ModelArtifacts:
    vectorizer: TfidfVectorizer
    model: LogisticRegression


class FakeNewsModelService:
    def __init__(self, dataset_csv: str) -> None:
        self.dataset_csv = dataset_csv
        self.artifacts: ModelArtifacts | None = None

    def load_or_train(self) -> None:
        if not os.path.exists(self.dataset_csv):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_csv}")

        df = pd.read_csv(self.dataset_csv)
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("Dataset must contain 'text' and 'label' columns")

        df = df[["text", "label"]].dropna().copy()
        df["text"] = df["text"].astype(str)
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

        X = self.artifacts.vectorizer.transform([text])
        proba = self.artifacts.model.predict_proba(X)[0]
        label = int(proba.argmax())
        confidence = float(proba[label])
        return label, confidence
