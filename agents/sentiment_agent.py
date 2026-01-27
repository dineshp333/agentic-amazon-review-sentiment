"""SentimentAgent runs sentiment prediction using pre-trained assets."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class SentimentAgent:
    model_path: str = "models/model.keras"
    title_vectorizer_path: str = "models/cv1.pkl"
    body_vectorizer_path: str = "models/cv2.pkl"
    model: Any | None = None
    title_vectorizer: Any | None = None
    body_vectorizer: Any | None = None

    def load(self) -> None:
        """Placeholder for loading model/vectorizers from disk."""
        # Implement actual load logic (e.g., keras.models.load_model, joblib.load)
        self.model = "loaded-model"
        self.title_vectorizer = "title-cv"
        self.body_vectorizer = "body-cv"

    def predict(self, title: str, body: str) -> Dict[str, Any]:
        """Run a dummy sentiment prediction until model is wired."""
        if self.model is None:
            self.load()
        score = 0.5  # placeholder score
        label = "positive" if score >= 0.5 else "negative"
        return {"label": label, "score": score}
