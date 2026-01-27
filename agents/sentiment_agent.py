"""SentimentAgent runs sentiment prediction using pre-trained Keras model and CountVectorizers."""

import os
from typing import Dict, Any
import numpy as np
import joblib
from tensorflow import keras


class SentimentAgent:
    """
    SentimentAgent loads a pretrained Keras sentiment model and CountVectorizers.

    It accepts preprocessed text and returns sentiment predictions (Positive/Negative)
    with confidence scores.

    Attributes:
        model_path (str): Path to the pretrained Keras model.
        title_vectorizer_path (str): Path to the CountVectorizer for review titles.
        body_vectorizer_path (str): Path to the CountVectorizer for review bodies.
        model: Loaded Keras model instance.
        title_vectorizer: Loaded CountVectorizer for titles.
        body_vectorizer: Loaded CountVectorizer for bodies.
    """

    def __init__(
        self,
        model_path: str = "models/model.keras",
        title_vectorizer_path: str = "models/cv1.pkl",
        body_vectorizer_path: str = "models/cv2.pkl",
    ) -> None:
        """
        Initialize SentimentAgent with paths to model and vectorizers.

        Args:
            model_path (str): Path to the Keras sentiment model. Defaults to "models/model.keras".
            title_vectorizer_path (str): Path to the title CountVectorizer. Defaults to "models/cv1.pkl".
            body_vectorizer_path (str): Path to the body CountVectorizer. Defaults to "models/cv2.pkl".
        """
        self.model_path = model_path
        self.title_vectorizer_path = title_vectorizer_path
        self.body_vectorizer_path = body_vectorizer_path

        self.model = None
        self.title_vectorizer = None
        self.body_vectorizer = None

    def load(self) -> None:
        """
        Load the pretrained Keras model and CountVectorizers from disk.

        Raises:
            FileNotFoundError: If any model or vectorizer file is not found.
            Exception: If loading fails due to file corruption or format issues.
        """
        # Load Keras model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        self.model = keras.models.load_model(self.model_path)

        # Load title CountVectorizer
        if not os.path.exists(self.title_vectorizer_path):
            raise FileNotFoundError(
                f"Title vectorizer not found at {self.title_vectorizer_path}"
            )
        self.title_vectorizer = joblib.load(self.title_vectorizer_path)

        # Load body CountVectorizer
        if not os.path.exists(self.body_vectorizer_path):
            raise FileNotFoundError(
                f"Body vectorizer not found at {self.body_vectorizer_path}"
            )
        self.body_vectorizer = joblib.load(self.body_vectorizer_path)

    def predict(self, title: str, body: str) -> Dict[str, Any]:
        """
        Predict sentiment for a review given its title and body.

        Accepts preprocessed text (from DataAgent) and returns a sentiment label
        (Positive/Negative) with confidence score.

        Args:
            title (str): Preprocessed review title.
            body (str): Preprocessed review body.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - "label" (str): Sentiment label ("positive" or "negative")
                - "score" (float): Confidence score between 0 and 1
                - "title_features" (int): Number of features from title vectorizer
                - "body_features" (int): Number of features from body vectorizer

        Raises:
            ValueError: If title or body is empty after preprocessing.
            RuntimeError: If model or vectorizers are not loaded.
        """
        # Ensure models are loaded
        if (
            self.model is None
            or self.title_vectorizer is None
            or self.body_vectorizer is None
        ):
            self.load()

        # Validate input
        if not title or not isinstance(title, str):
            raise ValueError("Title must be a non-empty string")
        if not body or not isinstance(body, str):
            raise ValueError("Body must be a non-empty string")

        # Vectorize title and body
        title_features = self.title_vectorizer.transform([title]).toarray()
        body_features = self.body_vectorizer.transform([body]).toarray()

        # Concatenate features
        combined_features = np.hstack([title_features, body_features])

        # Make prediction
        raw_score = self.model.predict(combined_features, verbose=0)[0][0]

        # Convert to label and confidence
        confidence = float(raw_score) if raw_score >= 0.5 else float(1 - raw_score)
        label = "positive" if raw_score >= 0.5 else "negative"

        return {
            "label": label,
            "score": confidence,
            "title_features": title_features.shape[1],
            "body_features": body_features.shape[1],
        }

    def predict_batch(
        self, titles: list[str], bodies: list[str]
    ) -> list[Dict[str, Any]]:
        """
        Predict sentiment for multiple reviews in batch.

        Args:
            titles (list[str]): List of preprocessed review titles.
            bodies (list[str]): List of preprocessed review bodies.

        Returns:
            list[Dict[str, Any]]: List of sentiment predictions.

        Raises:
            ValueError: If titles and bodies lists have different lengths.
        """
        if len(titles) != len(bodies):
            raise ValueError("titles and bodies must have the same length")

        return [self.predict(title, body) for title, body in zip(titles, bodies)]
