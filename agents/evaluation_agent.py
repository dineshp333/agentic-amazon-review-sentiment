"""EvaluationAgent evaluates sentiment predictions and calculates performance metrics."""

import logging
from typing import Dict, List, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure logging for low-confidence predictions
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EvaluationAgent:
    """
    EvaluationAgent evaluates sentiment predictions from SentimentAgent.

    Calculates performance metrics (accuracy, precision, recall, F1 score),
    identifies low-confidence predictions for manual review, and logs insights.

    Attributes:
        confidence_threshold (float): Minimum confidence score to consider a prediction reliable.
    """

    def __init__(self, confidence_threshold: float = 0.6) -> None:
        """
        Initialize EvaluationAgent.

        Args:
            confidence_threshold (float): Predictions below this score are flagged as low-confidence.
                                         Defaults to 0.6.
        """
        self.confidence_threshold = confidence_threshold

    def evaluate(
        self, predictions: List[Dict[str, Any]], ground_truth: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate predictions against ground truth and calculate metrics.

        Args:
            predictions (List[Dict[str, Any]]): List of prediction dictionaries from SentimentAgent,
                                               each containing "label" and "score".
            ground_truth (List[str]): List of true labels ("positive" or "negative").

        Returns:
            Dict[str, Any]: Dictionary containing:
                - "accuracy" (float): Overall accuracy score
                - "precision" (float): Precision score
                - "recall" (float): Recall score
                - "f1" (float): F1 score
                - "total_predictions" (int): Total number of predictions
                - "low_confidence_count" (int): Number of low-confidence predictions
                - "low_confidence_predictions" (List[Dict]): Low-confidence prediction details

        Raises:
            ValueError: If predictions and ground_truth have different lengths or invalid data.
        """
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Predictions ({len(predictions)}) and ground_truth ({len(ground_truth)}) "
                "must have the same length"
            )

        if not predictions:
            raise ValueError("Predictions list cannot be empty")

        # Extract predicted labels
        predicted_labels = [p.get("label") for p in predictions]

        # Validate labels
        valid_labels = {"positive", "negative"}
        for label in predicted_labels + ground_truth:
            if label not in valid_labels:
                raise ValueError(
                    f"Invalid label '{label}'. Expected 'positive' or 'negative'"
                )

        # Calculate metrics using scikit-learn
        accuracy = accuracy_score(ground_truth, predicted_labels)
        precision = precision_score(
            ground_truth, predicted_labels, average="weighted", zero_division=0
        )
        recall = recall_score(
            ground_truth, predicted_labels, average="weighted", zero_division=0
        )
        f1 = f1_score(
            ground_truth, predicted_labels, average="weighted", zero_division=0
        )

        # Identify low-confidence predictions
        low_confidence = self._identify_low_confidence(predictions)

        # Log results
        logger.info(
            f"Evaluation Summary - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, F1: {f1:.4f}"
        )
        logger.info(
            f"Low-confidence predictions: {len(low_confidence)} / {len(predictions)}"
        )

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "total_predictions": len(predictions),
            "low_confidence_count": len(low_confidence),
            "low_confidence_predictions": low_confidence,
        }

    def _identify_low_confidence(
        self, predictions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Identify predictions below the confidence threshold.

        Args:
            predictions (List[Dict[str, Any]]): List of prediction dictionaries.

        Returns:
            List[Dict[str, Any]]: List of low-confidence predictions with details.
        """
        low_conf = []
        for idx, pred in enumerate(predictions):
            score = pred.get("score", 0.0)
            if score < self.confidence_threshold:
                low_conf.append(
                    {
                        "index": idx,
                        "label": pred.get("label"),
                        "score": score,
                        "gap": self.confidence_threshold - score,
                    }
                )
                logger.warning(
                    f"Low-confidence prediction at index {idx}: "
                    f"label='{pred.get('label')}', score={score:.4f}"
                )
        return low_conf

    def assess_batch(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess a batch of predictions without ground truth (descriptive statistics only).

        Useful for monitoring in production when true labels are unavailable.

        Args:
            predictions (List[Dict[str, Any]]): List of prediction dictionaries from SentimentAgent.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - "avg_confidence" (float): Average confidence score
                - "min_confidence" (float): Minimum confidence score
                - "max_confidence" (float): Maximum confidence score
                - "total_predictions" (int): Total number of predictions
                - "positive_count" (int): Number of positive predictions
                - "negative_count" (int): Number of negative predictions
                - "low_confidence_count" (int): Number of low-confidence predictions
                - "low_confidence_predictions" (List[Dict]): Low-confidence prediction details

        Raises:
            ValueError: If predictions list is empty.
        """
        if not predictions:
            raise ValueError("Predictions list cannot be empty")

        scores = [p.get("score", 0.0) for p in predictions]
        labels = [p.get("label") for p in predictions]

        low_confidence = self._identify_low_confidence(predictions)

        return {
            "avg_confidence": float(sum(scores) / len(scores)),
            "min_confidence": float(min(scores)),
            "max_confidence": float(max(scores)),
            "total_predictions": len(predictions),
            "positive_count": sum(1 for label in labels if label == "positive"),
            "negative_count": sum(1 for label in labels if label == "negative"),
            "low_confidence_count": len(low_confidence),
            "low_confidence_predictions": low_confidence,
        }

    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Update the confidence threshold dynamically.

        Args:
            threshold (float): New confidence threshold (0.0 to 1.0).

        Raises:
            ValueError: If threshold is not between 0 and 1.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.confidence_threshold = threshold
        logger.info(f"Confidence threshold updated to {threshold:.2f}")
