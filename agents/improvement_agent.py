"""ImprovementAgent analyzes evaluation results and suggests improvements."""

import json
from datetime import datetime
from typing import Dict, List, Any
from enum import Enum


class RecommendationType(Enum):
    """Enumeration of recommendation types."""

    RETRAIN = "retrain"
    AUGMENT_DATA = "augment_data"
    ADJUST_THRESHOLD = "adjust_threshold"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_REPLACEMENT = "model_replacement"
    PROMPT_MODIFICATION = "prompt_modification"
    MONITOR = "monitor"


class ImprovementAgent:
    """
    ImprovementAgent analyzes evaluation results and generates improvement recommendations.

    Examines metrics like accuracy, precision, recall, and F1 score to suggest
    actionable improvements such as retraining, data augmentation, threshold adjustment,
    or model replacement. Generates structured JSON reports.

    Attributes:
        min_acceptable_accuracy (float): Threshold below which retraining is recommended.
        min_acceptable_f1 (float): Threshold below which model improvements are needed.
        max_low_confidence_ratio (float): Max acceptable ratio of low-confidence predictions.
    """

    def __init__(
        self,
        min_acceptable_accuracy: float = 0.75,
        min_acceptable_f1: float = 0.70,
        max_low_confidence_ratio: float = 0.15,
    ) -> None:
        """
        Initialize ImprovementAgent.

        Args:
            min_acceptable_accuracy (float): Minimum acceptable accuracy. Defaults to 0.75.
            min_acceptable_f1 (float): Minimum acceptable F1 score. Defaults to 0.70.
            max_low_confidence_ratio (float): Max ratio of low-confidence predictions (0-1).
                                             Defaults to 0.15.
        """
        self.min_acceptable_accuracy = min_acceptable_accuracy
        self.min_acceptable_f1 = min_acceptable_f1
        self.max_low_confidence_ratio = max_low_confidence_ratio

    def analyze(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze evaluation results and generate recommendations.

        Args:
            evaluation_results (Dict[str, Any]): Evaluation output from EvaluationAgent
                                                containing accuracy, precision, recall, f1, etc.

        Returns:
            Dict[str, Any]: Analysis report containing:
                - "timestamp" (str): Analysis timestamp
                - "overall_status" (str): "good", "warning", or "critical"
                - "recommendations" (List[Dict]): List of structured recommendations
                - "metrics_summary" (Dict): Key metrics with pass/fail status
                - "priority_actions" (List[str]): Top 3 priority actions

        Raises:
            ValueError: If evaluation_results lacks required keys.
        """
        # Validate required keys
        required_keys = {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "total_predictions",
            "low_confidence_count",
        }
        if not all(key in evaluation_results for key in required_keys):
            raise ValueError(f"evaluation_results must contain keys: {required_keys}")

        accuracy = evaluation_results["accuracy"]
        f1 = evaluation_results["f1"]
        precision = evaluation_results["precision"]
        recall = evaluation_results["recall"]
        low_conf_ratio = (
            evaluation_results["low_confidence_count"]
            / evaluation_results["total_predictions"]
        )

        # Determine overall status
        overall_status = self._determine_status(accuracy, f1, low_conf_ratio)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            accuracy, precision, recall, f1, low_conf_ratio
        )

        # Create metrics summary
        metrics_summary = {
            "accuracy": {
                "value": float(accuracy),
                "threshold": self.min_acceptable_accuracy,
                "status": "pass"
                if accuracy >= self.min_acceptable_accuracy
                else "fail",
            },
            "f1": {
                "value": float(f1),
                "threshold": self.min_acceptable_f1,
                "status": "pass" if f1 >= self.min_acceptable_f1 else "fail",
            },
            "precision": {
                "value": float(precision),
                "status": "pass" if precision >= 0.70 else "warning",
            },
            "recall": {
                "value": float(recall),
                "status": "pass" if recall >= 0.70 else "warning",
            },
            "low_confidence_ratio": {
                "value": float(low_conf_ratio),
                "threshold": self.max_low_confidence_ratio,
                "status": "pass"
                if low_conf_ratio <= self.max_low_confidence_ratio
                else "fail",
            },
        }

        # Extract priority actions
        priority_actions = self._extract_priority_actions(recommendations)

        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "recommendations": recommendations,
            "metrics_summary": metrics_summary,
            "priority_actions": priority_actions,
            "full_evaluation": evaluation_results,
        }

    def _determine_status(
        self, accuracy: float, f1: float, low_conf_ratio: float
    ) -> str:
        """
        Determine overall status based on key metrics.

        Args:
            accuracy (float): Model accuracy.
            f1 (float): F1 score.
            low_conf_ratio (float): Ratio of low-confidence predictions.

        Returns:
            str: "good", "warning", or "critical".
        """
        if (
            accuracy >= self.min_acceptable_accuracy
            and f1 >= self.min_acceptable_f1
            and low_conf_ratio <= self.max_low_confidence_ratio
        ):
            return "good"
        elif accuracy >= 0.70 and f1 >= 0.60 and low_conf_ratio <= 0.25:
            return "warning"
        else:
            return "critical"

    def _generate_recommendations(
        self,
        accuracy: float,
        precision: float,
        recall: float,
        f1: float,
        low_conf_ratio: float,
    ) -> List[Dict[str, Any]]:
        """
        Generate structured recommendations based on metrics.

        Args:
            accuracy (float): Model accuracy.
            precision (float): Precision score.
            recall (float): Recall score.
            f1 (float): F1 score.
            low_conf_ratio (float): Ratio of low-confidence predictions.

        Returns:
            List[Dict[str, Any]]: List of recommendation dictionaries.
        """
        recommendations = []

        # Accuracy check
        if accuracy < self.min_acceptable_accuracy:
            recommendations.append(
                {
                    "type": RecommendationType.RETRAIN.value,
                    "severity": "high",
                    "description": f"Accuracy ({accuracy:.4f}) below acceptable threshold "
                    f"({self.min_acceptable_accuracy:.4f})",
                    "action": "Collect more labeled data and retrain the model with improved features.",
                    "estimated_impact": "10-20% improvement",
                }
            )

        # F1 score check
        if f1 < self.min_acceptable_f1:
            recommendations.append(
                {
                    "type": RecommendationType.FEATURE_ENGINEERING.value,
                    "severity": "high" if f1 < 0.60 else "medium",
                    "description": f"F1 score ({f1:.4f}) below acceptable threshold "
                    f"({self.min_acceptable_f1:.4f})",
                    "action": "Improve feature engineering or consider model architecture changes.",
                    "estimated_impact": "5-15% improvement",
                }
            )

        # Precision check
        if precision < 0.70:
            recommendations.append(
                {
                    "type": RecommendationType.PROMPT_MODIFICATION.value,
                    "severity": "medium",
                    "description": f"Low precision ({precision:.4f}) - too many false positives",
                    "action": "Adjust decision threshold or refine model prompts to be more conservative.",
                    "estimated_impact": "3-8% improvement",
                }
            )

        # Recall check
        if recall < 0.70:
            recommendations.append(
                {
                    "type": RecommendationType.ADJUST_THRESHOLD.value,
                    "severity": "medium",
                    "description": f"Low recall ({recall:.4f}) - missing positive cases",
                    "action": "Lower confidence threshold to catch more positive sentiments.",
                    "estimated_impact": "5-10% improvement",
                }
            )

        # Low confidence check
        if low_conf_ratio > self.max_low_confidence_ratio:
            recommendations.append(
                {
                    "type": RecommendationType.AUGMENT_DATA.value,
                    "severity": "medium",
                    "description": f"High low-confidence ratio ({low_conf_ratio:.4f}) - "
                    f"model is uncertain on {low_conf_ratio * 100:.1f}% of predictions",
                    "action": "Review uncertain cases, augment training data, or consider ensemble methods.",
                    "estimated_impact": "5-12% improvement",
                }
            )

        # Model replacement if multiple issues
        if accuracy < 0.60 and f1 < 0.50:
            recommendations.append(
                {
                    "type": RecommendationType.MODEL_REPLACEMENT.value,
                    "severity": "critical",
                    "description": "Multiple critical metrics below threshold - model fundamentally underperforming",
                    "action": "Consider replacing with a different model architecture (e.g., LSTM, Transformer).",
                    "estimated_impact": "20-40% improvement",
                }
            )

        # Monitor if good performance
        if not recommendations:
            recommendations.append(
                {
                    "type": RecommendationType.MONITOR.value,
                    "severity": "low",
                    "description": "Model performance acceptable",
                    "action": "Continue monitoring in production; schedule periodic retraining (every 30 days).",
                    "estimated_impact": "Maintain current performance",
                }
            )

        return recommendations

    def _extract_priority_actions(
        self, recommendations: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract top priority actions from recommendations.

        Args:
            recommendations (List[Dict[str, Any]]): List of recommendations.

        Returns:
            List[str]: Top 3 priority actions.
        """
        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_recs = sorted(
            recommendations,
            key=lambda x: severity_order.get(x.get("severity", "low"), 3),
        )

        return [rec["action"] for rec in sorted_recs[:3]]

    def generate_json_report(self, analysis: Dict[str, Any]) -> str:
        """
        Generate a formatted JSON report from analysis results.

        Args:
            analysis (Dict[str, Any]): Analysis output from analyze() method.

        Returns:
            str: Pretty-printed JSON report.
        """
        report = {
            "timestamp": analysis["timestamp"],
            "overall_status": analysis["overall_status"],
            "metrics_summary": analysis["metrics_summary"],
            "recommendations": analysis["recommendations"],
            "priority_actions": analysis["priority_actions"],
        }
        return json.dumps(report, indent=2)

    def save_report(self, analysis: Dict[str, Any], filepath: str) -> None:
        """
        Save analysis report to a JSON file.

        Args:
            analysis (Dict[str, Any]): Analysis output from analyze() method.
            filepath (str): Path to save the JSON report.
        """
        report_json = self.generate_json_report(analysis)
        with open(filepath, "w") as f:
            f.write(report_json)
        print(f"Report saved to {filepath}")
