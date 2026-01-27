"""Evaluate sentiment predictions against ground truth and export results."""

import csv
import logging
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path
from agents.evaluation_agent import EvaluationAgent
from agents.improvement_agent import ImprovementAgent
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

# Initialize agents
_evaluator = EvaluationAgent()
_improver = ImprovementAgent()


def evaluate_predictions(
    predictions: List[Dict[str, Any]],
    ground_truth: List[str],
) -> Dict[str, Any]:
    """
    Evaluate predictions against ground truth labels.

    Calculates accuracy, precision, recall, and F1 score.

    Args:
        predictions (List[Dict[str, Any]]): List of prediction dictionaries
                                           containing "label" and "score" keys.
        ground_truth (List[str]): List of true labels ("positive" or "negative").

    Returns:
        Dict[str, Any]: Evaluation results including:
            - "accuracy" (float): Accuracy score
            - "precision" (float): Precision score
            - "recall" (float): Recall score
            - "f1" (float): F1 score
            - "total_predictions" (int): Total predictions
            - "low_confidence_count" (int): Number of low-confidence predictions
            - "low_confidence_predictions" (List): Details of low-confidence preds
            - "confusion_matrix" (Dict): Confusion matrix breakdown

    Raises:
        ValueError: If predictions and ground_truth have different lengths.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Predictions ({len(predictions)}) and ground_truth ({len(ground_truth)}) "
            "must have the same length"
        )

    # Use EvaluationAgent for metrics
    results = _evaluator.evaluate(predictions, ground_truth)

    # Add confusion matrix
    predicted_labels = [p.get("label") for p in predictions]
    cm = confusion_matrix(
        ground_truth, predicted_labels, labels=["positive", "negative"]
    )

    results["confusion_matrix"] = {
        "true_positive": int(cm[0, 0]),
        "false_negative": int(cm[0, 1]),
        "false_positive": int(cm[1, 0]),
        "true_negative": int(cm[1, 1]),
    }

    logger.info(
        f"Evaluation complete: Accuracy={results['accuracy']:.4f}, F1={results['f1']:.4f}"
    )
    return results


def assess_predictions(
    predictions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Assess predictions without ground truth (descriptive statistics).

    Useful for production monitoring when true labels are unavailable.

    Args:
        predictions (List[Dict[str, Any]]): List of prediction dictionaries.

    Returns:
        Dict[str, Any]: Descriptive statistics including confidence levels and counts.
    """
    return _evaluator.assess_batch(predictions)


def save_evaluation_results(
    predictions: List[Dict[str, Any]],
    ground_truth: List[str] | None = None,
    output_filepath: str = "evaluation_results.csv",
) -> str:
    """
    Save evaluation results to a CSV file.

    Args:
        predictions (List[Dict[str, Any]]): List of prediction dictionaries.
        ground_truth (List[str] | None): Optional true labels for comparison.
        output_filepath (str): Path to save CSV file. Defaults to "evaluation_results.csv".

    Returns:
        str: Path to saved CSV file.

    Raises:
        ValueError: If predictions list is empty.
    """
    if not predictions:
        raise ValueError("Predictions list cannot be empty")

    # Ensure output directory exists
    output_path = Path(output_filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare rows for CSV
    rows = []
    for idx, pred in enumerate(predictions):
        row = {
            "index": idx,
            "predicted_label": pred.get("label"),
            "confidence_score": pred.get("score"),
            "preprocessed_title": pred.get("preprocessed_title", ""),
            "preprocessed_body": pred.get("preprocessed_body", ""),
        }

        # Add ground truth if provided
        if ground_truth and idx < len(ground_truth):
            row["true_label"] = ground_truth[idx]
            row["correct"] = pred.get("label") == ground_truth[idx]

        rows.append(row)

    # Write to CSV
    fieldnames = list(rows[0].keys()) if rows else []
    with open(output_filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Evaluation results saved to {output_filepath}")
    return str(output_filepath)


def generate_full_report(
    predictions: List[Dict[str, Any]],
    ground_truth: List[str] | None = None,
    output_dir: str = "evaluation_reports",
) -> Dict[str, Any]:
    """
    Generate a comprehensive evaluation report with metrics, analysis, and recommendations.

    Args:
        predictions (List[Dict[str, Any]]): List of prediction dictionaries.
        ground_truth (List[str] | None): Optional true labels.
        output_dir (str): Directory to save reports. Defaults to "evaluation_reports".

    Returns:
        Dict[str, Any]: Complete report including evaluation metrics and recommendations.
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate evaluation results
    if ground_truth:
        eval_results = evaluate_predictions(predictions, ground_truth)
    else:
        eval_results = assess_predictions(predictions)

    # Generate improvement recommendations
    improvement_analysis = _improver.analyze(eval_results)

    # Save CSV results
    csv_path = output_path / "predictions.csv"
    save_evaluation_results(predictions, ground_truth, str(csv_path))

    # Save JSON report
    json_path = output_path / "analysis_report.json"
    _improver.save_report(improvement_analysis, str(json_path))

    # Create summary report
    timestamp = datetime.now().isoformat()
    report = {
        "timestamp": timestamp,
        "evaluation_metrics": eval_results,
        "improvement_analysis": improvement_analysis,
        "files_saved": {
            "csv": str(csv_path),
            "json": str(json_path),
        },
    }

    logger.info(f"Full report generated in {output_dir}")
    return report


def print_evaluation_summary(evaluation_results: Dict[str, Any]) -> None:
    """
    Print a formatted summary of evaluation results to console.

    Args:
        evaluation_results (Dict[str, Any]): Evaluation output from evaluate_predictions().
    """
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Predictions: {evaluation_results.get('total_predictions')}")
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:  {evaluation_results.get('accuracy', 0):.4f}")
    print(f"  Precision: {evaluation_results.get('precision', 0):.4f}")
    print(f"  Recall:    {evaluation_results.get('recall', 0):.4f}")
    print(f"  F1 Score:  {evaluation_results.get('f1', 0):.4f}")
    print(f"\nConfusion Matrix:")
    cm = evaluation_results.get("confusion_matrix", {})
    print(f"  True Positive:  {cm.get('true_positive', 0)}")
    print(f"  True Negative:  {cm.get('true_negative', 0)}")
    print(f"  False Positive: {cm.get('false_positive', 0)}")
    print(f"  False Negative: {cm.get('false_negative', 0)}")
    print(
        f"\nLow-Confidence Predictions: {evaluation_results.get('low_confidence_count', 0)}"
    )
    print("=" * 60 + "\n")
