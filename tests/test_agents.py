"""Unit tests for all agents - no model files required."""

from agents.data_agent import DataAgent
from agents.evaluation_agent import EvaluationAgent
from agents.improvement_agent import ImprovementAgent


def test_data_agent_cleaning():
    """Test DataAgent text cleaning."""
    agent = DataAgent()
    cleaned = agent.clean_text("Hello!!! This is a TEST.")
    assert "hello" in cleaned.lower()
    assert "!" not in cleaned


def test_data_agent_preprocessing():
    """Test DataAgent review preprocessing."""
    agent = DataAgent()
    title, body = agent.preprocess_review(
        "Great Product!", "I loved this product. It works perfectly!"
    )
    assert len(title) > 0
    assert len(body) > 0


def test_data_agent_validation():
    """Test DataAgent input validation."""
    agent = DataAgent()
    assert agent.validate_review("Good Title", "Good Body")
    assert not agent.validate_review("", "")


def test_data_agent_batch():
    """Test DataAgent batch processing."""
    agent = DataAgent()
    titles = ["Title 1", "Title 2"]
    bodies = ["Body 1", "Body 2"]
    results = agent.preprocess_reviews_batch(titles, bodies)
    assert len(results) == 2


def test_evaluation_agent_assess():
    """Test EvaluationAgent batch assessment."""
    evaluator = EvaluationAgent()
    predictions = [
        {"label": "positive", "score": 0.9},
        {"label": "negative", "score": 0.8},
    ]
    result = evaluator.assess_batch(predictions)
    assert result["total_predictions"] == 2


def test_evaluation_agent_metrics():
    """Test EvaluationAgent metrics with ground truth."""
    evaluator = EvaluationAgent()
    predictions = [
        {"label": "positive", "score": 0.92},
        {"label": "negative", "score": 0.85},
        {"label": "positive", "score": 0.78},
    ]
    ground_truth = ["positive", "negative", "positive"]
    result = evaluator.evaluate(predictions, ground_truth)
    assert result["accuracy"] > 0
    assert "precision" in result
    assert "recall" in result


def test_improvement_agent_analysis():
    """Test ImprovementAgent recommendation generation."""
    improver = ImprovementAgent()
    eval_results = {
        "accuracy": 0.80,
        "f1": 0.75,
        "precision": 0.80,
        "recall": 0.72,
        "total_predictions": 100,
        "low_confidence_count": 10,
    }
    analysis = improver.analyze(eval_results)
    assert "recommendations" in analysis
    assert "overall_status" in analysis
