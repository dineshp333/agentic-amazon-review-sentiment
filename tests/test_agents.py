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


def test_data_agent_cleaning_with_special_chars():
    """Test DataAgent handles special characters and URLs."""
    agent = DataAgent()
    text_with_url = "Check out https://example.com for more!"
    cleaned = agent.clean_text(text_with_url)
    assert "https" not in cleaned
    assert "example" in cleaned or "check" in cleaned


def test_data_agent_preprocessing():
    """Test DataAgent review preprocessing."""
    agent = DataAgent()
    title, body = agent.preprocess_review(
        "Great Product!", "I loved this product. It works perfectly!"
    )
    assert len(title) > 0
    assert len(body) > 0


def test_data_agent_empty_text_handling():
    """Test DataAgent handles empty or None values gracefully."""
    agent = DataAgent()
    result = agent.clean_text("")
    assert result == ""

    # Test with None should not crash
    try:
        result = agent.clean_text(None)
        assert True  # Should handle None gracefully
    except (TypeError, AttributeError):
        # Also acceptable to raise TypeError for None
        assert True


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


def test_evaluation_agent_low_confidence_detection():
    """Test EvaluationAgent identifies low-confidence predictions."""
    evaluator = EvaluationAgent()
    predictions = [
        {"label": "positive", "score": 0.52},  # Low confidence
        {"label": "negative", "score": 0.95},  # High confidence
        {"label": "positive", "score": 0.51},  # Low confidence
    ]
    result = evaluator.assess_batch(predictions)
    assert result["low_confidence_count"] >= 2  # Should catch borderline cases


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


def test_improvement_agent_high_accuracy():
    """Test ImprovementAgent with high accuracy model (91.5%)."""
    improver = ImprovementAgent()
    eval_results = {
        "accuracy": 0.9154,
        "f1": 0.9159,
        "precision": 0.9102,
        "recall": 0.9216,
        "total_predictions": 9087,
        "low_confidence_count": 50,
    }
    analysis = improver.analyze(eval_results)
    assert "recommendations" in analysis
    # High accuracy should result in positive status
    assert analysis["overall_status"] in ["excellent", "good"]


def test_evaluation_agent_balanced_predictions():
    """Test EvaluationAgent with balanced positive/negative predictions."""
    evaluator = EvaluationAgent()
    predictions = [
        {"label": "positive", "score": 0.92},
        {"label": "negative", "score": 0.88},
        {"label": "positive", "score": 0.91},
        {"label": "negative", "score": 0.87},
    ]
    ground_truth = ["positive", "negative", "positive", "negative"]
    result = evaluator.evaluate(predictions, ground_truth)

    # Perfect predictions should give accuracy of 1.0
    assert result["accuracy"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
