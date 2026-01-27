"""Basic smoke tests for agent placeholders."""

from agents.data_agent import DataAgent
from agents.sentiment_agent import SentimentAgent
from agents.evaluation_agent import EvaluationAgent
from agents.improvement_agent import ImprovementAgent


def test_data_agent_roundtrip():
    agent = DataAgent()
    record = agent.preprocess("Great", "Loved the product")
    assert agent.validate(record)
    assert record["title"]
    assert record["body"]


def test_sentiment_agent_predict():
    agent = SentimentAgent()
    result = agent.predict("Title", "Body")
    assert "label" in result and "score" in result


def test_evaluation_agent_assess():
    evaluator = EvaluationAgent()
    summary = evaluator.assess([{"score": 0.5}, {"score": 0.9}])
    assert summary["avg_score"] > 0


def test_improvement_agent_suggest():
    improver = ImprovementAgent()
    suggestions = improver.suggest({"avg_score": 0.4, "low_confidence": [{}]})
    assert suggestions
