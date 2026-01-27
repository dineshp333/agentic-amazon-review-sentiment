"""Run sentiment inference for reviews."""

from __future__ import annotations
from typing import Dict, Any
from agents.sentiment_agent import SentimentAgent
from scripts.preprocess import preprocess_record

sentiment_agent = SentimentAgent()


def run_inference(title: str, body: str) -> Dict[str, Any]:
    record = preprocess_record(title, body)
    prediction = sentiment_agent.predict(record["title"], record["body"])
    return {**prediction, "meta": record.get("meta", {})}
