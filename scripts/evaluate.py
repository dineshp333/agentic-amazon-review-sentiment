"""Evaluate sentiment runs and confidence."""

from __future__ import annotations
from typing import List, Dict, Any
from agents.evaluation_agent import EvaluationAgent

_evaluator = EvaluationAgent()


def summarize(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    return _evaluator.assess(predictions)
