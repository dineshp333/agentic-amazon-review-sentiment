"""EvaluationAgent checks model confidence and evaluation metrics."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class EvaluationAgent:
    threshold: float = 0.6

    def assess(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute simple aggregates for now."""
        if not predictions:
            return {"avg_score": 0.0, "low_confidence": []}
        scores = [p.get("score", 0.0) for p in predictions]
        avg = sum(scores) / len(scores)
        low_conf = [p for p in predictions if p.get("score", 0.0) < self.threshold]
        return {"avg_score": avg, "low_confidence": low_conf}
