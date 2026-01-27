"""ImprovementAgent suggests next steps like retraining or prompt tweaks."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ImprovementAgent:
    def suggest(self, evaluation: Dict[str, Any]) -> List[str]:
        """Return placeholder recommendations based on evaluation output."""
        suggestions: List[str] = []
        if evaluation.get("avg_score", 0) < 0.6:
            suggestions.append("Consider retraining with more labeled data.")
        if evaluation.get("low_confidence"):
            suggestions.append(
                "Review low-confidence cases and augment prompts/models."
            )
        if not suggestions:
            suggestions.append("Performance acceptable; monitor in production.")
        return suggestions
