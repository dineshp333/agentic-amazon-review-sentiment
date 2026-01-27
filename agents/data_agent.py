"""DataAgent handles review preprocessing and validation steps."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List


def clean_text(text: str) -> str:
    """Basic text cleaner placeholder; extend with real normalization."""
    return text.strip()


@dataclass
class DataAgent:
    config: Dict[str, Any] | None = None

    def preprocess(self, title: str, body: str) -> Dict[str, Any]:
        """Return cleaned title/body plus any derived metadata."""
        return {
            "title": clean_text(title),
            "body": clean_text(body),
            "meta": {
                "title_tokens": len(title.split()),
                "body_tokens": len(body.split()),
            },
        }

    def validate(self, record: Dict[str, Any]) -> bool:
        """Lightweight validation placeholder."""
        required = ["title", "body"]
        return all(record.get(key) for key in required)
