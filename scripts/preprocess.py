"""Utility functions for cleaning and validating review data."""

from __future__ import annotations
from typing import Dict, Any
from agents.data_agent import DataAgent

data_agent = DataAgent()


def preprocess_record(title: str, body: str) -> Dict[str, Any]:
    record = data_agent.preprocess(title, body)
    if not data_agent.validate(record):
        raise ValueError("Invalid review payload")
    return record
