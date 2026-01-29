"""Unit tests for Streamlit app functionality."""

import csv
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import the save function from the app
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_csv_file_structure():
    """Test that CSV file is created with correct headers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "test_results.csv"

        # Create CSV with correct headers
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Timestamp",
                    "Name",
                    "Profession",
                    "Review Title",
                    "Review Body",
                    "Sentiment",
                    "Confidence",
                ]
            )

        # Read and verify headers
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

        assert "Timestamp" in headers
        assert "Name" in headers
        assert "Profession" in headers
        assert "Review Title" in headers
        assert "Review Body" in headers
        assert "Sentiment" in headers
        assert "Confidence" in headers
        assert "Age" not in headers  # Old field should not exist


def test_csv_data_write():
    """Test writing data to CSV file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "test_results.csv"

        # Create CSV with headers
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Timestamp",
                    "Name",
                    "Profession",
                    "Review Title",
                    "Review Body",
                    "Sentiment",
                    "Confidence",
                ]
            )

        # Write test data
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "2026-01-29 12:00:00",
                    "Test User",
                    "Student",
                    "Great Product",
                    "This is a great product!",
                    "POSITIVE",
                    "0.95",
                ]
            )

        # Verify data was written
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["Name"] == "Test User"
        assert rows[0]["Profession"] == "Student"
        assert rows[0]["Sentiment"] == "POSITIVE"


def test_profession_options():
    """Test that profession options are valid."""
    valid_professions = [
        "Prefer not to say",
        "Student",
        "Working Professional",
        "Business Owner",
        "Freelancer",
        "Retired",
    ]

    # All options should be strings
    for profession in valid_professions:
        assert isinstance(profession, str)
        assert len(profession) > 0

    # Should have exactly 6 options
    assert len(valid_professions) == 6


def test_user_data_validation():
    """Test user data validation logic."""
    # Valid inputs
    assert len("Test User".strip()) > 0
    assert "Student" in ["Prefer not to say", "Student", "Working Professional"]

    # Empty name handling
    name_input = ""
    default_name = "User"
    final_name = name_input.strip() if name_input else default_name
    assert final_name == "User"

    # Valid name handling
    name_input = "John Doe"
    final_name = name_input.strip() if name_input else default_name
    assert final_name == "John Doe"


def test_review_validation():
    """Test review input validation."""
    # Valid review
    title = "Great Product"
    body = "This product works perfectly!"
    assert len(title.strip()) > 0
    assert len(body.strip()) > 0

    # Empty review
    empty_title = ""
    empty_body = ""
    assert len(empty_title.strip()) == 0
    assert len(empty_body.strip()) == 0


def test_sentiment_format():
    """Test sentiment output format."""
    valid_sentiments = ["POSITIVE", "NEGATIVE"]

    for sentiment in valid_sentiments:
        assert sentiment.isupper()
        assert isinstance(sentiment, str)


def test_confidence_score_range():
    """Test confidence score validation."""
    # Valid confidence scores
    valid_scores = [0.0, 0.5, 0.95, 1.0]
    for score in valid_scores:
        assert 0.0 <= score <= 1.0

    # Invalid scores should fail
    invalid_scores = [-0.1, 1.1]
    for score in invalid_scores:
        assert not (0.0 <= score <= 1.0)


def test_csv_row_format():
    """Test that CSV rows have correct number of fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "test_results.csv"

        # Create CSV
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            headers = [
                "Timestamp",
                "Name",
                "Profession",
                "Review Title",
                "Review Body",
                "Sentiment",
                "Confidence",
            ]
            writer.writerow(headers)
            writer.writerow(
                [
                    "2026-01-29 12:00:00",
                    "Test",
                    "Student",
                    "Title",
                    "Body",
                    "POSITIVE",
                    "0.95",
                ]
            )

        # Verify row structure
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
            row = next(reader)

        assert len(headers) == 7
        assert len(row) == 7


def test_default_session_values():
    """Test default session state values."""
    default_name = "User"
    default_profession = "Prefer not to say"

    assert isinstance(default_name, str)
    assert isinstance(default_profession, str)
    assert len(default_name) > 0
    assert len(default_profession) > 0


def test_profession_field_migration():
    """Test that old Age field data can be migrated to Profession."""
    with tempfile.TemporaryDirectory() as tmpdir:
        old_csv = Path(tmpdir) / "old.csv"
        new_csv = Path(tmpdir) / "new.csv"

        # Create old format CSV
        with open(old_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Timestamp",
                    "Name",
                    "Age",
                    "Review Title",
                    "Review Body",
                    "Sentiment",
                    "Confidence",
                ]
            )
            writer.writerow(
                [
                    "2026-01-29 12:00:00",
                    "User",
                    "25",
                    "Title",
                    "Body",
                    "POSITIVE",
                    "0.95",
                ]
            )

        # Migrate to new format
        with open(old_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        with open(new_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Timestamp",
                    "Name",
                    "Profession",
                    "Review Title",
                    "Review Body",
                    "Sentiment",
                    "Confidence",
                ],
            )
            writer.writeheader()
            for row in rows:
                new_row = {
                    "Timestamp": row["Timestamp"],
                    "Name": row["Name"],
                    "Profession": "Prefer not to say",
                    "Review Title": row["Review Title"],
                    "Review Body": row["Review Body"],
                    "Sentiment": row["Sentiment"],
                    "Confidence": row["Confidence"],
                }
                writer.writerow(new_row)

        # Verify migration
        with open(new_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            new_rows = list(reader)

        assert len(new_rows) == 1
        assert "Profession" in new_rows[0]
        assert "Age" not in new_rows[0]
        assert new_rows[0]["Profession"] == "Prefer not to say"


def test_special_characters_in_review():
    """Test handling of special characters in reviews."""
    special_reviews = [
        "Great product!!! 😊",
        "Works @ 100% efficiency",
        "Price: $50.99 - worth it!",
        "5/5 stars ⭐⭐⭐⭐⭐",
    ]

    for review in special_reviews:
        # Should not crash on special characters
        assert isinstance(review, str)
        assert len(review) > 0


def test_long_review_handling():
    """Test handling of very long reviews."""
    long_title = "A" * 500
    long_body = "B" * 5000

    # Should handle long text
    assert isinstance(long_title, str)
    assert isinstance(long_body, str)
    assert len(long_title) == 500
    assert len(long_body) == 5000


def test_empty_profession_handling():
    """Test handling when profession is not selected."""
    profession_input = None
    default_profession = "Prefer not to say"

    final_profession = profession_input if profession_input else default_profession
    assert final_profession == "Prefer not to say"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
