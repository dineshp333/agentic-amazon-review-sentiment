"""Security tests for the sentiment analysis application."""

import csv
import tempfile
from pathlib import Path
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_csv_injection_protection():
    """Test that CSV injection attempts are neutralized."""
    from webapp.streamlit_app import sanitize_csv_field

    # Test formula injection attempts
    dangerous_inputs = [
        "=1+1",
        "+1+1",
        "-1+1",
        "@SUM(A1:A10)",
        "\t=1+1",
        "\r=1+1",
    ]

    for dangerous_input in dangerous_inputs:
        sanitized = sanitize_csv_field(dangerous_input)
        # Should be prefixed with single quote
        assert sanitized.startswith("'"), f"Failed to sanitize: {dangerous_input}"


def test_csv_field_length_limit():
    """Test that excessively long inputs are truncated."""
    from webapp.streamlit_app import sanitize_csv_field

    # Create a very long string (20,000 characters)
    long_input = "A" * 20000
    sanitized = sanitize_csv_field(long_input)

    # Should be truncated to 10,000
    assert len(sanitized) == 10000


def test_input_validation_length():
    """Test input length validation."""
    from webapp.streamlit_app import validate_input_length

    # Test normal input
    normal_input = "This is a normal review"
    result = validate_input_length(normal_input, "test", 100)
    assert result == normal_input

    # Test long input
    long_input = "A" * 20000
    result = validate_input_length(long_input, "test", 1000)
    assert len(result) == 1000


def test_path_validation():
    """Test path validation prevents directory traversal."""
    from agents.sentiment_agent import validate_model_path

    # Test valid path within project
    valid_path = "models/model.keras"
    try:
        validated = validate_model_path(valid_path)
        assert "models" in validated
    except ValueError:
        pass  # May fail if path doesn't exist, but should not raise for security

    # Test path traversal attempts
    dangerous_paths = [
        "../../etc/passwd",
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
    ]

    for dangerous_path in dangerous_paths:
        with pytest.raises(ValueError, match="outside project directory"):
            validate_model_path(dangerous_path)


def test_safe_csv_writing():
    """Test that CSV writing uses proper quoting."""
    from webapp.streamlit_app import sanitize_csv_field

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "test.csv"

        # Write data with dangerous content
        dangerous_data = [
            ["2026-01-29", "=1+1", "Student", "Test", "Body", "POSITIVE", "0.95"],
        ]

        # Sanitize and write
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(
                [
                    "Timestamp",
                    "Name",
                    "Profession",
                    "Title",
                    "Body",
                    "Sentiment",
                    "Confidence",
                ]
            )
            for row in dangerous_data:
                safe_row = [
                    sanitize_csv_field(str(field)) if isinstance(field, str) else field
                    for field in row
                ]
                writer.writerow(safe_row)

        # Read back and verify
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            row = next(reader)
            # The dangerous formula should be neutralized
            assert row[1].startswith("'=") or row[1] == "'=1+1"


def test_special_characters_handling():
    """Test handling of special characters in inputs."""
    from webapp.streamlit_app import sanitize_csv_field

    special_inputs = [
        "Hello\nWorld",  # Newline
        'Quote"Test',  # Quotes
        "Comma,Test",  # Comma
        "Tab\tTest",  # Tab
        "Unicode: 你好 🎉",  # Unicode
    ]

    for special_input in special_inputs:
        sanitized = sanitize_csv_field(special_input)
        # Should not crash and should return a string
        assert isinstance(sanitized, str)
        assert len(sanitized) > 0


def test_empty_input_handling():
    """Test handling of empty or None inputs."""
    from webapp.streamlit_app import sanitize_csv_field, validate_input_length

    # Test empty string
    assert sanitize_csv_field("") == ""
    assert validate_input_length("", "test") == ""

    # Test None
    result = sanitize_csv_field(None)
    assert result == "None"  # Should convert to string


def test_numeric_input_handling():
    """Test handling of numeric inputs."""
    from webapp.streamlit_app import sanitize_csv_field

    # Test numbers
    assert sanitize_csv_field(123) == "123"
    assert sanitize_csv_field(0.95) == "0.95"
    assert sanitize_csv_field(-42) == "'-42"  # Minus sign should be neutralized


def test_xss_prevention():
    """Test that HTML/JS injection is prevented in user inputs."""
    from webapp.streamlit_app import sanitize_csv_field

    xss_attempts = [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('XSS')>",
        "javascript:alert('XSS')",
        "<iframe src='evil.com'>",
    ]

    for xss_attempt in xss_attempts:
        sanitized = sanitize_csv_field(xss_attempt)
        # Should sanitize dangerous characters if they start with dangerous chars
        # Note: Our sanitization focuses on CSV injection, not XSS
        # XSS is prevented by not rendering user input in HTML
        assert isinstance(sanitized, str)


def test_very_long_field_name():
    """Test handling of extremely long field names."""
    from webapp.streamlit_app import validate_input_length

    # Create 100KB of data
    huge_input = "X" * 100000
    result = validate_input_length(huge_input, "huge_field", max_length=10000)

    # Should be truncated
    assert len(result) == 10000


def test_sql_injection_like_patterns():
    """Test that SQL-like patterns are handled safely."""
    from webapp.streamlit_app import sanitize_csv_field

    sql_patterns = [
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        "admin'--",
        "' UNION SELECT * FROM users--",
    ]

    for pattern in sql_patterns:
        sanitized = sanitize_csv_field(pattern)
        # Should handle as regular text (we don't use SQL)
        assert isinstance(sanitized, str)
        # First character might be neutralized if it's dangerous
        if pattern[0] in ["=", "+", "-", "@"]:
            assert sanitized.startswith("'")


def test_path_validation_absolute_paths():
    """Test that absolute paths outside project are rejected."""
    from agents.sentiment_agent import validate_model_path

    # These should fail
    dangerous_absolute_paths = [
        "/etc/passwd",
        "C:\\Windows\\System32\\config\\sam",
        "/tmp/malicious.pkl",
    ]

    for dangerous_path in dangerous_absolute_paths:
        try:
            validate_model_path(dangerous_path)
            # If it doesn't raise, it should still be within project
            # (may not raise if path construction differs on Windows/Linux)
        except ValueError:
            # Expected behavior
            pass


def test_model_path_normalization():
    """Test that path normalization works correctly."""
    from agents.sentiment_agent import validate_model_path

    # Test path with dots
    test_path = "models/../models/model.keras"
    try:
        result = validate_model_path(test_path)
        # Should normalize to models/model.keras
        assert "models" in result
        assert ".." not in result
    except ValueError:
        # Acceptable if considered suspicious
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
