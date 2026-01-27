"""DataAgent handles review preprocessing and validation of Amazon review data."""

import re
from typing import List, Tuple
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data (run once)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


class DataAgent:
    """
    DataAgent handles preprocessing and validation of Amazon review data.

    Responsibilities:
    - Clean review text (title + body)
    - Remove punctuation, lowercase, and stopwords
    - Return preprocessed text ready for model inference
    """

    def __init__(self, remove_stopwords: bool = True) -> None:
        """
        Initialize DataAgent.

        Args:
            remove_stopwords (bool): Whether to remove English stopwords. Defaults to True.
        """
        self.remove_stopwords = remove_stopwords
        self.stop_words = set(stopwords.words("english")) if remove_stopwords else set()

    def clean_text(self, text: str) -> str:
        """
        Clean a single text string by removing punctuation, lowercasing, and optionally removing stopwords.

        Args:
            text (str): Raw text to clean.

        Returns:
            str: Cleaned text.
        """
        if not text or not isinstance(text, str):
            return ""

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove punctuation and special characters (keep spaces)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Tokenize and remove stopwords if enabled
        if self.remove_stopwords:
            tokens = word_tokenize(text)
            tokens = [word for word in tokens if word not in self.stop_words]
            text = " ".join(tokens)

        return text

    def preprocess_review(self, title: str, body: str) -> Tuple[str, str]:
        """
        Preprocess a review by cleaning both title and body separately.

        Args:
            title (str): Review title.
            body (str): Review body text.

        Returns:
            Tuple[str, str]: (cleaned_title, cleaned_body)
        """
        cleaned_title = self.clean_text(title)
        cleaned_body = self.clean_text(body)
        return cleaned_title, cleaned_body

    def preprocess_reviews_batch(
        self, titles: List[str], bodies: List[str]
    ) -> List[Tuple[str, str]]:
        """
        Preprocess a batch of reviews.

        Args:
            titles (List[str]): List of review titles.
            bodies (List[str]): List of review bodies.

        Returns:
            List[Tuple[str, str]]: List of (cleaned_title, cleaned_body) tuples.

        Raises:
            ValueError: If titles and bodies lists have different lengths.
        """
        if len(titles) != len(bodies):
            raise ValueError("titles and bodies must have the same length")

        results = [
            self.preprocess_review(title, body) for title, body in zip(titles, bodies)
        ]
        return results

    def validate_review(self, title: str, body: str) -> bool:
        """
        Validate that a review has sufficient content after preprocessing.

        Args:
            title (str): Review title.
            body (str): Review body text.

        Returns:
            bool: True if review is valid (non-empty after cleaning), False otherwise.
        """
        cleaned_title, cleaned_body = self.preprocess_review(title, body)
        combined = f"{cleaned_title} {cleaned_body}".strip()
        return len(combined) > 0
