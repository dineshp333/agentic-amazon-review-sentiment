"""Utility functions for cleaning, tokenizing, and preprocessing review data."""

import re
from typing import List, Tuple
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4", quiet=True)


def clean_text(
    text: str, remove_punctuation: bool = True, lowercase: bool = True
) -> str:
    """
    Clean text by removing URLs, emails, special characters, and normalizing whitespace.

    Args:
        text (str): Raw text to clean.
        remove_punctuation (bool): Whether to remove punctuation. Defaults to True.
        lowercase (bool): Whether to convert to lowercase. Defaults to True.

    Returns:
        str: Cleaned text.
    """
    if not text or not isinstance(text, str):
        return ""

    # Lowercase
    if lowercase:
        text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove punctuation and special characters
    if remove_punctuation:
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words.

    Args:
        text (str): Text to tokenize.

    Returns:
        List[str]: List of tokens.
    """
    tokens = word_tokenize(text)
    return tokens


def remove_stopwords(tokens: List[str], language: str = "english") -> List[str]:
    """
    Remove stopwords from token list.

    Args:
        tokens (List[str]): List of tokens.
        language (str): Language for stopwords. Defaults to 'english'.

    Returns:
        List[str]: Tokens with stopwords removed.
    """
    stop_words = set(stopwords.words(language))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return filtered_tokens


def stem_tokens(tokens: List[str]) -> List[str]:
    """
    Apply Porter stemming to tokens.

    Args:
        tokens (List[str]): List of tokens.

    Returns:
        List[str]: Stemmed tokens.
    """
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(token) for token in tokens]
    return stemmed


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Apply lemmatization to tokens.

    Args:
        tokens (List[str]): List of tokens.

    Returns:
        List[str]: Lemmatized tokens.
    """
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized


def preprocess_text(
    text: str,
    remove_stopwords_flag: bool = True,
    use_stemming: bool = False,
    use_lemmatization: bool = True,
) -> str:
    """
    Full preprocessing pipeline: clean, tokenize, remove stopwords, and optionally stem/lemmatize.

    Args:
        text (str): Raw text to preprocess.
        remove_stopwords_flag (bool): Whether to remove stopwords. Defaults to True.
        use_stemming (bool): Whether to apply stemming. Defaults to False.
        use_lemmatization (bool): Whether to apply lemmatization. Defaults to True.

    Returns:
        str: Fully preprocessed text.
    """
    # Clean text
    text = clean_text(text)

    # Tokenize
    tokens = tokenize(text)

    # Remove stopwords
    if remove_stopwords_flag:
        tokens = remove_stopwords(tokens)

    # Apply stemming or lemmatization
    if use_stemming:
        tokens = stem_tokens(tokens)
    elif use_lemmatization:
        tokens = lemmatize_tokens(tokens)

    # Join back to string
    return " ".join(tokens)


def preprocess_review(
    title: str,
    body: str,
    remove_stopwords_flag: bool = True,
    use_stemming: bool = False,
    use_lemmatization: bool = True,
) -> Tuple[str, str]:
    """
    Preprocess both title and body of a review.

    Args:
        title (str): Review title.
        body (str): Review body.
        remove_stopwords_flag (bool): Whether to remove stopwords. Defaults to True.
        use_stemming (bool): Whether to apply stemming. Defaults to False.
        use_lemmatization (bool): Whether to apply lemmatization. Defaults to True.

    Returns:
        Tuple[str, str]: (preprocessed_title, preprocessed_body).
    """
    preprocessed_title = preprocess_text(
        title, remove_stopwords_flag, use_stemming, use_lemmatization
    )
    preprocessed_body = preprocess_text(
        body, remove_stopwords_flag, use_stemming, use_lemmatization
    )
    return preprocessed_title, preprocessed_body


def preprocess_batch(
    titles: List[str],
    bodies: List[str],
    remove_stopwords_flag: bool = True,
    use_stemming: bool = False,
    use_lemmatization: bool = True,
) -> List[Tuple[str, str]]:
    """
    Preprocess a batch of reviews.

    Args:
        titles (List[str]): List of review titles.
        bodies (List[str]): List of review bodies.
        remove_stopwords_flag (bool): Whether to remove stopwords. Defaults to True.
        use_stemming (bool): Whether to apply stemming. Defaults to False.
        use_lemmatization (bool): Whether to apply lemmatization. Defaults to True.

    Returns:
        List[Tuple[str, str]]: List of (preprocessed_title, preprocessed_body) tuples.

    Raises:
        ValueError: If titles and bodies have different lengths.
    """
    if len(titles) != len(bodies):
        raise ValueError("titles and bodies must have the same length")

    results = [
        preprocess_review(
            title, body, remove_stopwords_flag, use_stemming, use_lemmatization
        )
        for title, body in zip(titles, bodies)
    ]
    return results
