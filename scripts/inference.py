"""Run sentiment inference for reviews using pretrained models."""

import logging
from typing import Dict, Any, List
from agents.sentiment_agent import SentimentAgent
from scripts.preprocess import preprocess_review, preprocess_batch

logger = logging.getLogger(__name__)

# Initialize global sentiment agent
_sentiment_agent: SentimentAgent | None = None


def initialize_model(
    model_path: str = "models/model.keras",
    title_vectorizer_path: str = "models/cv1.pkl",
    body_vectorizer_path: str = "models/cv2.pkl",
) -> SentimentAgent:
    """
    Initialize and load the sentiment model and vectorizers.

    Args:
        model_path (str): Path to Keras model. Defaults to "models/model.keras".
        title_vectorizer_path (str): Path to title vectorizer. Defaults to "models/cv1.pkl".
        body_vectorizer_path (str): Path to body vectorizer. Defaults to "models/cv2.pkl".

    Returns:
        SentimentAgent: Initialized sentiment agent with loaded models.

    Raises:
        FileNotFoundError: If any model file is not found.
    """
    global _sentiment_agent
    _sentiment_agent = SentimentAgent(
        model_path=model_path,
        title_vectorizer_path=title_vectorizer_path,
        body_vectorizer_path=body_vectorizer_path,
    )
    _sentiment_agent.load()
    logger.info("Sentiment model loaded successfully")
    return _sentiment_agent


def get_model() -> SentimentAgent:
    """
    Get the global sentiment model instance, initializing if necessary.

    Returns:
        SentimentAgent: Global sentiment agent instance.
    """
    global _sentiment_agent
    if _sentiment_agent is None:
        _sentiment_agent = initialize_model()
    return _sentiment_agent


def run_inference(
    title: str,
    body: str,
    remove_stopwords: bool = True,
    use_stemming: bool = False,
    use_lemmatization: bool = True,
) -> Dict[str, Any]:
    """
    Run sentiment inference on a single review.

    Preprocesses the review text and returns sentiment prediction with confidence.

    Args:
        title (str): Review title.
        body (str): Review body.
        remove_stopwords (bool): Whether to remove stopwords. Defaults to True.
        use_stemming (bool): Whether to apply stemming. Defaults to False.
        use_lemmatization (bool): Whether to apply lemmatization. Defaults to True.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - "label" (str): Sentiment label ("positive" or "negative")
            - "score" (float): Confidence score (0-1)
            - "title_features" (int): Number of features from title
            - "body_features" (int): Number of features from body
            - "preprocessed_title" (str): Cleaned and processed title
            - "preprocessed_body" (str): Cleaned and processed body

    Raises:
        ValueError: If title or body is empty or invalid.
    """
    # Preprocess input
    preprocessed_title, preprocessed_body = preprocess_review(
        title, body, remove_stopwords, use_stemming, use_lemmatization
    )

    # Get model and run prediction
    sentiment_agent = get_model()
    prediction = sentiment_agent.predict(preprocessed_title, preprocessed_body)

    # Add preprocessing info
    return {
        **prediction,
        "preprocessed_title": preprocessed_title,
        "preprocessed_body": preprocessed_body,
    }


def run_batch_inference(
    titles: List[str],
    bodies: List[str],
    remove_stopwords: bool = True,
    use_stemming: bool = False,
    use_lemmatization: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run sentiment inference on multiple reviews in batch.

    Args:
        titles (List[str]): List of review titles.
        bodies (List[str]): List of review bodies.
        remove_stopwords (bool): Whether to remove stopwords. Defaults to True.
        use_stemming (bool): Whether to apply stemming. Defaults to False.
        use_lemmatization (bool): Whether to apply lemmatization. Defaults to True.

    Returns:
        List[Dict[str, Any]]: List of sentiment predictions.

    Raises:
        ValueError: If titles and bodies have different lengths.
    """
    if len(titles) != len(bodies):
        raise ValueError("titles and bodies must have the same length")

    predictions = []
    for title, body in zip(titles, bodies):
        try:
            pred = run_inference(
                title, body, remove_stopwords, use_stemming, use_lemmatization
            )
            predictions.append(pred)
        except Exception as e:
            logger.error(f"Error processing review: {e}")
            predictions.append({"label": None, "score": None, "error": str(e)})

    return predictions
