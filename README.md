# Agentic Amazon Review Sentiment Analysis

Minimal scaffold for an agentic workflow that scores Amazon reviews. Built for Python 3.11 and easy to extend on a laptop.

## Layout
- agents/: modular agents for data prep, sentiment, evaluation, improvement
- scripts/: thin wrappers used by CLI/web
- webapp/: Streamlit UI
- models/: placeholders for model and vectorizers
- prompts/: LLM prompt templates
- tests/: pytest smoke tests and sample data
- docs/: design notes and usage docs

## Quick start
1) Create/activate a Python 3.11 env.
2) Install deps: `pip install -r requirements.txt`.
3) Run tests: `pytest`.
4) Launch UI: `streamlit run webapp/streamlit_app.py`.

## Next steps
- Replace model/vectorizer placeholders in models/.
- Flesh out agents with real logic and load paths.
- Expand prompts and docs as you experiment.
