# 📊 Agentic Amazon Review Sentiment Analysis

A modular, production-ready Python application that uses AI agents to analyze Amazon product reviews and predict sentiment (positive/negative). Built with a clean architecture, comprehensive type hints, and designed to demonstrate core AI engineering skills.

## 🚀 Live Demo

**Try the app now:** [https://agentic-amazon-review-sentiment-2llhu4jyed2jvdduooq8yn.streamlit.app/](https://agentic-amazon-review-sentiment-2llhu4jyed2jvdduooq8yn.streamlit.app/)

Analyze Amazon reviews instantly with AI-powered sentiment detection!

## 🎯 Project Overview

This project implements an **agentic workflow** for sentiment analysis using specialized Python agents that work together:

- **DataAgent**: Cleans and preprocesses review text
- **SentimentAgent**: Predicts sentiment using a pre-trained Keras model
- **EvaluationAgent**: Calculates performance metrics and identifies low-confidence predictions
- **ImprovementAgent**: Generates recommendations for model improvements

The application includes a **Streamlit web interface** for interactive sentiment analysis and comprehensive **evaluation tools** for batch processing and reporting.

### Key Characteristics
✅ **Modular Design**: Each agent has a single responsibility  
✅ **Type Safety**: Full type hints and docstrings throughout  
✅ **Production-Ready**: Error handling, logging, and validation  
✅ **Scalable**: Supports batch processing and multiple workflows  
✅ **Laptop-Safe**: Minimal dependencies, efficient resource usage  

---

## ✨ Features

### 1. **Four Specialized Agents**

#### DataAgent
- Text cleaning (removes URLs, emails, punctuation)
- Tokenization with NLTK
- Optional stopword removal, stemming, and lemmatization
- Batch processing support
- Input validation

#### SentimentAgent
- Loads pre-trained Keras sentiment model
- Vectorizes text using scikit-learn CountVectorizers
- Returns predictions with confidence scores
- Batch inference capability

#### EvaluationAgent
- Calculates accuracy, precision, recall, F1 score
- Confusion matrix analysis
- Identifies and logs low-confidence predictions
- Production monitoring (assess without ground truth)

#### ImprovementAgent
- Analyzes evaluation metrics
- Generates structured recommendations (retrain, augment data, adjust threshold, etc.)
- Severity-based prioritization
- JSON report generation

### 2. **Utility Scripts**

| Script | Purpose |
|--------|---------|
| `scripts/preprocess.py` | Text cleaning and normalization functions |
| `scripts/inference.py` | Model loading and prediction interface |
| `scripts/evaluate.py` | Metrics calculation and CSV export |

### 3. **Streamlit Web Application**

- Interactive review input (title + body)
- Real-time sentiment prediction with confidence scores
- Configurable text preprocessing options
- Detailed results with processed text visualization
- Example reviews for testing
- Responsive, professional UI with color-coded results

### 4. **Comprehensive Testing & Documentation**

- Unit tests for all agents
- Sample review data (CSV)
- Architecture documentation
- AI engineer skill mapping
- Setup instructions

---

## 📦 Installation

### Prerequisites
- **Python 3.11+** (tested on 3.11 and 3.12)
- **pip** (Python package manager)
- **Virtual environment** (recommended)

### Step 1: Clone or Download the Project
```bash
cd "d:\AI Journey\agentic-amazon-review-sentiment"
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- **streamlit** - Web interface
- **tensorflow-cpu** - Keras sentiment model
- **scikit-learn** - Text vectorization & metrics
- **nltk** - Text preprocessing
- **pandas** - Data handling
- **joblib** - Model serialization
- **pytest** - Testing framework

### Step 4: Download NLTK Data (One-time Setup)
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## 🚀 Usage

### Option A: Launch Streamlit Web App (Recommended)

```bash
streamlit run webapp/streamlit_app.py
```

Then open your browser to: **http://localhost:8501**

**Features:**
- Input review title and body
- Adjust preprocessing options (stopword removal, stemming, lemmatization)
- View sentiment prediction with confidence score
- See processed text and feature extraction details
- Read usage instructions in the sidebar

### Option B: Use Python Scripts Directly

```python
from scripts.inference import run_inference

# Single review
result = run_inference(
    title="Great product!",
    body="Works as advertised and arrived quickly."
)

print(f"Sentiment: {result['label']}")
print(f"Confidence: {result['score']:.2%}")

# Batch predictions
from scripts.inference import run_batch_inference

titles = ["Great!", "Terrible", "Okay"]
bodies = ["Text here", "Text here", "Text here"]
results = run_batch_inference(titles, bodies)
```

### Option C: Evaluate with Ground Truth

```python
from scripts.evaluate import evaluate_predictions, generate_full_report

# Prepare predictions and true labels
predictions = [
    {"label": "positive", "score": 0.92},
    {"label": "negative", "score": 0.87},
]
ground_truth = ["positive", "negative"]

# Evaluate
results = evaluate_predictions(predictions, ground_truth)
print(f"Accuracy: {results['accuracy']:.4f}")

# Generate comprehensive report
report = generate_full_report(predictions, ground_truth)
```

### Option D: Run Unit Tests

```bash
pytest tests/test_agents.py -v
```

---

## 📁 Project Structure

```
amazon-review-agentic-ai/
│
├── agents/                              # Core AI agents
│   ├── __init__.py
│   ├── data_agent.py                   # Text preprocessing & validation
│   ├── sentiment_agent.py               # Sentiment prediction
│   ├── evaluation_agent.py              # Metrics calculation
│   └── improvement_agent.py             # Recommendations & reporting
│
├── models/                              # Pre-trained assets
│   ├── model.keras                      # Keras sentiment model (placeholder)
│   ├── cv1.pkl                          # CountVectorizer for titles
│   └── cv2.pkl                          # CountVectorizer for bodies
│
├── scripts/                             # Utility functions
│   ├── preprocess.py                    # Text cleaning pipeline
│   ├── inference.py                     # Model loading & prediction
│   └── evaluate.py                      # Evaluation & reporting
│
├── webapp/                              # Web interface
│   └── streamlit_app.py                 # Interactive Streamlit application
│
├── prompts/                             # LLM prompts (for future use)
│   ├── sentiment_reasoning.md           # Prompt for sentiment explanation
│   └── error_analysis.md                # Prompt for error analysis
│
├── tests/                               # Unit tests & samples
│   ├── test_agents.py                   # Agent functionality tests
│   └── sample_reviews.csv               # Sample review data
│
├── docs/                                # Documentation
│   ├── README.md                        # Project overview
│   ├── agent_architecture.md            # Agent design & workflows
│   ├── copilot_usage.md                 # How GitHub Copilot was used
│   └── ai_engineer_mapping.md           # Skill demonstration
│
├── requirements.txt                     # Python dependencies
├── .gitignore                           # Git ignore rules
└── LICENSE                              # Open-source license
```

---

## 🔧 Configuration & Customization

### Preprocessing Options

Control text preprocessing behavior when running inference:

```python
run_inference(
    title="Review Title",
    body="Review body...",
    remove_stopwords=True,      # Remove common words
    use_stemming=False,         # Use PorterStemmer (faster)
    use_lemmatization=True      # Use lemmatization (better quality)
)
```

### Confidence Threshold

Adjust the confidence threshold for identifying uncertain predictions:

```python
from agents.evaluation_agent import EvaluationAgent

evaluator = EvaluationAgent(confidence_threshold=0.65)
evaluator.set_confidence_threshold(0.70)  # Update dynamically
```

### Model Paths

Point to different model files if needed:

```python
from scripts.inference import initialize_model

agent = initialize_model(
    model_path="path/to/model.keras",
    title_vectorizer_path="path/to/cv1.pkl",
    body_vectorizer_path="path/to/cv2.pkl"
)
```

---

## 🎓 AI Engineer Skills Demonstrated

This project showcases essential AI engineering competencies:

### 1. **Data Preparation**
- ✅ Text cleaning and normalization
- ✅ Tokenization and lemmatization
- ✅ Feature vectorization (CountVectorizer)
- ✅ Batch processing

### 2. **Model Inference**
- ✅ Loading pre-trained models (Keras)
- ✅ Making predictions on new data
- ✅ Handling batch vs. single inference
- ✅ Error handling and validation

### 3. **Evaluation & Metrics**
- ✅ Calculating accuracy, precision, recall, F1
- ✅ Confusion matrix analysis
- ✅ Confidence score interpretation
- ✅ Low-confidence detection

### 4. **Software Engineering**
- ✅ Modular architecture (agent pattern)
- ✅ Type hints and docstrings
- ✅ Error handling and logging
- ✅ Unit testing (pytest)
- ✅ Configuration management

### 5. **Monitoring & Improvement**
- ✅ Performance tracking
- ✅ Generating actionable recommendations
- ✅ JSON report generation
- ✅ CSV export for analysis

### 6. **User Interfaces**
- ✅ Building interactive web apps (Streamlit)
- ✅ Professional UI/UX design
- ✅ User input validation
- ✅ Results visualization

### 7. **DevOps & Documentation**
- ✅ Requirements management
- ✅ Git version control (.gitignore)
- ✅ Project documentation
- ✅ README and code comments
- ✅ Reproducible setup instructions

---

## 📊 Example Workflows

### Workflow 1: Analyze Single Review (Web)
1. Open Streamlit app: `streamlit run webapp/streamlit_app.py`
2. Enter review title and body
3. Click "Analyze Review"
4. View sentiment and confidence

### Workflow 2: Batch Analysis with Evaluation
```python
import pandas as pd
from scripts.inference import run_batch_inference
from scripts.evaluate import generate_full_report

# Load reviews
df = pd.read_csv("reviews.csv")

# Run inference
predictions = run_batch_inference(
    titles=df['title'].tolist(),
    bodies=df['body'].tolist()
)

# Generate report
report = generate_full_report(
    predictions,
    ground_truth=df['label'].tolist(),
    output_dir="reports/"
)

print(f"Accuracy: {report['evaluation_metrics']['accuracy']:.4f}")
```

### Workflow 3: Monitor Production Predictions
```python
from scripts.evaluate import assess_predictions

# Get predictions from production
predictions = [...]  # Your predictions

# Assess without ground truth
stats = assess_predictions(predictions)

print(f"Average confidence: {stats['avg_confidence']:.2%}")
print(f"Low-confidence predictions: {stats['low_confidence_count']}")
```

---

## 🤝 Contributing

To extend the project:

1. **Add new preprocessing methods** in `scripts/preprocess.py`
2. **Create new agents** by following the pattern in `agents/`
3. **Add test cases** in `tests/test_agents.py`
4. **Update documentation** in `docs/`

---

## 📝 License

This project is provided as-is for educational and commercial use.  
See [LICENSE](LICENSE) for details.

---

## 🆘 Troubleshooting

### Issue: "Model files not found"
**Solution:** Ensure these files exist:
- `models/model.keras`
- `models/cv1.pkl`
- `models/cv2.pkl`

### Issue: "NLTK data not found"
**Solution:** Download NLTK data:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Issue: Port 8501 already in use
**Solution:** Use a different port:
```bash
streamlit run webapp/streamlit_app.py --server.port 8502
```

### Issue: Import errors
**Solution:** Ensure you're in the project root directory and virtual environment is activated:
```bash
cd "d:\AI Journey\agentic-amazon-review-sentiment"
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

---

## 📧 Support & Questions

For issues, questions, or suggestions, please refer to the documentation in the `docs/` folder or review the inline code comments.

---

## 🎉 Quick Reference

| Task | Command |
|------|---------|
| Install deps | `pip install -r requirements.txt` |
| Launch web app | `streamlit run webapp/streamlit_app.py` |
| Run tests | `pytest tests/` |
| Download NLTK data | `python -c "import nltk; nltk.download('all')"` |
| Check Python version | `python --version` |
| Activate venv | `venv\Scripts\activate` (Windows) |

---

**Built with ❤️ for AI Engineers | Python 3.11+ | Production-Ready**
