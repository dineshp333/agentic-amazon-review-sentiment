"""
Streamlit web interface for Agentic Amazon Review Sentiment Analysis.

Run with: streamlit run streamlit_app.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Download NLTK data before importing anything else
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

import streamlit as st
from scripts.inference import run_inference
from scripts.evaluate import print_evaluation_summary
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1000px;
        margin: 0 auto;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
        font-size: 24px;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
        font-size: 24px;
    }
    .confidence-bar {
        padding: 10px;
        border-radius: 5px;
        background-color: #e9ecef;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and description
st.title("📊 Amazon Review Sentiment Analyzer")
st.markdown(
    """
    This tool uses AI agents to analyze Amazon product reviews and predict sentiment.
    Enter a review title and body to get started!
    """
)

# Sidebar with instructions
with st.sidebar:
    st.header("📝 Instructions")
    st.markdown(
        """
        ### How to use:
        1. **Enter Review Title**: A short headline for the review
        2. **Enter Review Body**: The full review text
        3. **Click "Analyze Review"**: The AI will process your input
        4. **View Results**: See sentiment prediction and confidence score
        
        ### Sentiment Labels:
        - **Positive** 😊: Review expresses satisfaction
        - **Negative** 😞: Review expresses dissatisfaction
        
        ### Confidence Score:
        - Higher score = more confident prediction
        - Range: 0.0 to 1.0
        - Scores above 0.7 are highly reliable
        """
    )

    st.divider()

    st.header("⚙️ Settings")

    # Preprocessing options
    st.subheader("Text Processing")
    remove_stopwords = st.checkbox(
        "Remove stopwords (common words like 'the', 'a', 'is')",
        value=True,
        help="Removing stopwords reduces noise in the text",
    )

    use_stemming = st.checkbox(
        "Use stemming",
        value=False,
        help="Stems words to their root form (e.g., 'running' → 'run')",
    )

    use_lemmatization = st.checkbox(
        "Use lemmatization",
        value=True,
        help="Converts words to their base form (e.g., 'better' → 'good')",
    )

# Main content area
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📥 Review Input")

    # Review input form
    with st.form("review_form", border=True):
        title = st.text_input(
            "Review Title",
            placeholder="e.g., Great quality product",
            help="A short headline for the review",
        )

        body = st.text_area(
            "Review Body",
            placeholder="e.g., I really enjoyed this product. It arrived quickly and works as advertised...",
            height=150,
            help="The full review text",
        )

        # Submit button
        submitted = st.form_submit_button(
            "🔍 Analyze Review", use_container_width=True, type="primary"
        )

# Process form submission
if submitted:
    # Validate inputs
    if not title.strip() or not body.strip():
        st.error("❌ Please enter both a title and body for the review.")
    else:
        with col2:
            st.subheader("📊 Analysis Results")

            try:
                with st.spinner("🤖 Running sentiment analysis..."):
                    # Run inference with selected options
                    result = run_inference(
                        title=title,
                        body=body,
                        remove_stopwords=remove_stopwords,
                        use_stemming=use_stemming,
                        use_lemmatization=use_lemmatization,
                    )

                # Display results in a nice layout
                # Sentiment prediction
                sentiment = result.get("label", "unknown").upper()
                confidence = result.get("score", 0.0)

                # Color-coded sentiment display
                if sentiment == "POSITIVE":
                    st.markdown(
                        f'<div class="sentiment-positive">✅ {sentiment}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="sentiment-negative">❌ {sentiment}</div>',
                        unsafe_allow_html=True,
                    )

                # Confidence score with progress bar
                st.markdown("**Confidence Score:**")
                st.progress(min(confidence, 1.0), text=f"{confidence:.1%}")

                # Additional details in expandable section
                with st.expander("📋 Detailed Information"):
                    col_detail1, col_detail2 = st.columns(2)

                    with col_detail1:
                        st.metric(
                            "Prediction Confidence",
                            f"{confidence:.4f}",
                            help="Higher values indicate stronger confidence",
                        )

                    with col_detail2:
                        st.metric(
                            "Prediction Type",
                            sentiment,
                        )

                    st.divider()

                    st.markdown("**Processed Text:**")
                    col_proc1, col_proc2 = st.columns(2)

                    with col_proc1:
                        st.markdown("*Processed Title:*")
                        preprocessed_title = result.get("preprocessed_title", "N/A")
                        st.code(preprocessed_title, language="text")

                    with col_proc2:
                        st.markdown("*Processed Body:*")
                        preprocessed_body = result.get("preprocessed_body", "N/A")
                        st.code(preprocessed_body, language="text")

                    st.markdown("**Features:**")
                    col_feat1, col_feat2 = st.columns(2)

                    with col_feat1:
                        st.metric(
                            "Title Features",
                            result.get("title_features", 0),
                            help="Number of features extracted from title",
                        )

                    with col_feat2:
                        st.metric(
                            "Body Features",
                            result.get("body_features", 0),
                            help="Number of features extracted from body",
                        )

                # Success message
                st.success("✅ Analysis complete!")

            except FileNotFoundError as e:
                st.error(
                    f"❌ Model files not found: {str(e)}\n\n"
                    "Please ensure the following files exist:\n"
                    "- `models/model.keras`\n"
                    "- `models/cv1.pkl`\n"
                    "- `models/cv2.pkl`"
                )
                logger.error(f"FileNotFoundError: {str(e)}")

            except Exception as e:
                st.error(f"❌ An error occurred during analysis: {str(e)}")
                logger.error(f"Error: {str(e)}")

else:
    # Default state - show welcome message
    with col2:
        st.info(
            "👈 Enter a review on the left and click **'Analyze Review'** to get started!"
        )

# Footer with examples
st.divider()

st.subheader("📚 Example Reviews")

col_ex1, col_ex2 = st.columns(2)

with col_ex1:
    st.markdown("**✅ Positive Example:**")
    st.code(
        """
Title: Excellent Quality!
Body: This product exceeded my expectations. 
The build quality is outstanding, it works 
perfectly, and the customer service was amazing. 
Highly recommended!
        """,
        language="text",
    )

with col_ex2:
    st.markdown("**❌ Negative Example:**")
    st.code(
        """
Title: Very Disappointing
Body: Very disappointed with this purchase. 
It broke after just one week. The quality 
is poor and customer service was unhelpful. 
Do not recommend.
        """,
        language="text",
    )

st.divider()

st.markdown(
    """
    ### 🚀 How to Run Locally
    
    1. **Install dependencies:**
       ```bash
       pip install -r requirements.txt
       ```
    
    2. **Run the Streamlit app:**
       ```bash
       streamlit run webapp/streamlit_app.py
       ```
    
    3. **Access the app:**
       Open your browser and go to `http://localhost:8501`
    
    ### 📦 Requirements
    - Python 3.11+
    - TensorFlow/Keras for model inference
    - scikit-learn for text vectorization
    - NLTK for text preprocessing
    - Streamlit for the web interface
    
    ### 🔧 Notes
    - The app requires pre-trained model files in the `models/` directory
    - First run may download NLTK data for tokenization and lemmatization
    - Batch processing is available via the inference module
    """
)

st.markdown(
    """
    <hr>
    <p style="text-align: center; color: #666;">
    🔬 Agentic Amazon Review Sentiment Analysis | Built with Streamlit & Python
    </p>
    """,
    unsafe_allow_html=True,
)
