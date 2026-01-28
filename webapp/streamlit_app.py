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
import csv
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create data folder for storing user responses
DATA_FOLDER = Path(__file__).parent.parent / "user_data"
DATA_FOLDER.mkdir(exist_ok=True)
DATA_FILE = DATA_FOLDER / "analysis_results.csv"

# Initialize CSV file if it doesn't exist
if not DATA_FILE.exists():
    with open(DATA_FILE, "w", newline="", encoding="utf-8") as f:
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
    /* Main background gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Content container */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, #FF9A56 0%, #FF6B95 50%, #A855F7 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        animation: fadeIn 1s ease-in;
    }
    
    .hero-title {
        color: white;
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.3rem;
        margin-bottom: 1rem;
        font-weight: 400;
    }
    
    .hero-description {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        max-width: 800px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    /* Sentiment results */
    .sentiment-positive {
        color: #10B981;
        font-weight: 900;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(16, 185, 129, 0.2);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .sentiment-negative {
        color: #EF4444;
        font-weight: 900;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(239, 68, 68, 0.2);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        color: white;
        animation: slideIn 0.5s ease-out;
    }
    
    /* Stats badge */
    .stats-badge {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.5rem 1rem;
        border-radius: 50px;
        display: inline-block;
        margin: 0.5rem;
        backdrop-filter: blur(10px);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Full-screen glowing overlay animations */
    @keyframes glowGreenOverlay {
        0%, 100% { opacity: 0; }
        50% { opacity: 0.4; }
    }
    
    @keyframes glowRedOverlay {
        0%, 100% { opacity: 0; }
        50% { opacity: 0.4; }
    }
    
    .fullscreen-glow-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        pointer-events: none;
        z-index: 9999;
        transition: opacity 0.3s;
    }
    
    .glow-positive-overlay {
        background: radial-gradient(circle, #10B981 0%, #059669 100%);
        animation: glowGreenOverlay 1s ease-in-out 3;
    }
    
    .glow-negative-overlay {
        background: radial-gradient(circle, #EF4444 0%, #DC2626 100%);
        animation: glowRedOverlay 1s ease-in-out 3;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #E5E7EB;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Modal overlay */
    .modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.75);
        z-index: 999;
        pointer-events: none;
    }
        height: 100%;
        background-color: rgba(0, 0, 0, 0.75);
        z-index: 1000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for user info modal
if "user_info_submitted" not in st.session_state:
    st.session_state.user_info_submitted = False
if "modal_user_name" not in st.session_state:
    st.session_state.modal_user_name = ""
if "modal_user_age" not in st.session_state:
    st.session_state.modal_user_age = 25

# User Info Modal Popup - Show if not submitted
if not st.session_state.user_info_submitted:
    # Overlay background
    st.markdown(
        """
        <div class="modal-overlay"></div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
    
    # Center the modal with proper spacing
    col_left, col_modal, col_right = st.columns([0.5, 1.5, 0.5], gap="large")
    
    with col_modal:
        st.markdown(
            """
            <div style="background: white; border-radius: 25px; padding: 3rem; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);">
                <div style="text-align: center;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">👤</div>
                    <h1 style="color: #667eea; margin: 0.5rem 0; font-size: 2rem;">Welcome!</h1>
                    <p style="color: #666; font-size: 1.1rem; margin: 1rem 0 2rem 0;">Please tell us about yourself before analyzing reviews</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # User info form
        modal_name = st.text_input(
            "Your Name",
            placeholder="Enter your full name",
            key="modal_name_input",
        )
        
        modal_age = st.number_input(
            "Your Age",
            min_value=13,
            max_value=120,
            value=25,
            key="modal_age_input",
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([0.5, 1, 0.5])
        with col_btn2:
            if st.button("Continue to Analysis", use_container_width=True, key="modal_submit_btn", type="primary"):
                if modal_name and modal_name.strip() != "":
                    st.session_state.user_info_submitted = True
                    st.session_state.modal_user_name = modal_name.strip()
                    st.session_state.modal_user_age = modal_age
                    st.rerun()
                else:
                    st.error("Please enter your name to continue")
    
    st.stop()

# Get user info from session state
user_name = st.session_state.modal_user_name
user_age = st.session_state.modal_user_age

# Hero Section
st.markdown(
    """
    <div class="hero-section">
        <div class="hero-title">🌟 Amazon Review Sentiment Analyzer</div>
        <div class="hero-subtitle">AI-Powered Review Analysis in Seconds</div>
        <div class="hero-description">
            Harness the power of advanced AI agents to instantly analyze customer reviews and predict sentiment with confidence. 
            Built with state-of-the-art machine learning and natural language processing.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Feature highlights
st.markdown("### ✨ Why Use This Tool?")
feat_col1, feat_col2, feat_col3, feat_col4 = st.columns(4)

with feat_col1:
    st.markdown(
        """
        <div class="feature-card">
            <div style="font-size: 2.5rem; text-align: center;">🤖</div>
            <h4 style="text-align: center; color: #667eea;">AI Agents</h4>
            <p style="text-align: center; font-size: 0.9rem;">Specialized agents for cleaning, analyzing, and evaluating reviews</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with feat_col2:
    st.markdown(
        """
        <div class="feature-card">
            <div style="font-size: 2.5rem; text-align: center;">⚡</div>
            <h4 style="text-align: center; color: #667eea;">Lightning Fast</h4>
            <p style="text-align: center; font-size: 0.9rem;">Get instant sentiment predictions in under 2 seconds</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with feat_col3:
    st.markdown(
        """
        <div class="feature-card">
            <div style="font-size: 2.5rem; text-align: center;">🎯</div>
            <h4 style="text-align: center; color: #667eea;">Highly Accurate</h4>
            <p style="text-align: center; font-size: 0.9rem;">Trained on thousands of real Amazon reviews</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with feat_col4:
    st.markdown(
        """
        <div class="feature-card">
            <div style="font-size: 2.5rem; text-align: center;">🔧</div>
            <h4 style="text-align: center; color: #667eea;">Customizable</h4>
            <p style="text-align: center; font-size: 0.9rem;">Adjust preprocessing options to fit your needs</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Sidebar with instructions
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="color: #667eea;">📊</h1>
            <h3 style="color: #667eea; margin-top: -1rem;">Sentiment Analyzer</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### 👤 User Information")
    st.markdown(f"**Name:** {user_name}")
    st.markdown(f"**Age:** {user_age}")
    
    if st.button("🔄 Change User", help="Change name and age"):
        st.session_state.user_info_submitted = False
        st.rerun()

    st.divider()

    st.markdown("### 🚀 Quick Start")
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); 
                    padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <strong>Step 1:</strong> Enter your review details<br>
            <strong>Step 2:</strong> Click "Analyze Review"<br>
            <strong>Step 3:</strong> Get instant results!
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📖 Understanding Results")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**😊 Positive**")
        st.caption("Customer satisfaction")
    with col_s2:
        st.markdown("**😞 Negative**")
        st.caption("Dissatisfaction")

    st.markdown(
        """
        **Confidence Score:**  
        • 0.9-1.0: Very confident  
        • 0.7-0.9: Confident  
        • 0.5-0.7: Moderate  
        • Below 0.5: Low confidence
        """
    )

    st.divider()

    st.markdown("### ⚙️ Advanced Settings")

    # Preprocessing options
    remove_stopwords = st.checkbox(
        "🔤 Remove stopwords",
        value=True,
        help="Remove common words like 'the', 'a', 'is'",
    )

    use_stemming = st.checkbox(
        "✂️ Use stemming",
        value=False,
        help="Convert words to root form (running → run)",
    )

    use_lemmatization = st.checkbox(
        "📝 Use lemmatization",
        value=True,
        help="Convert words to base form (better → good)",
    )

    st.divider()

    st.markdown(
        """
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; 
                    border-radius: 10px; margin-top: 2rem;">
            <p style="font-size: 0.85rem; color: #666; margin: 0;">
                💡 <strong>Tip:</strong> Try different settings to see how preprocessing affects results!
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Function to save analysis results
def save_analysis_result(name, age, title, body, sentiment, confidence):
    """Save analysis result to CSV file"""
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(DATA_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, name, age, title, body, sentiment, confidence])
        return True
    except Exception as e:
        logger.error(f"Error saving analysis result: {e}")
        return False

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
                with st.spinner("🤖 Running AI sentiment analysis..."):
                    # Run inference with selected options
                    result = run_inference(
                        title=title,
                        body=body,
                        remove_stopwords=remove_stopwords,
                        use_stemming=use_stemming,
                        use_lemmatization=use_lemmatization,
                    )

                # Display results in enhanced layout
                sentiment = result.get("label", "unknown").upper()
                confidence = result.get("score", 0.0)

                # Create result card with gradient background
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div style="text-align: center; margin-bottom: 1.5rem;">
                            <h2 style="color: white; margin: 0;">Sentiment Analysis Complete</h2>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Color-coded sentiment display with large icons
                if sentiment == "POSITIVE":
                    # Full-screen glow effect - 3 blinks then disappear
                    st.markdown(
                        """
                        <script>
                            (function() {
                                var body = window.parent.document.body;
                                var existingOverlay = window.parent.document.getElementById('sentiment-glow-overlay');
                                
                                if (existingOverlay) {
                                    existingOverlay.remove();
                                }
                                
                                var overlay = window.parent.document.createElement('div');
                                overlay.id = 'sentiment-glow-overlay';
                                overlay.style.cssText = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: radial-gradient(circle, #10B981 0%, #059669 100%); z-index: 9999; pointer-events: none; opacity: 0;';
                                body.appendChild(overlay);
                                
                                var blinkCount = 0;
                                var blinkInterval = setInterval(function() {
                                    if (blinkCount < 6) {
                                        overlay.style.opacity = blinkCount % 2 === 0 ? '0.4' : '0';
                                        blinkCount++;
                                    } else {
                                        clearInterval(blinkInterval);
                                        overlay.remove();
                                    }
                                }, 500);
                            })();
                        </script>
                        <div style="background: linear-gradient(135deg, #10B981 0%, #059669 100%); 
                                    padding: 2rem; border-radius: 15px; text-align: center; margin: 1rem 0;">
                            <div style="font-size: 4rem;">😊</div>
                            <h1 style="color: white; font-size: 3rem; margin: 0.5rem 0;">POSITIVE</h1>
                            <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem;">This review expresses satisfaction</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    # Full-screen glow effect - 3 blinks then disappear
                    st.markdown(
                        """
                        <script>
                            (function() {
                                var body = window.parent.document.body;
                                var existingOverlay = window.parent.document.getElementById('sentiment-glow-overlay');
                                
                                if (existingOverlay) {
                                    existingOverlay.remove();
                                }
                                
                                var overlay = window.parent.document.createElement('div');
                                overlay.id = 'sentiment-glow-overlay';
                                overlay.style.cssText = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: radial-gradient(circle, #EF4444 0%, #DC2626 100%); z-index: 9999; pointer-events: none; opacity: 0;';
                                body.appendChild(overlay);
                                
                                var blinkCount = 0;
                                var blinkInterval = setInterval(function() {
                                    if (blinkCount < 6) {
                                        overlay.style.opacity = blinkCount % 2 === 0 ? '0.4' : '0';
                                        blinkCount++;
                                    } else {
                                        clearInterval(blinkInterval);
                                        overlay.remove();
                                    }
                                }, 500);
                            })();
                        </script>
                        <div style="background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); 
                                    padding: 2rem; border-radius: 15px; text-align: center; margin: 1rem 0;">
                            <div style="font-size: 4rem;">😞</div>
                            <h1 style="color: white; font-size: 3rem; margin: 0.5rem 0;">NEGATIVE</h1>
                            <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem;">This review expresses dissatisfaction</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Confidence score with enhanced progress bar
                st.markdown("### 🎯 Confidence Level")
                confidence_percent = confidence * 100

                # Determine confidence level
                if confidence >= 0.9:
                    conf_label = "🔥 Very High"
                    conf_color = "#10B981"
                elif confidence >= 0.7:
                    conf_label = "✅ High"
                    conf_color = "#3B82F6"
                elif confidence >= 0.5:
                    conf_label = "⚠️ Moderate"
                    conf_color = "#F59E0B"
                else:
                    conf_label = "❓ Low"
                    conf_color = "#EF4444"

                st.markdown(
                    f"""
                    <div style="background: white; padding: 1.5rem; border-radius: 10px; 
                                box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 1rem 0;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 1.2rem; font-weight: 600;">{conf_label}</span>
                            <span style="font-size: 1.5rem; font-weight: 700; color: {conf_color};">{confidence_percent:.1f}%</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(min(confidence, 1.0))

                # Additional details in expandable section
                with st.expander("📋 View Detailed Analysis", expanded=False):
                    st.markdown("#### Preprocessing Details")

                    col_detail1, col_detail2 = st.columns(2)

                    with col_detail1:
                        st.markdown("**Processed Title:**")
                        preprocessed_title = result.get("preprocessed_title", "N/A")
                        st.code(preprocessed_title, language="text")
                        st.metric(
                            "Title Features Extracted",
                            result.get("title_features", 0),
                        )

                    with col_detail2:
                        st.markdown("**Processed Body:**")
                        preprocessed_body = result.get("preprocessed_body", "N/A")
                        st.code(preprocessed_body, language="text")
                        st.metric(
                            "Body Features Extracted",
                            result.get("body_features", 0),
                        )

                    st.divider()

                    st.markdown("#### Model Information")
                    st.info(
                        f"""
                        **Prediction Confidence:** {confidence:.4f}  
                        **Model Type:** Neural Network (Keras)  
                        **Total Features:** {result.get("title_features", 0) + result.get("body_features", 0)}  
                        **Processing Options:**  
                        • Stopwords Removed: {"Yes" if remove_stopwords else "No"}  
                        • Stemming: {"Yes" if use_stemming else "No"}  
                        • Lemmatization: {"Yes" if use_lemmatization else "No"}
                        """
                    )

                # Success message with action
                st.success("✅ Analysis completed successfully!")
                
                # Save the analysis result to CSV
                save_success = save_analysis_result(
                    user_name.strip(),
                    user_age,
                    title,
                    body,
                    sentiment,
                    confidence
                )
                
                if save_success:
                    st.success(f"📁 Data saved successfully! Thank you for your feedback, {user_name}!")
                else:
                    st.warning("⚠️ Analysis complete but failed to save data.")
                
                st.markdown(
                    """
                    <div style="text-align: center; margin-top: 1.5rem; padding: 1rem; background: #fff3cd; border-radius: 10px;">
                        <p style="color: #856404; margin: 0;">⏱️ <strong>Page will reset in 10 seconds...</strong></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                # Auto-reset after 10 seconds
                import time
                time.sleep(10)
                st.rerun()

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
    # Default state - show welcome message with attractive design
    with col2:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 3rem 2rem; border-radius: 20px; text-align: center; 
                        color: white; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🚀</div>
                <h2 style="margin: 1rem 0;">Ready to Analyze!</h2>
                <p style="font-size: 1.1rem; opacity: 0.95; margin-bottom: 1.5rem;">
                    Enter your review details on the left and click <strong>"Analyze Review"</strong> 
                    to get instant AI-powered sentiment predictions.
                </p>
                <div style="background: rgba(255,255,255,0.2); padding: 1rem; 
                            border-radius: 10px; backdrop-filter: blur(10px);">
                    <p style="margin: 0; font-size: 0.95rem;">
                        💡 <strong>Pro Tip:</strong> Try the example reviews below to see the analyzer in action!
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Footer with enhanced examples
st.divider()

st.markdown(
    """
    <div style="text-align: center; margin: 2rem 0;">
        <h2 style="color: #667eea;">📚 Try These Example Reviews</h2>
        <p style="color: #666; font-size: 1.1rem;">Copy and paste these examples to test the analyzer</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_ex1, col_ex2 = st.columns(2)

with col_ex1:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #10B981 0%, #059669 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem; 
                    box-shadow: 0 5px 15px rgba(16, 185, 129, 0.3);">
            <h3 style="color: white; margin-top: 0;">😊 Positive Example</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code(
        """Title: Excellent Quality!

Body: This product exceeded my expectations. 
The build quality is outstanding, it works 
perfectly, and the customer service was 
amazing. Delivery was fast and packaging 
was secure. Highly recommended for anyone 
looking for a reliable product!""",
        language="text",
    )

with col_ex2:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem; 
                    box-shadow: 0 5px 15px rgba(239, 68, 68, 0.3);">
            <h3 style="color: white; margin-top: 0;">😞 Negative Example</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code(
        """Title: Very Disappointing

Body: Very disappointed with this purchase. 
It broke after just one week of normal use. 
The quality is poor and feels cheaply made. 
Customer service was unhelpful when I tried 
to get a refund. Save your money and look 
elsewhere. Do not recommend.""",
        language="text",
    )

st.divider()

# Enhanced footer with better styling
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 20px; margin: 2rem 0; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.2); color: white;">
        <h3 style="text-align: center; margin-top: 0;">🚀 About This Project</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                    gap: 1.5rem; margin: 1.5rem 0;">
            <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; 
                        border-radius: 10px; backdrop-filter: blur(10px);">
                <h4>🤖 AI Agents</h4>
                <p style="font-size: 0.9rem; opacity: 0.95;">
                    Specialized agents for data preprocessing, sentiment prediction, 
                    evaluation, and continuous improvement recommendations.
                </p>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; 
                        border-radius: 10px; backdrop-filter: blur(10px);">
                <h4>🧠 Machine Learning</h4>
                <p style="font-size: 0.9rem; opacity: 0.95;">
                    Powered by TensorFlow/Keras neural networks trained on real Amazon reviews 
                    with advanced NLP preprocessing.
                </p>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; 
                        border-radius: 10px; backdrop-filter: blur(10px);">
                <h4>📊 Features</h4>
                <p style="font-size: 0.9rem; opacity: 0.95;">
                    Real-time analysis, confidence scoring, customizable preprocessing, 
                    and detailed explanations for every prediction.
                </p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="text-align: center; padding: 2rem; background: #f8f9fa; 
                border-radius: 15px; margin: 1rem 0;">
        <p style="color: #666; font-size: 0.95rem; margin: 0;">
            🔬 <strong>Agentic Amazon Review Sentiment Analysis</strong><br>
            Built with ❤️ using Streamlit, Python, TensorFlow, and NLTK<br>
            <span style="font-size: 0.85rem; opacity: 0.8;">
                © 2026 | AI-Powered Sentiment Analysis Platform
            </span>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
