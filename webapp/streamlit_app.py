"""Streamlit interface for the sentiment demo."""

from __future__ import annotations
import streamlit as st
from scripts.inference import run_inference

st.set_page_config(page_title="Amazon Review Sentiment", page_icon="🛒")
st.title("Agentic Amazon Review Sentiment")

with st.form("review_form"):
    title = st.text_input("Review Title", "Great product")
    body = st.text_area("Review Body", "I loved using this item.")
    submitted = st.form_submit_button("Analyze")

if submitted:
    with st.spinner("Running agents..."):
        result = run_inference(title, body)
    st.success(f"Predicted: {result['label']} (score={result['score']:.2f})")
    st.caption(f"Meta: {result.get('meta', {})}")
else:
    st.info("Submit a review to get sentiment.")
