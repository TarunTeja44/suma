"""
AI Study Assistant (Updated for OpenAI Responses API with gpt-5-nano)
"""

import os
import json
import io
import requests
import streamlit as st
from typing import Optional

# API constants (replace with your actual key if you want to hardcode)
OPENAI_API_KEY = "YOUR_API_KEY_HERE"  
API_URL = "https://api.openai.com/v1/responses"
MODEL_NAME = "gpt-5-nano"

# PDF extraction libraries
try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except:
    _HAS_PDFPLUMBER = False

try:
    from PyPDF2 import PdfReader
    _HAS_PYPDF2 = True
except:
    _HAS_PYPDF2 = False


st.set_page_config(page_title="AI Study Assistant", page_icon=":mortar_board:", layout="wide")
st.title("AI Study Assistant (gpt-5-nano)")
st.markdown("Summarize text, create flashcards, Q&A, and mind-maps using the **Responses API** with `gpt-5-nano`.")

# Sidebar controls
st.sidebar.header("Settings")
num_bullets = st.sidebar.slider("Number of summary bullets", 3, 10, 5)
num_flashcards = st.sidebar.slider("Number of flashcards", 3, 30, 10)
num_short_qa = st.sidebar.slider("Number of short Q&A", 3, 20, 8)


# --- PDF Extraction ---
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    text_pages = []
    if _HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for p in pdf.pages:
                    text_pages.append(p.extract_text() or "")
            return "\n\n".join(text_pages).strip()
        except:
            pass
    if _HAS_PYPDF2:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for page in reader.pages:
                text_pages.append(page.extract_text() or "")
            return "\n\n".join(text_pages).strip()
        except:
            pass
    return ""


# --- OpenAI helper ---
def call_openai(prompt: str, max_tokens: int = 800, temperature: float = 0.3) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    payload = {
        "model": MODEL_NAME,
        "input": prompt,
        "max_output_tokens": max_tokens,
        "temperature": temperature
    }

    try:
        resp = requests.post(API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        # Response format: choices[0].message.content in Chat API,
        # here it's in 'output_text' (for Responses API).
        return data.get("output_text", "").strip()
    except Exception as e:
        return f"[API call failed] {e}"


# --- Input section ---
st.subheader("Input")
col1, col2 = st.columns([1, 2])
with col1:
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    paste_text = st.text_area("Or paste text", height=200)

if paste_text.strip():
    input_text = paste_text.strip()
elif uploaded_file:
    input_text = extract_text_from_pdf_bytes(uploaded_file.read())
else:
    input_text = ""

if not input_text:
    st.warning("Upload a PDF or paste text to continue.")
    st.stop()

st.subheader("Preview of text")
with st.expander("Show first 3000 chars"):
    st.text(input_text[:3000] + ("..." if len(input_text) > 3000 else ""))


# --- Generation ---
if st.button("Generate summary, flashcards, Q&A & mind-map"):
    with st.spinner("Talking to gpt-5-nano..."):
        context = input_text[:60000]

        # 1. Summary
        summary_prompt = f"Summarize the following in {num_bullets} concise bullet points:\n\n{context}"
        summary_raw = call_openai(summary_prompt, max_tokens=400)

        # 2. Flashcards
        flashcards_prompt = (
            f"Create {num_flashcards} flashcards in JSON format "
            f"with 'question' and 'answer' from this content:\n\n{context}"
        )
        flashcards_raw = call_openai(flashcards_prompt, max_tokens=800)

        # 3. Short Q&A
        qa_prompt = f"Generate {num_short_qa} short Q&A pairs (Q: ... A: ...):\n\n{context}"
        short_qa_raw = call_openai(qa_prompt, max_tokens=500)

        # 4. Mind-map
        mindmap_prompt = (
            "Create a mind-map in Graphviz DOT format for this content. "
            "Output ONLY DOT code.\n\n" + context
        )
        mindmap_raw = call_openai(mindmap_prompt, max_tokens=600)

    st.success("✅ Done!")

    # --- Display ---
    st.header("Summary")
    st.markdown(summary_raw)

    st.header("Flashcards")
    try:
        parsed_flashcards = json.loads(flashcards_raw)
        st.json(parsed_flashcards)
    except:
        st.text_area("Raw flashcards", flashcards_raw, height=200)

    st.header("Q&A")
    st.text_area("Quick Q&A", short_qa_raw, height=200)

    st.header("Mind-map")
    st.code(mindmap_raw, language="dot")
    try:
        st.graphviz_chart(mindmap_raw)
    except:
        st.error("Could not render Graphviz chart")
