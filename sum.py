"""
AI Study Assistant (Gemini Free API Version)
"""

import io
import json
import streamlit as st
import google.generativeai as genai

# 🔹 Configure Gemini API with your key
GEMINI_API_KEY = "AIzaSyAovY7_L07gri_mNN4Xod6-gmPEYiEJvjY"
genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-1.5-flash"

# Optional libraries
try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False

try:
    from PyPDF2 import PdfReader
    _HAS_PYPDF2 = True
except ImportError:
    _HAS_PYPDF2 = False

try:
    import graphviz
    _HAS_GRAPHVIZ = True
except ImportError:
    _HAS_GRAPHVIZ = False

# --- Streamlit setup ---
st.set_page_config(page_title="AI Study Assistant", page_icon=":mortar_board:", layout="wide")
st.title("AI Study Assistant (Gemini API)")
st.markdown("Summarize text, create flashcards, Q&A, and mind-maps using **Gemini free API**.")

# Sidebar controls
st.sidebar.header("Settings")
num_bullets = st.sidebar.slider("Number of summary bullets", 3, 10, 5)
num_flashcards = st.sidebar.slider("Number of flashcards", 3, 30, 10)
num_short_qa = st.sidebar.slider("Number of short Q&A", 3, 20, 8)

# --- PDF extraction ---
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    text_pages = []
    if _HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for p in pdf.pages:
                    text_pages.append(p.extract_text() or "")
            return "\n\n".join(text_pages).strip()
        except Exception:
            pass
    if _HAS_PYPDF2:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for page in reader.pages:
                text_pages.append(page.extract_text() or "")
            return "\n\n".join(text_pages).strip()
        except Exception:
            pass
    return ""

# --- Gemini helper ---
def call_gemini(prompt: str, model: str = MODEL_NAME) -> str:
    try:
        m = genai.GenerativeModel(model)
        resp = m.generate_content(prompt)
        return resp.text
    except Exception as e:
        st.error(f"Gemini API error: {e}")
        return ""

# --- Input section ---
st.subheader("Input")
col1, col2 = st.columns([1, 2])
with col1:
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    paste_text = st.text_area("Or paste text", height=200)

input_text = ""
if uploaded_file:
    input_text = extract_text_from_pdf_bytes(uploaded_file.read())
elif paste_text.strip():
    input_text = paste_text.strip()

if not input_text:
    st.warning("Upload a PDF or paste text to continue.")
    st.stop()

st.subheader("Preview of text")
with st.expander("Show first 3000 chars"):
    st.text(input_text[:3000] + ("..." if len(input_text) > 3000 else ""))

# --- Generation ---
if st.button("Generate summary, flashcards, Q&A & mind-map"):
    with st.spinner("Talking to Gemini..."):
        context = input_text[:60000]

        # 1️⃣ Summary
        summary_prompt = f"Summarize the following in {num_bullets} concise bullet points:\n\n{context}"
        summary_raw = call_gemini(summary_prompt)

        # 2️⃣ Flashcards
        flashcards_prompt = (
            f"Create {num_flashcards} flashcards in JSON format "
            f"with 'question' and 'answer' from this content:\n\n{context}"
        )
        flashcards_raw = call_gemini(flashcards_prompt)

        # 3️⃣ Short Q&A
        qa_prompt = f"Generate {num_short_qa} short Q&A pairs (Q: ... A: ...):\n\n{context}"
        short_qa_raw = call_gemini(qa_prompt)

        # 4️⃣ Mind-map
        mindmap_prompt = (
            "Create a mind-map in Graphviz DOT format for this content. "
            "Output ONLY DOT code.\n\n" + context
        )
        mindmap_raw = call_gemini(mindmap_prompt)

    st.success("✅ Done!")

    # --- Display results ---
    st.header("Summary")
    st.markdown(summary_raw)

    st.header("Flashcards")
    try:
        parsed_flashcards = json.loads(flashcards_raw)
        st.json(parsed_flashcards)
    except json.JSONDecodeError:
        st.text_area("Raw flashcards", flashcards_raw, height=200)

    st.header("Q&A")
    st.text_area("Quick Q&A", short_qa_raw, height=200)

    st.header("Mind-map")
    st.code(mindmap_raw, language="dot")
    if _HAS_GRAPHVIZ:
        try:
            st.graphviz_chart(mindmap_raw)
        except Exception:
            st.error("Could not render Graphviz chart. Check DOT format.")
    else:
        st.warning("Graphviz not installed, cannot render mind-map chart.")
