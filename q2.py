import streamlit as st
import requests
import pdfplumber
import docx
import io

# ----------------------------
# Config
# ----------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:latest"

st.set_page_config(page_title="Web LLM App", layout="centered")

# ----------------------------
# Simple text extraction
# ----------------------------
def extract_text(uploaded_file):
    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        text = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text.append(t)
        return "\n".join(text)

    if name.endswith(".docx"):
        document = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in document.paragraphs)

    if name.endswith(".html"):
        return data.decode("utf-8", errors="ignore")

    return ""

# ----------------------------
# Call Llama through Ollama
# ----------------------------
def call_llama(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    r = requests.post(OLLAMA_URL, json=payload)
    r.raise_for_status()
    return r.json().get("response", "")

# ----------------------------
# UI Layout
# ----------------------------
st.title("Input to AI")

question = st.text_input("Enter your question:", placeholder="Who are you?")
uploaded_file = st.file_uploader("Upload attachment:", type=["txt", "pdf", "docx", "html"])

st.markdown("---")
st.header("AI Response")

col1, col2 = st.columns(2)
qa_button = col1.button("Run QA")
abbr_button = col2.button("Generate Abbreviation Index")

# ----------------------------
# Q1 — Document-first QA
# ----------------------------
if qa_button:
    if not question.strip():
        st.error("Please enter a question.")
    else:
        doc_text = ""
        if uploaded_file:
            doc_text = extract_text(uploaded_file)

        if doc_text.strip():
            prompt = f"""
You are an assistant that MUST prioritize the provided document.
If the document contains the answer, use it and label sentences as:
- [DOC] from document
- [LM] from LLM memory

DOCUMENT:
{doc_text}

QUESTION: {question}

Provide the answer.
"""
        else:
            prompt = f"""
No document was provided.
Answer the question using your knowledge and label all claims with [LM].

QUESTION: {question}
"""

        with st.spinner("Thinking..."):
            answer = call_llama(prompt)
            st.write(answer)

# ----------------------------
# Q2 — Abbreviation Index (simple LLM approach)
# ----------------------------
if abbr_button:
    if not uploaded_file:
        st.error("Please upload an article.")
    else:
        doc_text = extract_text(uploaded_file)

        if not doc_text.strip():
            st.error("Could not extract any text from the document.")
        else:
            prompt = f"""
Extract all abbreviations from the scientific article below.

Return ONLY lines in the form:
ABBR: full expansion

Rules:
- ABBR must be 2-6 uppercase letters.
- If the article directly defines the abbreviation, use that.
- If not defined but you are confident, infer the expansion.
- If unsure, output: AMBIG: ABBR

ARTICLE TEXT:
{doc_text}

Now produce the abbreviation index.
"""

            with st.spinner("Extracting abbreviations..."):
                result = call_llama(prompt)

            st.subheader("Abbreviation Index")
            st.write(result)
