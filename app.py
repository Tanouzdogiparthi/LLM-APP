# app.py
"""
Streamlit RAG QA app that calls a local Ollama instance.
Supports PDF, DOCX, TXT, HTML uploads, naive retrieval, and Ollama generation.
Adjust OLLAMA_URL and OLLAMA_MODEL via environment variables or change the defaults below.
"""

import os
import io
import re
import json
import requests
import streamlit as st
from typing import List, Tuple

# Optional parsers
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import docx
except Exception:
    docx = None

# Config (change as needed or export env vars)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")  # change to 'mistral' or 'gptoss' as needed
CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "3000"))
MAX_CHUNKS_IN_PROMPT = int(os.getenv("MAX_CHUNKS_IN_PROMPT", "6"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.0"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "512"))

st.set_page_config(page_title="RAG QA — Streamlit + Ollama", layout="wide")
st.title("RAG / Document QA — Streamlit + Ollama (Local)")

# Sidebar settings
st.sidebar.header("Settings")
ollama_url_in = st.sidebar.text_input("Ollama URL", OLLAMA_URL)
model_in = st.sidebar.text_input("Ollama model name", OLLAMA_MODEL)
temperature_in = st.sidebar.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE)
max_tokens_in = st.sidebar.number_input("Max tokens (LLM response)", min_value=64, max_value=4096, value=DEFAULT_MAX_TOKENS)
chunk_size_in = st.sidebar.number_input("Chunk size (chars)", min_value=500, max_value=20000, value=CHUNK_SIZE_CHARS)
max_chunks_prompt_in = st.sidebar.number_input("Max chunks in prompt", min_value=1, max_value=12, value=MAX_CHUNKS_IN_PROMPT)

uploaded_files = st.file_uploader("Upload documents (PDF / DOCX / TXT / HTML)", accept_multiple_files=True)
question = st.text_area("Question", height=80)
st.write("Tip: Upload files and ask a question that refers to those documents. If no files are uploaded the model will answer from its own knowledge (not recommended for assignment).")

# ---------- Helpers ----------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    if not pdfplumber:
        raise RuntimeError("pdfplumber is required to parse PDFs. pip install pdfplumber")
    texts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                texts.append(t)
    return "\n".join(texts)

def extract_text_from_docx(file_bytes: bytes) -> str:
    if not docx:
        raise RuntimeError("python-docx is required to parse DOCX. pip install python-docx")
    doc = docx.Document(io.BytesIO(file_bytes))
    paras = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paras)

def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

def get_text_from_upload(f) -> Tuple[str, str]:
    """Return (filename, text)"""
    name = f.name
    b = f.read()
    lower = name.lower()
    if lower.endswith(".pdf"):
        txt = extract_text_from_pdf(b)
    elif lower.endswith(".docx") or lower.endswith(".doc"):
        txt = extract_text_from_docx(b)
    elif lower.endswith(".txt") or lower.endswith(".md") or lower.endswith(".html"):
        txt = extract_text_from_txt(b)
    else:
        # fallback to text
        txt = extract_text_from_txt(b)
    return name, txt

def chunk_text(text: str, chunk_size_chars: int) -> List[str]:
    """Simple character-based chunking. Returns list of chunk strings."""
    if not text:
        return []
    chunks = []
    text = text.replace("\r\n", "\n")
    for i in range(0, len(text), chunk_size_chars):
        chunk = text[i:i+chunk_size_chars]
        chunks.append(chunk.strip())
    return chunks

def score_chunk_for_query(chunk_text: str, query: str) -> int:
    words = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 3]
    t = chunk_text.lower()
    score = 0
    for w in words:
        score += t.count(w)
    return score

def build_prompt(chunks: List[Tuple[str,str]], question: str, max_chunks: int) -> str:
    """
    chunks: list of (filename, chunk_text) ordered by relevance
    Build a prompt that instructs the model to use only provided context and cite.
    """
    system = (
        "You are an assistant that must answer using ONLY the provided documents below. "
        "If the answer cannot be found in the documents, reply: \"I can't find that in the provided documents.\" "
        "Cite sources using [filename:chunk_index]. Keep answers concise and include citations for factual claims."
    )
    parts = [system, "\n\n=== CONTEXT START ===\n"]
    for idx, (fname, ctext) in enumerate(chunks[:max_chunks]):
        # include small header for traceability
        parts.append(f"[{fname} : chunk {idx}]\n{ctext}\n\n")
    parts.append("=== CONTEXT END ===\n")
    parts.append(f"Question: {question}\n")
    parts.append("Answer (concise, cite file chunks where relevant):")
    return "\n".join(parts)

def call_ollama(prompt: str, ollama_url: str, model: str, temperature: float, max_tokens: int) -> str:
    """
    Call Ollama's /api/generate endpoint with stream disabled.
    This matches current Ollama API (see: ollama /api/generate docs).
    """
    url = f"{ollama_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,          # important: so we get one JSON response
        "options": {
            "temperature": float(temperature),
            # ollama's API doesn't use max_tokens exactly like OpenAI; we can pass num_predict
            "num_predict": int(max_tokens),
        },
    }

    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers, timeout=120)

    if resp.status_code != 200:
        raise Exception(f"{url} returned status {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    # Current Ollama returns the text under "response" when stream=False
    if "response" in data and isinstance(data["response"], str):
        return data["response"]

    # Fallback if shape changes
    return str(data)

# ---------- Main UI / logic ----------
if st.button("Run QA"):
    if not question.strip():
        st.error("Please enter a question.")
    else:
        # get docs
        docs = []
        for f in uploaded_files:
            try:
                fname, txt = get_text_from_upload(f)
            except Exception as e:
                st.error(f"Failed to parse {f.name}: {e}")
                continue
            docs.append((fname, txt))

        if not docs:
            st.warning("No documents uploaded — the model will answer from its pretraining knowledge (not recommended).")

        # chunk docs and score by keyword overlap
        all_chunks = []
        for fname, text in docs:
            for chunk in chunk_text(text, chunk_size_in):
                all_chunks.append((fname, chunk))

        if all_chunks:
            # score each chunk
            scored = []
            for fname, ctext in all_chunks:
                sc = score_chunk_for_query(ctext, question)
                scored.append((sc, fname, ctext))
            # sort descending by score
            scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)
            # filter out zero-score chunks unless all are zero
            if scored_sorted[0][0] == 0:
                # fallback: use the first N chunks in document order
                top_chunks = [(fname, ctext) for (_, fname, ctext) in scored_sorted[:max_chunks_prompt_in]]
            else:
                top_chunks = [(fname, ctext) for (_, fname, ctext) in scored_sorted if _ > 0][:max_chunks_prompt_in]
        else:
            top_chunks = []

        # Build prompt
        prompt = build_prompt(top_chunks, question, max_chunks_prompt_in)
        st.subheader("Prompt preview")
        st.code(prompt[:4000] + ("\n\n... (truncated)" if len(prompt) > 4000 else ""), language="text")

        # Call Ollama
        with st.spinner("Calling Ollama..."):
            try:
                answer = call_ollama(prompt, ollama_url_in, model_in, temperature_in, int(max_tokens_in))
                st.subheader("Answer")
                st.write(answer)
                # show which chunks were used (simple list)
                if top_chunks:
                    st.subheader("Source chunks used (top selection)")
                    for i, (fname, ctext) in enumerate(top_chunks):
                        st.markdown(f"**{i+1}. {fname}** — snippet:")
                        st.write(ctext[:800] + ("... " if len(ctext) > 800 else ""))
            except Exception as e:
                st.error(f"Error calling Ollama: {e}")
