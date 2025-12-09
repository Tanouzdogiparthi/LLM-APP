# streamlit_app.py
"""
Streamlit app that supports:
1) QA over an uploaded document using a local Ollama model (llama3.2:latest),
   following a document-first policy where the model labels claims as [DOC] or [LM].
2) Generating an abbreviation index for an uploaded article using the LLM's memory
   (the model may infer expansions; ambiguous items are flagged as AMBIG).
Save as streamlit_app.py and run:
    streamlit run streamlit_app.py
Dependencies:
    pip install streamlit requests pdfplumber python-docx
"""
import streamlit as st
import requests
import io
import zipfile
import xml.etree.ElementTree as ET
import re
from typing import List

# optional imports
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import docx
except Exception:
    docx = None

# ---------- Config ----------
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "llama3.2:latest"
MAX_PROMPT_CHARS = 120000  # truncate document text to this many chars in prompts

st.set_page_config(page_title="Web based LLM-APP", layout="centered")
st.markdown("<h1 style='font-size:36px'>Input to AI</h1>", unsafe_allow_html=True)

# ---------- Helpers: text extraction ----------
def extract_text_from_bytes(filename: str, b: bytes) -> str:
    name = filename.lower()
    if name.endswith(".txt"):
        return b.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        if not pdfplumber:
            raise RuntimeError("pdfplumber is required to parse PDFs. pip install pdfplumber")
        pages = []
        with pdfplumber.open(io.BytesIO(b)) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    pages.append(t)
        return "\n".join(pages)
    if name.endswith(".docx") or name.endswith(".doc"):
        # try python-docx first
        try:
            if docx:
                document = docx.Document(io.BytesIO(b))
                paras = [p.text for p in document.paragraphs if p.text]
                return "\n".join(paras)
        except Exception:
            pass
        # fallback: unzip word/document.xml
        try:
            with zipfile.ZipFile(io.BytesIO(b)) as z:
                xml = z.read("word/document.xml")
            root = ET.fromstring(xml)
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            paragraphs = []
            for para in root.findall(".//w:p", ns):
                texts = [t.text for t in para.findall(".//w:t", ns) if t.text]
                if texts:
                    paragraphs.append("".join(texts))
            return "\n".join(paragraphs)
        except Exception:
            return ""
    if name.endswith(".html") or name.endswith(".htm"):
        return b.decode("utf-8", errors="ignore")
    # fallback
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return ""

# ---------- Helpers: call Ollama ----------
def call_ollama(prompt: str, model: str = MODEL_NAME, ollama_url: str = OLLAMA_URL, timeout: int = 120) -> str:
    """
    Call Ollama /api/generate with stream=False and return text result.
    Handles common response field variants.
    """
    url = ollama_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    # try common fields
    for key in ("response", "text", "result", "output"):
        if key in data and isinstance(data[key], str):
            return data[key]
    # sometimes data itself is a string or nested
    if isinstance(data, str):
        return data
    # fallback to str of json
    return str(data)

# ---------- Prompts ----------
def build_document_first_prompt(document_text: str, question: str) -> str:
    doc_short = document_text[:MAX_PROMPT_CHARS]
    prompt = f"""
You are an assistant that must prioritize the provided document as the primary source of truth.
Answer using the document when possible. If the document does not directly answer the question, you MAY supplement with your general knowledge, but you MUST clearly label whether each factual claim is:
- [DOC] if it came from the provided document, or
- [LM] if it came from your prior knowledge.

DOCUMENT:
---
{doc_short}
---

QUESTION: {question}

ANSWER: Provide a concise answer. For each sentence or key claim, prepend [DOC] or [LM]. 
If the document contains the answer exactly, prefer [DOC] and do NOT add [LM]. 
If you cannot find any relevant info in the document, start the answer with: "I cannot find that in the provided document."
"""
    return prompt

def build_abbr_infer_prompt(article_text: str) -> str:
    short = article_text[:MAX_PROMPT_CHARS]
    prompt = f"""
You are given the text of a scientific article. Produce an abbreviation index listing abbreviations used in the article and their expansions, one per line, EXACTLY in the format:
ABBR: full expansion

Rules:
- Include only abbreviations that appear in the article (case-insensitive).
- ABBR must be 2–6 uppercase letters.
- If the article explicitly defines an abbreviation (e.g., "weighted degree centrality (WDC)"), use that expansion.
- If the article does NOT explicitly define an abbreviation but you (the model) are confident about the standard expansion, you MAY infer the expansion from your knowledge. In that case return the line normally: "ABBR: expansion".
- If you are NOT confident, DO NOT invent it. Instead return: "AMBIG: ABBR".
- Do NOT output anything except lines of the form "ABBR: expansion" or "AMBIG: ABBR".
Article text:
---
{short}
---
Now produce the abbreviation index following the rules above.
"""
    return prompt

# ---------- UI elements ----------
question = st.text_input("Enter your question:", placeholder="Who are you?")
uploaded_file = st.file_uploader("Upload attachment:", type=["txt", "pdf", "docx", "html"])
st.markdown("---")
st.markdown("<h2 style='font-size:30px'>AI Response:</h2>", unsafe_allow_html=True)

# Buttons grouped
col1, col2 = st.columns(2)
with col1:
    run_qa = st.button("Run QA (document-first)")
with col2:
    gen_abbr = st.button("Generate abbreviation index (allow LLM inference)")

# ---------- Actions ----------
if run_qa:
    if not question or not question.strip():
        st.error("Please enter a question.")
    else:
        context_text = ""
        if uploaded_file:
            try:
                uploaded_file.seek(0)
                b = uploaded_file.read()
                context_text = extract_text_from_bytes(uploaded_file.name, b)
                if not context_text.strip():
                    st.warning("Uploaded file parsed but no text extracted.")
            except Exception as e:
                st.error(f"Failed to read document: {e}")
                context_text = ""
        else:
            st.info("No document uploaded; the model will answer from its general knowledge and label claims as [LM].")

        prompt = build_document_first_prompt(context_text, question) if context_text.strip() else f"QUESTION: {question}\nANSWER: (No document provided; answer using your knowledge and label claims [LM])"
        # show prompt preview (truncated)
        st.subheader("Prompt preview (truncated)")
        st.code(prompt[:3000] + ("\n\n... (truncated)" if len(prompt) > 3000 else ""), language="text")

        with st.spinner("Calling LLM..."):
            try:
                answer = call_ollama(prompt)
                # basic label check
                if "[DOC]" not in answer and "[LM]" not in answer:
                    st.warning("Model did not include explicit labels [DOC]/[LM]. Check prompt adherence.")
                st.markdown("**Answer:**")
                st.write(answer)
            except Exception as e:
                st.error(f"Error contacting LLM: {e}")

if gen_abbr:
    if not uploaded_file:
        st.error("Please upload the article file first.")
    else:
        try:
            uploaded_file.seek(0)
            b = uploaded_file.read()
            article_text = extract_text_from_bytes(uploaded_file.name, b)
            if not article_text.strip():
                st.info("No text could be extracted from the uploaded file.")
            else:
                prompt = build_abbr_infer_prompt(article_text)
                st.subheader("Abbreviation extraction prompt preview (truncated)")
                st.code(prompt[:3000] + ("\n\n... (truncated)" if len(prompt) > 3000 else ""), language="text")
                with st.spinner("Asking LLM to build abbreviation index..."):
                    try:
                        out = call_ollama(prompt)
                        # parse lines
                        lines = []
                        for ln in out.splitlines():
                            ln = ln.strip()
                            if not ln:
                                continue
                            if ':' not in ln:
                                continue
                            left, right = [p.strip() for p in ln.split(':', 1)]
                            if left == "AMBIG":
                                # right is ABBR
                                abbr = right
                                lines.append(f"AMBIG: {abbr}")
                                continue
                            if re.fullmatch(r'[A-Z]{2,6}', left):
                                lines.append(f"{left}: {right}")
                        # dedupe
                        seen = set()
                        final = []
                        for L in lines:
                            key = L.split(':', 1)[0].strip()
                            if key not in seen:
                                seen.add(key)
                                final.append(L)
                        if not final:
                            st.info("No abbreviations found (or model output did not match expected format).")
                        else:
                            st.success("Abbreviation index (LLM-inferred):")
                            for ln in final:
                                st.write(ln)
                    except Exception as e:
                        st.error(f"LLM request failed: {e}")
        except Exception as e:
            st.error(f"Failed to process uploaded file: {e}")

# Footer note
#st.markdown("---")
#st.markdown(
  #  "Notes: The app uses a document-first policy: answers prefer the uploaded document and label claims as [DOC] or [LLM]. "
 #   "Abbreviation generation is allowed to use the LLM's memory; ambiguous items are returned as AMBIG: ABBR for manual review."
#)
