# streamlit_app.py
"""
Final Streamlit app for Project 2 (Q1 + Q2 combined).

Features:
- Document-first QA using a local Ollama model (llama3.2:latest).
  The model is instructed to prefer the uploaded document and label claims as [DOC] or [LM].
- Abbreviation index generation that:
  1) extracts explicit definitions from the document (regex tolerant to newlines/wrapping)
  2) asks the LLM to infer expansions from its memory when needed
  3) merges, deduplicates, and flags ambiguous items as AMBIG: ABBR
- Supports TXT, PDF, DOCX, HTML uploads.
- Configure OLLAMA_URL and MODEL_NAME at top.

Run:
    pip install streamlit requests pdfplumber python-docx
    streamlit run streamlit_app.py
Make sure Ollama is running and the model is available:
    ollama pull llama3.2:latest
    ollama serve
"""

import streamlit as st
import requests
import io
import zipfile
import xml.etree.ElementTree as ET
import re
from typing import Dict, List

# Optional libs
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
MAX_PROMPT_CHARS = 120000  # keep prompt sizes reasonable
TIMEOUT_SECONDS = 120

st.set_page_config(page_title="Web based LLM-APP", layout="centered")
st.markdown("<h1 style='font-size:36px'>Input to AI</h1>", unsafe_allow_html=True)

# ---------- Helpers: text extraction ----------
def extract_text_from_bytes(filename: str, b: bytes) -> str:
    """
    Robust text extraction for several filetypes.
    Returns a text string (possibly empty).
    """
    name = filename.lower()
    if name.endswith(".txt"):
        try:
            return b.decode("utf-8", errors="ignore")
        except Exception:
            return ""
    if name.endswith(".pdf"):
        if not pdfplumber:
            raise RuntimeError("pdfplumber is required to parse PDFs. Install: pip install pdfplumber")
        pages = []
        with pdfplumber.open(io.BytesIO(b)) as pdf:
            for p in pdf.pages:
                try:
                    t = p.extract_text()
                except Exception:
                    t = None
                if t:
                    pages.append(t)
        return "\n".join(pages)
    if name.endswith(".docx") or name.endswith(".doc"):
        # Try python-docx first
        try:
            if docx:
                document = docx.Document(io.BytesIO(b))
                paras = [p.text for p in document.paragraphs if p.text]
                return "\n".join(paras)
        except Exception:
            pass
        # Fallback: unzip and parse word/document.xml
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
        try:
            return b.decode("utf-8", errors="ignore")
        except Exception:
            return ""
    # fallback decode
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return ""

# ---------- Helpers: call Ollama ----------
def call_ollama(prompt: str, model: str = MODEL_NAME, ollama_url: str = OLLAMA_URL, timeout: int = TIMEOUT_SECONDS) -> str:
    """
    Call Ollama's /api/generate endpoint with stream=False.
    Attempts to return a text result robustly across different Ollama response shapes.
    """
    url = ollama_url.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    # Common keys for textual output
    for key in ("response", "text", "result", "output"):
        if key in data and isinstance(data[key], str):
            return data[key]
    # If data contains a list or dict with nested text, attempt to find strings
    if isinstance(data, dict):
        # search recursively for first string value
        stack = [data]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                for v in item.values():
                    if isinstance(v, str):
                        return v
                    if isinstance(v, (dict, list)):
                        stack.append(v)
            elif isinstance(item, list):
                for v in item:
                    if isinstance(v, str):
                        return v
                    if isinstance(v, (dict, list)):
                        stack.append(v)
    # fallback to raw repr
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

# ---------- Abbreviation extraction helpers ----------
def extract_explicit_abbrs(text: str) -> Dict[str, str]:
    """
    Extract explicit 'Full term (ABBR)' and 'ABBR (Full term)' occurrences using tolerant regex.
    Returns dict {ABBR: full_term}
    """
    # Normalize whitespace to reduce line-break issues
    clean = re.sub(r'\s+', ' ', text)
    found: Dict[str, str] = {}
    # Full term (ABBR)
    for m in re.finditer(r'([A-Za-z][A-Za-z0-9\-\,:;\/\s]{2,200}?)\s*\(\s*([A-Z]{2,6})\s*\)', clean):
        full = ' '.join(m.group(1).split())
        abbr = m.group(2).strip()
        if abbr:
            found[abbr] = full
    # ABBR (Full term)
    for m in re.finditer(r'\b([A-Z]{2,6})\s*\(\s*([A-Za-z][A-Za-z0-9\-\,:;\/\s]{2,200}?)\s*\)', clean):
        abbr = m.group(1).strip()
        full = ' '.join(m.group(2).split())
        if abbr and abbr not in found:
            found[abbr] = full
    return found

def parse_llm_abbr_output(llm_text: str) -> Dict[str, str]:
    """
    Parse many LLM output styles and return dict {ABBR: expansion} for recognized forms.
    Accepts 'ABBR: expansion', 'ABBR - expansion', 'Full term (ABBR)', and others.
    """
    out: Dict[str, str] = {}
    lines = [l.strip() for l in llm_text.splitlines() if l.strip()]
    for ln in lines:
        # colon
        if ':' in ln:
            left, right = ln.split(':', 1)
            left = left.strip()
            right = right.strip(' .')
            if left == "AMBIG":
                # AMBIG: ABBR or AMBIG: XYZ
                abbr = right.strip()
                if re.fullmatch(r'[A-Z]{2,6}', abbr):
                    out[abbr] = ""  # mark ambiguous as empty
                continue
            if re.fullmatch(r'[A-Z]{2,6}', left):
                out[left] = right
                continue
        # dash variants
        m = re.match(r'^([A-Z]{2,6})\s*[-–—]\s*(.+)$', ln)
        if m:
            out[m.group(1)] = m.group(2).strip(' .')
            continue
        # ABBR (expansion)
        m = re.match(r'^([A-Z]{2,6})\s*\(\s*(.+?)\s*\)$', ln)
        if m:
            out[m.group(1)] = m.group(2).strip()
            continue
        # Full term (ABBR) -> invert
        m = re.match(r'^(.+?)\s*\(\s*([A-Z]{2,6})\s*\)$', ln)
        if m:
            abbr = m.group(2).strip()
            full = m.group(1).strip()
            out[abbr] = full
            continue
        # fallback: find any "(ABBR)" in the line and take preceding chunk as expansion
        m = re.search(r'\(([A-Z]{2,6})\)', ln)
        if m:
            a = m.group(1)
            idx = ln.find('(' + a + ')')
            candidate = ln[max(0, idx-120):idx].strip(' :-—,.')
            candidate = candidate.split('.')[-1].strip()
            if candidate:
                out[a] = candidate
    return out

def generate_abbr_index_combined(file_bytes: bytes, filename: str, use_llm: bool = True) -> List[str]:
    """
    Combine explicit extraction and LLM inference to produce final list of lines:
    - "ABBR: expansion" or "AMBIG: ABBR"
    """
    text = extract_text_from_bytes(filename, file_bytes)
    results: Dict[str, str] = {}

    # 1) explicit regex extraction
    explicit = extract_explicit_abbrs(text)
    results.update(explicit)

    # 2) LLM-assisted extraction/inference
    if use_llm:
        try:
            prompt = build_abbr_infer_prompt(text)
            llm_out = call_ollama(prompt)
            parsed = parse_llm_abbr_output(llm_out)
            # Merge, preferring explicit definitions
            for k, v in parsed.items():
                if k not in results:
                    results[k] = v
        except Exception as e:
            # If LLM call fails, continue with explicit only
            st.warning(f"LLM abbreviation step failed: {e}")

    # Build final lines sorted by ABBR
    final_lines: List[str] = []
    for k in sorted(results.keys()):
        val = results[k].strip() if results[k] else ""
        if not val:
            final_lines.append(f"AMBIG: {k}")
        else:
            final_lines.append(f"{k}: {val}")
    return final_lines

# ---------- UI ----------
question = st.text_input("Enter your question:", placeholder="Who are you?")
uploaded_file = st.file_uploader("Upload attachment:", type=["txt", "pdf", "docx", "html"])
st.markdown("---")
st.markdown("<h2 style='font-size:30px'>AI Response:</h2>", unsafe_allow_html=True)

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

        if context_text.strip():
            prompt = build_document_first_prompt(context_text, question)
        else:
            prompt = f"QUESTION: {question}\nANSWER: (No document provided; answer using your knowledge and label claims [LM])"

        st.subheader("Prompt preview (truncated)")
        st.code(prompt[:3000] + ("\n\n... (truncated)" if len(prompt) > 3000 else ""), language="text")

        with st.spinner("Calling LLM..."):
            try:
                answer = call_ollama(prompt)
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
            with st.spinner("Extracting abbreviations (regex + LLM inference)..."):
                lines = generate_abbr_index_combined(b, uploaded_file.name, use_llm=True)
                if not lines:
                    st.info("No abbreviations found (or the model output did not match expected formats).")
                else:
                    st.success("Abbreviation index (merged explicit + LLM-inferred):")
                    for ln in lines:
                        st.write(ln)
        except Exception as e:
            st.error(f"Failed to process uploaded file: {e}")

st.markdown("---")
st.markdown(
    "Notes: The app uses a document-first policy: answers prefer the uploaded document and label claims as [DOC] or [LM]. "
    "Abbreviation generation is allowed to use the LLM's memory; ambiguous items are returned as AMBIG: ABBR for manual review."
)
