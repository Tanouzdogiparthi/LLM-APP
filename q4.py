import streamlit as st
import requests
import pdfplumber
import docx
import io
import time

# -------------------------
# Gemini config (paste your key)
# -------------------------
GEMINI_API_KEY = "PASTE_YOUR_GEMINI_API_KEY_HERE"
# Use gemini-pro to reduce chance of strict rate limits
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

# -------------------------
# App settings
# -------------------------
MAX_PROMPT_CHARS = 20000   # truncate document text to this many chars
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0      # seconds

st.set_page_config(page_title="Web based LLM-APP (Gemini)", layout="centered")


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


def call_gemini_with_retries(prompt, max_retries=MAX_RETRIES):
    """
    Call Gemini with simple exponential backoff retry for 429 responses.
    Returns text or raises the last exception.
    """
    if GEMINI_API_KEY == "AIzaSyCMSr_Y35fyhrtOQ8ob4pb4x-Lfv0zNH7g":
        raise Exception("Please paste your Gemini API Key in GEMINI_API_KEY at the top of the file.")

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json", "X-goog-api-key": GEMINI_API_KEY}

    backoff = INITIAL_BACKOFF
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            # robust extraction
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            # retry on 429 (rate limit) or 503 (server busy)
            if status in (429, 503) and attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise
        except requests.RequestException:
            # network error, maybe retry
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise


# -------------------------
# UI
# -------------------------
st.markdown("<h1 style='font-size: 36px;'>Input to AI</h1>", unsafe_allow_html=True)

question = st.text_input("Enter your question:", placeholder="Who are you?")
uploaded_file = st.file_uploader("Upload attachment:", type=["txt", "pdf", "docx", "html"])

st.markdown("---")
st.markdown("<h2 style='font-size: 30px;'>AI Response:</h2>", unsafe_allow_html=True)

# Use an explicit button to avoid automatic re-runs and accidental quota usage
if st.button("Run QA (Gemini)"):
    if not question or not question.strip():
        st.error("Please enter a question.")
    else:
        context = ""
        if uploaded_file:
            try:
                uploaded_file.seek(0)
                context = extract_text(uploaded_file)
                if not context.strip():
                    st.warning("Uploaded file parsed but no text extracted.")
            except Exception as e:
                st.error(f"Failed to read document: {e}")
                context = ""

        # build prompt (document-first)
        if context.strip():
            truncated = context[:MAX_PROMPT_CHARS]
            prompt = f"""
You are an assistant that must prioritize the provided document as the primary source of truth.

Rules:
- If a fact comes from the document, prepend [DOC].
- If it comes from your own knowledge, prepend [LM].
- Be concise.
- If the document does not contain the answer, begin with:
  "I cannot find that in the provided document."

DOCUMENT:
---
{truncated}
---

QUESTION: {question}

Provide the final answer now.
"""
        else:
            prompt = f"""
No document was provided.

Answer the question using your own knowledge.
Label all claims as [LM].

QUESTION: {question}
"""

        st.subheader("Prompt preview (truncated)")
        st.code(prompt[:2000] + ("\n\n... (truncated)" if len(prompt) > 2000 else ""), language="text")

        with st.spinner("Calling Gemini (gemini-pro)..."):
            try:
                answer = call_gemini_with_retries(prompt)
                st.write(answer)
            except requests.HTTPError as e:
                code = e.response.status_code if e.response is not None else "N/A"
                st.error(f"HTTP error contacting Gemini API: {code} - {e}")
            except Exception as e:
                st.error(f"Error contacting Gemini API: {e}")

st.markdown("---")
st.markdown(
    "Notes: This app uses an explicit run button to avoid accidental repeated API calls (which consume quota). "
    "If you hit rate limits (429), wait a minute or switch to a different Gemini model/plan."
)
