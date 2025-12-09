import streamlit as st
import requests
import pdfplumber
import docx
import io


st.set_page_config(page_title="Web based LLM-APP", layout="centered")


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


def call_llama(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2:latest",
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["response"]

# UI

st.markdown("<h1 style='font-size: 36px;'>Input to AI</h1>", unsafe_allow_html=True)

question = st.text_input("Enter your question:", placeholder="Who are you?")

uploaded_file = st.file_uploader("Upload attachment:", type=["txt", "pdf", "docx", "html"])

st.markdown("---")
st.markdown("<h2 style='font-size: 30px;'>AI Response:</h2>", unsafe_allow_html=True)


# Logic

if question:
    context = ""

    if uploaded_file:
        try:
            context = extract_text(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read document: {e}")

    # Build prompt for model
    if context.strip():
        final_prompt  = f"""
    You are an assistant that must prioritize the provided document as the primary source of truth.
    Answer using the document when possible. If the document does not directly answer the question, you MAY supplement with your general knowledge, but you MUST clearly label whether each factual claim is:
    - [From DOC] if it came from the provided document, or
    - [LLM Memory] if it came from your prior knowledge.

    DOCUMENT:
    ---
    {context}
    ---

    QUESTION: {question}

    ANSWER: Provide a concise answer. For each sentence or key claim, prepend [DOC] or [LM]. If the document contains the answer exactly, prefer [DOC] and do NOT add [LM]. If you cannot find any relevant info in the document, start the answer with: "I cannot find that in the provided document."
    """
    else:
        final_prompt  = f"QUESTION: {question}\nANSWER: (No document provided; answer using your knowledge and label claims [LM])"
        #final_prompt = prompt

    try:
        answer = call_llama(final_prompt)
        st.write(answer)
    except Exception as e:
        st.error(f"Error contacting LLM: {e}")
