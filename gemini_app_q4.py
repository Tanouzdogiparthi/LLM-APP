import streamlit as st
import pdfplumber
import docx
import io

from google import genai

# Paste your API key here
API_KEY = st.secrets.get("GEMINI_API_KEY")

if not API_KEY:
    st.error("API key missing. Set GEMINI_API_KEY in secrets or env")


client = genai.Client(api_key=API_KEY)


st.set_page_config(page_title="Web based Closed Source LLM-APP-GEMINI", layout="centered")


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

# Gemini API Call 

def call_gemini(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )
    return response.text

# UI
st.markdown("<h1 style='font-size: 36px;'>Input to AI</h1>", unsafe_allow_html=True)

question = st.text_input("Enter your question:", placeholder="Who are you?")
uploaded_file = st.file_uploader("Upload attachment:", type=["txt", "pdf", "docx", "html"])

st.markdown("---")
st.markdown("<h2 style='font-size: 30px;'>AI Response:</h2>", unsafe_allow_html=True)


if question:
    context = ""

    if uploaded_file:
        try:
            context = extract_text(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read document: {e}")

    if context.strip():
        final_prompt = f"""
        You are an assistant that must prioritize the provided document as the primary source of truth.
        Answer using the document when possible. If the document does not directly answer the question, you MAY supplement with your general knowledge, but you MUST clearly label whether each factual claim is:
        - [From DOC] if it came from the provided document, or
        - [LLM Memory] if it came from your prior knowledge.

        DOCUMENT:
        ---
        {context}
        ---

        QUESTION: {question}

        ANSWER: Provide a concise answer. For each sentence or key claim, prepend [DOC] or [LM].
        If the document contains the answer exactly, prefer [DOC] and do NOT add [LM].
        If you cannot find any relevant info in the document, start with:
        "I cannot find that in the provided document."
        """
    else:
        final_prompt = f"""
        QUESTION: {question}
        ANSWER: (No document provided; answer using your knowledge and label claims [LM])
        """

    try:
        answer = call_gemini(final_prompt)
        st.write(answer)
    except Exception as e:
        st.error(f"Error contacting Gemini API: {e}")

st.markdown("""
---
### 👨‍💻 Developed by:
**Tanouz**, **Chandana**, **Pratyush**, **Sahithi**, **Susmitha**

### 🤖 LLM Used:
This application uses Google Gemini (Closed-Source LLM) accessed through the official Gemini API.
---
""")




