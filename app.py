import io
import os
from typing import List

import streamlit as st
import google.genai as genai
from google.genai.types import GenerateContentConfig
from pypdf import PdfReader
from dotenv import load_dotenv

from ingest import ingest_text
from query_engine import answer_question

GEMINI_MODEL = "gemini-2.0-flash"

# Load environment variables from .env so Qdrant and Gemini work
load_dotenv()


def get_gemini_client() -> genai.Client:
    """Create a Gemini client using the API key from the environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment.")
    return genai.Client(api_key=api_key)


def summarize_document(text: str) -> str:
    """
    Use Gemini to generate a short bullet point summary of the ingested text.

    To avoid sending extremely large payloads, we trim the document
    to a safe length before summarization.
    """
    trimmed_text = text[:8000]

    prompt = f"""
You are a data analyst.

You receive a raw text document.
Write a concise summary in bullet points with:

- 3 to 5 key insights
- Focus on practical findings and risks
- Use simple language

Document:
{trimmed_text}
"""

    client = get_gemini_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=GenerateContentConfig(
            temperature=0.4,
            max_output_tokens=256,
        ),
    )
    return response.text.strip()


def extract_text_from_uploaded_file(uploaded_file) -> str:
    """
    Read text from an uploaded txt or pdf file.

    For txt files we decode bytes as utf-8.
    For pdf files we use pypdf and limit the number of pages we read
    so that very large documents do not consume too many resources.
    """
    if uploaded_file is None:
        return ""

    # Try to reset pointer
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    file_name = uploaded_file.name.lower()

    # Plain text file
    if file_name.endswith(".txt") or uploaded_file.type == "text/plain":
        raw_bytes = uploaded_file.read()
        try:
            return raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return raw_bytes.decode("latin-1", errors="ignore")

    # PDF file
    if file_name.endswith(".pdf") or uploaded_file.type == "application/pdf":
        try:
            # Let PdfReader stream from the uploaded file instead of
            # reading the entire pdf into memory first.
            reader = PdfReader(uploaded_file)
        except Exception:
            # Fallback: read bytes into memory if the above fails
            raw_bytes = uploaded_file.read()
            pdf_bytes = io.BytesIO(raw_bytes)
            reader = PdfReader(pdf_bytes)

        pages_text: List[str] = []
        max_pages = 5  # safety limit
        for page_index, page in enumerate(reader.pages):
            if page_index >= max_pages:
                break
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            pages_text.append(page_text)

        text = "\n\n".join(pages_text)
        # Final safety limit on text size
        return text[:20000]

    # Unsupported type
    return ""


def init_session_state() -> None:
    """Prepare Streamlit session state variables."""
    if "history" not in st.session_state:
        st.session_state["history"] = []  # list of {question, answer, project}
    if "last_summary" not in st.session_state:
        st.session_state["last_summary"] = ""
    if "current_project" not in st.session_state:
        st.session_state["current_project"] = "demo"
    if "documents" not in st.session_state:
        # list of {project, document_name}
        st.session_state["documents"] = []


def main():
    st.set_page_config(
        page_title="Vector Insight Engine",
        layout="wide",
    )

    init_session_state()

    # Project selector at the top
    st.title("Vector Insight Engine")
    st.write(
        "Turn unstructured notes into searchable insights for data analysis and research."
    )

    project_label = st.text_input(
        "Project label",
        value=st.session_state["current_project"],
        help="Use the same label to group documents and questions by project or client.",
    )
    st.session_state["current_project"] = project_label.strip() or "default"

    col_ingest, col_query = st.columns(2)

    # Sidebar
    with st.sidebar:
        st.markdown("### About")
        st.write(
            "Vector Insight Engine turns messy text into searchable insights "
            "using Google Gemini for reasoning and Qdrant for vector search."
        )

        st.markdown("### Tech stack")
        st.markdown(
            """
- Language model: Gemini `gemini-2.0-flash`  
- Embeddings: `text-embedding-004`  
- Vector database: Qdrant Cloud  
- Web app: Streamlit  
- Language: Python 3.11  
"""
        )

        st.markdown("### Ingested documents (this session)")
        if not st.session_state["documents"]:
            st.write("No documents ingested yet.")
        else:
            for doc in reversed(st.session_state["documents"]):
                st.write(f"{doc['project']}  ·  {doc['document_name']}")

        st.markdown("### How to use")
        st.write(
            "1. Choose a project label at the top.\n"
            "2. Paste text or upload a txt/pdf file.\n"
            "3. Click **Ingest into Qdrant**.\n"
            "4. Ask project specific questions in Step 2.\n"
            "5. Inspect retrieved context and citations."
        )

    # Step 1 - ingest
    with col_ingest:
        st.subheader("Step 1. Ingest a document")

        default_text = (
            "Machine learning models depend on the quality and coverage of their training data.\n"
            "Poor inputs lead to unstable predictions and biased outcomes.\n"
            "Robust data pipelines improve accuracy and interpretability for high impact decisions.\n"
            "Vector Insight Engine is built to help analysts turn messy documents into searchable insights."
        )

        input_text = st.text_area(
            "Paste a document or notes (optional if you upload a file):",
            value=default_text,
            height=220,
        )

        uploaded_file = st.file_uploader(
            "Or upload a plain text or PDF file",
            type=["txt", "pdf"],
        )

        ingest_button = st.button("Ingest into Qdrant")

        if ingest_button:
            raw_text = ""
            document_name = "pasted_text"

            if uploaded_file is not None:
                document_name = uploaded_file.name
                with st.spinner("Reading file and embedding chunks..."):
                    raw_text = extract_text_from_uploaded_file(uploaded_file)
            else:
                raw_text = input_text

            if not raw_text or not raw_text.strip():
                st.error("There is no text to ingest. Please paste some text or upload a file.")
            else:
                with st.spinner("Embedding text and writing chunks to Qdrant..."):
                    written = ingest_text(
                        raw_text,
                        project=st.session_state["current_project"],
                        document_name=document_name,
                    )
                    try:
                        summary = summarize_document(raw_text)
                    except Exception as e:
                        summary = ""
                        st.warning(
                            f"Document ingested successfully, "
                            f"but summary generation failed: {e}"
                        )

                st.session_state["last_summary"] = summary
                st.session_state["documents"].append(
                    {
                        "project": st.session_state["current_project"],
                        "document_name": document_name,
                    }
                )
                st.success(
                    f"Ingested {written} chunks into project "
                    f"`{st.session_state['current_project']}`."
                )

        if st.session_state.get("last_summary"):
            st.markdown("#### Key insights from the last document")
            st.markdown(st.session_state["last_summary"])

    # Step 2 - query
    with col_query:
        st.subheader("Step 2. Ask a question")

        st.caption(
            f"Current project: `{st.session_state['current_project']}`. "
            "Questions will search only within this project."
        )

        question = st.text_input(
            "Ask a question about the ingested content:",
            value="What is the main goal of the Vector Insight Engine project?",
        )

        top_k = st.slider(
            "Number of context snippets to use",
            min_value=1,
            max_value=5,
            value=3,
        )

        if st.button("Get insight"):
            if not question.strip():
                st.error("Please enter a question.")
            else:
                with st.spinner(
                    "Retrieving context from Qdrant and querying Gemini..."
                ):
                    result = answer_question(
                        question,
                        top_k=top_k,
                        project=st.session_state["current_project"],
                    )

                answer_text = result["answer"]
                st.session_state["history"].append(
                    {
                        "question": question,
                        "answer": answer_text,
                        "project": st.session_state["current_project"],
                    }
                )

                st.markdown("### Answer")
                st.write(answer_text)

                with st.expander("Show retrieved context"):
                    for hit in result["hits"]:
                        st.markdown(
                            f"[{hit['index']}] "
                            f"(score: {hit['score']:.4f}) "
                            f"[project: {hit['project']} | doc: {hit['document_name']}] "
                            f"{hit['text']}"
                        )

    # History section
    st.markdown("---")
    st.subheader("Session history")

    if not st.session_state["history"]:
        st.write("No questions asked yet.")
    else:
        for i, item in enumerate(reversed(st.session_state["history"]), start=1):
            st.markdown(
                f"**Q{i} [{item['project']}]** {item['question']}"
            )
            st.write(item["answer"])
            st.markdown("")


if __name__ == "__main__":
    main()
