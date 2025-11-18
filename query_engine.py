import os
from typing import List, Dict, Optional

import google.genai as genai
from google.genai.types import GenerateContentConfig, EmbedContentConfig
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

COLLECTION_NAME = "vector_insight_chunks"
EMBEDDING_MODEL = "text-embedding-004"
GEMINI_MODEL = "gemini-2.0-flash"


def get_gemini_client() -> genai.Client:
    """Create a Gemini client using the API key from environment variables."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment.")
    return genai.Client(api_key=api_key)


def get_qdrant_client() -> QdrantClient:
    """
    Create a local Qdrant client.

    This uses embedded Qdrant storage in the current working directory,
    so it works both on your laptop and on Streamlit Community Cloud
    without any external Qdrant Cloud credentials.
    """
    print("[query] Using local Qdrant at ./qdrant_data")
    return QdrantClient(
        path="./qdrant_data",
        prefer_grpc=False,
    )


def embed_query(text: str) -> List[float]:
    """Embed a user question as a retrieval query vector."""
    client = get_gemini_client()
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values


def search_similar_chunks(
    question: str,
    top_k: int = 3,
    project: Optional[str] = None,
) -> List[Dict]:
    """
    Search Qdrant for the most relevant chunks to the given question.

    Returns a list of dictionaries with:
      - index (1 based, for citations)
      - score
      - text
      - project
      - document_name
    """
    qdrant = get_qdrant_client()
    query_vector = embed_query(question)

    query_filter = None
    if project:
        # Filter by project label so each project is its own semantic space
        query_filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="project",
                    match=rest.MatchValue(value=project),
                )
            ]
        )

    print("[query] searching similar chunks, top_k =", top_k)

    # Very important: use the argument name "query_filter", not "filter"
    resp = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    hits: List[Dict] = []
    for i, point in enumerate(resp.points, start=1):
        payload = point.payload or {}
        hits.append(
            {
                "index": i,
                "score": point.score,
                "text": payload.get("text", ""),
                "project": payload.get("project", ""),
                "document_name": payload.get("document_name", ""),
            }
        )

    return hits


def build_context_block(hits: List[Dict]) -> str:
    """
    Build a numbered context block for Gemini from Qdrant hits.

    Example:

    [1] Some text snippet here
    [2] Another relevant paragraph here
    """
    lines: List[str] = []
    for hit in hits:
        idx = hit["index"]
        text = hit["text"].replace("\n", " ").strip()
        if not text:
            continue
        lines.append(f"[{idx}] {text}")
    return "\n".join(lines)


def call_gemini_with_context(question: str, context_block: str) -> str:
    """
    Call Gemini with the user question and the retrieved context snippets.
    """
    prompt = f"""
You are an analytical assistant working in a retrieval augmented system.

You receive:
1. A user question.
2. A set of numbered context snippets from a vector database.
3. Each snippet may include different parts of one or more documents.

Your job:
- Answer the question only using the context snippets.
- Cite snippets in square brackets like [1], [2] wherever you use them.
- If the context is not sufficient, clearly say you are not sure instead of guessing.
- Keep the answer concise and focused on practical insight.

User question:
{question}

Context snippets:
{context_block}
"""

    client = get_gemini_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=GenerateContentConfig(
            temperature=0.35,
            max_output_tokens=512,
        ),
    )
    return (response.text or "").strip()


def answer_question(
    question: str,
    top_k: int = 3,
    project: Optional[str] = None,
) -> Dict:
    """
    Main entry point used by the Streamlit app.

    Returns a dictionary with:
      - answer: str
      - hits: list of {index, score, text, project, document_name}
      - raw_context: str
    """
    hits = search_similar_chunks(question, top_k=top_k, project=project)
    context_block = build_context_block(hits)

    if not context_block:
        answer_text = (
            "I could not find any relevant context in the current project. "
            "Try ingesting more documents first."
        )
        return {
            "answer": answer_text,
            "hits": [],
            "raw_context": "",
        }

    answer = call_gemini_with_context(question, context_block)

    return {
        "answer": answer,
        "hits": hits,
        "raw_context": context_block,
    }


if __name__ == "__main__":
    demo_question = "What is the main goal of the Vector Insight Engine project?"
    result = answer_question(demo_question, top_k=3, project="demo")
    print("Answer:")
    print(result["answer"])
    print("\nContext:")
    print(result["raw_context"])
