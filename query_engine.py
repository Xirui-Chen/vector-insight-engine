import os
from typing import List, Dict

import google.genai as genai
from google.genai.types import EmbedContentConfig, GenerateContentConfig
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from dotenv import load_dotenv

COLLECTION_NAME = "vector_insight_chunks"
EMBEDDING_MODEL = "text-embedding-004"
GENERATION_MODEL = "gemini-2.0-flash"


def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment.")
    return genai.Client(api_key=api_key)


def get_qdrant_client() -> QdrantClient:
    """
    Same logic as ingest.py

    Local dev:
      USE_LOCAL_QDRANT is not set or is "0" -> use Qdrant Cloud from .env
    Streamlit Cloud:
      USE_LOCAL_QDRANT = "1" in secrets -> use embedded Qdrant at ./qdrant_data
    """
    use_local = os.getenv("USE_LOCAL_QDRANT", "0") == "1"

    if not use_local:
        load_dotenv(".env")

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if use_local or not url:
        print("[query] Using local Qdrant at ./qdrant_data")
        return QdrantClient(path="qdrant_data")

    print(f"[query] Using remote Qdrant at {url}")
    return QdrantClient(
        url=url,
        api_key=api_key,
        timeout=10.0,
    )


def embed_query(text: str) -> List[float]:
    """
    Embed a user question as a retrieval query vector.
    """
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
    project: str | None = None,
) -> List[Dict]:
    """
    Search Qdrant for the chunks most relevant to the question.

    Returns a list of dict hits:
    {
        "index": int,
        "score": float,
        "text": str,
        "project": str,
        "document_name": str,
    }
    """
    qdrant = get_qdrant_client()
    query_vector = embed_query(question)

    query_filter = None
    if project:
        query_filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="project",
                    match=rest.MatchValue(value=project),
                )
            ]
        )

    print("[query] searching similar chunks, top_k =", top_k)

    # Newer clients have query_points, older ones only have search.
    if hasattr(qdrant, "query_points"):
        resp = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            filter=query_filter,
        )
        raw_hits = resp.points
    else:
        resp = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )
        raw_hits = resp

    hits: List[Dict] = []

    for idx, point in enumerate(raw_hits, start=1):
        payload = point.payload or {}
        hits.append(
            {
                "index": idx,
                "score": point.score,
                "text": payload.get("text", ""),
                "project": payload.get("project", ""),
                "document_name": payload.get("document_name", ""),
            }
        )

    return hits


def build_context_string(hits: List[Dict]) -> str:
    """
    Turn retrieved chunks into a numbered context block for Gemini.
    """
    lines: List[str] = []
    for hit in hits:
        idx = hit["index"]
        text = hit["text"]
        lines.append(f"[{idx}] {text}")
    return "\n\n".join(lines)


def call_gemini_with_context(
    question: str,
    context: str,
) -> str:
    """
    Ask Gemini to answer the question using only the provided context.
    """
    client = get_gemini_client()

    prompt = f"""
You are an AI research assistant.

You receive a user question and several numbered context snippets
retrieved from a vector database.

Rules:
- Answer the question using only the information in the context.
- When you use a snippet, cite it in square brackets like [1] or [2][3].
- If the answer is not in the context, say you do not know based on the provided documents.

Context:
{context}

Question:
{question}
"""

    resp = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=512,
        ),
    )
    return resp.text.strip()


def answer_question(
    question: str,
    top_k: int = 3,
    project: str | None = None,
) -> Dict:
    """
    Main entry point called from the Streamlit app.

    Returns:
    {
        "answer": str,
        "hits": List[Dict],
    }
    """
    hits = search_similar_chunks(question, top_k=top_k, project=project)
    context = build_context_string(hits)
    answer = call_gemini_with_context(question, context)

    return {
        "answer": answer,
        "hits": hits,
    }


if __name__ == "__main__":
    # Small CLI test
    q = "What is the main goal of the Vector Insight Engine project?"
    res = answer_question(q, top_k=3, project="demo")
    print("Answer:\n", res["answer"])
    print("\nHits:")
    for h in res["hits"]:
        print(h["index"], h["score"], h["document_name"])
