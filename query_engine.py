import os
from typing import List, Dict, Optional

import google.genai as genai
from google.genai.types import GenerateContentConfig
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

COLLECTION_NAME = "vector_insight_chunks"

GEMINI_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "text-embedding-004"


def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment.")
    return genai.Client(api_key=api_key)


def get_qdrant_client() -> QdrantClient:
    use_local = os.getenv("USE_LOCAL_QDRANT", "0") == "1"
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if use_local or not url:
        print("[query] Using local Qdrant at ./qdrant_data")
        return QdrantClient(path="qdrant_data")

    print(f"[query] Using remote Qdrant at {url}")
    return QdrantClient(url=url, api_key=api_key, timeout=10)


def embed_text(text: str) -> List[float]:
    client = get_gemini_client()
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return result.embeddings[0].values


def ensure_collection_exists() -> None:
    qdrant = get_qdrant_client()
    collections = qdrant.get_collections().collections
    names = {c.name for c in collections}
    if COLLECTION_NAME not in names:
        # Create with a test embedding to obtain the dimension
        example_vector = embed_text("test")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=rest.VectorParams(
                size=len(example_vector),
                distance=rest.Distance.COSINE,
            ),
        )


def search_similar_chunks(
    query_text: str,
    top_k: int = 3,
    project: Optional[str] = None,
) -> List[Dict]:
    """
    Embed the query and search for similar chunks in Qdrant.

    If project is provided, restrict to that project label.
    """
    ensure_collection_exists()
    qdrant = get_qdrant_client()
    query_vector = embed_text(query_text)

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

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=top_k,
    )

    hits: List[Dict] = []
    for point in results:
        hits.append(
            {
                "score": float(point.score),
                "text": point.payload.get("text", ""),
                "project": point.payload.get("project", ""),
                "document_name": point.payload.get("document_name", ""),
            }
        )
    return hits


def build_context_block(hits: List[Dict]) -> str:
    lines = []
    for idx, hit in enumerate(hits, start=1):
        lines.append(f"[{idx}] {hit['text']}")
    return "\n\n".join(lines)


def answer_question(
    question: str,
    top_k: int = 3,
    project: Optional[str] = None,
) -> Dict:
    """
    Full RAG pipeline.

    Returns:
    - answer: str
    - hits: list of {index, score, text, project, document_name}
    - raw_context: str
    """
    hits = search_similar_chunks(question, top_k=top_k, project=project)
    context_block = build_context_block(hits)

    prompt = f"""
You are an AI research assistant.

You receive:
1. A user question.
2. A context block made from retrieved snippets, each numbered like [1], [2].

Use only this context to answer.
If the context is insufficient, say you cannot answer confidently.
Cite snippets in square brackets, for example [1] or [1][2].

Current project label: {project}

User question:
{question}

Retrieved context:
{context_block}
"""

    client = get_gemini_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=512,
        ),
    )

    answer_text = response.text.strip()

    indexed_hits = []
    for idx, hit in enumerate(hits, start=1):
        indexed_hits.append(
            {
                "index": idx,
                "score": hit["score"],
                "text": hit["text"],
                "project": hit.get("project", ""),
                "document_name": hit.get("document_name", ""),
            }
        )

    return {
        "answer": answer_text,
        "hits": indexed_hits,
        "raw_context": context_block,
    }


if __name__ == "__main__":
    project_label = input("Project label to query (empty for all): ").strip() or None
    user_q = input("Question: ")
    result = answer_question(user_q, top_k=3, project=project_label)
    print("\nAnswer:\n")
    print(result["answer"])
    print("\nRetrieved context:\n")
    for h in result["hits"]:
        print(
            f"[{h['index']}] "
            f"(score: {h['score']:.4f}) "
            f"[project: {h['project']} | doc: {h['document_name']}] "
            f"{h['text']}"
        )
