import os
from typing import List, Dict

import google.genai as genai
from google.genai.types import EmbedContentConfig
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from dotenv import load_dotenv
from uuid import uuid4

COLLECTION_NAME = "vector_insight_chunks"
EMBEDDING_MODEL = "text-embedding-004"

# Simple character based chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 160


def get_gemini_client() -> genai.Client:
    """Create a Gemini client using the API key from environment variables."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment.")
    return genai.Client(api_key=api_key)


def get_qdrant_client() -> QdrantClient:
    """
    Create a Qdrant client using URL and API key from environment variables.

    We load .env here to make sure QDRANT_URL and QDRANT_API_KEY are available
    even when this module is used outside of Streamlit.
    """
    load_dotenv(".env")

    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY")

    print(f"[ingest] Using Qdrant URL: {url}")

    return QdrantClient(
        url=url,
        api_key=api_key,
        timeout=10.0,
    )


def split_into_chunks(text: str) -> List[str]:
    """
    Split long text into overlapping chunks.
    This is simple but good enough for our hackathon project.
    """
    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    length = len(text)

    step = CHUNK_SIZE - CHUNK_OVERLAP
    if step <= 0:
        step = CHUNK_SIZE

    print(
        f"[ingest] split_into_chunks: length={length}, "
        f"chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, step={step}"
    )

    while start < length:
        end = min(start + CHUNK_SIZE, length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += step

    print(f"[ingest] split_into_chunks: created {len(chunks)} chunks.")

    return chunks


def embed_text(text: str) -> List[float]:
    """Use Gemini embedding model to turn text into a vector."""
    client = get_gemini_client()
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return result.embeddings[0].values


def ensure_payload_indexes(qdrant: QdrantClient) -> None:
    """
    Ensure we have keyword indexes on 'project' and 'document_name'
    so that filters in search() work correctly on Qdrant Cloud.
    """
    print(
        "[ingest] ensure_payload_indexes: creating index on "
        "project and document_name if missing ..."
    )

    # Index for project
    try:
        qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="project",
            field_schema=rest.PayloadSchemaType.KEYWORD,
        )
        print("[ingest] Created payload index on 'project'.")
    except Exception as e:
        print(
            "[ingest] create_payload_index for 'project' may already exist "
            f"or failed: {repr(e)}"
        )

    # Index for document_name
    try:
        qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="document_name",
            field_schema=rest.PayloadSchemaType.KEYWORD,
        )
        print("[ingest] Created payload index on 'document_name'.")
    except Exception as e:
        print(
            "[ingest] create_payload_index for 'document_name' may already exist "
            f"or failed: {repr(e)}"
        )


def init_collection() -> None:
    """
    Create the Qdrant collection if it does not exist yet,
    and make sure we have payload indexes for filters.
    """
    print("[ingest] init_collection: creating Qdrant client ...")
    qdrant = get_qdrant_client()

    print("[ingest] init_collection: calling get_collections() ...")
    collections = qdrant.get_collections().collections
    names = {c.name for c in collections}
    print("[ingest] init_collection: existing collections:", names)

    if COLLECTION_NAME not in names:
        print(
            f"[ingest] Collection {COLLECTION_NAME} does not exist yet, creating ..."
        )
        example_vector = embed_text("test")
        vector_size = len(example_vector)

        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=rest.VectorParams(
                size=vector_size,
                distance=rest.Distance.COSINE,
            ),
        )
        print("[ingest] Collection created.")
    else:
        print(f"[ingest] Collection {COLLECTION_NAME} already exists.")

    # Important: make sure payload indexes exist so filters on project work
    ensure_payload_indexes(qdrant)


def ingest_text(
    text: str,
    project: str = "default",
    document_name: str = "manual",
) -> int:
    """
    Split text into chunks, embed them, and write to Qdrant.

    Returns the number of chunks written.
    """
    print("[ingest] ingest_text: start.")
    print("[ingest] ingest_text: calling init_collection() ...")
    init_collection()
    print("[ingest] ingest_text: init_collection() done.")

    qdrant = get_qdrant_client()

    print("[ingest] ingest_text: splitting text into chunks ...")
    chunks = split_into_chunks(text)
    print(f"[ingest] ingest_text: number of chunks = {len(chunks)}")

    if not chunks:
        print("[ingest] No chunks to ingest.")
        return 0

    points: List[rest.PointStruct] = []

    for idx, chunk in enumerate(chunks, start=1):
        print(f"[ingest] Embedding chunk {idx}/{len(chunks)} ...")
        vector = embed_text(chunk)

        payload: Dict = {
            "text": chunk,
            "project": project,
            "document_name": document_name,
            "chunk_index": idx - 1,
        }

        # Generate a unique string id for each point
        point_id = str(uuid4())

        points.append(
            rest.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        )

    print(
        f"[ingest] Calling qdrant.upsert(...) with points = {len(points)}"
    )

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        wait=True,
        points=points,
    )

    print(
        f"[ingest] Done. Ingested {len(chunks)} chunks into collection "
        f"{COLLECTION_NAME} for project={project}, document={document_name}"
    )

    return len(chunks)


if __name__ == "__main__":
    # Small manual test when running this file directly
    sample = (
        "Machine learning models depend on the quality and coverage of their"
        " training data. Poor inputs lead to unstable predictions."
    )
    written = ingest_text(sample, project="cli-test", document_name="sample")
    print(f"Ingested {written} chunks.")
