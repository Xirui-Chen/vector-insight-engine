# smoke_test.py
# This file checks that the Gemini API key and Qdrant connection work correctly.

import os

from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# 1. Load environment variables from the .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "vector_insight_chunks_test")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the .env file")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("QDRANT_URL or QDRANT_API_KEY is not set in the .env file")


def test_gemini():
    """
    Simple check that we can call Gemini and get a response.
    """
    print("Testing Gemini API...")
    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say a short greeting for a data science hackathon project.",
    )
    print("Gemini response:")
    print(response.text)
    print("Gemini API call succeeded.\n")


def test_qdrant():
    """
    Simple check that we can connect to Qdrant and create a collection.
    """
    print("Testing Qdrant connection...")
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    # We will create a small collection with 768 dimensional vectors.
    # Later we will set the dimension to match the embedding model we use.
    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qmodels.VectorParams(
                size=768,  # placeholder dimension
                distance=qmodels.Distance.COSINE,
            ),
        )
        print(f"Created collection: {QDRANT_COLLECTION}")
    else:
        print(f"Collection already exists: {QDRANT_COLLECTION}")

    print("Qdrant connection test succeeded.\n")


if __name__ == "__main__":
    test_gemini()
    test_qdrant()
    print("All smoke tests passed.")