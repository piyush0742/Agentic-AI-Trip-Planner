"""
DIAGNOSTIC SCRIPT - tests ONLY the Chroma retrieval, no LLM involved.
This isolates whether the search itself is fixed, or if something else
is going wrong.

Run with:
    python3 diagnose.py
"""

import requests
import chromadb

EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL_NAME = "nomic-embed-text"
CHROMA_DB_PATH = "./chroma_db"


def get_embedding(text):
    response = requests.post(
        EMBED_URL,
        json={"model": EMBED_MODEL_NAME, "prompt": text},
        timeout=60
    )
    response.raise_for_status()
    return response.json()["embedding"]


client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name="trip_guide")

# Print collection info FIRST - this tells us if the cosine fix
# actually took effect, and how many chunks are actually stored.
print(f"Collection count: {collection.count()}")
print(f"Collection metadata: {collection.metadata}")
print("-" * 60)

test_queries = [
    "San Francisco",
    "Golden Gate Bridge",
    "places to explore in San Francisco",
    "The Golden Gate Bridge is San Francisco's most iconic landmark, best viewed from Battery Spencer across the bay or by walking or biking across it.",
]
for query in test_queries:
    print(f"\nQuery: '{query}'")
    query_embedding = get_embedding("search_query: " + query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3  # top 3, so we can see if the RIGHT answer is at least close
    )

    for i in range(len(results["documents"][0])):
        doc_text = results["documents"][0][i]
        distance = results["distances"][0][i]
        print(f"  [{i+1}] distance={distance:.4f}")
        print(f"      {doc_text[:100]}...")