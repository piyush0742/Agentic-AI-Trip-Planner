"""
DIAGNOSTIC SCRIPT (LangChain version) - tests ONLY the Chroma retrieval,
no LLM/agent involved. Isolates whether search itself is working well,
or whether an issue is actually in the agent's reasoning/tool-calling.

Run with:
    python3 diagnose.py
"""

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "trip_guide"


class PrefixedOllamaEmbeddings(OllamaEmbeddings):
    """Must match ingest.py and chat.py exactly."""
    def embed_documents(self, texts):
        prefixed = ["search_document: " + t for t in texts]
        return super().embed_documents(prefixed)

    def embed_query(self, text):
        return super().embed_query("search_query: " + text)


embeddings = PrefixedOllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_PATH,
)

# Print collection info first - confirms how many chunks are actually
# stored, and (indirectly) whether ingest.py ran successfully.
collection = vectorstore._collection  # accessing the underlying chromadb collection
print(f"Collection count: {collection.count()}")
print(f"Collection metadata: {collection.metadata}")
print("-" * 60)

test_queries = [
    "sports in Seattle",
    "food in Portland",
    "Golden Gate Bridge",
    "historic places in Philadelphia",
    "sports on the West Coast",   # deliberately vague - tests the relevance threshold
    "The Golden Gate Bridge is San Francisco's most iconic landmark, best viewed from Battery Spencer across the bay or by walking or biking across it.",  # EXACT sentence from the PDF - should score near-perfect if embeddings are aligned correctly
]

for query in test_queries:
    print(f"\nQuery: '{query}'")

    # (document, score) pairs - score closer to 1.0 = more similar,
    # closer to 0 = less similar. Same threshold idea used in chat.py.
    results = vectorstore.similarity_search_with_relevance_scores(query, k=3)

    for i, (document, score) in enumerate(results):
        print(f"  [{i+1}] score={score:.4f}")
        print(f"      {document.page_content[:100]}...")