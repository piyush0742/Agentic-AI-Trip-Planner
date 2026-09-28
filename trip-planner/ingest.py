"""
INGESTION SCRIPT — run this ONCE (or whenever trip_guide.pdf changes).

This is the missing piece from before: separating "build the knowledge
base" (slow, only needs to happen occasionally) from "chat with the
user" (fast, happens every time). The chat script will no longer
re-read the PDF or re-compute embeddings - it'll just query what THIS
script already saved to disk.

Run with:
    pip install chromadb pdfplumber requests
    python3 ingest.py
"""

import requests
import pdfplumber
import chromadb

EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL_NAME = "nomic-embed-text"
PDF_PATH = "trip_guide_expanded.pdf"
SENTENCES_PER_CHUNK = 3  # group 3 full sentences per chunk - no more mid-sentence cuts
CHROMA_DB_PATH = "./chroma_db"  # a folder Chroma will create, holds the saved vectors


def extract_text_from_pdf(filepath):
    full_text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"
    return full_text


def split_into_sentences(text):
    """Splits text on '. ' (period + space) instead of on word count.
    This keeps every sentence WHOLE - fixing the mid-sentence cuts we
    saw with fixed word-count chunking, which was hurting search quality."""
    text = text.replace("\n", " ")  # treat line breaks as just spaces
    raw_sentences = text.split(". ")

    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if s:
            if not s.endswith("."):
                s += "."  # split() removes the period - add it back
            sentences.append(s)
    return sentences


def group_sentences_into_chunks(sentences, sentences_per_chunk):
    """Groups whole sentences together, a fixed NUMBER of sentences at
    a time (instead of a fixed number of words). Every chunk now ends
    on a real sentence boundary."""
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        group = sentences[i:i + sentences_per_chunk]
        chunks.append(" ".join(group))
    return chunks


def get_embedding(text):
    response = requests.post(
        EMBED_URL,
        json={"model": EMBED_MODEL_NAME, "prompt": text},
        timeout=60
    )
    response.raise_for_status()
    return response.json()["embedding"]


print("Reading PDF...")
pdf_text = extract_text_from_pdf(PDF_PATH)
sentences = split_into_sentences(pdf_text)
chunks = group_sentences_into_chunks(sentences, SENTENCES_PER_CHUNK)
print(f"Split into {len(sentences)} sentences, grouped into {len(chunks)} chunks.")

# ---------------------------------------------------------------------
# PersistentClient means Chroma SAVES to disk (in CHROMA_DB_PATH),
# instead of only living in memory while the script runs. This is
# what makes it survive between separate script runs.
# ---------------------------------------------------------------------
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# get_or_create_collection: if this collection already exists from a
# previous run, we reuse it (and will overwrite below). A "collection"
# is just Chroma's name for a table/group of documents.
#
# IMPORTANT: metadata={"hnsw:space": "cosine"} - by DEFAULT, Chroma uses
# L2 (Euclidean) distance, not cosine similarity. Our embeddings
# (from nomic-embed-text) are NOT normalized to length 1, so L2
# distance gives genuinely wrong rankings. We must explicitly request
# cosine here, and it can only be set at CREATION time (not changed later).
collection = client.get_or_create_collection(
    name="trip_guide",
    metadata={"hnsw:space": "cosine"}
)

print("Computing embeddings and storing in Chroma...")
for i, chunk_text in enumerate(chunks):
    # IMPORTANT: nomic-embed-text is an "asymmetric" embedding model -
    # it needs to know whether text is a DOCUMENT or a QUERY, using a
    # prefix. Without this, search results come back wrong/misaligned.
    embedding = get_embedding("search_document: " + chunk_text)

    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[chunk_text]  # store the ORIGINAL text, without the prefix
    )

print(f"Done! {len(chunks)} chunks saved to '{CHROMA_DB_PATH}'.")
print("You do NOT need to run this again unless trip_guide.pdf changes.")