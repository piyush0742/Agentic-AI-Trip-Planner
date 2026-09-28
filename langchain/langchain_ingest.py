"""
INGESTION SCRIPT (LangChain version) — run ONCE, or whenever the PDF changes.

Reads trip_guide_expanded.pdf (10 East/West coast cities), splits it into
sentence-based chunks (same approach as the raw project - avoids cutting
sentences mid-way), embeds each chunk using Ollama's nomic-embed-text
(kept local/free even though chat now uses Vertex AI), and stores
everything in a persistent Chroma vector database.

Run with:
    pip install -r requirements.txt
    python3 ingest.py
"""

import langchain_chroma
import pdfplumber
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

PDF_PATH = "trip_guide_expanded.pdf"
SENTENCES_PER_CHUNK = 3
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "trip_guide"


def extract_text_from_pdf(filepath):
    full_text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"
    return full_text


def split_into_sentences(text):
    text = text.replace("\n", " ")
    raw_sentences = text.split(". ")
    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if s:
            if not s.endswith("."):
                s += "."
            sentences.append(s)
    return sentences


def group_sentences_into_chunks(sentences, sentences_per_chunk):
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        group = sentences[i:i + sentences_per_chunk]
        chunks.append(" ".join(group))
    return chunks





print("Reading PDF...")
pdf_text = extract_text_from_pdf(PDF_PATH)
sentences = split_into_sentences(pdf_text)
chunks = group_sentences_into_chunks(sentences, SENTENCES_PER_CHUNK)
print(f"Split into {len(sentences)} sentences, grouped into {len(chunks)} chunks.")

embeddings = GoogleGenerativeAIEmbeddings(
      model="gemini-embedding-001",
      project="agentic-ai-510016",   # jo variable pehle se define hai
      vertexai=True,
  )

print("Computing embeddings and storing in Chroma...")
vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DB_PATH,
    collection_metadata={"hnsw:space": "cosine"},  # same fix as raw project needed
)

print(f"Done! {len(chunks)} chunks saved to '{CHROMA_DB_PATH}'.")
