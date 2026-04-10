import pdfplumber
import chromadb
from openai import OpenAI
from backend.config import (
    OPENAI_API_KEY, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP,
    CHROMA_DB_PATH, COLLECTION_NAME
)

client = OpenAI(api_key=OPENAI_API_KEY)

def get_chroma_collection():
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return db.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    "page": page_num,
                    "text": text.strip()
                })
    return pages

def chunk_text(pages: list[dict], filename: str) -> list[dict]:
    chunks = []
    chunk_id = 0
    for page_data in pages:
        text = page_data["text"]
        page_num = page_data["page"]
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]
            if chunk.strip():
                chunks.append({
                    "id":       f"{filename}_p{page_num}_c{chunk_id}",
                    "text":     chunk,
                    "filename": filename,
                    "page":     page_num,
                })
                chunk_id += 1
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    texts = [c["text"] for c in chunks]
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]

def is_document_ingested(filename: str) -> bool:
    collection = get_chroma_collection()
    results = collection.get(where={"filename": filename})
    return len(results["ids"]) > 0

def ingest_document(pdf_path: str, filename: str) -> int:
    if is_document_ingested(filename):
        print(f"{filename} already ingested — skipping")
        collection = get_chroma_collection()
        return collection.count()

    collection = get_chroma_collection()

    print(f"Extracting text from {filename}...")
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        raise ValueError(f"No text extracted from {filename}. Is it a scanned PDF?")

    print(f"Chunking {len(pages)} pages...")
    chunks = chunk_text(pages, filename)

    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embed_chunks(chunks)

    print("Storing in ChromaDB...")
    collection.add(
        ids        = [c["id"]       for c in chunks],
        documents  = [c["text"]     for c in chunks],
        embeddings = embeddings,
        metadatas  = [{"filename": c["filename"], "page": c["page"]} for c in chunks]
    )

    print(f"Done — {len(chunks)} chunks stored for {filename}")
    return len(chunks)
