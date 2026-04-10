import chromadb
from openai import OpenAI
from backend.config import (
    OPENAI_API_KEY, EMBEDDING_MODEL,
    CHROMA_DB_PATH, COLLECTION_NAME, TOP_K_RESULTS
)

client = OpenAI(api_key=OPENAI_API_KEY)

def get_chroma_collection():
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    # Use cosine similarity explicitly
    return db.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

def embed_query(query: str) -> list[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query]
    )
    return response.data[0].embedding

def retrieve_relevant_chunks(query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
    collection = get_chroma_collection()

    if collection.count() == 0:
        return []

    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        # With cosine space: distance is 0 (identical) to 2 (opposite)
        # Convert to similarity: 1 - (distance / 2) gives 0 to 1
        distance = results["distances"][0][i]
        score = round(1 - (distance / 2), 4)
        chunks.append({
            "text":     results["documents"][0][i],
            "filename": results["metadatas"][0][i]["filename"],
            "page":     results["metadatas"][0][i]["page"],
            "score":    score,
        })

    return chunks
