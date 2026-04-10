from openai import OpenAI
from backend.config import OPENAI_API_KEY, CHAT_MODEL
from backend.retriever import retrieve_relevant_chunks

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided document context.

Rules:
- Base your answers primarily on the context provided below
- Always cite your source using the filename and page number like this: [filename, page X]
- If asked what a document is about, summarize the context you have been given
- If the answer is truly not in the context, say: "I don't have enough information in the uploaded documents to answer that."
- Be concise and direct
- Never make up information that contradicts the context"""

def build_prompt(query: str, chunks: list[dict], history: list[dict]) -> list[dict]:
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Source {i}: {chunk['filename']}, page {chunk['page']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Document context:\n\n{context}"},
    ]

    for turn in history:
        messages.append({"role": "user",      "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})

    messages.append({"role": "user", "content": query})
    return messages

def get_answer(query: str, history: list[dict] = []) -> dict:
    chunks = retrieve_relevant_chunks(query)

    if not chunks:
        return {
            "answer":      "No documents have been uploaded yet. Please upload a PDF first.",
            "sources":     [],
            "chunks_used": 0,
        }

    messages = build_prompt(query, chunks, history)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk["filename"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "filename": chunk["filename"],
                "page":     chunk["page"],
                "score":    chunk["score"],
            })

    return {
        "answer":      answer,
        "sources":     sources,
        "chunks_used": len(chunks),
    }
