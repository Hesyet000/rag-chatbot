import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.ingest import ingest_document, is_document_ingested
from backend.chat import get_answer
from backend.retriever import get_chroma_collection

app = FastAPI(
    title="RAG Chatbot API",
    description="Upload PDFs and ask questions grounded in their content",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    history:  list[dict] = []

class ChatResponse(BaseModel):
    answer:      str
    sources:     list[dict]
    chunks_used: int

# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    if is_document_ingested(file.filename):
        return {
            "message":  f"{file.filename} was already uploaded and indexed",
            "filename": file.filename,
            "chunks":   0,
            "skipped":  True,
        }

    # Save uploaded file to a temp location then ingest
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        chunk_count = ingest_document(tmp_path, file.filename)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {
        "message":  f"Successfully indexed {file.filename}",
        "filename": file.filename,
        "chunks":   chunk_count,
        "skipped":  False,
    }

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = get_answer(request.question, request.history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

    return ChatResponse(
        answer      = result["answer"],
        sources     = result["sources"],
        chunks_used = result["chunks_used"],
    )

@app.get("/documents")
def list_documents():
    try:
        collection = get_chroma_collection()
        results    = collection.get(include=["metadatas"])
        filenames  = sorted(set(m["filename"] for m in results["metadatas"]))
        return {"documents": filenames, "total": len(filenames)}
    except Exception:
        return {"documents": [], "total": 0}
