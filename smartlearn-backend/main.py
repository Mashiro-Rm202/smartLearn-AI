import os

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pypdf.errors import PyPdfError

from services.rag import (
    answer_chat_turn,
    build_upload_response,
    extract_pages_from_bytes_for_rag,
    prepare_rag_chat_record,
)

app = FastAPI(title="SmartLearn Lite API")

ALLOWED_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
)

# 内存存储，按 chat_id 组织 (Day 3: 存储 RAG 文档记录而非页面列表)
documents: dict[str, dict] = {}

UPLOAD_ROOT = os.path.join(
    os.path.dirname(__file__), "uploads",
)


class ChatRequest(BaseModel):
    message: str
    smart_mode: bool = False


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/")
async def root():
    return {"message": "SmartLearn Lite API is running"}


@app.post("/upload")
async def upload_pdf(chat_id: str = Query(...), file: UploadFile = File(...)):
    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Validate PDF
    try:
        pages = extract_pages_from_bytes_for_rag(contents)
    except PyPdfError:
        raise HTTPException(status_code=400, detail="File is not a valid PDF")

    if not pages or sum(len(p["text"]) for p in pages) == 0:
        raise HTTPException(
            status_code=422,
            detail="PDF contains no extractable text — OCR is not supported",
        )

    # Build the Day 3 RAG record
    try:
        record = prepare_rag_chat_record(
            chat_id=chat_id,
            filename=file.filename or "upload.pdf",
            pdf_bytes=contents,
            pages=pages,
            upload_root=UPLOAD_ROOT,
            artifact_root=os.path.join(
                os.path.dirname(__file__), "artifacts", "rag",
            ),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"Upstream LLM error: {e}")

    documents[chat_id] = record
    return build_upload_response(record)


@app.get("/documents/{chat_id}/file")
async def serve_pdf(chat_id: str):
    """Serve the uploaded PDF so the frontend preview can open it."""
    if chat_id not in documents:
        raise HTTPException(
            status_code=404,
            detail="No PDF uploaded for this chat_id",
        )
    saved_path = documents[chat_id].get("saved_pdf_path")
    if not saved_path or not os.path.isfile(saved_path):
        raise HTTPException(status_code=404, detail="Saved PDF file not found")
    return FileResponse(saved_path, media_type="application/pdf")


@app.post("/chat")
async def chat(chat_id: str = Query(...), body: ChatRequest = ...):
    if chat_id not in documents:
        raise HTTPException(
            status_code=404,
            detail="No PDF uploaded for this chat_id — upload a PDF first",
        )

    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        result = answer_chat_turn(
            documents[chat_id],
            body.message,
            smart_mode=body.smart_mode,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"Upstream LLM error: {e}")

    return {
        "chat_id": chat_id,
        "answer": result["answer"],
        "citations": result["citations"],
        "sources": result["sources"],
    }
