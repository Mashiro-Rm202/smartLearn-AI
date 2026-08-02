import os

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf.errors import PyPdfError

from services.llm import answer_from_pages
from services.pdf import extract_pages

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

# 内存存储，按 chat_id 组织
chat_store: dict[str, list[dict]] = {}


class ChatRequest(BaseModel):
    message: str


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

    try:
        pages = extract_pages(contents)
    except PyPdfError:
        raise HTTPException(status_code=400, detail="File is not a valid PDF")
    except ValueError:
        raise HTTPException(status_code=400, detail="PDF exceeds 30 page limit")

    total_chars = sum(len(p["text"]) for p in pages)

    if total_chars == 0:
        raise HTTPException(status_code=422, detail="PDF contains no extractable text — OCR is not supported")

    chat_store[chat_id] = pages
    return {
        "status": "ok",
        "filename": file.filename,
        "pages": len(pages),
        "characters": total_chars,
    }


@app.post("/chat")
async def chat(chat_id: str = Query(...), body: ChatRequest = ...):
    if chat_id not in chat_store:
        raise HTTPException(
            status_code=404,
            detail="No PDF uploaded for this chat_id — upload a PDF first",
        )

    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        answer = answer_from_pages(chat_store[chat_id], body.message)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"Upstream LLM error: {e}")

    return {
        "chat_id": chat_id,
        "answer": answer,
    }
