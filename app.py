"""
app.py — Production FastAPI entry point for the RAG Chatbot.
Run with: uvicorn app:app --reload --port 8000
"""
import sys
import os

# Ensure rag_app directory is importable
_root = os.path.dirname(os.path.abspath(__file__))
_rag  = os.path.join(_root, "rag_app")
for _p in [_root, _rag]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import asyncio
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

# ── RAG modules  (resolved via sys.path above) ────────────────────────────
from config   import get_settings
from database import Neo4jDatabase
from chain    import RAGChain, ingest_document

settings = get_settings()

# ── FastAPI app ─────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-ready RAG Chatbot powered by Ollama + Neo4j Graph DB",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (chatbot UI)
static_dir = Path(_root) / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ── Lazy RAG chain ─────────────────────────────────────────────────────
_rag_chain: Optional[RAGChain] = None

def get_rag_chain() -> RAGChain:
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain

# ── Startup / shutdown ─────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    logger.info("🚀 Starting {} v{}", settings.app_name, settings.app_version)
    try:
        Neo4jDatabase.setup_schema()
        logger.info("✅ Neo4j schema ready.")
    except Exception as e:
        logger.error("Neo4j setup failed — is Neo4j running? Error: {}", e)

@app.on_event("shutdown")
async def shutdown():
    Neo4jDatabase.close()
    logger.info("Server shut down cleanly.")

# ── Pydantic models ───────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: list
    num_sources: int

# ── Routes ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    html_path = static_dir / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>RAG Bot is running. Place index.html in /static/</h1>")


@app.get("/health")
async def health():
    try:
        docs = Neo4jDatabase.list_documents()
        neo4j_ok = "ok"
    except Exception as e:
        neo4j_ok = f"error: {e}"
        docs = []

    return {
        "status": "ok",
        "neo4j": neo4j_ok,
        "documents_indexed": len(docs),
        "model": settings.ollama_model,
        "embed_model": settings.ollama_embed_model,
    }


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)):
    """Upload and ingest a PDF/TXT document into Neo4j."""
    allowed = {".pdf", ".txt", ".md"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {allowed}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, ingest_document, tmp_path)
        result["original_filename"] = file.filename
        return result
    except Exception as e:
        logger.exception("Ingestion error: {}", e)
        raise HTTPException(500, f"Ingestion failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Ask a question and receive a full answer with sources."""
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    try:
        chain = get_rag_chain()
        loop  = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, chain.query, req.question)
        return result
    except Exception as e:
        logger.exception("Chat error: {}", e)
        raise HTTPException(500, f"Chat failed: {str(e)}")


@app.get("/api/chat/stream")
async def chat_stream(question: str):
    """Stream LLM answer tokens via Server-Sent Events."""
    if not question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    chain = get_rag_chain()

    async def generate():
        try:
            loop = asyncio.get_event_loop()

            def _run():
                return list(chain.stream_query(question))

            tokens = await loop.run_in_executor(None, _run)
            for token in tokens:
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: ERROR: {e}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/documents")
async def list_documents():
    try:
        docs = Neo4jDatabase.list_documents()
        return {"documents": docs, "total": len(docs)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    try:
        deleted = Neo4jDatabase.delete_document(doc_id)
        if not deleted:
            raise HTTPException(404, f"Document '{doc_id}' not found.")
        return {"status": "deleted", "doc_id": doc_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
