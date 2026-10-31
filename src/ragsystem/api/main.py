"""
RAG chatbot API.

POST /chat   question -> answer with numbered citations + references
GET  /health status and collection size
GET  /docs   interactive page to test the API in a browser

Retrieval uses the best method of the evaluation: dense embedding +
SPLADE sparse vector on the chunk text (see docs/rag_evaluation.md).

Run from the repository root:

    uv run uvicorn ragsystem.api.main:app --host 0.0.0.0 --port 8000

Configuration (environment variables):

    RAG_CHROMA_DIR    Chroma folder           (default ./chroma_db_v1)
    RAG_COLLECTION    collection name         (default documents)
    RAG_GROQ_CONFIG   file with GROQ_API_KEY  (default ./secrets/config.yaml)
    RAG_API_KEY       if set, every /chat request must send it in the
                      X-API-Key header (protects your Groq quota)
"""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ragsystem.core.dense_sparse_retriever import DenseSparseRetriever, ScoredChunk
from ragsystem.prompts.chatbot import CHATBOT_PROMPT
from ragsystem.utils.llm_connector import GroqLLM


CHROMA_DIR = os.getenv("RAG_CHROMA_DIR", "./chroma_db_v1")
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "documents")
GROQ_CONFIG = os.getenv("RAG_GROQ_CONFIG", "./secrets/config.yaml")
API_KEY = os.getenv("RAG_API_KEY")

# Characters of each chunk sent to the LLM, and shown as excerpt.
MAX_SOURCE_CHARS = 2500
EXCERPT_CHARS = 300

LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 1024


# =============================================================
# Request / response models
# =============================================================

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(5, ge=1, le=10, description="Number of chunks used as sources")


class Reference(BaseModel):
    number: int = Field(..., description="Number used in the answer, e.g. [1]")
    chunk_id: str
    source: str | None
    title: str | None
    pages: list[int] | None
    headings: list[str] | None
    score: float
    excerpt: str


class ChatResponse(BaseModel):
    question: str
    answer: str | None
    references: list[Reference]
    error: str | None = None
    latency_ms: int


# =============================================================
# Startup: load models and index once
# =============================================================

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading retriever from {CHROMA_DIR} ...")
    state["retriever"] = DenseSparseRetriever(
        chroma_dir=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )
    print(f"Retriever ready: {state['retriever'].size} chunks.")

    state["llm"] = GroqLLM(config_path=GROQ_CONFIG)
    yield
    state.clear()


app = FastAPI(
    title="RAG Chatbot API",
    description=(
        "Ask a question about the document collection. The answer cites its "
        "sources as [1], [2]... and the matching references are returned."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow a browser front-end hosted elsewhere to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def check_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


# =============================================================
# Helpers
# =============================================================

def parse_json_list(value: Any) -> list | None:
    """
    Lists are stored as JSON strings in Chroma metadata.
    """

    if value is None:
        return None

    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, list) else None


def build_sources(chunks: list[ScoredChunk]) -> str:
    blocks = []

    for number, chunk in enumerate(chunks, start=1):
        title = chunk.metadata.get("title") or chunk.metadata.get("source") or "unknown"
        text = chunk.text.strip()[:MAX_SOURCE_CHARS]
        blocks.append(f"[{number}] (document: {title})\n{text}")

    return "\n\n".join(blocks)


def normalize_citations(answer: str) -> str:
    """
    gpt-oss models often cite as 【1】; return the [1] format promised by the API.
    """

    return re.sub(r"【\s*(\d+)\s*】", r"[\1]", answer)


def to_reference(number: int, chunk: ScoredChunk) -> Reference:
    excerpt = " ".join(chunk.text.split())

    return Reference(
        number=number,
        chunk_id=chunk.id,
        source=chunk.metadata.get("source"),
        title=chunk.metadata.get("title"),
        pages=parse_json_list(chunk.metadata.get("pages")),
        headings=parse_json_list(chunk.metadata.get("headings")),
        score=round(chunk.score, 4),
        excerpt=excerpt[:EXCERPT_CHARS] + ("…" if len(excerpt) > EXCERPT_CHARS else ""),
    )


# =============================================================
# Endpoints
# =============================================================

CHAT_PAGE = Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def chat_page() -> FileResponse:
    """
    Web chat interface calling /chat.
    """

    return FileResponse(CHAT_PAGE)


@app.get("/health")
def health() -> dict[str, Any]:
    retriever: DenseSparseRetriever | None = state.get("retriever")

    return {
        "status": "ok" if retriever else "loading",
        "chunks": retriever.size if retriever else 0,
        "chunks_without_sparse_vector": retriever.missing_sparse if retriever else 0,
        "retrieval": "dense + sparse (50/50)",
        "api_key_required": bool(API_KEY),
    }


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(check_api_key)])
def chat(request: ChatRequest) -> ChatResponse:
    start = time.perf_counter()

    retriever: DenseSparseRetriever = state["retriever"]
    llm: GroqLLM = state["llm"]

    chunks = retriever.search(request.question, top_k=request.top_k)
    references = [to_reference(n, chunk) for n, chunk in enumerate(chunks, start=1)]

    answer, error = None, None

    if not chunks:
        error = "No document found for this question."
    else:
        prompt = CHATBOT_PROMPT.format(
            SOURCES=build_sources(chunks),
            QUESTION=request.question.strip(),
        )

        try:
            answer = normalize_citations(
                llm.generate(
                    prompt,
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                ).strip()
            )
        except Exception as exc:
            # Still return the references so the caller sees what was found.
            error = f"Answer generation failed: {exc}"

    return ChatResponse(
        question=request.question,
        answer=answer,
        references=references,
        error=error,
        latency_ms=int((time.perf_counter() - start) * 1000),
    )
