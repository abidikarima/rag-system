"""
RAG chatbot API.

POST /chat      question (+ conversation history) -> answer with numbered
                citations + the references it cites
GET  /documents overview of the indexed documents (groups, main topics)
GET  /health    status and collection size
GET  /docs   interactive page to test the API in a browser

Retrieval uses the best method of the evaluation, "Common first: embedding
+ sparse": dense embedding + SPLADE sparse vector on the chunk text, chunks
found by both signals ranked first (see docs/rag_evaluation.md).

Each message first goes through a router (one LLM call) that uses the
conversation to:
    - rewrite follow-ups ("and in 2023?") into a standalone search question
    - answer greetings directly, and decline off-topic requests
      (poems, code, ...) without searching the documents

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
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ragsystem.core.dense_sparse_retriever import DenseSparseRetriever, ScoredChunk
from ragsystem.prompts.chatbot import (
    CHATBOT_PROMPT,
    COLLECTION_DESCRIPTION,
    QUERY_ROUTER_PROMPT,
)
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

# Conversation sent to the LLM: last messages only, each one shortened.
MAX_HISTORY_MESSAGES = 8
MAX_HISTORY_CHARS = 1200

# Document overview shown in the interface. The OHR-Bench category is not
# stored in Chroma, so documents are grouped by their file name.
DOCUMENT_GROUPS = (
    ("News", "Newspaper pages (May to August 2024)", re.compile(r"^\d{2}_\d{2}\.pdf$")),
    ("Contracts", "Commercial agreements filed with the SEC", re.compile(r"agreement", re.IGNORECASE)),
    ("Financial reports", "Company 10-K and 10-Q filings", re.compile(r"10K|10Q")),
    ("Scientific papers", "arXiv research papers", re.compile(r"^\d{4}\.\d{4,5}(v\d+)?\.pdf$")),
)
OTHER_GROUP = ("Other documents", "Forms, reports, manuals and administrative documents")

TOPICS_PER_GROUP = 8
TOPICS_OVERALL = 16

# Keywords produced by the metadata extractor that are not real topics.
IGNORED_KEYWORDS = {
    "volume", "number", "clxxiii", "array", "hne", "new york times",
    "the new york times", "page", "table", "figure", "contact information",
    "date", "agreement", "accuracy", "acknowledgments", "canada prices",
}


# =============================================================
# Request / response models
# =============================================================

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    history: list[Message] = Field(
        default_factory=list,
        max_length=50,
        description="Previous messages of the conversation, oldest first",
    )
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
    intent: Literal["documents", "chitchat", "off_topic"] = Field(
        ..., description="documents = answered from the documents"
    )
    search_query: str | None = Field(
        None, description="Standalone question used for the search"
    )
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

    state["overview"] = build_overview(state["retriever"].metadatas)

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


def format_history(history: list[Message]) -> str:
    lines = []

    for message in history[-MAX_HISTORY_MESSAGES:]:
        content = " ".join(message.content.split())
        if len(content) > MAX_HISTORY_CHARS:
            content = content[:MAX_HISTORY_CHARS] + "…"
        lines.append(f"{message.role.upper()}: {content}")

    return "\n".join(lines) or "(no previous messages)"


def route(llm: GroqLLM, question: str, history: str) -> dict[str, str]:
    """
    Classify the message and rewrite it as a standalone search question.

    Any failure falls back to a plain document search with the original
    question, so the router can never block an answer.
    """

    fallback = {"intent": "documents", "question": question, "reply": ""}

    try:
        response = llm.generate(
            QUERY_ROUTER_PROMPT.format(
                COLLECTION=COLLECTION_DESCRIPTION,
                HISTORY=history,
                QUESTION=question,
            ),
            temperature=0,
            max_tokens=LLM_MAX_TOKENS,
        )
        match = re.search(r"\{.*\}", response or "", re.DOTALL)
        result = json.loads(match.group(0)) if match else {}
    except Exception as exc:
        print(f"[WARN] Router failed, searching with the original question: {exc}")
        return fallback

    intent = result.get("intent")
    reply = str(result.get("reply") or "").strip()

    if intent in ("chitchat", "off_topic") and reply:
        return {"intent": intent, "question": "", "reply": reply}

    return {
        "intent": "documents",
        "question": str(result.get("question") or "").strip() or question,
        "reply": "",
    }


def cited_numbers(answer: str) -> set[int]:
    return {int(n) for n in re.findall(r"\[(\d+)\]", answer)}


def build_sources(chunks: list[ScoredChunk]) -> str:
    blocks = []

    for number, chunk in enumerate(chunks, start=1):
        title = chunk.metadata.get("title") or chunk.metadata.get("source") or "unknown"
        text = chunk.text.strip()[:MAX_SOURCE_CHARS]
        blocks.append(f"[{number}] (document: {title})\n{text}")

    return "\n\n".join(blocks)


def normalize_citations(answer: str) -> str:
    """
    gpt-oss models often cite as 【1】 or 【1†L1-L4】; return the [1] format
    promised by the API.
    """

    answer = re.sub(r"【\s*(\d+)[^】]*】", r"[\1]", answer)

    return re.sub(r"\[(\d+)†[^\]]*\]", r"[\1]", answer)


def document_group(file_name: str) -> tuple[str, str]:
    for name, description, pattern in DOCUMENT_GROUPS:
        if pattern.search(file_name):
            return name, description

    return OTHER_GROUP


def top_topics(keyword_docs: dict[str, set[str]], labels: dict[str, Counter], limit: int) -> list[dict]:
    """
    Keywords found in the most documents, with their most frequent spelling.
    """

    ranked = sorted(keyword_docs.items(), key=lambda item: (-len(item[1]), item[0]))

    return [
        {"topic": labels[key].most_common(1)[0][0], "documents": len(docs)}
        for key, docs in ranked[:limit]
        if len(docs) > 1
    ]


def build_overview(metadatas: list[dict]) -> dict[str, Any]:
    """
    Documents, groups and main topics of the indexed collection.
    """

    groups: dict[str, dict[str, Any]] = {}
    group_of: dict[str, str] = {}
    labels: dict[str, Counter] = defaultdict(Counter)
    all_keyword_docs: dict[str, set[str]] = defaultdict(set)
    pages: dict[str, int] = defaultdict(int)

    for metadata in metadatas:
        source = metadata.get("source")
        if not source:
            continue

        file_name = source.split("/")[-1]

        if source not in group_of:
            name, description = document_group(file_name)
            group_of[source] = name
            groups.setdefault(
                name,
                {
                    "name": name,
                    "description": description,
                    "documents": set(),
                    "keyword_docs": defaultdict(set),
                },
            )
            groups[name]["documents"].add(source)

        for page in parse_json_list(metadata.get("pages")) or []:
            if isinstance(page, int):
                pages[source] = max(pages[source], page)

        try:
            keywords = json.loads(metadata.get("keywords") or "[]")
        except (TypeError, json.JSONDecodeError):
            continue

        for keyword in keywords if isinstance(keywords, list) else []:
            label = " ".join(str(keyword).split())
            key = label.lower()

            if len(key) < 3 or key in IGNORED_KEYWORDS:
                continue

            labels[key][label] += 1
            all_keyword_docs[key].add(source)
            groups[group_of[source]]["keyword_docs"][key].add(source)

    group_order = [name for name, _, _ in DOCUMENT_GROUPS] + [OTHER_GROUP[0]]

    return {
        "documents": len(group_of),
        "pages": sum(pages.values()),
        "groups": [
            {
                "name": group["name"],
                "description": group["description"],
                "documents": len(group["documents"]),
                "topics": top_topics(group["keyword_docs"], labels, TOPICS_PER_GROUP),
            }
            for name in group_order
            if (group := groups.get(name))
        ],
        "topics": top_topics(all_keyword_docs, labels, TOPICS_OVERALL),
    }


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
        "retrieval": "dense + sparse, common first",
        "api_key_required": bool(API_KEY),
    }


@app.get("/documents")
def documents() -> dict[str, Any]:
    """
    Overview of the indexed documents: groups and main topics.
    """

    if "overview" not in state:
        raise HTTPException(status_code=503, detail="Index is still loading.")

    return state["overview"]


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(check_api_key)])
def chat(request: ChatRequest) -> ChatResponse:
    start = time.perf_counter()

    retriever: DenseSparseRetriever = state["retriever"]
    llm: GroqLLM = state["llm"]

    question = request.question.strip()
    history = format_history(request.history)

    def response(**fields: Any) -> ChatResponse:
        return ChatResponse(
            question=request.question,
            latency_ms=int((time.perf_counter() - start) * 1000),
            **fields,
        )

    routed = route(llm, question, history)

    # Greetings and off-topic requests: no search, no references.
    if routed["intent"] != "documents":
        return response(intent=routed["intent"], answer=routed["reply"], references=[])

    search_query = routed["question"]
    chunks = retriever.search(search_query, top_k=request.top_k)
    references = [to_reference(n, chunk) for n, chunk in enumerate(chunks, start=1)]

    if not chunks:
        return response(
            intent="documents",
            search_query=search_query,
            answer=None,
            references=[],
            error="No document found for this question.",
        )

    prompt = CHATBOT_PROMPT.format(
        HISTORY=history,
        SOURCES=build_sources(chunks),
        QUESTION=search_query,
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
        return response(
            intent="documents",
            search_query=search_query,
            answer=None,
            references=references,
            error=f"Answer generation failed: {exc}",
        )

    # Only the sources the answer actually cites; none when the answer
    # says the documents do not contain the information.
    cited = cited_numbers(answer)

    return response(
        intent="documents",
        search_query=search_query,
        answer=answer,
        references=[ref for ref in references if ref.number in cited],
    )
