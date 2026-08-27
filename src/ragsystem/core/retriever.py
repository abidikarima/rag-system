from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chromadb

from ragsystem.core.rag_pipeline.vectorizer import (
    DoclingChunkEmbedder,
)


@dataclass
class RetrievedChunk:
    """
    A single result returned by the retriever.
    """

    id: str
    text: str
    distance: float | None
    metadata: dict[str, Any]


class ChromaRetriever:
    """
    Retriever for the Chroma collection created by the RAG ingestion pipeline.
    """

    def __init__(
        self,
        chroma_dir: str = "./chroma_db",
        collection_name: str = "documents",
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        top_k: int = 5,
    ):
        self.top_k = top_k

        self.client = chromadb.PersistentClient(
            path=chroma_dir
        )

        self.collection = self.client.get_collection(
            name=collection_name
        )

        self.embedder = DoclingChunkEmbedder(
            model_name=model_name,
            batch_size=32,
            normalize_embeddings=True,
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:

        if not query or not query.strip():
            return []

        k = top_k if top_k is not None else self.top_k

        # ---------------------------------------------------------
        # Embed query
        # ---------------------------------------------------------

        query_embedding = self._embed_query(query)

        # ---------------------------------------------------------
        # Search Chroma
        # ---------------------------------------------------------

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        chunks: list[RetrievedChunk] = []

        for index, chunk_id in enumerate(ids):

            chunks.append(
                RetrievedChunk(
                    id=chunk_id,
                    text=documents[index],
                    distance=(
                        distances[index]
                        if index < len(distances)
                        else None
                    ),
                    metadata=(
                        metadatas[index]
                        if index < len(metadatas)
                        else {}
                    ),
                )
            )

        return chunks

    def _embed_query(
        self,
        query: str,
    ) -> list[float]:

        # embed_vectors expects a sequence.
        # Therefore pass [query], not query.
        embedding = self.embedder.embed_vectors(
            [query]
        )

        # embedding shape is:
        # (1, embedding_dimension)

        return embedding[0].tolist()
