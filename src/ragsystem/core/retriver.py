from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chromadb

from ragsystem.core.vectorizer import DoclingChunkEmbedder


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

        # ---------------------------------------------------------
        # Chroma
        # ---------------------------------------------------------

        self.client = chromadb.PersistentClient(
            path=chroma_dir
        )

        self.collection = self.client.get_collection(
            name=collection_name
        )

        # ---------------------------------------------------------
        # Embedding model
        # ---------------------------------------------------------

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
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query:
                User's natural-language query.

            top_k:
                Number of chunks to retrieve.
                Falls back to the value configured on the class.

            where:
                Optional Chroma metadata filter.

        Returns:
            List of RetrievedChunk objects ordered by similarity.
        """

        if not query or not query.strip():
            return []

        k = top_k or self.top_k

        # ---------------------------------------------------------
        # Embed query
        # ---------------------------------------------------------

        # DoclingChunkEmbedder.embed() expects chunks rather than
        # plain strings, so use its underlying embedding interface
        # if available.
        query_embedding = self._embed_query(query)

        # ---------------------------------------------------------
        # Similarity search
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

    def _embed_query(self, query: str) -> list[float]:
        """
        Convert a query into the same embedding space used during
        document ingestion.
        """

        # Adapt this method to the public API exposed by your
        # DoclingChunkEmbedder.
        #
        # The preferred implementation is for the embedder to expose
        # something like:
        #
        #     embedder.embed_query(query)
        #
        if hasattr(self.embedder, "embed_query"):
            return self.embedder.embed_query(query)

        raise NotImplementedError(
            "DoclingChunkEmbedder must expose an "
            "`embed_query()` method for query retrieval."
        )
