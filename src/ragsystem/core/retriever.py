from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import re

import chromadb
from rank_bm25 import BM25Okapi

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

    # Hybrid retrieval information
    bm25_score: float | None = None
    vector_score: float | None = None
    hybrid_score: float | None = None

    # Where the document came from
    retrieval_sources: list[str] | None = None


class ChromaRetriever:
    """
    Hybrid BM25 + Vector retriever.

    Retrieval pipeline:

        Query
          |
          +---- BM25 -------------------+
          |                              |
          +---- Vector / Chroma ---------+
                                         |
                                  Merge candidates
                                         |
                                  Common documents first
                                         |
                                  Hybrid reranking
                                         |
                                  Return top-k
    """

    def __init__(
        self,
        chroma_dir: str = "./chroma_db",
        collection_name: str = "documents",
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        top_k: int = 5,

        # Number of candidates retrieved by each method
        bm25_top_k: int = 10,
        vector_top_k: int = 10,

        # Weight used during final hybrid ranking
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
    ):
        self.top_k = top_k

        self.bm25_top_k = bm25_top_k
        self.vector_top_k = vector_top_k

        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        if bm25_weight < 0 or vector_weight < 0:
            raise ValueError(
                "bm25_weight and vector_weight must be >= 0"
            )

        if bm25_weight + vector_weight == 0:
            raise ValueError(
                "At least one hybrid weight must be > 0"
            )

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

        # ---------------------------------------------------------
        # BM25 index
        # ---------------------------------------------------------

        self._bm25 = None
        self._bm25_ids: list[str] = []
        self._bm25_documents: list[str] = []
        self._bm25_metadatas: list[dict[str, Any]] = []

        self._build_bm25_index()

    # =========================================================
    # PUBLIC API
    # =========================================================

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Basic vector retrieval.

        Kept for backwards compatibility.
        """

        if not query or not query.strip():
            return []

        k = top_k if top_k is not None else self.top_k

        query_embedding = self._embed_query(query)

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

        return self._parse_chroma_results(result)

    # =========================================================
    # HYBRID SEARCH
    # =========================================================

    def hybrid_search(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
        bm25_top_k: int | None = None,
        vector_top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Hybrid BM25 + vector search.

        Strategy:

        1. Retrieve BM25 candidates.
        2. Retrieve vector candidates.
        3. Find documents appearing in both.
        4. Give common documents priority.
        5. Merge remaining candidates from both methods.
        6. Calculate normalized BM25 + vector scores.
        7. Rerank.
        8. Return exactly top_k documents.
        """

        if not query or not query.strip():
            return []

        k = top_k if top_k is not None else self.top_k

        bm25_k = (
            bm25_top_k
            if bm25_top_k is not None
            else self.bm25_top_k
        )

        vector_k = (
            vector_top_k
            if vector_top_k is not None
            else self.vector_top_k
        )

        # ---------------------------------------------------------
        # 1. BM25 retrieval
        # ---------------------------------------------------------

        bm25_results = self._bm25_search(
            query=query,
            top_k=bm25_k,
            where=where,
        )

        # ---------------------------------------------------------
        # 2. Vector retrieval
        # ---------------------------------------------------------

        vector_results = self._vector_search(
            query=query,
            top_k=vector_k,
            where=where,
        )

        if not bm25_results and not vector_results:
            return []

        # ---------------------------------------------------------
        # 3. Index results by ID
        # ---------------------------------------------------------

        bm25_by_id = {
            result.id: result
            for result in bm25_results
        }

        vector_by_id = {
            result.id: result
            for result in vector_results
        }

        bm25_ids = set(bm25_by_id)
        vector_ids = set(vector_by_id)

        common_ids = bm25_ids & vector_ids

        # ---------------------------------------------------------
        # 4. Merge candidates
        #
        # Common documents are explicitly preferred.
        # ---------------------------------------------------------

        candidates: dict[str, RetrievedChunk] = {}

        # First: documents found by BOTH retrievers
        for doc_id in common_ids:

            bm25_doc = bm25_by_id[doc_id]
            vector_doc = vector_by_id[doc_id]

            candidate = self._merge_results(
                bm25_doc=bm25_doc,
                vector_doc=vector_doc,
            )

            candidates[doc_id] = candidate

        # ---------------------------------------------------------
        # 5. Add unique BM25/vector documents
        # ---------------------------------------------------------

        for result in bm25_results:

            if result.id not in candidates:

                candidates[result.id] = result

        for result in vector_results:

            if result.id not in candidates:

                candidates[result.id] = result

        # ---------------------------------------------------------
        # 6. Normalize scores
        # ---------------------------------------------------------

        self._normalize_bm25_scores(
            list(candidates.values())
        )

        self._normalize_vector_scores(
            list(candidates.values())
        )

        # ---------------------------------------------------------
        # 7. Calculate hybrid score
        # ---------------------------------------------------------

        total_weight = (
            self.bm25_weight +
            self.vector_weight
        )

        for candidate in candidates.values():

            bm25_score = (
                candidate.bm25_score
                if candidate.bm25_score is not None
                else 0.0
            )

            vector_score = (
                candidate.vector_score
                if candidate.vector_score is not None
                else 0.0
            )

            score = (
                self.bm25_weight * bm25_score
                + self.vector_weight * vector_score
            ) / total_weight

            # Explicit bonus when both retrievers found
            # the same document.
            if candidate.id in common_ids:
                score += 0.10

            candidate.hybrid_score = score

        # ---------------------------------------------------------
        # 8. Rank
        # ---------------------------------------------------------

        ranked = sorted(
            candidates.values(),
            key=lambda x: (
                x.hybrid_score
                if x.hybrid_score is not None
                else 0.0
            ),
            reverse=True,
        )

        # ---------------------------------------------------------
        # 9. Return top-k
        # ---------------------------------------------------------

        return ranked[:k]

    # =========================================================
    # BM25
    # =========================================================

    def _build_bm25_index(self) -> None:
        """
        Load all Chroma documents and build BM25 index.

        For a large collection, this should eventually be moved
        to a persistent BM25 index rather than rebuilt at startup.
        """

        data = self.collection.get(
            include=[
                "documents",
                "metadatas",
            ]
        )

        ids = data.get("ids", [])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])

        if not documents:
            self._bm25 = None
            return

        self._bm25_ids = ids
        self._bm25_documents = documents
        self._bm25_metadatas = metadatas

        tokenized_documents = [
            self._tokenize(document or "")
            for document in documents
        ]

        self._bm25 = BM25Okapi(
            tokenized_documents
        )

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:

        if self._bm25 is None:
            return []

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self._bm25.get_scores(
            query_tokens
        )

        # ---------------------------------------------------------
        # Sort BM25 scores
        # ---------------------------------------------------------

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results: list[RetrievedChunk] = []

        for index in ranked_indexes:

            if len(results) >= top_k:
                break

            # -----------------------------------------------------
            # Apply metadata filtering.
            #
            # BM25 is implemented locally, so where filtering
            # needs to happen here.
            # -----------------------------------------------------

            metadata = (
                self._bm25_metadatas[index]
                if index < len(self._bm25_metadatas)
                else {}
            )

            if not self._matches_where(
                metadata,
                where,
            ):
                continue

            document_id = self._bm25_ids[index]

            results.append(
                RetrievedChunk(
                    id=document_id,
                    text=self._bm25_documents[index],
                    distance=None,
                    metadata=metadata,
                    bm25_score=float(scores[index]),
                    vector_score=None,
                    hybrid_score=None,
                    retrieval_sources=["bm25"],
                )
            )

        return results

    # =========================================================
    # VECTOR SEARCH
    # =========================================================

    def _vector_search(
        self,
        query: str,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:

        query_embedding = self._embed_query(query)

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        chunks = self._parse_chroma_results(
            result
        )

        for chunk in chunks:

            if chunk.distance is not None:

                # Chroma cosine distance:
                #
                # similarity ~= 1 - distance
                #
                # This is useful for ranking but we normalize
                # again before combining with BM25.

                chunk.vector_score = (
                    1.0 - chunk.distance
                )

            chunk.retrieval_sources = [
                "vector"
            ]

        return chunks

    # =========================================================
    # MERGING
    # =========================================================

    def _merge_results(
        self,
        bm25_doc: RetrievedChunk,
        vector_doc: RetrievedChunk,
    ) -> RetrievedChunk:
        """
        Merge the same document found by BM25 and vector search.
        """

        return RetrievedChunk(
            id=bm25_doc.id,
            text=(
                vector_doc.text
                or bm25_doc.text
            ),
            distance=vector_doc.distance,
            metadata=(
                vector_doc.metadata
                or bm25_doc.metadata
            ),
            bm25_score=bm25_doc.bm25_score,
            vector_score=vector_doc.vector_score,
            hybrid_score=None,
            retrieval_sources=[
                "bm25",
                "vector",
            ],
        )

    # =========================================================
    # SCORE NORMALIZATION
    # =========================================================

    def _normalize_bm25_scores(
        self,
        candidates: list[RetrievedChunk],
    ) -> None:
        """
        Min-max normalize BM25 scores to [0, 1].
        """

        scores = [
            c.bm25_score
            for c in candidates
            if c.bm25_score is not None
        ]

        if not scores:
            return

        min_score = min(scores)
        max_score = max(scores)

        if math.isclose(
            min_score,
            max_score,
        ):
            for candidate in candidates:

                if candidate.bm25_score is not None:
                    candidate.bm25_score = 1.0

            return

        for candidate in candidates:

            if candidate.bm25_score is not None:

                candidate.bm25_score = (
                    candidate.bm25_score - min_score
                ) / (
                    max_score - min_score
                )

    def _normalize_vector_scores(
        self,
        candidates: list[RetrievedChunk],
    ) -> None:
        """
        Normalize vector scores to [0, 1].
        """

        scores = [
            c.vector_score
            for c in candidates
            if c.vector_score is not None
        ]

        if not scores:
            return

        min_score = min(scores)
        max_score = max(scores)

        if math.isclose(
            min_score,
            max_score,
        ):
            for candidate in candidates:

                if candidate.vector_score is not None:
                    candidate.vector_score = 1.0

            return

        for candidate in candidates:

            if candidate.vector_score is not None:

                candidate.vector_score = (
                    candidate.vector_score - min_score
                ) / (
                    max_score - min_score
                )

    # =========================================================
    # CHROMA RESULT PARSER
    # =========================================================

    def _parse_chroma_results(
        self,
        result: dict[str, Any],
    ) -> list[RetrievedChunk]:

        ids = result.get(
            "ids",
            [[]],
        )[0]

        documents = result.get(
            "documents",
            [[]],
        )[0]

        metadatas = result.get(
            "metadatas",
            [[]],
        )[0]

        distances = result.get(
            "distances",
            [[]],
        )[0]

        chunks: list[RetrievedChunk] = []

        for index, chunk_id in enumerate(ids):

            chunks.append(
                RetrievedChunk(
                    id=chunk_id,
                    text=(
                        documents[index]
                        if index < len(documents)
                        else ""
                    ),
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
                    bm25_score=None,
                    vector_score=None,
                    hybrid_score=None,
                    retrieval_sources=[],
                )
            )

        return chunks

    # =========================================================
    # EMBEDDING
    # =========================================================

    def _embed_query(
        self,
        query: str,
    ) -> list[float]:

        # Qwen3-Embedding expects queries to be encoded with its
        # "query" instruction prompt (documents are encoded without).
        embedding = self.embedder.embed_vectors(
            [query],
            prompt_name="query",
        )

        return embedding[0].tolist()

    # =========================================================
    # TOKENIZATION
    # =========================================================

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Simple BM25 tokenizer.

        Keeps words and numbers such as:
            prior
            approval
            HNE
            800
            842
            4464
        """

        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )

    # =========================================================
    # METADATA FILTER
    # =========================================================

    @staticmethod
    def _matches_where(
        metadata: dict[str, Any],
        where: dict[str, Any] | None,
    ) -> bool:

        if not where:
            return True

        for key, expected_value in where.items():

            actual_value = metadata.get(key)

            if isinstance(
                expected_value,
                dict,
            ):
                for operator, value in expected_value.items():

                    if operator == "$eq":
                        if actual_value != value:
                            return False

                    elif operator == "$ne":
                        if actual_value == value:
                            return False

                    elif operator == "$in":
                        if actual_value not in value:
                            return False

                    elif operator == "$nin":
                        if actual_value in value:
                            return False

                    else:
                        raise ValueError(
                            f"Unsupported where operator: {operator}"
                        )

            else:

                if actual_value != expected_value:
                    return False

        return True
