from __future__ import annotations

import json
import re

from dataclasses import dataclass
from typing import Any, Iterator, Literal, Optional

from ragsystem.prompts.metadata_extractor import EXTRACTOR_METADATA
from ragsystem.utils.llm_connector import GroqLLM

from docling.chunking import (
    HierarchicalChunker,
    HybridChunker,
)

from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)

from docling_core.transforms.serializer.markdown import (
    MarkdownTableSerializer,
)

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocItemLabel


ChunkerType = Literal[
    "hierarchical",
    "hybrid",
    "line",
]


class MarkdownTableSerializerProvider(
    ChunkingSerializerProvider
):
    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
        )


@dataclass
class ChunkingOptions:
    chunker: ChunkerType = "hybrid"

    # ---------------------------------------------------------
    # Page filtering
    # ---------------------------------------------------------

    pages: Optional[list[int]] = None

    # ---------------------------------------------------------
    # Document metadata
    # ---------------------------------------------------------

    source: Optional[str] = None
    title: Optional[str] = None

    # ---------------------------------------------------------
    # HierarchicalChunker
    # ---------------------------------------------------------

    merge_list_items: bool = True

    # ---------------------------------------------------------
    # HybridChunker
    # ---------------------------------------------------------

    max_tokens: Optional[int] = None
    merge_peers: bool = True
    repeat_table_header: bool = True
    omit_header_on_overflow: bool = False

    # ---------------------------------------------------------
    # Other
    # ---------------------------------------------------------

    max_tokens_per_chunk: Optional[int] = None
    prefix: Optional[str] = None
    omit_prefix_on_overflow: bool = False

    tokenizer: Any = None
    serializer_provider: Any = None

    # ---------------------------------------------------------
    # Small chunk grouping
    # ---------------------------------------------------------

    max_words: int = 250
    small_chunk_words: int = 100
    min_group_words: int = 250


class DoclingChunker:

    def __init__(
        self,
        options: Optional[ChunkingOptions] = None,
    ):
        self.options = options or ChunkingOptions()

        self._validate_options()
        self._chunker = self._build_chunker()

        self.llm = GroqLLM()

    # =========================================================
    # Validation
    # =========================================================

    def _validate_options(self):

        pages = self.options.pages

        if pages is not None:

            if not pages:
                raise ValueError(
                    "pages cannot be empty. "
                    "Use pages=None for all pages."
                )

            invalid = [
                page
                for page in pages
                if not isinstance(page, int) or page < 1
            ]

            if invalid:
                raise ValueError(
                    f"Invalid page numbers: {invalid}"
                )

            self.options.pages = list(
                dict.fromkeys(pages)
            )

        if self.options.max_words <= 0:
            raise ValueError(
                "max_words must be greater than 0."
            )

        if self.options.small_chunk_words <= 0:
            raise ValueError(
                "small_chunk_words must be greater than 0."
            )

        if self.options.min_group_words <= 0:
            raise ValueError(
                "min_group_words must be greater than 0."
            )

    # =========================================================
    # Build chunker
    # =========================================================

    def _build_chunker(self):

        opts = self.options

        if opts.chunker == "hierarchical":

            kwargs = {
                "merge_list_items": opts.merge_list_items,
            }

            if opts.serializer_provider is not None:
                kwargs["serializer_provider"] = (
                    opts.serializer_provider
                )

            return HierarchicalChunker(
                **kwargs
            )

        if opts.chunker == "hybrid":

            kwargs = {
                "merge_peers": opts.merge_peers,
                "repeat_table_header": (
                    opts.repeat_table_header
                ),
                "omit_header_on_overflow": (
                    opts.omit_header_on_overflow
                ),
            }

            if opts.tokenizer is not None:
                kwargs["tokenizer"] = opts.tokenizer

            if opts.max_tokens is not None:
                kwargs["max_tokens"] = opts.max_tokens

            if opts.serializer_provider is not None:
                kwargs["serializer_provider"] = (
                    opts.serializer_provider
                )

            return HybridChunker(
                **kwargs
            )

        if opts.chunker == "line":

            raise NotImplementedError(
                "Line chunker is not implemented."
            )

        raise ValueError(
            f"Unsupported chunker: {opts.chunker!r}"
        )

    # =========================================================
    # Raw chunks
    # =========================================================

    def _raw_chunks(
        self,
        document: DoclingDocument,
    ):

        selected_pages = self.options.pages

        if selected_pages is None:

            yield from self._chunker.chunk(
                document
            )

            return

        selected_pages = set(
            selected_pages
        )

        for chunk in self._chunker.chunk(
            document
        ):

            chunk_pages = self.pages(
                chunk
            )

            if selected_pages.intersection(
                chunk_pages
            ):

                yield chunk

    # =========================================================
    # Chunk
    # =========================================================

    def chunk(
        self,
        document: DoclingDocument,
    ):

        raw_chunks = list(
            self._raw_chunks(
                document
            )
        )

        grouped_chunks = (
            self._group_small_chunks(
                raw_chunks
            )
        )

        yield from grouped_chunks

    # =========================================================
    # Chunks
    # =========================================================

    def chunks(
        self,
        document: DoclingDocument,
    ) -> list:

        return list(
            self.chunk(
                document
            )
        )

    # =========================================================
    # Word Count
    # =========================================================

    @staticmethod
    def _word_count(
        text: str,
    ) -> int:

        if not text:
            return 0

        return len(
            re.findall(
                r"\S+",
                text,
            )
        )

    # =========================================================
    # Chunk Text
    # =========================================================

    def _chunk_text(
        self,
        chunk,
    ) -> str:

        grouped_text = getattr(
            chunk,
            "text",
            None,
        )

        if grouped_text:
            return str(
                grouped_text
            ).strip()

        text = self._chunker.contextualize(
            chunk
        )

        if text is None:
            return ""

        return str(
            text
        ).strip()

    # =========================================================
    # Chunk Type
    #
    # IMPORTANT:
    # Type is determined by Docling, NOT by the LLM.
    # =========================================================

    @staticmethod
    def _chunk_kind(
        chunk,
    ) -> str:

        grouped_kind = getattr(
            chunk,
            "_kind",
            None,
        )

        if grouped_kind:
            return grouped_kind

        meta = getattr(
            chunk,
            "meta",
            None,
        )

        if meta is None:
            return "text"

        labels = []

        for item in getattr(
            meta,
            "doc_items",
            [],
        ):

            label = getattr(
                item,
                "label",
                None,
            )

            if label is not None:

                labels.append(
                    str(
                        label
                    ).lower()
                )

        # -----------------------------------------------------
        # Table
        # -----------------------------------------------------

        if any(
            "table" in label
            for label in labels
        ):
            return "table"

        # -----------------------------------------------------
        # Images / figures
        # -----------------------------------------------------

        if any(
            (
                "picture" in label
                or "figure" in label
                or "image" in label
                or "chart" in label
            )
            for label in labels
        ):
            return "image"

        return "text"

    # =========================================================
    # Unique values preserving order
    # =========================================================

    @staticmethod
    def _unique_preserve_order(
        values: list,
    ) -> list:

        result = []
        seen = set()

        for value in values:

            if value is None:
                continue

            try:
                key = value
                hash(key)

            except TypeError:
                key = repr(value)

            if key in seen:
                continue

            seen.add(key)

            result.append(
                value
            )

        return result

    # =========================================================
    # Can Merge
    # =========================================================

    def _can_merge(
        self,
        current,
        candidate,
        current_words: int,
    ) -> bool:

        current_kind = (
            self._chunk_kind(
                current
            )
        )

        candidate_kind = (
            self._chunk_kind(
                candidate
            )
        )

        # Different types cannot be merged.
        if current_kind != candidate_kind:
            return False

        # Tables and images remain independent.
        if current_kind != "text":
            return False

        candidate_text = (
            self._chunk_text(
                candidate
            )
        )

        candidate_words = (
            self._word_count(
                candidate_text
            )
        )

        if candidate_words == 0:
            return False

        if (
            current_words
            + candidate_words
            > self.options.max_words
        ):
            return False

        return True

    # =========================================================
    # Merge Docling Metadata
    # =========================================================

    def _merge_chunk_metadata(
        self,
        chunks: list,
    ):

        if not chunks:
            return None

        class GroupedMeta:
            pass

        merged_meta = GroupedMeta()

        # -----------------------------------------------------
        # DOC ITEMS
        # -----------------------------------------------------

        all_doc_items = []

        for chunk in chunks:

            meta = getattr(
                chunk,
                "meta",
                None,
            )

            if meta is None:
                continue

            for item in getattr(
                meta,
                "doc_items",
                [],
            ):

                if item not in all_doc_items:

                    all_doc_items.append(
                        item
                    )

        merged_meta.doc_items = (
            all_doc_items
        )

        # -----------------------------------------------------
        # HEADINGS
        # -----------------------------------------------------

        all_headings = []

        for chunk in chunks:

            meta = getattr(
                chunk,
                "meta",
                None,
            )

            if meta is None:
                continue

            headings = getattr(
                meta,
                "headings",
                None,
            )

            if not headings:
                continue

            if isinstance(
                headings,
                (list, tuple),
            ):

                all_headings.extend(
                    headings
                )

            else:

                all_headings.append(
                    headings
                )

        merged_meta.headings = (
            self._unique_preserve_order(
                all_headings
            )
        )

        # -----------------------------------------------------
        # CAPTIONS
        # -----------------------------------------------------

        all_captions = []

        for chunk in chunks:

            meta = getattr(
                chunk,
                "meta",
                None,
            )

            if meta is None:
                continue

            captions = getattr(
                meta,
                "captions",
                None,
            )

            if not captions:
                continue

            if isinstance(
                captions,
                (list, tuple),
            ):

                all_captions.extend(
                    captions
                )

            else:

                all_captions.append(
                    captions
                )

        merged_meta.captions = (
            self._unique_preserve_order(
                all_captions
            )
        )

        return merged_meta

    # =========================================================
    # Create Grouped Chunk
    # =========================================================

    def _create_grouped_chunk(
        self,
        chunks: list,
        texts: list,
    ):

        if len(chunks) == 1:
            return chunks[0]

        class GroupedChunk:
            pass

        grouped = GroupedChunk()

        grouped.chunks = list(
            chunks
        )

        grouped.text = "\n\n".join(
            text.strip()
            for text in texts
            if text.strip()
        )

        grouped._kind = (
            self._chunk_kind(
                chunks[0]
            )
        )

        grouped.meta = (
            self._merge_chunk_metadata(
                chunks
            )
        )

        pages = set()

        for chunk in chunks:

            pages.update(
                self.pages(
                    chunk
                )
            )

        grouped._pages = sorted(
            pages
        )

        grouped._headings = getattr(
            grouped.meta,
            "headings",
            [],
        )

        grouped._captions = getattr(
            grouped.meta,
            "captions",
            [],
        )

        return grouped

    # =========================================================
    # Group Small Successive Chunks
    # =========================================================

    def _group_small_chunks(
        self,
        chunks: list,
    ) -> list:

        if not chunks:
            return []

        result = []

        current_chunks = []
        current_texts = []
        current_words = 0

        for chunk in chunks:

            text = self._chunk_text(
                chunk
            )

            words = self._word_count(
                text
            )

            if words == 0:
                continue

            if not current_chunks:

                current_chunks = [
                    chunk
                ]

                current_texts = [
                    text
                ]

                current_words = words

                continue

            previous = (
                current_chunks[-1]
            )

            can_merge = (
                self._can_merge(
                    previous,
                    chunk,
                    current_words,
                )
            )

            is_small = (
                current_words
                < self.options.small_chunk_words
                or
                words
                < self.options.small_chunk_words
            )

            group_is_small = (
                current_words
                < self.options.min_group_words
            )

            if (
                can_merge
                and (
                    is_small
                    or group_is_small
                )
            ):

                current_chunks.append(
                    chunk
                )

                current_texts.append(
                    text
                )

                current_words += words

                continue

            result.append(
                self._create_grouped_chunk(
                    current_chunks,
                    current_texts,
                )
            )

            current_chunks = [
                chunk
            ]

            current_texts = [
                text
            ]

            current_words = words

        if current_chunks:

            result.append(
                self._create_grouped_chunk(
                    current_chunks,
                    current_texts,
                )
            )

        return result

    # =========================================================
    # Page numbers
    # =========================================================

    @staticmethod
    def pages(
        chunk,
    ) -> list[int]:

        grouped_pages = getattr(
            chunk,
            "_pages",
            None,
        )

        if grouped_pages is not None:

            return list(
                grouped_pages
            )

        pages = set()

        meta = getattr(
            chunk,
            "meta",
            None,
        )

        if meta is None:
            return []

        for item in getattr(
            meta,
            "doc_items",
            [],
        ):

            for prov in getattr(
                item,
                "prov",
                [],
            ):

                page_no = getattr(
                    prov,
                    "page_no",
                    None,
                )

                if page_no is not None:

                    pages.add(
                        page_no
                    )

        return sorted(
            pages
        )

    # =========================================================
    # Document title
    # =========================================================

    def document_title(
        self,
        document: DoclingDocument,
    ) -> Optional[str]:

        if self.options.title:
            return self.options.title

        for attribute in (
            "title",
            "name",
            "document_title",
        ):

            value = getattr(
                document,
                attribute,
                None,
            )

            if isinstance(
                value,
                str,
            ):

                value = value.strip()

                if value:
                    return value

        return None

    # =========================================================
    # Text for LLM
    # =========================================================

    def _text_for_llm(
        self,
        chunk,
    ) -> str:

        grouped_text = getattr(
            chunk,
            "text",
            None,
        )

        if grouped_text:

            return str(
                grouped_text
            ).strip()

        return self._chunk_text(
            chunk
        )

    # =========================================================
    # Extract JSON from LLM response
    # =========================================================

    @staticmethod
    def _parse_llm_json(
        response,
    ) -> Optional[dict]:

        if isinstance(
            response,
            dict,
        ):

            return response

        if response is None:
            return None

        text = str(
            response
        ).strip()

        if not text:
            return None

        # -----------------------------------------------------
        # Remove Markdown code fences
        # -----------------------------------------------------

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

        text = text.strip()

        # -----------------------------------------------------
        # Direct JSON
        # -----------------------------------------------------

        try:

            parsed = json.loads(
                text
            )

            if isinstance(
                parsed,
                dict,
            ):

                return parsed

        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ):
            pass

        # -----------------------------------------------------
        # JSON embedded inside text
        # -----------------------------------------------------

        start = text.find(
            "{"
        )

        end = text.rfind(
            "}"
        )

        if (
            start >= 0
            and end > start
        ):

            candidate = text[
                start:end + 1
            ]

            try:

                parsed = json.loads(
                    candidate
                )

                if isinstance(
                    parsed,
                    dict,
                ):

                    return parsed

            except (
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ):
                pass

        return None

    # =========================================================
    # Validate LLM Metadata
    #
    # LLM only produces:
    #
    # {
    #     "summary": "...",
    #     "keywords": [...]
    # }
    #
    # Type is determined by Docling.
    # =========================================================

    @staticmethod
    def _validate_llm_metadata(
        result,
        chunk_kind: str,
    ) -> Optional[dict]:

        if not isinstance(
            result,
            dict,
        ):

            print(
                "[DEBUG] Metadata is not a dict:",
                repr(result),
            )

            return None

        summary = result.get(
            "summary"
        )

        keywords = result.get(
            "keywords",
            [],
        )

        # -----------------------------------------------------
        # Summary
        # -----------------------------------------------------

        if not isinstance(
            summary,
            str,
        ):

            print(
                "[DEBUG] Summary is not a string:",
                repr(summary),
            )

            return None

        summary = summary.strip()

        if not summary:

            print(
                "[DEBUG] Summary is empty"
            )

            return None

        # -----------------------------------------------------
        # Reject generic LLM responses
        # -----------------------------------------------------

        normalized_summary = (
            re.sub(
                r"\s+",
                " ",
                summary.lower(),
            ).strip(
                " .!?,;:"
            )
        )

        invalid_summaries = {
            "no content provided",
            "no content",
            "no information provided",
            "no information",
            "no text provided",
            "no text",
            "unable to summarize",
            "cannot summarize",
            "cannot provide a summary",
            "not enough information",
            "insufficient information",
        }

        if normalized_summary in (
            invalid_summaries
        ):

            print(
                "[DEBUG] LLM returned generic "
                f"summary: {summary!r}"
            )

            return None

        # -----------------------------------------------------
        # Keywords
        # -----------------------------------------------------

        if not isinstance(
            keywords,
            list,
        ):

            print(
                "[DEBUG] Keywords is not a list:",
                repr(keywords),
            )

            return None

        clean_keywords = []

        for keyword in keywords:

            if keyword is None:
                continue

            keyword = str(
                keyword
            ).strip()

            if keyword:
                clean_keywords.append(
                    keyword
                )

        # -----------------------------------------------------
        # Table / image
        # -----------------------------------------------------

        if chunk_kind in (
            "table",
            "image",
        ):

            clean_keywords = []

        return {
            "type": chunk_kind,
            "summary": summary,
            "keywords": clean_keywords,
        }

    # =========================================================
    # Fallback Metadata
    # =========================================================

    def _fallback_metadata(
        self,
        chunk,
    ) -> dict:

        kind = self._chunk_kind(
            chunk
        )

        # -----------------------------------------------------
        # Table
        # -----------------------------------------------------

        if kind == "table":

            return {
                "type": "table",
                "summary": (
                    "Table containing "
                    "structured information."
                ),
                "keywords": [],
            }

        # -----------------------------------------------------
        # Image
        # -----------------------------------------------------

        if kind == "image":

            return {
                "type": "image",
                "summary": (
                    "Visual content extracted "
                    "from the document."
                ),
                "keywords": [],
            }

        # -----------------------------------------------------
        # Text fallback
        #
        # IMPORTANT:
        # Do not pretend that the first 40 words are a summary.
        # -----------------------------------------------------

        text = self._text_for_llm(
            chunk
        )

        words = text.split()

        if words:

            summary_words = words[:40]

            summary = " ".join(
                summary_words
            )

            if len(words) > 40:
                summary += "..."

        else:

            summary = (
                "Text content extracted "
                "from the document."
            )

        return {
            "type": "text",
            "summary": summary,
            "keywords": [],
        }

    # =========================================================
    # LLM Metadata Extraction
    # =========================================================

    def _extract_llm_metadata(
        self,
        chunk,
    ) -> dict:

        chunk_text = (
            self._text_for_llm(
                chunk
            )
        )

        kind = self._chunk_kind(
            chunk
        )

        # -----------------------------------------------------
        # No text
        # -----------------------------------------------------

        if not chunk_text:

            print(
                "[WARNING] Empty chunk text."
            )

            return (
                self._fallback_metadata(
                    chunk
                )
            )

        word_count = (
            self._word_count(
                chunk_text
            )
        )

        print(
            "[INFO] "
            f"Processing chunk: "
            f"{word_count} words | "
            f"pages={self.pages(chunk)} | "
            f"type={kind}"
        )

        # -----------------------------------------------------
        # Safety limit
        # -----------------------------------------------------

        words = chunk_text.split()

        if len(words) > self.options.max_words:

            chunk_text = " ".join(
                words[
                    :self.options.max_words
                ]
            )

            print(
                "[WARNING] Chunk exceeded "
                f"{self.options.max_words} words. "
                "Truncated before LLM."
            )

        # -----------------------------------------------------
        # Build prompt
        # -----------------------------------------------------

        try:

            prompt = (
                EXTRACTOR_METADATA.format(
                    TEXT_CHUNK=chunk_text
                )
            )

        except Exception as exc:

            print(
                "[ERROR] Failed to build "
                f"metadata prompt: {exc}"
            )

            return (
                self._fallback_metadata(
                    chunk
                )
            )

        # -----------------------------------------------------
        # DEBUG:
        #
        # This is useful for finding cases where the Python
        # chunk has text but the actual prompt does not.
        # -----------------------------------------------------

        print(
            "[DEBUG] Prompt content length:",
            len(prompt),
        )

        print(
            "[DEBUG] Chunk text length:",
            len(chunk_text),
        )

        # Uncomment if you need the complete prompt:
        #
        # print("========== PROMPT ==========")
        # print(prompt)
        # print("========== END PROMPT ======")

        # -----------------------------------------------------
        # Call LLM
        # -----------------------------------------------------

        try:

            response = (
                self.llm.generate(
                    prompt
                )
            )

        except Exception as exc:

            print(
                "[WARNING] LLM metadata "
                f"extraction failed: {exc}"
            )

            return (
                self._fallback_metadata(
                    chunk
                )
            )

        # -----------------------------------------------------
        # Parse JSON
        # -----------------------------------------------------

        result = (
            self._parse_llm_json(
                response
            )
        )

        if result is None:

            print(
                "[WARNING] LLM returned "
                "non-JSON metadata."
            )

            print(
                "[DEBUG] Raw response:",
                response,
            )

            return (
                self._fallback_metadata(
                    chunk
                )
            )

        # -----------------------------------------------------
        # Validate
        # -----------------------------------------------------

        validated = (
            self._validate_llm_metadata(
                result,
                kind,
            )
        )

        if validated is None:

            print(
                "[WARNING] LLM returned "
                "invalid metadata:"
            )

            print(
                result
            )

            return (
                self._fallback_metadata(
                    chunk
                )
            )

        return validated

    # =========================================================
    # Metadata
    # =========================================================

    def metadata(
        self,
        chunk,
        document: Optional[
            DoclingDocument
        ] = None,
    ) -> dict:

        meta = getattr(
            chunk,
            "meta",
            None,
        )

        # -----------------------------------------------------
        # Document title
        # -----------------------------------------------------

        title = None

        if document is not None:

            title = (
                self.document_title(
                    document
                )
            )

        # -----------------------------------------------------
        # Headings
        # -----------------------------------------------------

        headings = getattr(
            meta,
            "headings",
            None,
        )

        # -----------------------------------------------------
        # Captions
        # -----------------------------------------------------

        captions = getattr(
            meta,
            "captions",
            None,
        )

        # -----------------------------------------------------
        # Doc items
        # -----------------------------------------------------

        doc_items = []

        if meta is not None:

            for item in getattr(
                meta,
                "doc_items",
                [],
            ):

                item_pages = []

                for prov in getattr(
                    item,
                    "prov",
                    [],
                ):

                    page_no = getattr(
                        prov,
                        "page_no",
                        None,
                    )

                    if page_no is not None:

                        item_pages.append(
                            page_no
                        )

                doc_items.append(
                    {
                        "ref": getattr(
                            item,
                            "self_ref",
                            None,
                        ),
                        "label": str(
                            getattr(
                                item,
                                "label",
                                "",
                            )
                        ),
                        "pages": sorted(
                            set(
                                item_pages
                            )
                        ),
                    }
                )

        # -----------------------------------------------------
        # LLM metadata
        # -----------------------------------------------------

        llm_metadata = (
            self._extract_llm_metadata(
                chunk
            )
        )

        # -----------------------------------------------------
        # Final metadata
        # -----------------------------------------------------

        return {
            "title": title,

            "source": (
                self.options.source
            ),

            "pages": self.pages(
                chunk
            ),

            "headings": headings,

            "captions": captions,

            "doc_items": doc_items,

            "type": (
                llm_metadata[
                    "type"
                ]
            ),

            "summary": (
                llm_metadata[
                    "summary"
                ]
            ),

            "keywords": (
                llm_metadata[
                    "keywords"
                ]
            ),
        }

    # =========================================================
    # Table detection
    # =========================================================

    @staticmethod
    def is_table(
        chunk,
    ) -> bool:

        meta = getattr(
            chunk,
            "meta",
            None,
        )

        if meta is None:
            return False

        return any(
            getattr(
                item,
                "label",
                None,
            )
            == DocItemLabel.TABLE
            for item in getattr(
                meta,
                "doc_items",
                [],
            )
        )

    # =========================================================
    # Contextualized chunks
    # =========================================================

    def contextualized_chunks(
        self,
        document: DoclingDocument,
    ) -> Iterator[str]:

        for chunk in self.chunk(
            document
        ):

            yield self._text_for_llm(
                chunk
            )

    # =========================================================
    # Callable
    # =========================================================

    def __call__(
        self,
        document: DoclingDocument,
    ):

        return self.chunk(
            document
        )
