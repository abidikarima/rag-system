# Data: Corpus, Index and Test Data

Source documents, how they are turned into chunks stored in Chroma, and the
content of a stored chunk.

---

## 1. Source corpus

The documents come from the **OHR-Bench** dataset on the Hugging Face Hub:

| | |
| --- | --- |
| Dataset | [`opendatalab/OHR-Bench`](https://huggingface.co/datasets/opendatalab/OHR-Bench) |
| File | `pdfs.zip` (~1.5 GB) |
| Revision (pinned) | `7f833e3eda9a571a9ea545a8f6d476fa1685033d` |
| Local folder | `data/` (not tracked by git) |

OHR-Bench contains real-world PDFs grouped into 7 categories. By default all
categories except `textbook` are used, giving **757 PDFs (~1.3 GB)**:

| Category | PDFs | Size | Examples of file names |
| --- | ---: | ---: | --- |
| academic | 85 | 173 MB | `2305.02437v3.pdf`, `DUDE_*.pdf` |
| administration | 146 | 170 MB | `DUDE_*.pdf` |
| finance | 65 | 60 MB | `3M_2023Q2_10Q.pdf`, `AMAZON_2017_10K.pdf` |
| law | 95 | 39 MB | `*-EX-10.*-AGREEMENT.pdf` (contracts) |
| manual | 87 | 331 MB | product manuals |
| news | 279 | 550 MB | `05_01.pdf` … `08_31.pdf` (newspaper pages by date) |
| textbook | 504 | — | *not downloaded by default* |

All PDFs are stored flat in `data/` (no category sub-folders); the category
is not kept in the file path or in the chunk metadata.

### How the data is downloaded

[`ragsystem/utils/data_downloader.py`](../src/ragsystem/utils/data_downloader.py),
called at the start of `update_rag_data.py`:

1. If `data/` already contains at least one PDF, the download is skipped.
2. Otherwise `pdfs.zip` is downloaded with `huggingface_hub` (Hugging Face
   cache).
3. The PDFs of the selected categories are extracted into `data/`; existing
   files are not overwritten.

Command-line options: see [Build the index](../README.md#build-the-index).

---

## 2. How the index is created

Script: [`update_rag_data.py`](../scripts/update_rag_data.py).

```
data/*.pdf
   │  1. Parse        Docling  → DoclingDocument
   │  2. Chunk        Docling HybridChunker + grouping of small text chunks
   │  3. Metadata     title, pages, headings, … + LLM summary/keywords (Groq)
   │  4. Embed        dense (Qwen3) + sparse SPLADE (chunk text and summary)
   ▼  5. Store        Chroma collection "documents"
chroma_db/
```

### 2.1 Parsing

[`injestor.py`](../src/ragsystem/core/rag_pipeline/injestor.py) /
[`parser.py`](../src/ragsystem/utils/parser.py), using **Docling**:

- Supported files: `.pdf`, `.docx`, `.html`, `.htm` (other files are ignored).
- OCR **disabled** (`enable_ocr=False`): scanned pages without a text layer
  produce little or no text.
- Table structure recognition **enabled**, in the fast TableFormer mode.

### 2.2 Chunking

[`chunker.py`](../src/ragsystem/core/rag_pipeline/chunker.py):

1. Docling **HybridChunker** with `max_tokens=2000`, `merge_peers=True` and
   `repeat_table_header=True` (a table split across chunks keeps its header).
2. **Grouping of small chunks**: consecutive *text* chunks are merged when one
   of them is shorter than 100 words or the group is shorter than 250 words,
   as long as the merged group does not exceed 250 words. Tables and images are
   never merged. A single Docling chunk can still be longer than 250 words.

### 2.3 Metadata

For every chunk, `DoclingChunker.metadata()` produces:

- **From Docling**: document title (the file name without extension), pages, headings, captions and the list of
  Docling items (`doc_items`) that make up the chunk.
- **Chunk type** (`text`, `table` or `image`): decided from the Docling item
  labels, not by the LLM.
- **LLM summary and keywords**: one call per chunk to Groq (`openai/gpt-oss-120b`,
  falling back to `openai/gpt-oss-20b`), using the prompt in
  [`prompts/metadata_extractor.py`](../src/ragsystem/prompts/metadata_extractor.py).
  - Text longer than 250 words is truncated before being sent to the LLM.
  - Tables and images get a short description and an empty keyword list.
  - If the LLM fails or answers with something unusable, a fallback is stored:
    - text: the first 40 words of the chunk;
    - table: `"Table containing structured information."`;
    - image: `"Visual content extracted from the document."`.

### 2.4 Embeddings

[`vectorizer.py`](../src/ragsystem/core/rag_pipeline/vectorizer.py):

| Vector | Model | Input | Format |
| --- | --- | --- | --- |
| Dense | `Qwen/Qwen3-Embedding-0.6B` | chunk text | 1024 floats, L2-normalized |
| Sparse (text) | `naver/splade-cocondenser-selfdistil` | chunk text | `{"indices": [...], "values": [...]}` |
| Sparse (summary) | `naver/splade-cocondenser-selfdistil` | chunk summary | same as above |

- The Chroma collection uses **cosine** distance (`hnsw:space = cosine`).
- At search time, queries are encoded with Qwen3's `"query"` instruction
  prompt; documents are encoded without a prompt.
- SPLADE vectors live in a 30,522-term vocabulary (BERT word pieces). Only
  non-zero terms are stored (typically 50–500 per chunk), with values rounded
  to 4 decimals.
- SPLADE reads at most **512 tokens**, so for long chunks the text sparse vector
  only covers the beginning of the chunk.
- The local Chroma engine does not support native sparse vectors, so both
  sparse vectors are stored as JSON strings in the metadata. The API
  retriever uses the text sparse vector.

### 2.5 Storage and incremental runs

- Chunk ID: `uuid5(NAMESPACE_URL, "<source>:<chunk_index>")`, for example
  `uuid5(NAMESPACE_URL, "data/05_01.pdf:3")`. Re-indexing a file overwrites
  (`upsert`) its chunks.
- Runs are incremental: files whose `source` already exists in the
  collection are skipped.

---

## 3. Content of a stored chunk

Each Chroma record contains:

| Field | Content |
| --- | --- |
| `id` | Deterministic UUID (see 2.5) |
| `document` | Chunk text as produced by Docling. Tables use Docling's default serialization (cells written as `row, column = value` text, not Markdown tables) |
| `embedding` | Dense Qwen3 vector (1024 dimensions) |
| `metadata` | Fields below |

Metadata fields (lists and dicts are stored as JSON strings because Chroma
only accepts scalar metadata values):

| Key | Type | Description |
| --- | --- | --- |
| `source` | str | File path, e.g. `data/05_01.pdf` |
| `chunk_index` | int | Position of the chunk in its file |
| `title` | str | Docling document name, i.e. the file name without extension (e.g. `05_01`) |
| `pages` | JSON list | Page numbers covered by the chunk |
| `headings` | JSON list | Section headings above the chunk (may be missing) |
| `captions` | JSON list | Table/figure captions (may be missing) |
| `doc_items` | JSON list | Docling items: `ref`, `label`, `pages` |
| `type` | str | `text`, `table` or `image` |
| `is_table` | str | `"True"` / `"False"` |
| `summary` | str | LLM summary, or fallback (see 2.3) |
| `keywords` | JSON list | 3–8 LLM keywords (empty for tables/images) |
| `sparse_embedding` | JSON | SPLADE vector of the chunk text |
| `summary_sparse_embedding` | JSON | SPLADE vector of the summary (missing if no summary) |

Example (long values shortened):

```json
{
  "source": "data/05_01.pdf",
  "chunk_index": 1,
  "title": "05_01",
  "pages": "[1]",
  "headings": "[\"Cease-Fire  Won't  Stop It, Netanyahu Says\"]",
  "doc_items": "[{\"ref\": \"#/texts/4\", \"label\": \"text\", \"pages\": [1]}, …]",
  "type": "text",
  "is_table": "False",
  "summary": "Prime Minister Benjamin Netanyahu reaffirmed Israel's intention to invade Rafah in souther…",
  "keywords": "[\"Netanyahu\", \"Rafah\", \"Gaza\", \"cease-fire\", \"Hamas\", \"Blinken\", \"Israel-Hamas war\"]",
  "sparse_embedding": "{\"indices\": [1042, 1043, 1046, …], \"values\": [0.2648, 0.4953, 0.2856, …]}",
  "summary_sparse_embedding": "{\"indices\": [1057, 2020, 2040, …], \"values\": [0.3604, 0.4031, 0.6052, …]}"
}
```

> **Older indexes.** Chunks indexed before the summary/sparse fields were added
> only have `source`, `chunk_index`, `headings`, `captions`, `doc_items` and
> `is_table`. Re-index with `--rebuild` (or into a new `--chroma-dir`) to get
> all fields.

### Reading the sparse vectors

```python
import json, chromadb

col = chromadb.PersistentClient(path="./chroma_db").get_collection("documents")
meta = col.get(limit=1, include=["metadatas"])["metadatas"][0]
vector = json.loads(meta["sparse_embedding"])   # {"indices": [...], "values": [...]}
```

SPLADE similarity between a query vector `q` (from `SpladeEmbedder.embed_query`)
and a chunk vector `d` is their dot product over the shared indices:

```python
def sparse_dot(q, d):
    d_map = dict(zip(d["indices"], d["values"]))
    return sum(v * d_map.get(i, 0.0) for i, v in zip(q["indices"], q["values"]))
```

---

## 4. Other data folders

| Folder | Content |
| --- | --- |
| `test_data/` | `groq_test_questions.jsonl`, `claude_test_questions.jsonl`: 300 retrieval questions each, see [test_data.md](test_data.md) |
| `results/` | Evaluation outputs (JSON), see [rag_evaluation.md](rag_evaluation.md) |

Files in `results/`:

| File | Written by |
| --- | --- |
| `combination_evaluation_claude_test_questions.json`, `combination_evaluation_groq_test_questions.json` | [`evaluate_combinations.py`](../src/ragsystem/evaluation/evaluate_combinations.py) |
| `combination_evaluation.json`, `combination_evaluation_claude_data.json`, `combination_evaluation_gro_data.json`, `hybrid_retrieval_evaluation.json` | Earlier runs |
