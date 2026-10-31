# RAG System

Retrieval-Augmented Generation over a PDF corpus (OHR-Bench), with
Docling parsing, hybrid BM25 + vector retrieval (Chroma, Qwen3-Embedding)
and a Streamlit chat interface backed by Groq.

## Project structure

```
ragsystem
├── update_rag_data.py            # Pipeline: download data -> parse -> chunk -> embed -> Chroma
├── run.py                        # Shortcut for update_rag_data.py
├── data/                         # PDF corpus (downloaded automatically, not in git)
├── chroma_db/                    # Vector store (generated, not in git)
├── test_data/                    # Retrieval evaluation questions and results
├── secrets/config.yaml           # GROQ_API_KEY (not in git)
└── src
    ├── app/app.py                # Streamlit chatbot
    └── ragsystem
        ├── core
        │   ├── rag_pipeline      # injestor (Docling), chunker, vectorizer
        │   └── retriever.py      # Hybrid BM25 + vector retriever
        ├── evaluation            # Retrieval evaluation
        ├── prompts               # LLM prompts
        └── utils                 # parser, Groq connector, data downloader, test data
```

## Documentation

- [Data: corpus, index and test data](src/ragsystem/docs/data.md): what the
  data contains, how it is downloaded, chunked, embedded and stored.
- [Test data](src/ragsystem/docs/test_data.md): how the 300 retrieval test
  questions are generated (random documents, 3 difficulty levels).
- [RAG evaluation](src/ragsystem/docs/rag_evaluation.md): evaluation method
  and results.

## Setup

```bash
uv sync
```

Create `secrets/config.yaml`:

```yaml
GROQ_API_KEY: "your-key"
```

## Build the index

```bash
uv run python update_rag_data.py
```

On first run the corpus is downloaded from the Hugging Face dataset
[`opendatalab/OHR-Bench`](https://huggingface.co/datasets/opendatalab/OHR-Bench)
(`pdfs.zip`, ~1.5 GB, cached by `huggingface_hub`) and the PDFs are extracted
flat into `data/`. If `data/` already contains PDFs the download is skipped.

Indexing is incremental: files already present in the Chroma collection are
skipped, so an interrupted run can simply be restarted.

Useful options:

| Option | Effect |
| --- | --- |
| `--categories academic law ...` | OHR-Bench categories to download (default: all except `textbook`) |
| `--limit N` | Process at most N new documents per run (default `MAX_DOCUMENTS = 50`, `0` = all) |
| `--skip-download` | Only index what is already in `data/` |
| `--force-download` | Download even if `data/` is not empty (adds missing files) |
| `--backfill-sparse` | Add the SPLADE sparse vector to chunks indexed before it existed |
| `--rebuild` | Drop the collection and re-index everything |

To only download the data: `uv run python -m ragsystem.utils.data_downloader`.

## Chat interface

```bash
uv run streamlit run src/app/app.py
```

## Evaluation

See [src/ragsystem/docs/rag_evaluation.md](src/ragsystem/docs/rag_evaluation.md).
