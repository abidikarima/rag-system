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

## Chatbot API

A REST API that answers questions from the documents and returns the
references it used. Retrieval uses the best method of the evaluation:
dense embedding + SPLADE sparse vector on the chunk text.

```bash
uv run uvicorn ragsystem.api.main:app --host 0.0.0.0 --port 8000
```

Startup takes about a minute (models and index are loaded once). Then open
<http://localhost:8000> for the chat interface, or
<http://localhost:8000/docs> for the raw API.

| Endpoint | Description |
| --- | --- |
| `GET /` | Chat interface: answers with clickable citations and a sources panel. `/?q=your+question` asks a question directly |
| `POST /chat` | `{"question": "...", "top_k": 5}` → answer with citations `[1]`, `[2]`… and the matching references |
| `GET /health` | Status and number of indexed chunks |
| `GET /docs` | Interactive documentation |

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your key>" \
  -d '{"question": "How many pilgrims died during the hajj in Saudi Arabia?"}'
```

Each reference contains the number used in the answer, the chunk id, the
source PDF, title, pages, headings, retrieval score and an excerpt.

Configuration (environment variables):

| Variable | Default | Meaning |
| --- | --- | --- |
| `RAG_CHROMA_DIR` | `./chroma_db_v1` | Chroma folder |
| `RAG_COLLECTION` | `documents` | Collection name |
| `RAG_GROQ_CONFIG` | `./secrets/config.yaml` | File containing `GROQ_API_KEY` |
| `RAG_API_KEY` | not set | If set, `/chat` requires the `X-API-Key` header |

### Sharing the API with a tester

To give someone a public link to the API running on your machine, use a
Cloudflare quick tunnel (free, no account):

```bash
brew install cloudflared                      # once

# terminal 1: start the API, protected by a key
RAG_API_KEY=choose-a-secret uv run uvicorn ragsystem.api.main:app --port 8000

# terminal 2: open the tunnel
cloudflared tunnel --url http://localhost:8000
```

`cloudflared` prints a URL like `https://xxxx.trycloudflare.com`. Send the
tester that URL and the key:

- **Chat interface:** the URL itself. The API key field appears at the top
  right when a key is required.
- **Raw API:** the URL followed by `/docs`. They open `POST /chat`, click
  **Try it out**, fill in the `x-api-key` field and the question, then click
  **Execute**.

The link works only while your computer is on, awake and running both
commands, and it changes every time the tunnel restarts.

## Evaluation

See [src/ragsystem/docs/rag_evaluation.md](src/ragsystem/docs/rag_evaluation.md).
