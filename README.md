# RAG System

Question answering over a collection of PDF documents (the OHR-Bench corpus:
news, contracts, financial reports, manuals, scientific papers, administrative
documents).

- **Indexing:** PDFs are parsed with Docling, split into chunks, summarized by
  an LLM (Groq) and stored in Chroma with three vectors per chunk: a dense
  embedding (Qwen3-Embedding) and SPLADE sparse vectors of the text and of
  the summary.
- **Retrieval:** dense embedding + SPLADE sparse vector on the chunk text, the
  best method in the [evaluation](docs/rag_evaluation.md).
- **Answering:** a FastAPI service with a web chat interface. The LLM answers
  from the retrieved chunks and cites them; the references are returned with
  the answer.

## Project structure

```
ragsystem
├── README.md
├── pyproject.toml / uv.lock      # Dependencies (uv)
├── docs/                         # Documentation (data, test data, evaluation)
├── scripts/
│   ├── update_rag_data.py        # Pipeline: download -> parse -> chunk -> embed -> Chroma
│   └── run.py                    # Shortcut for update_rag_data.py
├── src/
│   ├── ragsystem/
│   │   ├── api/                  # FastAPI service + web chat interface (static/index.html)
│   │   ├── core/
│   │   │   ├── rag_pipeline/     # Docling ingestion, chunker, vectorizer (dense + SPLADE)
│   │   │   ├── dense_sparse_retriever.py  # Retriever used by the API
│   │   │   └── retriever.py      # Older BM25 + vector retriever
│   │   ├── evaluation/           # Retrieval evaluation scripts
│   │   ├── prompts/              # LLM prompts (metadata, questions, chatbot)
│   │   └── utils/                # Parser, Groq connector, data downloader, test question generator
│   └── app/app.py                # Legacy Streamlit app (old retriever, see below)
├── test_data/                    # Evaluation questions (Groq and Claude sets)
├── results/                      # Evaluation results (JSON)
│
│   Not in git (generated or private):
├── data/                         # PDF corpus, downloaded automatically
├── chroma_db_v1/                 # Current Chroma index
└── secrets/config.yaml           # GROQ_API_KEY
```

## Documentation

- [Data](docs/data.md): what the corpus contains, and how it is downloaded,
  chunked, embedded and stored.
- [Test data](docs/test_data.md): how the 300 retrieval test questions are
  generated (random documents, 3 difficulty levels).
- [Evaluation](docs/rag_evaluation.md): comparison of 7 retrieval methods on
  the Claude and Groq test sets.

## Setup

```bash
uv sync
```

Create `secrets/config.yaml`:

```yaml
GROQ_API_KEY: "your-key"
```

## Build the index

Run the scripts from the repository root:

```bash
uv run python scripts/update_rag_data.py --chroma-dir ./chroma_db_v1
```

Without `--chroma-dir`, the index is written to `./chroma_db`. The API reads
`./chroma_db_v1` by default (see `RAG_CHROMA_DIR` below).

On first run the corpus is downloaded from the Hugging Face dataset
[`opendatalab/OHR-Bench`](https://huggingface.co/datasets/opendatalab/OHR-Bench)
(`pdfs.zip`, ~1.5 GB, cached by `huggingface_hub`) and the PDFs are extracted
flat into `data/`. If `data/` already contains PDFs the download is skipped.

Indexing is incremental: files already present in the Chroma collection are
skipped, so an interrupted run can simply be restarted.

Useful options:

| Option | Effect |
| --- | --- |
| `--limit N` | Process at most N new documents per run (default `MAX_DOCUMENTS` in the script, `0` = all) |
| `--categories academic law ...` | OHR-Bench categories to download (default: all except `textbook`) |
| `--skip-download` | Only index what is already in `data/` |
| `--force-download` | Download even if `data/` is not empty (adds missing files) |
| `--rebuild` | Drop the collection and re-index everything |

To only download the data: `uv run python -m ragsystem.utils.data_downloader`.

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

```bash
# Generate a test set (Groq LLM, 300 questions)
uv run python -m ragsystem.utils.generate_test_questions

# Compare the 7 retrieval methods on a test set
uv run python -m ragsystem.evaluation.evaluate_combinations --test-file test_data/groq_test_questions.jsonl
```

Results are written to `results/`. Method and conclusions:
[docs/rag_evaluation.md](docs/rag_evaluation.md).

## Legacy Streamlit app

`src/app/app.py` is the first chat interface. It uses the older BM25 + vector
retriever and reads `./chroma_db`, so it does not work with the current index.
Use the chat interface of the API instead.
