# Test Data: Retrieval Questions

Question sets used to evaluate retrieval ([rag_evaluation.md](rag_evaluation.md)).

| File | Generator | Questions |
| --- | --- | ---: |
| `test_data/groq_test_questions.jsonl` | Groq (`openai/gpt-oss-120b`, fallback `openai/gpt-oss-20b`) with [`generate_test_questions.py`](../src/ragsystem/utils/generate_test_questions.py) | 300 |
| `test_data/claude_test_questions.jsonl` | Claude (`claude-opus-5-5`), same chunk selection and rules | 300 |

Each set: one question per document (300 different PDFs), one text chunk per
document, 100 `easy`, 100 `medium` and 100 `hard` questions, taken from
`chroma_db_v1` (collection `documents`, 5,608 chunks).

---

## 1. Generation

```bash
uv run python -m ragsystem.utils.generate_test_questions
```

```
Chroma collection
   │ 1. keep eligible chunks (text, 60–600 words)
   │ 2. group by document, shuffle documents (seed 42)
   │ 3. one random chunk per document, each document used once
   │ 4. assign a difficulty (easy / medium / hard, balanced, shuffled)
   │ 5. LLM writes one question + short answer from the chunk
   │ 6. automatic checks, retry up to 3 times
   ▼ 7. append to the output file
```

Prompt: [`prompts/question_generator.py`](../src/ragsystem/prompts/question_generator.py).

### 1.1 Chunk selection

- Eligible chunks: `type` is `text`, 60 to 600 words.
- Documents are shuffled with `SEED = 42`; one random eligible chunk is taken
  from each document; each document is used at most once.
- If the LLM fails on a document, the next document is used.

### 1.2 Difficulty levels

Levels are balanced (100 each) and assigned in random order.

| Level | Instruction to the LLM | Example |
| --- | --- | --- |
| `easy` | One fact stated explicitly; names may be reused, the rest rephrased | *How many participants were involved in the Wiki Loves the Olympics editing contest?* |
| `medium` | A fact or reason from the chunk, paraphrased with synonyms instead of the chunk's key phrases | *Which institution issues the Uniform Standards of Professional Appraisal Practice annually each January?* |
| `hard` | Combine two pieces of information or make a simple inference (comparison, cause, purpose, condition), with different vocabulary from the chunk | *Which U.S. official met Israel's president and hostage families, then exited his Tel Aviv hotel to shake hands with protesters amid Netanyahu's Rafah invasion talk?* |

### 1.3 Question rules

- At most 25 words, one sentence, ending with `?`.
- Natural user query, not a copied sentence.
- Answerable from the chunk only.
- Names its subject (organisation, product, person, event); the chunk's
  headings are given to the LLM.
- No reference to the source ("the passage", "the document", "according
  to"…), no page numbers or formatting, no yes/no questions.
- The LLM also returns a short answer from the chunk.

### 1.4 Automatic checks

An output is rejected and regenerated (up to 3 attempts) when:

- the response is not valid JSON, or the question or answer is missing;
- the question does not end with `?` or has more than 25 words;
- the question contains a word referring to the source (`passage`,
  `excerpt`, `the text`, `the document`, `this article`, `according to`,
  `mentioned`, `chunk`).

On a Groq error the script waits 10 seconds and retries; after 3 failed
attempts the document is skipped.

### 1.5 Options

Each question is appended as soon as it is created. Running the command
again resumes: documents already in the file are skipped and only missing
levels are generated.

| Option | Default | Effect |
| --- | --- | --- |
| `--max-questions N` | 300 | Size of the test set |
| `--chroma-dir` | `./chroma_db_v1` | Index used to pick chunks |
| `--output` | `test_data/test_questions.jsonl` | Output file |
| `--seed` | 42 | Random selection |
| `--overwrite` | off | Delete the output file and start from zero |

---

## 2. Claude set

- Chunks and levels exported with the script's own functions
  (`pick_chunks`, `complexity_plan`, `SEED = 42`) from `chroma_db_v1` at
  5,608 chunks.
- Questions written by Claude following the rules of 1.3; all 300 pass the
  script's `validate()` checks.
- Claude read at most the first 200 words of each chunk.
- Extra field: `"generator": "claude-opus-5-5"`.
- Overlap with the Groq set: 294 shared chunks, 124 of them with the same
  difficulty level.

---

## 3. File format

One JSON object per line. Example (chunk text shortened):

```json
{
  "id": "3f27b0c0-0d86-5945-a4e7-ecc39063c94a",
  "question": "How many participants were involved in the Wiki Loves the Olympics editing contest?",
  "answer": "45 participants",
  "complexity": "easy",
  "text": "●\n●\nGoal Improving the quality and increasing the amount of articles about the O…",
  "metadata": {
    "source": "data/DUDE_7a0762c63314c1755fa6f02a19da5587.pdf",
    "chunk_index": 0,
    "title": "DUDE_7a0762c63314c1755fa6f02a19da5587",
    "pages": "[1]",
    "headings": "[\"Challenges\", \"Bonus contest\"]",
    "type": "text"
  }
}
```

| Field | Content |
| --- | --- |
| `id` | Chroma ID of the chunk (expected search result) |
| `question` | Question |
| `answer` | Short answer from the chunk |
| `complexity` | `easy`, `medium` or `hard` |
| `text` | Chunk text at generation time |
| `metadata.source` | Source PDF (expected document) |
| other `metadata` | Copied from Chroma, without the sparse vectors |
| `generator` | Claude set only |

The evaluation only uses questions whose chunk exists in the index with the
same text.
