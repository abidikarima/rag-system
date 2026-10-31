# Test Data: Retrieval Questions

This document explains how the retrieval test set
`test_data/test_questions.jsonl` is created and what it contains.

The test set is used to measure whether the search finds the right chunk
for a question (see
[`evaluation/evaluate_combinations.py`](../src/ragsystem/evaluation/evaluate_combinations.py)).

---

## 1. Summary

| | |
| --- | --- |
| Questions | up to **300** (`MAX_QUESTIONS`) |
| Documents | **one question per document**: 300 questions come from 300 different PDFs |
| Chunks | one random text chunk per selected document |
| Difficulty | balanced: 100 `easy`, 100 `medium`, 100 `hard` |
| Generator | Groq LLM (`openai/gpt-oss-120b`, falling back to `openai/gpt-oss-20b`) |
| Source index | Chroma `./chroma_db_v1`, collection `documents` |
| Script | [`utils/generate_test_questions.py`](../src/ragsystem/utils/generate_test_questions.py) |
| Prompt | [`prompts/question_generator.py`](../src/ragsystem/prompts/question_generator.py) |

```bash
uv run python -m ragsystem.utils.generate_test_questions
```

---

## 2. How the questions are created

```
Chroma collection
   │ 1. keep eligible chunks (text, 60–600 words)
   │ 2. group by document, shuffle documents (seed 42)
   │ 3. one random chunk per document, each document used once
   │ 4. assign a difficulty (easy / medium / hard, balanced, shuffled)
   │ 5. LLM writes one question + short answer from the chunk
   │ 6. automatic checks, retry up to 3 times
   ▼ 7. append to test_data/test_questions.jsonl
```

### 2.1 Chunk selection

1. All chunks are read from the Chroma collection.
2. A chunk is **eligible** if:
   - its `type` is `text`: tables and images are excluded, because questions
     on table cells are hard to generate and check reliably;
   - it has between **60 and 600 words**: shorter chunks rarely contain a full
     fact, and longer ones are unusual.
3. Eligible chunks are grouped by source document. The documents are shuffled
   with a fixed seed (`SEED = 42`), and **one chunk is picked at random in each
   document**.
4. Documents are used in that random order, **each at most once**. With 300
   questions, the test set covers 300 different documents. If the LLM fails on
   a document, that document is skipped and the next one is used.

The fixed seed makes the selection reproducible as long as the collection does
not change. If documents are added to the index, the selection changes.

### 2.2 Difficulty levels

Each question gets one level. The levels are balanced (300 / 3 = 100 each) and
assigned in random order.

| Level | What the LLM is asked to do | Example (generated) |
| --- | --- | --- |
| `easy` | One fact stated explicitly; names may be reused, the rest rephrased | *How many participants were involved in the Wiki Loves the Olympics editing contest?* |
| `medium` | A fact or reason from the chunk, paraphrased with synonyms instead of the chunk's key phrases | *Which institution issues the Uniform Standards of Professional Appraisal Practice annually each January?* |
| `hard` | Combine two pieces of information or make a simple inference (comparison, cause, purpose, condition), with different vocabulary from the chunk | *Which U.S. official met Israel's president and hostage families, then exited his Tel Aviv hotel to shake hands with protesters amid Netanyahu's Rafah invasion talk?* |

Harder questions share fewer words with the chunk. They are therefore harder
for keyword search (BM25, SPLADE) and test whether the semantic search
(dense embeddings) understands the meaning.

### 2.3 Question rules (prompt)

Every question must:

- be **short**: at most 25 words, one sentence, ending with `?`;
- sound like a real user query: natural and fairly general, not a copied
  sentence;
- be **answerable from the chunk only**, with no outside knowledge;
- be **specific enough** that the chunk is the best match in the whole corpus:
  it must name the subject (organisation, product, person, event). The
  chunk's section headings are given to the LLM to help identify the subject;
- **never refer to the source**: no "the passage", "the document",
  "according to"…;
- not be about page numbers, headers or formatting, and not be yes/no.

The LLM also returns a **short answer** taken from the chunk. It is not used by
the retrieval evaluation, but can be used later to evaluate generated answers.

### 2.4 Automatic checks

An LLM output is rejected, and the LLM is asked again (up to 3 attempts), when:

- the response is not valid JSON, or the question or answer is missing;
- the question does not end with `?`;
- the question has more than 25 words;
- the question contains a word that refers to the source (`passage`, `excerpt`,
  `the text`, `the document`, `this article`, `according to`, `mentioned`,
  `chunk`…).

If Groq fails (for example a rate limit), the script waits 10 seconds and
retries. After 3 failed attempts the document is skipped.

### 2.5 Saving and resuming

- Each question is **appended to the file as soon as it is created**. Stopping
  the script (Ctrl+C) keeps every question already written.
- Running the command again **resumes**: documents already in the file are not
  reused, and only the missing questions of each level are generated.
- `--overwrite` deletes the file and starts again from zero.

| Option | Default | Effect |
| --- | --- | --- |
| `--max-questions N` | 300 | Size of the test set |
| `--chroma-dir` | `./chroma_db_v1` | Index used to pick chunks |
| `--output` | `test_data/test_questions.jsonl` | Output file |
| `--seed` | 42 | Random selection |
| `--overwrite` | off | Start from zero |

---

## 3. File format

`test_data/test_questions.jsonl`: one JSON object per line.

Example of a generated record (chunk text shortened):

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

| Field | Meaning |
| --- | --- |
| `id` | Chroma ID of the chunk: **the expected search result** |
| `question` | Generated question |
| `answer` | Short answer from the chunk |
| `complexity` | `easy`, `medium` or `hard` |
| `text` | Chunk text at generation time |
| `metadata.source` | Source PDF: the expected document |
| other `metadata` | Copied from Chroma (sparse vectors are left out) |

---

## 4. Using it for evaluation

```bash
uv run python -m ragsystem.evaluation.evaluate_combinations
```

Use `--test-file` to choose the question set. Results are saved to
`results/combination_evaluation_<test file name>.json` (for example
`results/combination_evaluation_groq_test_questions.json`), never in
`test_data/`. For each retrieval method it reports:

- `chunk@5`, `chunk@10`: the expected chunk (`id`) is in the top 5 / top 10;
- `doc@5`, `doc@10`: a chunk of the expected document is in the top 5 / top 10;
- MRR (mean of 1 / rank of the expected chunk);
- `chunk@k` **per difficulty level**.

A question is only evaluated if its chunk still exists in the index with the
same text. Chunk IDs depend on the source path and the chunk position, so a
rebuilt index with different chunking invalidates questions; regenerate the
test set in that case.

---

## 5. Claude-written comparison set

`test_data/claude_test_questions.jsonl` contains 300 questions written by
Claude (`claude-opus-5-5`) instead of the Groq LLM, so the two generators can
be compared.

- **Same selection method.** The selection was exported with the script's own
  functions (`pick_chunks`, `complexity_plan`, `SEED = 42`) from
  `chroma_db_v1` when it contained 5,608 chunks.
- **Overlap with the Groq set (`groq_test_questions.jsonl`).** The two sets
  share 294 of their 300 chunks, but only 124 of them have the same difficulty
  level. The Groq run assigns a level to the next unused document, so a
  skipped document or a resumed run shifts the levels. Compare the two files
  by `id`.
- **Same rules.** Claude followed the rules of section 2.3, and every question
  was checked with the script's `validate()` function. All 300 pass, with an
  average of 15 words per question.
- **Same format**, plus a `"generator": "claude-opus-5-5"` field.
- Claude read at most the first 200 words of each chunk. Every answer is in
  the chunk, but questions never use text beyond those 200 words.

First evaluation (`evaluate_combinations`, `chroma_db_v1`, 300 questions):

| Method | chunk@5 | chunk@10 | MRR |
| --- | ---: | ---: | ---: |
| Embedding only | 95.3% | 97.7% | 0.862 |
| Sparse only | 95.7% | 97.7% | 0.879 |
| Embedding + sparse | **97.7%** | **98.7%** | **0.894** |
| Summary embedding | 90.0% | 93.3% | 0.800 |
| Summary sparse | 91.3% | 94.7% | 0.798 |
| Summary embedding + sparse | 91.0% | 95.0% | 0.823 |
| All four | 96.7% | 98.0% | 0.883 |

Observations:

- **`hard` questions were not harder to retrieve.** Combining two facts puts
  more distinctive names into the question, which helps the search. The
  levels describe how hard the question is to *answer*, not to *retrieve*.
- **Scores are high.** Each question names its subject (rule 2.3), which
  makes retrieval easier than real user queries usually are.
- **Some chunks are poor OCR text** (for example scanned forms or tables of
  contents). Questions on them rely on the few readable words.

---

## 6. Limitations

- **Generated by an LLM.** Some questions can be ambiguous (several chunks
  could answer them) or slightly wrong. Reviewing a sample by hand is
  recommended before drawing conclusions.
- **One chunk is the "correct" answer.** If another chunk also answers the
  question, the evaluation counts it as a miss. `doc@k` helps when that other
  chunk is in the same document.
- **Text chunks only.** Retrieval of tables and images is not tested.
- **The LLM sees the chunk.** Even `hard` questions may reuse some words from
  it, which favours keyword-based methods.
- **Previous test set.** The earlier `test_data/clean_test_data.jsonl`
  (200 questions, removed, still in the git history) was built from **a
  single document**, so its scores are not comparable with these sets.
  Results for the current sets are in [rag_evaluation.md](rag_evaluation.md).
