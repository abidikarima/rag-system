# RAG Evaluation Dataset and First Evaluation Results

## 1. Overview

The evaluation dataset is created from documents already stored in the RAG (Retrieval-Augmented Generation) collection.

The objective is to measure how accurately the RAG system can retrieve the document that contains the information relevant to a given question.

For the first evaluation, a total of **200 evaluation questions** were created.

---

## 2. Evaluation Dataset Creation

The evaluation dataset is generated directly from the documents available in the RAG collection.

The process is:

1. Select a document randomly from the RAG collection.
2. Extract the text/content of the selected document.
3. Send the document text to ChatGPT.
4. Ask ChatGPT to generate a **very general question** based on the content of the document.
5. Save the generated question together with the original document as the expected/source document.
6. Repeat the process until **200 evaluation questions** have been generated.

The important point is that every evaluation question originates from an actual document in the RAG collection. Therefore, the original document can be used as the **ground-truth document** when evaluating the retrieval system.

### Dataset Generation Process

RAG Collection
      |
      v
Randomly select a document
      |
      v
Extract document text
      |
      v
Send document text to ChatGPT
      |
      v
Ask ChatGPT to generate a general question
      |
      v
Question + Source Document
      |
      v
Evaluation Dataset

---

## 3. Example of Evaluation Data Creation

For example, suppose the RAG collection contains a document about an employee vacation policy.

The document is randomly selected from the RAG collection and its text is provided to ChatGPT.

ChatGPT can generate a general question such as:

> What are the main rules regarding employee vacation?

The evaluation dataset stores the relationship between the question and the document from which the question was generated.

Example:

| Field             | Value                                                |
| ----------------- | ---------------------------------------------------- |
| Question          | What are the main rules regarding employee vacation? |
| Expected Document | The company vacation policy document                 |

Later, the question is sent to the RAG retrieval system.

The RAG system returns a ranked list of documents, and the evaluation checks whether the original document appears in the retrieved results.

---

## 4. Why Generate General Questions?

The questions are intentionally generated as **general questions** rather than questions that directly copy text from the document.

This makes the evaluation closer to a real user interaction.

For example, instead of asking:

> What does the document say about annual leave requests?

the generated question could be:

> How does the company handle employee vacations?

The second question is more general and requires the retrieval system to understand the semantic relationship between the question and the document.

This allows the evaluation to test the semantic retrieval capabilities of the RAG system.

---

## 5. Evaluation Dataset

The first evaluation dataset contains:

- **Total Questions: 200**

Each evaluation item contains, at minimum:

- The generated question
- The expected/source document

Conceptually, the dataset contains a relationship between each question and the document from which the question was generated.

| Question                                     | Expected Document |
| -------------------------------------------- | ----------------- |
| General question generated from document 1   | Document 1        |
| General question generated from document 2   | Document 2        |
| General question generated from document 3   | Document 3        |
| ...                                          | ...               |
| General question generated from document 200 | Document 200      |

The expected document is used as the ground truth for evaluating retrieval performance.

---

## 6. Retrieval Evaluation

For each of the 200 questions, the question is sent to the RAG retrieval system.

The system returns documents ranked according to their relevance to the question.

The evaluation checks whether the expected/source document is present within different retrieval depths:

- **Top 5**
- **Top 10**
- **Top 15**

For each question:

- **Hit**: the expected document is found within the selected Top K results.
- **Miss**: the expected document is not found within the selected Top K results.

---

## 7. Accuracy@K

The main metric used for this evaluation is **Accuracy@K**.

The formula is:

**Accuracy@K = (Number of Hits / Total Number of Questions) × 100**

For example:

**Accuracy@5 = (169 / 200) × 100 = 84.50%**

This means that the expected document was found in the first 5 retrieved documents for 169 out of the 200 questions.

---

# 8. First Evaluation Results

The first evaluation was performed on **200 questions**.

| Retrieval Level | Questions | Hits | Misses |   Accuracy |
| --------------- | --------: | ---: | -----: | ---------: |
| Top 5           |       200 |  169 |     31 | **84.50%** |
| Top 10          |       200 |  177 |     23 | **88.50%** |
| Top 15          |       200 |  180 |     20 | **90.00%** |

---

## 9. Accuracy@5

For the Top 5 evaluation:

| Metric     |     Result |
| ---------- | ---------: |
| Questions  |        200 |
| Hits       |        169 |
| Misses     |         31 |
| Accuracy@5 | **84.50%** |

The expected/source document was found within the first 5 retrieved documents for **169 questions**.

Therefore:

**Accuracy@5 = 169 / 200 × 100 = 84.50%**

There were **31 questions** where the expected document was not found in the Top 5 results.

---

## 10. Accuracy@10

For the Top 10 evaluation:

| Metric      |     Result |
| ----------- | ---------: |
| Questions   |        200 |
| Hits        |        177 |
| Misses      |         23 |
| Accuracy@10 | **88.50%** |

The expected/source document was found within the first 10 retrieved documents for **177 questions**.

Therefore:

**Accuracy@10 = 177 / 200 × 100 = 88.50%**

Compared with Top 5, the system successfully retrieved the expected document for **8 additional questions**:

**177 - 169 = 8**

The accuracy improved by:

**88.50% - 84.50% = 4.00 percentage points**

---

## 11. Accuracy@15

For the Top 15 evaluation:

| Metric      |     Result |
| ----------- | ---------: |
| Questions   |        200 |
| Hits        |        180 |
| Misses      |         20 |
| Accuracy@15 | **90.00%** |

The expected/source document was found within the first 15 retrieved documents for **180 questions**.

Therefore:

**Accuracy@15 = 180 / 200 × 100 = 90.00%**

Compared with Top 10, the system successfully retrieved the expected document for **3 additional questions**:

**180 - 177 = 3**

The accuracy improved by:

**90.00% - 88.50% = 1.50 percentage points**

---

# 12. Results Comparison

The retrieval performance increases as the number of retrieved documents increases:

| Metric    |      Top 5 |     Top 10 |     Top 15 |
| --------- | ---------: | ---------: | ---------: |
| Questions |        200 |        200 |        200 |
| Hits      |        169 |        177 |        180 |
| Misses    |         31 |         23 |         20 |
| Accuracy  | **84.50%** | **88.50%** | **90.00%** |

---

## 13. Improvement Between Retrieval Levels

### Top 5 → Top 10

Accuracy:

**84.50% → 88.50%**

Improvement:

**+4.00 percentage points**

Additional successful questions:

**177 - 169 = 8**

### Top 10 → Top 15

Accuracy:

**88.50% → 90.00%**

Improvement:

**+1.50 percentage points**

Additional successful questions:

**180 - 177 = 3**

### Top 5 → Top 15

Accuracy:

**84.50% → 90.00%**

Overall improvement:

**+5.50 percentage points**

Additional successful questions:

**180 - 169 = 11**

---

# 14. Interpretation of the Results

The first evaluation shows that the RAG retrieval system performs well on the generated evaluation dataset.

At Top 5, the system achieves:

**84.50% Accuracy@5**

Increasing the retrieval depth to Top 10 improves the result to:

**88.50% Accuracy@10**

Increasing it again to Top 15 results in:

**90.00% Accuracy@15**

This means that increasing the number of retrieved candidates allows the system to recover documents that were ranked below the first few results.

However, the improvement becomes smaller as K increases:

| Retrieval Change |             Improvement |
| ---------------- | ----------------------: |
| Top 5 → Top 10   | +4.00 percentage points |
| Top 10 → Top 15  | +1.50 percentage points |
| Top 5 → Top 15   | +5.50 percentage points |

This suggests that a large portion of the relevant documents are already retrieved within the Top 5 or Top 10 results.

---

# 15. Miss Analysis

At Top 15, there are still:

**20 misses out of 200 questions**

These 20 failed retrieval cases should be analyzed individually to understand why the expected document was not retrieved.

Potential causes include:

### 15.1 Question Generation

The question generated by ChatGPT may be too general or ambiguous.

For example:

> What are the important rules?

Such a question may not contain enough semantic information to identify the correct document.

### 15.2 Similar Documents

The RAG collection may contain multiple documents with similar content.

The system may retrieve a similar document instead of the exact source document used to generate the question.

### 15.3 Document Chunking

The relevant information may be split across multiple chunks.

Poor chunking can make it harder for the retrieval system to identify the original document.

### 15.4 Embedding Quality

The generated question and the relevant document may not have sufficiently similar embeddings.

This can result in the correct document being ranked lower than unrelated documents.

### 15.5 Ranking

The correct document may be retrieved but ranked below the selected Top K threshold.

For example, if:

**Correct document rank = 18**

then:

| Evaluation | Result |
| ---------- | ------ |
| Top 5      | Miss   |
| Top 10     | Miss   |
| Top 15     | Miss   |

The document may still exist somewhere in the larger result set, but it is not within the evaluated Top 15.

### 15.6 Document Content

The original document may not contain enough distinctive information to allow the retrieval system to identify it reliably.

---

# 16. Baseline

These results should be considered the **baseline evaluation** for the current RAG retrieval system.

| Metric       |      Baseline |
| ------------ | ------------: |
| Dataset size | 200 questions |
| Accuracy@5   |    **84.50%** |
| Accuracy@10  |    **88.50%** |
| Accuracy@15  |    **90.00%** |

Future changes to the RAG pipeline can be evaluated against this baseline.

For example, after changing:

- Embedding models
- Chunk size
- Chunk overlap
- Retrieval parameters
- Similarity search
- Reranking
- Query processing
- Question-generation prompts

the same evaluation dataset can be used again to determine whether retrieval performance improves or decreases.

---

# 17. Complete Evaluation Workflow

The complete evaluation workflow is:

**Step 1 — RAG Collection**

Start with the existing RAG collection containing the documents.

**Step 2 — Random Document Selection**

Randomly select a document from the RAG collection.

**Step 3 — Extract Document Text**

Extract the text/content of the selected document.

**Step 4 — Question Generation**

Send the document text to ChatGPT and ask it to generate a very general question based on the document.

**Step 5 — Create Evaluation Entry**

Store:

- Generated question
- Source/expected document

**Step 6 — Repeat**

Repeat the process until the evaluation dataset contains **200 questions**.

**Step 7 — RAG Retrieval**

Send each of the 200 questions to the RAG retrieval system.

**Step 8 — Retrieve Ranked Documents**

For each question, retrieve the ranked list of relevant documents.

**Step 9 — Compare With Ground Truth**

Check whether the expected/source document is present in:

- Top 5
- Top 10
- Top 15

**Step 10 — Calculate Hits and Misses**

For each retrieval level, count:

- Hits
- Misses

**Step 11 — Calculate Accuracy**

Calculate Accuracy@5, Accuracy@10, and Accuracy@15.

---

# 18. Evaluation Workflow Summary

```text
RAG Collection
      |
      v
Random Document Selection
      |
      v
Extract Document Text
      |
      v
Send Text to ChatGPT
      |
      v
Generate General Question
      |
      v
Store Question + Source Document
      |
      v
Repeat Until 200 Questions
      |
      v
Evaluation Dataset
      |
      v
Send Question to RAG
      |
      v
Retrieve Ranked Documents
      |
      +-------------------+
      |                   |
      v                   v
   Expected           Retrieved
   Document           Documents
      |                   |
      +---------+---------+
                |
                v
          Hit or Miss
                |
                v
        Top 5 / Top 10 / Top 15
                |
                v
        Calculate Accuracy