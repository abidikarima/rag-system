
import json
import random
import chromadb
from pathlib import Path


def export_random_chunks_to_jsonl(
    chroma_dir: str,
    collection_name: str,
    output_file: str,
    n_chunks: int = 100,
    seed: int | None = None,
) -> None:

    client = chromadb.PersistentClient(
        path=chroma_dir
    )

    collection = client.get_collection(
        name=collection_name
    )

    total = collection.count()

    if total == 0:
        raise ValueError("La collection est vide.")

    if seed is not None:
        random.seed(seed)

    # Get ALL chunks with metadata
    data = collection.get(
        include=[
            "documents",
            "metadatas",
        ]
    )
    print(f"Total chunks in collection: {len(data['ids'])}")
    ids = data["ids"]
    documents = data["documents"]
    metadatas = data["metadatas"]

    # ---------------------------------
    # Group chunks by document
    # ---------------------------------

    chunks_by_document = {}

    for chunk_id, text, metadata in zip(
        ids,
        documents,
        metadatas,
    ):

        metadata = metadata or {}

        # Change "source" to whatever field identifies
        # the original document in your metadata.
        document_id = metadata.get("source")

        if not document_id:
            continue

        chunks_by_document.setdefault(
            document_id,
            []
        ).append({
            "id": chunk_id,
            "text": text,
            "metadata": metadata,
        })

    if not chunks_by_document:
        raise ValueError(
            "Aucun chunk ne possède de metadata 'source'."
        )

    # ---------------------------------
    # Keep only documents having enough
    # chunks to provide n_chunks
    # ---------------------------------

    eligible_documents = [
        document_id
        for document_id, chunks in chunks_by_document.items()
        if len(chunks) >= n_chunks
    ]

    if not eligible_documents:
        max_chunks = max(
            len(chunks)
            for chunks in chunks_by_document.values()
        )

        raise ValueError(
            f"Aucun document ne contient {n_chunks} chunks. "
            f"Le document ayant le plus de chunks en contient "
            f"{max_chunks}."
        )

    # ---------------------------------
    # Randomly choose ONE document
    # ---------------------------------

    selected_document = random.choice(
        eligible_documents
    )

    document_chunks = chunks_by_document[
        selected_document
    ]

    # ---------------------------------
    # Randomly choose 100 chunks
    # from this ONE document
    # ---------------------------------

    selected_chunks = random.sample(
        document_chunks,
        n_chunks,
    )

    # ---------------------------------
    # Save JSONL
    # ---------------------------------

    output_path = Path(output_file)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for chunk in selected_chunks:

            item = {
                "id": chunk["id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
            }

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"{len(selected_chunks)} chunks exportés."
    )

    print(
        f"Document sélectionné : {selected_document}"
    )

    print(
        f"Output : {output_path.resolve()}"
    )


def load_json_objects(filename):
    """
    Robust JSON loader.

    Handles:


    3. Multiple JSON objects without proper line separation:
       {...}{...}
    """

    text = Path(filename).read_text(encoding="utf-8").strip()

    if not text:
        return []

    # ---------------------------------
    # First try normal JSON
    # ---------------------------------
    try:
        data = json.loads(text)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            return [data]

    except json.JSONDecodeError:
        pass

    # ---------------------------------
    # Parse multiple JSON objects
    # ---------------------------------
    decoder = json.JSONDecoder()

    records = []
    position = 0
    length = len(text)

    while position < length:

        # Skip whitespace
        while position < length and text[position].isspace():
            position += 1

        if position >= length:
            break

        try:
            obj, end_position = decoder.raw_decode(text, position)

            if isinstance(obj, dict):
                records.append(obj)

            elif isinstance(obj, list):
                records.extend(obj)

            position = end_position

        except json.JSONDecodeError as e:
            print(
                f"Warning: could not parse JSON near "
                f"character {position}: {e}"
            )

            # Move forward until we find the next {
            next_object = text.find("{", position + 1)

            if next_object == -1:
                break

            position = next_object

    return records


def get_base_id(question_id):
    """
    Convert:
        abc123-q1
        abc123-q2
        abc123-q10

    into:
        abc123
    """
    if "-q" in question_id:
        return question_id.rsplit("-q", 1)[0]

    return question_id

    # =========================
    # Load files
    # =========================


def filter_questions(
    questions_file: str,
    answers_file: str,
) -> list[dict]:

    FIRST_FILE = "random_chunks.json"
    SECOND_FILE = "testdata.json"
    OUTPUT_FILE = "output_test_data.jsonl"

    first_records = load_json_objects(FIRST_FILE)
    second_records = load_json_objects(SECOND_FILE)

    first_by_id = {}

    for record in first_records:
        record_id = record.get("id")

        if record_id:
            first_by_id[record_id] = record

    output_records = []
    unmatched_questions = []

    for question_record in second_records:

        question_id = question_record.get("question_id")
        question = question_record.get("question")
        source = question_record.get("source")

        if not question_id:
            continue

        # Example:
        # b5d35cfc-2f61-543c-a5bc-d45aec61c0e4-q1
        #
        # becomes:
        # b5d35cfc-2f61-543c-a5bc-d45aec61c0e4

        base_id = get_base_id(question_id)

        first_record = first_by_id.get(base_id)

        if first_record is None:
            unmatched_questions.append(question_record)
            continue

        # Create a NEW record for every question
        output_record = {
            "id": first_record["id"],
            "text": first_record.get("text", ""),
            "metadata": first_record.get("metadata", {}),
            "question_id": question_id,
            "question": question,
            "source": source
        }

        output_records.append(output_record)

    # =========================
    # Save output as JSONL
    # =========================

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # =========================
    # Statistics
    # =========================

    print(f"First file records: {len(first_records)}")
    print(f"Questions: {len(second_records)}")
    print(f"Matched questions: {len(output_records)}")
    print(f"Unmatched questions: {len(unmatched_questions)}")
    print(f"Output saved to: {OUTPUT_FILE}")

    # Optional: save unmatched questions
    if unmatched_questions:
        with open("unmatched_questions.jsonl", "w", encoding="utf-8") as f:
            for record in unmatched_questions:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Unmatched questions saved to: unmatched_questions.jsonl")


def load_random_chunks(file_path):
    """
    Load random_chunks.json as JSONL.
    Returns:
        {
            "id1": {"id": "id1", "source": "...", ...},
            "id2": {"id": "id2", "source": "...", ...}
        }
    """
    chunks_by_id = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                chunk = json.loads(line)

                if "id" in chunk:
                    chunks_by_id[chunk["id"]] = chunk

            except json.JSONDecodeError as e:
                print(
                    f"Skipping invalid line {line_number}: {e}"
                )

    return chunks_by_id


def load_jsonl(file_path):
    """Load a JSONL file and index it by ID."""
    data_by_id = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)

                if "id" in item:
                    data_by_id[item["id"]] = item

            except json.JSONDecodeError as e:
                print(
                    f"Skipping invalid line {line_number} "
                    f"in {file_path}: {e}"
                )

    return data_by_id


def format_questions(
    questions_file,
    random_chunks_file,
    output_file
):
    # -----------------------------------------
    # Load random_chunks.json
    # -----------------------------------------
    chunks = load_jsonl(random_chunks_file)

    print(f"Loaded {len(chunks)} chunks")

    # -----------------------------------------
    # Load questions JSON
    # -----------------------------------------
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # If it's one object instead of a list
    if isinstance(questions, dict):
        questions = [questions]

    created = 0
    skipped = 0

    # -----------------------------------------
    # Create output JSONL
    # -----------------------------------------
    with open(output_file, "w", encoding="utf-8") as out:

        for item in questions:

            item_id = item.get("id")

            # -------------------------------------
            # Check if ID exists in random_chunks
            # -------------------------------------
            if item_id not in chunks:
                print(f"ID not found: {item_id}")
                skipped += 1
                continue

            # Get ALL data from random_chunks
            chunk = chunks[item_id]

            # -------------------------------------
            # Add hard question
            # -------------------------------------
            if item.get("hard_question"):

                result = chunk.copy()

                result["question"] = item["hard_question"]

                out.write(
                    json.dumps(
                        result,
                        ensure_ascii=False
                    ) + "\n"
                )

                created += 1

            # -------------------------------------
            # Add general question
            # -------------------------------------
            if item.get("general_question"):

                result = chunk.copy()

                result["question"] = item["general_question"]

                out.write(
                    json.dumps(
                        result,
                        ensure_ascii=False
                    ) + "\n"
                )

                created += 1

    print()
    print(f"Created: {created}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    # Select some random chunks to ask gpt to generate question
    DATA_DIR = "./data"
    CHROMA_DIR = "./chroma_db"
    COLLECTION_NAME = "documents"
    # export_random_chunks_to_jsonl(
    #    chroma_dir=CHROMA_DIR,
    #   collection_name=COLLECTION_NAME,
    #    output_file="random_chunks.json",
    #    n_chunks=100,
    #   seed=42,
    # )
    format_questions("test_data/llm_questions.json",
                     "random_chunks.json", "clean_test_data.jsonl")
