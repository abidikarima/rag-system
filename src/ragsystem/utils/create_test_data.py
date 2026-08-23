
import json
import random
import chromadb
from pathlib import Path


def export_random_chunks_to_jsonl(
    chroma_dir: str,
    collection_name: str,
    output_file: str,
    n_chunks: int = 20,
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

    n_chunks = min(n_chunks, total)

    all_ids = collection.get(
        include=[]
    )["ids"]

    if seed is not None:
        random.seed(seed)

    selected_ids = random.sample(
        all_ids,
        n_chunks,
    )

    selected = collection.get(
        ids=selected_ids,
        include=[
            "documents",
            "metadatas",
        ],
    )

    output_path = Path(output_file)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for chunk_id, text, metadata in zip(
            selected["ids"],
            selected["documents"],
            selected["metadatas"],
        ):
            item = {
                "id": chunk_id,
                "text": text,
                "metadata": metadata,
            }

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"{n_chunks} chunks exportés vers "
        f"{output_path.resolve()}"
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


if __name__ == "__main__":
    # Select some random chunks to ask gpt to generate question
    DATA_DIR = "./data"
    CHROMA_DIR = "./chroma_db"
    COLLECTION_NAME = "documents"
    # export_random_chunks_to_jsonl(
    #   chroma_dir=CHROMA_DIR,
    #   collection_name=COLLECTION_NAME,
    #   output_file="random_chunks.json",
    #   n_chunks=30,
    #    seed=42,
    # )

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
