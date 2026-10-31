from __future__ import annotations

import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable

from huggingface_hub import hf_hub_download


# OHR-Bench PDF corpus on the Hugging Face Hub.
# The revision is pinned so the corpus (and therefore chunk ids
# and the test data built from it) stays reproducible.
REPO_ID = "opendatalab/OHR-Bench"
REPO_FILENAME = "pdfs.zip"
REPO_REVISION = "7f833e3eda9a571a9ea545a8f6d476fa1685033d"

ALL_CATEGORIES = (
    "academic",
    "administration",
    "finance",
    "law",
    "manual",
    "news",
    "textbook",
)

# "textbook" is excluded by default: it is the largest category
# and was never part of the indexed corpus.
DEFAULT_CATEGORIES = (
    "academic",
    "administration",
    "finance",
    "law",
    "manual",
    "news",
)


def has_documents(data_dir: str | Path) -> bool:
    """Return True if the data directory already contains PDFs."""
    data_dir = Path(data_dir)

    return data_dir.is_dir() and any(data_dir.glob("*.pdf"))


def download_data(
    data_dir: str | Path = "./data",
    categories: Iterable[str] = DEFAULT_CATEGORIES,
) -> int:
    """
    Download the OHR-Bench PDFs and extract them into `data_dir`.

    PDFs from every selected category are written flat into
    `data_dir` (no category sub-folders), which is the layout the
    ingestion pipeline and the existing chunk ids expect.

    Files that already exist are left untouched.

    Returns the number of newly extracted files.
    """

    data_dir = Path(data_dir)
    categories = set(categories)

    unknown = categories - set(ALL_CATEGORIES)

    if unknown:
        raise ValueError(
            f"Unknown categories: {sorted(unknown)}. "
            f"Available: {list(ALL_CATEGORIES)}"
        )

    # Cached by huggingface_hub, so the ~1.5 GB archive is only
    # downloaded once per machine.
    print(f"Downloading {REPO_FILENAME} from {REPO_ID} ...")

    zip_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=REPO_FILENAME,
        repo_type="dataset",
        revision=REPO_REVISION,
    )

    data_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():

            if member.is_dir():
                continue

            parts = PurePosixPath(member.filename).parts

            if len(parts) < 2 or parts[0] not in categories:
                continue

            # Only keep the file name: flattens the layout and
            # prevents path traversal from archive entries.
            name = parts[-1]

            if not name.lower().endswith(".pdf"):
                continue

            target = data_dir / name

            if target.exists():
                continue

            tmp_target = target.with_name(target.name + ".part")

            with archive.open(member) as src, open(tmp_target, "wb") as dst:
                shutil.copyfileobj(src, dst)

            tmp_target.replace(target)
            extracted += 1

    print(f"Extracted {extracted} PDFs into {data_dir.resolve()}")

    return extracted


def ensure_data(
    data_dir: str | Path = "./data",
    categories: Iterable[str] = DEFAULT_CATEGORIES,
    force: bool = False,
) -> None:
    """
    Make sure `data_dir` contains the corpus.

    Downloads only when the folder has no PDFs yet, unless
    `force` is True (useful to fill in missing categories).
    """

    if has_documents(data_dir) and not force:
        print(f"Data already present in {Path(data_dir).resolve()}, skipping download.")
        return

    download_data(data_dir=data_dir, categories=categories)


if __name__ == "__main__":
    ensure_data()
