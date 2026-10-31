from __future__ import annotations

from pathlib import Path
from typing import Iterator
from ragsystem.utils.parser import DocumentParser
from docling_core.types.doc import DoclingDocument


class RepositoryDocumentParser:
    def __init__(
        self,
        data_dir: str | Path = "./data",
        enable_ocr: bool = True,
        enable_tables: bool = True,
        accurate_tables: bool = False,
    ):
        self.data_dir = Path(data_dir)

        self.parser = DocumentParser(
            enable_ocr=enable_ocr,
            enable_tables=enable_tables,
            accurate_tables=accurate_tables,
        )

    def files(self) -> Iterator[Path]:
        """Return all supported files in the repository data directory."""
        yield from sorted(
            path
            for path in self.data_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in DocumentParser.SUPPORTED_EXTENSIONS
        )

    def parse_file(self, path: str | Path) -> DoclingDocument:
        """Parse one file and return its DoclingDocument."""
        result = self.parser.parse(str(path))
        return result.document

    def parse_all(self) -> list[DoclingDocument]:
        """Parse every file in the data directory."""
        documents = []

        for path in self.files():
            try:
                doc = self.parse_file(path)
                documents.append(doc)

                print(f"Parsed: {path}")

            except Exception as e:
                print(f"Failed: {path}")
                print(f"Error: {e}")

        return documents
