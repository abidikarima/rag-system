from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption


@dataclass
class ParsedDocument:
    """
    Result returned by DocumentParser.

    `document` is the original DoclingDocument object.
    Nothing is flattened or converted to plain text here.
    """

    document: Any
    source: str
    filename: str
    extension: str
    pages: Optional[int] = None
    markdown: Optional[str] = None

    def to_markdown(self) -> str:
        """Convert the DoclingDocument to Markdown on demand."""
        if self.markdown is None:
            self.markdown = self.document.export_to_markdown()

        return self.markdown

    def to_dict(self) -> dict:
        """Lossless-ish Docling representation."""
        return self.document.export_to_dict()

    def text(self) -> str:
        """
        Convenience method for getting plain text.

        For RAG, you will normally want to use Docling's
        structure/chunking rather than simply doing this.
        """
        return self.document.export_to_markdown()


class DocumentParser:
    """
    Generic document parser based on Docling.

    Supported:
        - PDF
        - DOCX
        - HTML
        - optionally other formats supported by Docling

    The parser returns the native DoclingDocument object.
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf": InputFormat.PDF,
        ".docx": InputFormat.DOCX,
        ".html": InputFormat.HTML,
        ".htm": InputFormat.HTML,
    }

    def __init__(
        self,
        enable_ocr: bool = True,
        enable_tables: bool = True,
        accurate_tables: bool = False,
    ):
        """
        Args:
            enable_ocr:
                Enable OCR for PDFs/images.

            enable_tables:
                Enable table structure recognition.

            accurate_tables:
                Use the more accurate TableFormer mode.
                This is slower.
        """

        pdf_options = PdfPipelineOptions(
            do_ocr=enable_ocr,
            do_table_structure=enable_tables,
        )

        if accurate_tables:
            from docling.datamodel.pipeline_options import TableFormerMode

            pdf_options.table_structure_options.mode = (
                TableFormerMode.ACCURATE
            )

        self.converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.DOCX,
                InputFormat.HTML,
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_options
                )
            },
        )

    def parse(
        self,
        source: str | Path,
        *,
        max_pages: Optional[int] = None,
    ) -> ParsedDocument:
        """
        Parse one document.

        Args:
            source:
                Local file path or URL.

            max_pages:
                Optional maximum number of pages for PDFs.

        Returns:
            ParsedDocument containing the native DoclingDocument.
        """

        source = str(source)

        path = Path(source)

        # ---------------------------------------------------------
        # Local file
        # ---------------------------------------------------------

        if path.exists():
            extension = path.suffix.lower()

            if extension not in self.SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"Unsupported file type: {extension}. "
                    f"Supported: {list(self.SUPPORTED_EXTENSIONS)}"
                )

            filename = path.name

        # ---------------------------------------------------------
        # URL
        # ---------------------------------------------------------

        elif source.startswith(("http://", "https://")):
            extension = Path(source.split("?")[0]).suffix.lower()

            # URLs don't always have an extension.
            # Let Docling try to detect them.
            if extension not in self.SUPPORTED_EXTENSIONS:
                extension = ""

            filename = Path(source.split("?")[0]).name or "remote_document"

        else:
            raise FileNotFoundError(
                f"Document does not exist: {source}"
            )

        # ---------------------------------------------------------
        # Convert
        # ---------------------------------------------------------

        if max_pages is not None:
            result = self.converter.convert(
                source,
                max_num_pages=max_pages,
            )
        else:
            result = self.converter.convert(source)

        # ---------------------------------------------------------
        # Check conversion
        # ---------------------------------------------------------

        if result.document is None:
            raise RuntimeError(
                f"Docling failed to parse: {source}"
            )

        # ---------------------------------------------------------
        # Number of pages
        # ---------------------------------------------------------

        pages = None

        try:
            pages = len(result.document.pages)
        except Exception:
            pass

        return ParsedDocument(
            document=result.document,
            source=source,
            filename=filename,
            extension=extension,
            pages=pages,
        )

    def parse_many(
        self,
        sources: list[str | Path],
    ) -> list[ParsedDocument]:
        """
        Parse multiple documents.
        """

        results = []

        for source in sources:
            try:
                result = self.parse(source)
                results.append(result)

            except Exception as exc:
                print(
                    f"[ERROR] Failed to parse {source}: {exc}"
                )

        return results


parser = DocumentParser(
    enable_ocr=True,
    enable_tables=True,
    accurate_tables=False,
)


result = parser.parse("./data/3M_2023Q2_10Q.pdf")


print(result.filename)
print(result.extension)
print(result.pages)


# This is the actual DoclingDocument
doc = result.document

print(type(doc))


# Export only when you need it
markdown = result.to_markdown()

print(markdown[:2000])
