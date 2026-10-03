"""Import project spreadsheets and CSV files into the persistent RAG store.

Usage
-----
# Index default project (aryashakti) from default data/csv_files/ folder:
    python scripts/import_project_knowledge.py

# Index a NEW project from a custom directory WITHOUT touching existing data:
    python scripts/import_project_knowledge.py --project mynewapp --dir C:\\path\\to\\mynewapp\\data

Each project's records are stored with a unique project= metadata tag in ChromaDB.
Re-indexing a project ONLY deletes that project's old records — other projects are untouched.
"""

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path
from typing import Iterable

from openpyxl import load_workbook
import fitz
from pypdf import PdfReader
from tqdm import tqdm
from langchain_core.documents import Document

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from src.tools.file_tools import PROJECT_ROOT
from src.tools.rag_tools import get_vector_store
from src.tools.code_intelligence import index_project_code


# ---------------------------------------------------------------------------
# Defaults (used when no CLI args are given)
# ---------------------------------------------------------------------------
DEFAULT_SOURCE_DIRECTORY = PROJECT_DIRECTORY / "data" / "csv_files"
DEFAULT_PROJECT_NAME     = "aryashakti"

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xlsm", ".md", ".txt", ".pdf"}
SENSITIVE_MARKERS    = ("password", "secret", "token", "api_key", "apikey", "private_key", "credential")

# Database GIS/framework boilerplate files to exclude from RAG indexing
IGNORED_TABLE_PREFIXES = (
    "spatial_ref_sys",
    "django_session",
    "django_migrations",
    "django_admin_log",
    "django_content_type",
    "auth_permission",
)

MAX_ROWS_PER_TABLE = 20  # Cap sample rows per table to keep indexing fast & memory-efficient (10-20 rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(value: object) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _stable_id(project: str, source: str, content: str) -> str:
    """Stable, deterministic ID scoped to (project, source, content)."""
    digest = hashlib.sha256(f"{project}\n{source}\n{content}".encode("utf-8")).hexdigest()
    return f"spreadsheet-{project}-{digest}"


def _row_document(
    source: str,
    sheet: str,
    row_number: int,
    fields: Iterable[tuple[str, object]],
    *,
    project: str,
) -> Document | None:
    raw_values = [(str(name), _clean(value)) for name, value in fields if _clean(value)]
    if not raw_values:
        return None

    # Check cell values (not column header names) for actual secrets
    values_text = " ".join(val for _, val in raw_values).lower()
    sensitive_value_markers = ("password=", "api_key=", "private_key=", "secret_key=")
    if any(marker in values_text for marker in sensitive_value_markers):
        return None

    content = "\n".join(f"{name}: {value}" for name, value in raw_values)
    page_content = (
        f"Project knowledge\nProject: {project}\nWorkbook: {source}\nSheet: {sheet}\n"
        f"Row: {row_number}\n{content}"
    )
    return Document(
        page_content=page_content,
        metadata={
            "record_type": "spreadsheet",
            "source":       source,
            "project":      project,          # ← scoped to project
            "sheet":        sheet,
            "row":          row_number,
        },
    )


# ---------------------------------------------------------------------------
# File parsers
# ---------------------------------------------------------------------------

def _is_spec_file(path: Path) -> bool:
    name = path.name.lower()
    return any(k in name for k in ("audit", "kt", "api", "overview", "mapping", "schema", "role"))


def _xlsx_documents(path: Path, project: str) -> list[Document]:
    workbook  = load_workbook(path, read_only=True, data_only=True)
    documents: list[Document] = []
    is_spec = _is_spec_file(path)
    row_limit = None if is_spec else MAX_ROWS_PER_TABLE

    for sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]
        rows  = list(sheet.iter_rows(values_only=True))
        if not rows:
            continue
        headers = [_clean(value) or f"Column {i + 1}" for i, value in enumerate(rows[0])]
        row_slice = rows[1:] if row_limit is None else rows[1 : row_limit + 1]
        for row_number, row in enumerate(row_slice, start=2):
            doc = _row_document(path.name, sheet_name, row_number, zip(headers, row), project=project)
            if doc:
                documents.append(doc)
    return documents


def _csv_documents(path: Path, project: str) -> list[Document]:
    documents: list[Document] = []
    is_spec = _is_spec_file(path)
    row_limit = None if is_spec else MAX_ROWS_PER_TABLE

    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []

        # Index table schema metadata (table name + columns) so even 0-row tables are known
        if headers and not is_spec:
            clean_table_name = re.sub(r"_\d{12}$", "", path.stem)
            schema_text = (
                f"Database Table Schema\n"
                f"Project: {project}\n"
                f"Table name: {clean_table_name}\n"
                f"File: {path.name}\n"
                f"Columns ({len(headers)}): {', '.join(headers)}\n"
                f"Schema description: Table {clean_table_name} has columns {', '.join(headers)}"
            )
            documents.append(
                Document(
                    page_content=schema_text,
                    metadata={
                        "record_type": "table_schema",
                        "source": path.name,
                        "project": project,
                        "table": clean_table_name,
                    },
                )
            )

        for row_number, row in enumerate(reader, start=2):
            if row_limit is not None and row_number > row_limit + 1:
                break
            doc = _row_document(path.name, path.stem, row_number, row.items(), project=project)
            if doc:
                documents.append(doc)
    return documents


def _markdown_documents(path: Path, project: str) -> list[Document]:
    text     = path.read_text(encoding="utf-8")
    sections = re.split(r"(?=^#{1,3} )", text, flags=re.MULTILINE)
    documents: list[Document] = []
    for section_number, section in enumerate(sections, start=1):
        content = section.strip()
        if not content:
            continue
        doc = _row_document(path.name, "Markdown", section_number, [("Section", content)], project=project)
        if doc:
            documents.append(doc)
    return documents


def _text_documents(path: Path, project: str) -> list[Document]:
    text       = path.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    documents: list[Document] = []
    for section_number, paragraph in enumerate(paragraphs, start=1):
        doc = _row_document(path.name, "Text", section_number, [("Content", paragraph)], project=project)
        if doc:
            documents.append(doc)
    return documents


def _pdf_documents(path: Path, project: str) -> list[Document]:
    documents: list[Document] = []
    reader    = PdfReader(str(path))
    ocr_engine = None
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if len(text) < 40:
            if ocr_engine is None:
                from rapidocr_onnxruntime import RapidOCR
                ocr_engine = RapidOCR()
            pdf_document = fitz.open(str(path))
            pixmap       = pdf_document[page_number - 1].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            import numpy as np
            image_array  = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
            ocr_result, _ = ocr_engine(image_array)
            if ocr_result:
                text = "\n".join(item[1] for item in ocr_result)
            pdf_document.close()
        doc = _row_document(path.name, f"PDF page {page_number}", page_number, [("Content", text)], project=project)
        if doc:
            documents.append(doc)
    return documents


# ---------------------------------------------------------------------------
# Main indexing logic
# ---------------------------------------------------------------------------

def build_documents(source_directory: Path, project: str) -> list[Document]:
    target_files = [
        path for path in sorted(source_directory.iterdir())
        if path.is_file()
        and not path.name.startswith("~$")
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not any(path.name.lower().startswith(prefix) for prefix in IGNORED_TABLE_PREFIXES)
    ]
    if not target_files:
        return []

    print(f"Reading {len(target_files)} file(s) from {source_directory} (project={project})...")
    documents: list[Document] = []
    for path in tqdm(target_files, desc="Parsing files", unit="file"):
        ext = path.suffix.lower()
        if ext in {".xlsx", ".xlsm"}:
            documents.extend(_xlsx_documents(path, project))
        elif ext == ".md":
            documents.extend(_markdown_documents(path, project))
        elif ext == ".txt":
            documents.extend(_text_documents(path, project))
        elif ext == ".pdf":
            documents.extend(_pdf_documents(path, project))
        else:
            documents.extend(_csv_documents(path, project))
    return documents


def refresh_project_knowledge(source_directory: Path, project: str) -> int:
    """Re-index ONE project's records without touching any other project's data.

    Flow:
      1. Delete ONLY records tagged project=<project> and record_type=spreadsheet
      2. Parse all files from source_directory
      3. Tag every new document with project=<project>
      4. Insert into ChromaDB
    """
    documents = build_documents(source_directory, project)
    if not documents:
        return 0

    vector_store = get_vector_store()

    # Delete ONLY this project's old spreadsheet records — other projects untouched
    old = vector_store.get(
        where={"$and": [{"record_type": "spreadsheet"}, {"project": project}]}
    )
    old_ids = old.get("ids", [])
    if old_ids:
        print(f"Removing {len(old_ids)} old records for project '{project}'...")
        vector_store.delete(ids=old_ids)

    ids = [_stable_id(project, doc.metadata["source"], doc.page_content) for doc in documents]

    batch_size = 50
    print(f"Indexing {len(documents)} records for project '{project}'...")
    for i in tqdm(range(0, len(documents), batch_size), desc="Embedding batches", unit="batch"):
        vector_store.add_documents(documents[i : i + batch_size], ids=ids[i : i + batch_size])

    return len(documents)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import project knowledge into ChromaDB RAG store.")
    parser.add_argument(
        "--project", "-p",
        default=DEFAULT_PROJECT_NAME,
        help=f"Project name tag (default: {DEFAULT_PROJECT_NAME})",
    )
    parser.add_argument(
        "--dir", "-d",
        default=str(DEFAULT_SOURCE_DIRECTORY),
        help=f"Source directory containing CSV/XLSX/MD files (default: {DEFAULT_SOURCE_DIRECTORY})",
    )
    parser.add_argument(
        "--code", "-c",
        action="store_true",
        default=True,
        help="Also index Python code structure (classes, functions, imports) via AST (default: True)",
    )
    args = parser.parse_args()

    source_dir = Path(args.dir)
    if not source_dir.exists():
        print(f"ERROR: Directory not found: {source_dir}")
        sys.exit(1)

    count = refresh_project_knowledge(source_dir, args.project)
    if count:
        print(f"Done. Indexed {count} records for project '{args.project}' from {source_dir}")
    else:
        print(f"No supported files found in {source_dir}")

    if args.code:
        print(f"\nIndexing Python code architecture for project '{args.project}'...")
        code_result = index_project_code(args.project)
        print(code_result)


if __name__ == "__main__":
    main()