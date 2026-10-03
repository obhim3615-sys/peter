"""Index project documents without deleting saved conversation memories."""

import sys
from pathlib import Path

# Allow both `python scripts/build_rag.py` and `python -m scripts.build_rag`.
PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.tools.file_tools import PROJECT_ROOT
from src.tools.rag_tools import get_vector_store

load_dotenv()

DOCUMENT_EXTENSIONS = {".txt", ".md"}


def build_database() -> None:
    doc_dir = PROJECT_ROOT / "data" / "documents"
    doc_dir.mkdir(parents=True, exist_ok=True)

    paths = [p for p in sorted(doc_dir.iterdir()) if p.is_file() and p.suffix.lower() in DOCUMENT_EXTENSIONS]
    if not paths:
        print(f"No documents found. Add .txt or .md files to {doc_dir}.")
        return

    documents = []
    print(f"Reading {len(paths)} document(s) from {doc_dir}...")
    for path in tqdm(paths, desc="Reading files", unit="file"):
        documents.append(
            Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name})
        )

    print(f"Splitting {len(documents)} document(s) into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    raw_chunks = splitter.split_documents(documents)
    chunks = [
        Document(
            page_content=chunk.page_content,
            metadata={**chunk.metadata, "record_type": "document", "chunk_index": index},
        )
        for index, chunk in enumerate(raw_chunks)
    ]

    vector_store = get_vector_store()
    old_document_ids = vector_store.get(where={"record_type": "document"})["ids"]
    if old_document_ids:
        print(f"Clearing {len(old_document_ids)} previous document records...")
        vector_store.delete(ids=old_document_ids)

    document_ids = [f"document-{chunk.metadata['source']}-{chunk.metadata['chunk_index']}" for chunk in chunks]

    batch_size = 50
    print(f"Indexing {len(chunks)} document chunk(s) into ChromaDB...")
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches", unit="batch"):
        batch_docs = chunks[i : i + batch_size]
        batch_ids = document_ids[i : i + batch_size]
        vector_store.add_documents(batch_docs, ids=batch_ids)

    print(f"Indexed {len(chunks)} document chunks. Existing conversation memories were preserved.")



if __name__ == "__main__":
    build_database()
