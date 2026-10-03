"""Inspect records stored in the persistent Chroma knowledge base."""

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from src.tools.rag_tools import get_vector_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect records in the local Chroma vector database.")
    parser.add_argument("--project", help="Filter by project, such as aryaq or aryashakti.")
    parser.add_argument("--record-type", help="Filter by record type, such as spreadsheet or manual_memory.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum records to display.")
    args = parser.parse_args()

    vector_store = get_vector_store()
    data = vector_store.get(include=["metadatas", "documents"])
    ids = data.get("ids", [])
    metadatas = data.get("metadatas", [])
    documents = data.get("documents", [])

    filtered = []
    for record_id, metadata, document in zip(ids, metadatas, documents):
        metadata = metadata or {}
        if args.project and metadata.get("project") != args.project:
            continue
        if args.record_type and metadata.get("record_type") != args.record_type:
            continue
        filtered.append((record_id, metadata, document or ""))

    print(f"Collection: {vector_store._collection.name}")
    print(f"Total records: {len(ids)}")
    print(f"Matching records: {len(filtered)}")
    print(f"Record types: {dict(Counter(item[1].get('record_type', 'unknown') for item in filtered))}")
    print()

    for index, (record_id, metadata, document) in enumerate(filtered[: max(args.limit, 0)], start=1):
        print(f"[{index}] id={record_id}")
        print(f"metadata={metadata}")
        print(document[:500].replace("\n", " "))
        if len(document) > 500:
            print("...")
        print()


if __name__ == "__main__":
    main()