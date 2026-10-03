"""Persistent RAG storage for project documents and completed conversations."""

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import re
from pathlib import Path
from typing import List

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.tools.file_tools import PROJECT_ROOT
from src.utils.project_config import infer_project as _infer_project
from src.utils.logger import setup_logger

logger = setup_logger("rag-tools")

APP_ROOT = Path(__file__).resolve().parents[2]

CHROMA_DIRECTORY = APP_ROOT / "data" / "chroma_db"
# Use a separate collection for the local ONNX embeddings because the previous
# Google-backed vectors were created with a different embedding dimension.
COLLECTION_NAME = "orchestrator_knowledge_local_onnx"
LOCAL_EMBEDDING_DIR = APP_ROOT / "data" / "vector_store"
LOCAL_MODEL_PATH = LOCAL_EMBEDDING_DIR / "model.onnx"
LOCAL_TOKENIZER_DIR = LOCAL_EMBEDDING_DIR / "local_tokenizer"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "for",
    "with",
    "from",
    "into",
    "that",
    "this",
    "these",
    "those",
    "their",
    "there",
    "about",
    "have",
    "has",
    "been",
    "what",
    "when",
    "where",
    "why",
    "who",
    "how",
    "which",
    "your",
    "you",
    "we",
    "our",
    "are",
    "is",
    "was",
    "were",
    "it",
    "its",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "as",
    "if",
    "then",
    "than",
    "not",
    "but",
    "can",
    "could",
    "should",
    "would",
    "will",
    "may",
    "also",
}


class LocalONNXEmbedding:
    """Embedding function backed by the local ONNX model stored in the project vector_store folder."""

    def __init__(self) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        if not LOCAL_MODEL_PATH.exists():
            raise FileNotFoundError(f"Local embedding model not found: {LOCAL_MODEL_PATH}")
        if not LOCAL_TOKENIZER_DIR.exists():
            raise FileNotFoundError(f"Local tokenizer folder not found: {LOCAL_TOKENIZER_DIR}")

        self.ort = ort
        self.tokenizer = Tokenizer.from_file(str(LOCAL_TOKENIZER_DIR / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)

        session_options = self.ort.SessionOptions()
        session_options.log_severity_level = 3
        session_options.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = self.ort.InferenceSession(
            str(LOCAL_MODEL_PATH),
            providers=["CPUExecutionProvider"],
            sess_options=session_options,
        )

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vectors, axis=1)
        norm[norm == 0] = 1e-12
        return vectors / norm[:, np.newaxis]

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        encoded = [self.tokenizer.encode(text) for text in texts]
        input_ids = np.array([item.ids for item in encoded], dtype=np.int64)
        attention_mask = np.array([item.attention_mask for item in encoded], dtype=np.int64)
        token_type_ids = np.array([np.zeros(len(item), dtype=np.int64) for item in input_ids], dtype=np.int64)

        onnx_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        last_hidden_state = self.session.run(None, onnx_input)[0]
        input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask, -1), last_hidden_state.shape)
        pooled = np.sum(last_hidden_state * input_mask_expanded, axis=1) / np.clip(
            input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None
        )
        vectors = self._normalize(pooled).astype(np.float32)
        return [vector.tolist() for vector in vectors]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_batch(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def __call__(self, texts: List[str]):
        return self.embed_documents(texts)


def get_vector_store() -> Chroma:
    """Return the single persistent collection shared by documents and memories."""
    CHROMA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    embeddings = LocalONNXEmbedding()
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIRECTORY),
        embedding_function=embeddings,
    )


def _stable_id(prefix: str, content: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"{prefix}-{digest}"


def _normalize_text(value: str) -> str:
    return " ".join(str(value).split())



# Fix #4 — shared splitter instance (created once, reused across all calls)
# RecursiveCharacterTextSplitter tries to split on paragraphs first (\n\n),
# then single newlines, then sentences (". "), then words — so chunks always
# respect natural text boundaries.  chunk_overlap gives a true sliding window
# that the old hand-rolled function did not correctly implement.
_TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    length_function=len,
    is_separator_regex=False,
)


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split *text* into overlapping chunks suitable for vector indexing.

    Uses LangChain's RecursiveCharacterTextSplitter so boundaries are always
    respected in priority order: paragraph → newline → sentence → word.
    A new splitter is created only when the caller overrides the defaults;
    the module-level ``_TEXT_SPLITTER`` instance is reused otherwise.
    """
    cleaned = _normalize_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    # Use the pre-built instance for default sizes (fast path)
    if chunk_size == CHUNK_SIZE and chunk_overlap == CHUNK_OVERLAP:
        splitter = _TEXT_SPLITTER
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

    chunks = splitter.split_text(cleaned)
    return [c for c in chunks if c.strip()]



def _keyword_overlap_score(query: str, document_text: str) -> float:
    words = {
        token
        for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_/-]{2,}", (query or "").lower())
        if token not in STOP_WORDS and len(token) > 2
    }
    if not words:
        return 0.0
    text = (document_text or "").lower()

    # Exact Identifier Boost: table names (e.g., app_farm_crop_master) get strong priority
    identifiers = [w for w in words if "_" in w or "/" in w]
    id_boost = 0.0
    for ident in identifiers:
        if ident in text:
            id_boost += 1.5

    matches = sum(1 for word in words if word in text)
    base_score = matches / max(len(words), 1)
    return base_score + id_boost



def _build_metadata(source: str | None, project: str | None = None, *, source_file: str | None = None, line_number: int | None = None, page_number: int | None = None, chunk_index: int | None = None, chunk_count: int | None = None) -> dict:
    metadata = {
        "record_type": "manual_memory",
        "source": source or "manual memory",
        "source_file": source_file or source or "manual_memory",
        "project": project or "general",
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }
    if line_number is not None:
        metadata["line_number"] = int(line_number)
    if page_number is not None:
        metadata["page_number"] = int(page_number)
    if chunk_index is not None:
        metadata["chunk_index"] = int(chunk_index)
    if chunk_count is not None:
        metadata["chunk_count"] = int(chunk_count)
    return metadata


def _collection_filter(project: str | None = None) -> dict:
    if project and project != "general":
        return {"$or": [{"project": project}, {"project": "general"}]}
    return {}



def _cleanup_conversation_memory(vector_store: Chroma, max_records: int = 50) -> None:
    existing = vector_store.get(where={"record_type": "conversation"})
    ids = existing.get("ids", [])
    metadatas = existing.get("metadatas", [])
    if not ids:
        return

    grouped: dict[str, list[str]] = defaultdict(list)
    for record_id, metadata in zip(ids, metadatas):
        if not metadata:
            continue
        fingerprint = metadata.get("fingerprint") or metadata.get("source") or record_id
        grouped[fingerprint].append(record_id)

    to_delete: list[str] = []
    for fingerprint, record_ids in grouped.items():
        if len(record_ids) <= 1:
            continue
        for stale_id in record_ids[:-1]:
            to_delete.append(stale_id)

    if len(ids) > max_records:
        to_delete.extend(ids[: max(0, len(ids) - max_records)])

    if to_delete:
        vector_store.delete(ids=sorted(set(to_delete)))


def search_internal_knowledge(query: str, project: str | None = None) -> str:
    """Searches indexed project documents and completed conversation memories."""
    try:
        if not query or not query.strip():
            return "No relevant information found in the knowledge base."

        selected_project = project or _infer_project(query)
        vector_store = get_vector_store()
        search_filter = _collection_filter(selected_project)

        # --- Fix #5: use built-in scored search — no manual re-embedding ---
        # similarity_search_with_relevance_scores returns (Document, float) pairs
        # where the score is a cosine similarity already computed by ChromaDB.
        scored_results: list[tuple[Document, float]] = (
            vector_store.similarity_search_with_relevance_scores(
                query, k=15, filter=search_filter or None
            )
        )

        # Build a combined dict keyed by content hash for deduplication
        combined: dict[str, dict] = {}
        for doc, score in scored_results:
            key = hashlib.sha256(
                f"{doc.page_content}\n{doc.metadata}".encode("utf-8")
            ).hexdigest()
            combined[key] = {"doc": doc, "score": float(score) + 0.15}

        # Keyword-overlap pass (boosts exact-term matches on top of vector score)
        collection = getattr(vector_store, "_collection", None)
        if collection is not None:
            raw = collection.get(
                where=search_filter or None, include=["documents", "metadatas"]
            )
            for idx, doc_text in enumerate(raw.get("documents", [])):
                metadata = (
                    raw.get("metadatas", [])[idx]
                    if idx < len(raw.get("metadatas", []))
                    else {}
                )
                if not doc_text:
                    continue
                kw_score = _keyword_overlap_score(query, doc_text)
                if kw_score <= 0:
                    continue
                key = hashlib.sha256(
                    f"{doc_text}\n{metadata}".encode("utf-8")
                ).hexdigest()
                if key in combined:
                    combined[key]["score"] += kw_score * 1.35
                else:
                    combined[key] = {
                        "doc": Document(page_content=doc_text, metadata=metadata or {}),
                        "score": kw_score * 1.25,
                    }

        all_ranked = sorted(combined.values(), key=lambda item: item["score"], reverse=True)

        # Source diversity: prevent one large table CSV from hogging all top slots.
        # Allow at most 2 chunks per unique source file.
        ranked: list[dict] = []
        source_counts: dict[str, int] = {}
        for item in all_ranked:
            src = item["doc"].metadata.get("source", item["doc"].metadata.get("source_file", "unknown"))
            if source_counts.get(src, 0) < 2:
                ranked.append(item)
                source_counts[src] = source_counts.get(src, 0) + 1
                if len(ranked) >= 6:
                    break

        if not ranked:
            return "No relevant information found in the knowledge base."

        # --- Efficient result assembly ---
        # Each LLM call has a limited context window (~4k-8k tokens for local Ollama).
        # Strategy:
        #   1. Cap each chunk at MAX_CHUNK_CHARS, trimmed at last sentence boundary.
        #   2. Stop adding chunks once cumulative size exceeds TOKEN_BUDGET.
        # This keeps the BEST (highest-scored) chunks fully intact instead of
        # blindly cutting the final concatenated string mid-word.
        MAX_CHUNK_CHARS = 1_500   # ~375 tokens per chunk — enough for a full table row
        TOKEN_BUDGET    = 8_000   # ~2,000 tokens total — safe for Ollama 4k–8k context

        def _trim_to_sentence(text: str, limit: int) -> str:
            """Trim text to `limit` chars, ending at the last sentence boundary."""
            if len(text) <= limit:
                return text
            cut = text[:limit]
            # Find last sentence-ending punctuation before the cut
            for sep in (". ", ".\n", "! ", "? ", "\n\n", "\n"):
                pos = cut.rfind(sep)
                if pos > limit // 2:          # avoid trimming too aggressively
                    return cut[: pos + len(sep)].rstrip()
            return cut.rstrip()               # fallback: word-boundary trim

        report_parts: list[str] = [f"Found {len(ranked)} relevant knowledge snippets:\n"]
        total_chars = len(report_parts[0])

        for item in ranked:
            doc = item["doc"]
            source      = doc.metadata.get("source", doc.metadata.get("source_file", "Unknown"))
            record_type = doc.metadata.get("record_type", "document")
            source_file = doc.metadata.get("source_file")
            line_number = doc.metadata.get("line_number")
            page_number = doc.metadata.get("page_number")
            source_label = source_file or source
            if line_number is not None:
                source_label = f"{source_label} (line {line_number})"
            if page_number is not None:
                source_label = f"{source_label} (page {page_number})"

            header  = f"\n--- {record_type.title()} source: {source_label} ---\n"
            content = _trim_to_sentence(doc.page_content, MAX_CHUNK_CHARS)
            block   = header + content + "\n"

            if total_chars + len(block) > TOKEN_BUDGET:
                # Budget exhausted — skip remaining lower-scored chunks
                logger.debug(
                    f"Token budget reached after {len(report_parts) - 1} chunks "
                    f"({total_chars} chars). Skipping remaining."
                )
                break

            report_parts.append(block)
            total_chars += len(block)

        return "".join(report_parts)
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"



def save_to_knowledge_base(
    topic: str,
    information: str | None = None,
    content: str | None = None,
    *,
    project: str | None = None,
    source_file: str | None = None,
    line_number: int | None = None,
    page_number: int | None = None,
) -> str:
    """Immediately saves supplied knowledge to the persistent vector database."""
    try:
        if information is None and content is not None:
            information = content
        elif information is None:
            raise ValueError("Either 'information' or 'content' must be provided.")

        selected_project = project or _infer_project(topic, information)
        source_name = source_file or topic
        vector_store = get_vector_store()
        existing = vector_store.get(where={"$and": [{"source": source_name}, {"record_type": "manual_memory"}]})
        existing_ids = existing.get("ids", [])
        if existing_ids:
            vector_store.delete(ids=existing_ids)

        combined_text = f"Topic: {topic}\nInformation: {information}"
        chunks = _chunk_text(combined_text)
        documents: list[Document] = []
        for chunk_index, chunk in enumerate(chunks):
            metadata = _build_metadata(
                topic,
                selected_project,
                source_file=source_name,
                line_number=line_number,
                page_number=page_number,
                chunk_index=chunk_index,
                chunk_count=len(chunks),
            )
            metadata["record_type"] = "manual_memory"
            documents.append(
                Document(
                    page_content=chunk,
                    metadata=metadata,
                )
            )

        ids = [_stable_id(f"manual-memory-{index}", f"{source_name}:{index}:{chunk}") for index, chunk in enumerate(chunks)]
        vector_store.add_documents(documents, ids=ids)
        return f"Success! '{topic}' is now stored and searchable in the knowledge base."
    except Exception as e:
        return f"Error saving knowledge: {str(e)}"


def save_conversation_memory(user_request: str, assistant_response: str) -> str:
    """Stores one completed user/assistant exchange for future semantic retrieval."""
    try:
        content_lower = f"{user_request}\n{assistant_response}".lower()
        sensitive_markers = ("password", "secret", "token", "api_key", "apikey", "private_key", "credential")
        if any(marker in content_lower for marker in sensitive_markers):
            return "Conversation skipped because it may contain credential information."

        content = f"User: {user_request}\nAssistant: {assistant_response}"
        project = _infer_project(user_request, assistant_response)
        fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()
        vector_store = get_vector_store()
        existing = vector_store.get(where={"record_type": "conversation"})
        for record_id, metadata in zip(existing.get("ids", []), existing.get("metadatas", [])):
            if metadata and metadata.get("fingerprint") == fingerprint:
                _cleanup_conversation_memory(vector_store)
                return "Conversation saved to the vector database."

        document = Document(
            page_content=content,
            metadata={
                "record_type": "conversation",
                "source": "orchestrator conversation",
                "fingerprint": fingerprint,
                "source_file": "conversation_memory",
                "project": project,
                "stored_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        vector_store.add_documents([document], ids=[_stable_id("conversation", content)])
        _cleanup_conversation_memory(vector_store)
        return "Conversation saved to the vector database."
    except Exception as e:
        return f"Error saving conversation memory: {str(e)}"
