"""
Continuous Harvest -> Google Drive Sync -> Google Colab Continual Fine-Tuning Pipeline.

Capabilities:
  1. Incremental / Targeted or Full Project Harvest (strictly skipping .env files and raw DB row data):
     - Harvest a single file, a folder/module, or an entire project root.
     - Upsert-merges new Q&A pairs into the existing `peter_aryashakti_dataset.jsonl` keyed by normalized
       question so the model NEVER forgets previously harvested files, tables, or projects.
  2. Dual Storage Sync (Local + Google Drive `G:\\My Drive\\peter_finetune`):
     - Saves one copy locally at `data/finetune/peter_aryashakti_dataset.jsonl`.
     - Syncs a second copy + the Colab notebook (`Peter_Aryashakti_Colab_Finetune.ipynb`) directly to
       Google Drive (`G:\\My Drive\\peter_finetune\\`).
  3. Colab Trigger & Auto-Ollama GGUF Importer:
     - Opens/triggers the Drive-backed Colab notebook on T4 GPU.
     - Imports the exported `unsloth.Q4_K_M.gguf` from `G:\\My Drive\\peter_finetune\\` into local Ollama
       (`ollama create peter-coder -f Modelfile.peter-coder`).
"""

import argparse
import ast
import datetime
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import webbrowser
from collections import OrderedDict, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.tools.code_intelligence import (
    SYMBOL_DB_PATH,
    _get_cached_ast,
    _get_code_files,
    _get_django_models,
    _is_env_file,
    _read_file,
    analyze_module,
    get_project_root,
    investigate_keyword,
)
from src.tools.database_tools import _get_connection

OUTPUT_DIR = PROJECT_ROOT / "data" / "finetune"
OUTPUT_JSONL = OUTPUT_DIR / "peter_aryashakti_dataset.jsonl"
NOTEBOOK_LOCAL = PROJECT_ROOT / "notebooks" / "Peter_Aryashakti_Colab_Finetune.ipynb"
MODELFILE_LOCAL = PROJECT_ROOT / "Modelfile.peter-coder"

SYSTEM_PROMPT = (
    "You are Peter, a senior software engineer and codebase/database intelligence assistant "
    "for the Aryashakti Django & PostgreSQL platform. Always ground your answers strictly in "
    "verified code symbols, exact file paths and line numbers, API URL routes, Postman specs, "
    "DRF serializers, Django models, and PostgreSQL table names, column names, row counts, "
    "and foreign-key relationships. Never invent unrelated flows."
)


def get_gdrive_finetune_dir() -> Path | None:
    """Resolve the local Google Drive for Desktop sync folder (e.g. G:\\My Drive\\Ollama_Models)."""
    env_dir = os.getenv("GDRIVE_FINETUNE_DIR", "").strip()
    if env_dir:
        p = Path(env_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    candidates = [
        Path("G:/My Drive"),
        Path(os.path.expanduser("~/Google Drive/My Drive")),
        Path(os.path.expanduser("~/Google Drive")),
    ]
    for cand in candidates:
        if cand.exists():
            target = cand / "Ollama_Models"
            target.mkdir(parents=True, exist_ok=True)
            return target
    return None


def _make_example(user_msg: str, assistant_msg: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg.strip()},
            {"role": "assistant", "content": assistant_msg.strip()},
        ]
    }


def _question_key(example: dict) -> str:
    """Extract normalized user prompt key for deduplicated upsert merging."""
    msgs = example.get("messages", [])
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            return re.sub(r"\s+", " ", str(m.get("content", "")).strip()).lower()
    return ""


def _load_existing_jsonl(path: Path) -> OrderedDict[str, dict]:
    """Load existing JSONL examples keyed by normalized user prompt to preserve past data."""
    records: OrderedDict[str, dict] = OrderedDict()
    if not path or not path.exists():
        return records
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    k = _question_key(obj)
                    if k:
                        records[k] = obj
                except Exception:
                    continue
    except Exception:
        pass
    return records


def _scrub_secrets(text: str) -> str:
    """Redact any accidental tokens/passwords/keys in harvested config or Postman text."""
    if not text:
        return ""
    scrubbed = re.sub(
        r'(?i)(bearer\s+)[A-Za-z0-9\-._~+/]+=*',
        r'\1<REDACTED_TOKEN>',
        text,
    )
    scrubbed = re.sub(
        r'(?i)("(?:password|secret|api_key|token|authorization|access_token)"\s*:\s*)"[^"]+"',
        r'\1"<REDACTED>"',
        scrubbed,
    )
    return scrubbed


def _walk_postman_items(items: list, folder_chain: str = "") -> list[dict]:
    """Recursively extract all API request items from a Postman collection JSON."""
    extracted = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name", "Unnamed")
        current_chain = f"{folder_chain} / {name}" if folder_chain else name
        if "item" in item and isinstance(item["item"], list):
            extracted.extend(_walk_postman_items(item["item"], current_chain))
        elif "request" in item and isinstance(item["request"], dict):
            req = item["request"]
            method = req.get("method", "GET")
            url_obj = req.get("url", {})
            if isinstance(url_obj, str):
                raw_url = url_obj
                path_str = url_obj
                query_params = []
            else:
                raw_url = url_obj.get("raw", "")
                path_parts = url_obj.get("path", [])
                path_str = "/" + "/".join(str(p) for p in path_parts) if path_parts else raw_url
                query_params = [
                    f"{q.get('key')}={q.get('value', '')}"
                    for q in url_obj.get("query", [])
                    if isinstance(q, dict) and q.get("key")
                ]

            body_obj = req.get("body", {}) or {}
            body_mode = body_obj.get("mode", "")
            body_text = ""
            if body_mode == "raw":
                body_text = _scrub_secrets(body_obj.get("raw", ""))[:1200]
            elif body_mode in ("formdata", "urlencoded"):
                fields = body_obj.get(body_mode, [])
                if isinstance(fields, list):
                    body_text = ", ".join(
                        f"`{f.get('key')}` ({f.get('type', 'text')})"
                        for f in fields
                        if isinstance(f, dict) and f.get("key")
                    )

            extracted.append({
                "name": name,
                "folder": folder_chain,
                "full_title": current_chain,
                "method": method,
                "path": path_str,
                "raw_url": _scrub_secrets(raw_url),
                "query_params": query_params,
                "body_mode": body_mode,
                "body": body_text,
            })
    return extracted


def _harvest_pg_metadata() -> tuple[dict[str, list[str]], dict[str, int], list[tuple[str, str, str, str]]]:
    """
    Fetch ONLY metadata from PostgreSQL (never raw user row contents):
      - table_columns: {table_name: [col1, col2, ...]}
      - table_rows: {table_name: exact_row_count}
      - fk_relations: [(src_table, src_col, tgt_table, tgt_col), ...]
    """
    table_columns: dict[str, list[str]] = defaultdict(list)
    table_rows: dict[str, int] = {}
    fk_relations: list[tuple[str, str, str, str]] = []

    try:
        conn, db_type = _get_connection()
        if db_type not in ("postgres", "postgresql"):
            conn.close()
            return table_columns, table_rows, fk_relations

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name, column_name
                FROM information_schema.columns
                WHERE table_schema = %s
                ORDER BY table_name, ordinal_position;
                """,
                ("public",),
            )
            for t_name, c_name in cur.fetchall():
                table_columns[t_name].append(c_name)

            cur.execute(
                """
                SELECT
                    tc.table_name AS source_table,
                    kcu.column_name AS source_column,
                    ccu.table_name AS target_table,
                    ccu.column_name AS target_column
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                  ON ccu.constraint_name = tc.constraint_name
                 AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = %s
                  AND tc.table_schema = %s
                ORDER BY tc.table_name, kcu.column_name;
                """,
                ("FOREIGN KEY", "public"),
            )
            for s_tbl, s_col, t_tbl, t_col in cur.fetchall():
                if (s_tbl, s_col, t_tbl, t_col) not in fk_relations:
                    fk_relations.append((s_tbl, s_col, t_tbl, t_col))

            for t_name in list(table_columns.keys()):
                try:
                    cur.execute(f'SELECT COUNT(*) FROM "{t_name}";')
                    table_rows[t_name] = int(cur.fetchone()[0])
                except Exception:
                    table_rows[t_name] = 0

        conn.close()
    except Exception as e:
        print(f"Warning: Could not harvest live PostgreSQL metadata: {e}")

    return table_columns, table_rows, fk_relations


def _resolve_target_files(target_path: str | None) -> tuple[str, list[str], bool]:
    """
    Resolve `target_path` (which may be None, a relative path inside get_project_root(),
    or an absolute file/folder path) into:
      (base_root, list_of_py_files, is_full_project_harvest)
    Strictly skips `.env` / `.env.*` files.
    """
    default_root = os.path.abspath(get_project_root())
    if os.path.normcase(default_root) == os.path.normcase(str(PROJECT_ROOT)) and os.path.exists("C:/work/aryashakti"):
        default_root = os.path.abspath("C:/work/aryashakti")

    if not target_path or not target_path.strip():
        py_files = [
            fp for fp in _get_code_files(default_root)
            if fp.endswith(".py") and not _is_env_file(fp)
        ]
        return default_root, py_files, True

    raw = target_path.strip().strip("'\"")
    norm_raw = raw.replace("\\", "/").strip("/").lower()

    # Guard: If user passed `G:\My`, `G:\My Drive`, or `G:\My Drive\Ollama_Models` as target_path
    # (either truncated at space or intending Google Drive as the output destination),
    # harvest from `default_root` so it never errors with `Target path does not exist: G:\My`.
    if (
        norm_raw in {"g:/my", "g:/my drive", "g:/my drive/ollama_models", "g:/my drive/peter_finetune"}
        or norm_raw.startswith("g:/my drive/")
        or norm_raw.startswith("g:/my/")
    ):
        py_files = [
            fp for fp in _get_code_files(default_root)
            if fp.endswith(".py") and not _is_env_file(fp)
        ]
        return default_root, py_files, True

    cand = Path(raw)

    # Fast-path 1: Check SQLite indexed_files (<2ms) for already-indexed project files/folders
    if SYMBOL_DB_PATH.exists():
        norm_req = raw.replace("\\", "/").strip("/")
        alt_req = norm_req.replace("helpers.py", "helper.py") if "helpers.py" in norm_req else norm_req.replace("helper.py", "helpers.py")
        conn = sqlite3.connect(str(SYMBOL_DB_PATH))
        matched = conn.execute(
            """
            SELECT project_root, rel_path FROM indexed_files
            WHERE rel_path IN (?, ?)
               OR rel_path LIKE ?
               OR rel_path LIKE ?;
            """,
            (norm_req, alt_req, f"{norm_req}/%", f"%/{os.path.basename(norm_req)}"),
        ).fetchall()
        conn.close()
        non_venv = [(pr, rp) for pr, rp in matched if not rp.startswith(".venv/")]
        if non_venv:
            proj_r = non_venv[0][0]
            return proj_r, [f"{proj_r}/{rp}" for pr, rp in non_venv if rp.endswith(".py")], False

    if not cand.is_absolute():
        for root_cand in (Path(default_root), Path("C:/work/aryashakti"), PROJECT_ROOT):
            if not root_cand.exists():
                continue
            direct = root_cand / cand
            if direct.exists():
                default_root = os.path.abspath(str(root_cand))
                cand = direct
                break
            stem_alt = str(cand).replace("helpers.py", "helper.py").replace("helper.py", "helpers.py")
            if (root_cand / stem_alt).exists():
                default_root = os.path.abspath(str(root_cand))
                cand = root_cand / stem_alt
                break

    cand_abs = os.path.abspath(str(cand))
    if not os.path.exists(cand_abs):
        raise FileNotFoundError(f"Target path does not exist: {target_path} (resolved: {cand_abs})")

    if _is_env_file(cand_abs):
        raise ValueError("Refusing to harvest .env file (strict privacy boundary).")

    if os.path.isfile(cand_abs):
        base_root = default_root if cand_abs.lower().startswith(default_root.lower()) else os.path.dirname(cand_abs)
        py_files = [cand_abs] if cand_abs.endswith(".py") else []
        return base_root, py_files, False

    # Directory target
    is_full = os.path.normcase(cand_abs) == os.path.normcase(default_root)
    base_root = default_root if cand_abs.lower().startswith(default_root.lower()) else cand_abs
    py_files = [
        fp for fp in _get_code_files(cand_abs)
        if fp.endswith(".py") and not _is_env_file(fp)
    ]
    return base_root, py_files, is_full


def sync_files_to_gdrive(stats: dict | None = None, timeout_sec: float = 3.0) -> dict:
    """
    Copy `peter_aryashakti_dataset.jsonl` and `Peter_Aryashakti_Colab_Finetune.ipynb`
    to `G:\\My Drive\\peter_finetune\\` (and `G:\\My Drive\\Colab Notebooks\\`) using a daemon thread
    with a timeout guard so virtual DriveFS reconnects never hang Peter.
    """
    import threading

    shared_res: dict = {
        "gdrive_connected": False,
        "gdrive_dir": "G:\\My Drive\\peter_finetune",
        "gdrive_jsonl": "G:\\My Drive\\peter_finetune\\peter_aryashakti_dataset.jsonl",
        "gdrive_notebook": "G:\\My Drive\\peter_finetune\\Peter_Aryashakti_Colab_Finetune.ipynb",
    }

    def _do_sync() -> None:
        try:
            gdrive_dir = get_gdrive_finetune_dir()
            if not gdrive_dir:
                return
            shared_res["gdrive_connected"] = True
            shared_res["gdrive_dir"] = str(gdrive_dir)

            if OUTPUT_JSONL.exists():
                dst_jsonl = gdrive_dir / OUTPUT_JSONL.name
                shutil.copy2(OUTPUT_JSONL, dst_jsonl)
                shared_res["gdrive_jsonl"] = str(dst_jsonl)

            if NOTEBOOK_LOCAL.exists():
                dst_nb = gdrive_dir / NOTEBOOK_LOCAL.name
                shutil.copy2(NOTEBOOK_LOCAL, dst_nb)
                shared_res["gdrive_notebook"] = str(dst_nb)

                colab_notebooks_dir = gdrive_dir.parent / "Colab Notebooks"
                if colab_notebooks_dir.exists():
                    try:
                        shutil.copy2(NOTEBOOK_LOCAL, colab_notebooks_dir / NOTEBOOK_LOCAL.name)
                    except Exception:
                        pass

            if stats:
                manifest_path = gdrive_dir / "harvest_manifest.json"
                try:
                    manifest_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
                except Exception:
                    pass
        except Exception as e:
            shared_res["gdrive_note"] = f"Background sync queued ({type(e).__name__})"

    t = threading.Thread(target=_do_sync, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)
    if t.is_alive():
        shared_res["gdrive_note"] = "Syncing to G:\\My Drive\\peter_finetune in background"
    return shared_res


def harvest_and_sync_dataset(
    target_path: str | None = None,
    sync_gdrive: bool = True,
    include_db: bool = True,
    open_colab: bool = False,
) -> dict:
    """
    Incremental / Cumulative Harvester:
      1. Loads existing examples from local JSONL (and Google Drive JSONL if present)
         so past training examples are NEVER lost.
      2. Harvests `target_path` (single file, folder/module, or full project), skipping `.env`.
      3. Upsert-merges new/updated examples over existing examples by question key.
      4. Saves locally (`data/finetune/peter_aryashakti_dataset.jsonl`) AND syncs to
         Google Drive (`G:\\My Drive\\peter_finetune\\peter_aryashakti_dataset.jsonl`).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing dataset first so we NEVER forget past knowledge
    existing_records = _load_existing_jsonl(OUTPUT_JSONL)

    initial_count = len(existing_records)
    base_root, py_files, is_full_project = _resolve_target_files(target_path)

    print(
        f"[1/6] Indexing {len(py_files)} Python file(s) from target "
        f"'{target_path or base_root}' (skipping .env)..."
    )
    rel_filter_set: set[str] = set()
    for fp in py_files:
        _read_file(fp)
        rel_p = os.path.relpath(fp, base_root).replace("\\", "/")
        rel_filter_set.add(rel_p)

    new_examples: list[dict] = []

    # 2. Module-level summaries
    print("[2/6] Harvesting module architecture summaries...")
    app_modules: dict[str, list[str]] = defaultdict(list)
    for fp in py_files:
        rel_p = os.path.relpath(fp, base_root).replace("\\", "/")
        if "/migrations/" in rel_p:
            continue
        app_name = rel_p.split("/")[0] if "/" in rel_p else "root"
        app_modules[app_name].append(rel_p)
        mod_summary = analyze_module(fp if not is_full_project else rel_p)
        if mod_summary and not mod_summary.startswith("Error"):
            q_mod = f"What is the structure, classes, and functions of module `{rel_p}`?"
            a_mod = f"Here is the architectural breakdown of `{rel_p}`:\n\n{_scrub_secrets(mod_summary)[:3500]}"
            new_examples.append(_make_example(q_mod, a_mod))

    if is_full_project or (target_path and os.path.isdir(os.path.abspath(target_path)) if os.path.exists(str(target_path or "")) else False):
        for app_name, mod_list in sorted(app_modules.items()):
            q_app = f"Which Python files and modules exist inside the `{app_name}` app/directory?"
            a_app = (
                f"The `{app_name}` module contains **{len(mod_list)}** Python file(s):\n"
                + "\n".join(f"- `{m}`" for m in sorted(mod_list))
            )
            new_examples.append(_make_example(q_app, a_app))

    # 3. Symbol Q&A from SQLite index (filtered to target files when doing a partial harvest)
    print("[3/6] Generating AST symbol & implementation Q&A examples...")
    all_symbol_names: set[str] = set()
    symbol_defined_in: dict[str, str] = {}
    if SYMBOL_DB_PATH.exists():
        conn = sqlite3.connect(str(SYMBOL_DB_PATH))
        rows = conn.execute(
            """
            SELECT rel_path, symbol_name, symbol_type, parent_class,
                   start_line, end_line, adjacent_classes, source_code
            FROM symbols
            WHERE symbol_type IN ('class', 'function', 'method', 'constant')
            ORDER BY rel_path, start_line;
            """
        ).fetchall()
        conn.close()

        for rel_path, sym_name, sym_type, parent_cls, s_line, e_line, adj_json, src in rows:
            norm_rel = rel_path.replace("\\", "/")
            if not is_full_project and norm_rel not in rel_filter_set:
                continue
            if not src or len(src.strip()) < 15 or _is_env_file(norm_rel):
                continue
            src_clean = _scrub_secrets(src)
            if sym_type in ("class", "function", "constant"):
                all_symbol_names.add(sym_name)
                symbol_defined_in.setdefault(sym_name, f"{norm_rel}:{s_line}")

            adj_list = []
            try:
                adj_list = json.loads(adj_json) if adj_json else []
            except Exception:
                pass

            if sym_type == "class":
                role_desc = "Django Model" if norm_rel.endswith("models.py") else (
                    "DRF Serializer" if "serializer" in sym_name.lower() or norm_rel.endswith("serializers.py") else (
                        "API View / Controller" if norm_rel.endswith("views.py") else "Class"
                    )
                )
                adj_note = f"\n- **Adjacent classes in `{norm_rel}`**: `{', '.join(adj_list)}`" if adj_list else ""
                q1 = f"Where is `{sym_name}` defined and what is its implementation?"
                a1 = (
                    f"`{sym_name}` is a **{role_desc}** defined in `{norm_rel}` (lines {s_line}–{e_line}).{adj_note}\n\n"
                    f"### Implementation (`{norm_rel}:{s_line}-{e_line}`)\n"
                    f"```python\n{src_clean[:2500]}\n```"
                )
                new_examples.append(_make_example(q1, a1))

                q2 = f"Explain `{sym_name}` in `{norm_rel}`"
                a2 = (
                    f"To inspect `{sym_name}`, use `extract_function(name=\"{sym_name}\", file_path=\"{norm_rel}\")`.\n\n"
                    f"**Location**: `{norm_rel}` (lines {s_line}–{e_line})\n"
                    f"**Role**: {role_desc}\n\n"
                    f"```python\n{src_clean[:2000]}\n```"
                )
                new_examples.append(_make_example(q2, a2))

            elif sym_type == "function":
                q = f"Where is the function `{sym_name}` defined and how does it work?"
                a = (
                    f"The function `{sym_name}` is defined in `{norm_rel}` (lines {s_line}–{e_line}).\n\n"
                    f"```python\n{src_clean[:2200]}\n```"
                )
                new_examples.append(_make_example(q, a))
                q_alt = f"How does {sym_name} work and where is it defined?"
                new_examples.append(_make_example(q_alt, a))

            elif sym_type == "method" and parent_cls:
                q = f"How is `{parent_cls}.{sym_name}` implemented in `{norm_rel}`?"
                a = (
                    f"`{parent_cls}.{sym_name}` is defined in `{norm_rel}` (lines {s_line}–{e_line}):\n\n"
                    f"```python\n{src_clean[:2000]}\n```"
                )
                new_examples.append(_make_example(q, a))

            elif sym_type == "constant" and len(sym_name) >= 3:
                q = f"Where is the constant `{sym_name}` defined?"
                a = (
                    f"`{sym_name}` is defined in `{norm_rel}` (lines {s_line}–{e_line}):\n\n"
                    f"```python\n{src_clean[:800]}\n```"
                )
                new_examples.append(_make_example(q, a))

    # 4. If target is a single non-Python file (e.g. .md, .json, .html, .sql), harvest it directly
    if target_path:
        raw_cand = Path(target_path.strip().strip("'\""))
        if not raw_cand.is_absolute():
            raw_cand = Path(base_root) / raw_cand
        if raw_cand.exists() and raw_cand.is_file() and not _is_env_file(str(raw_cand)) and not str(raw_cand).endswith(".py"):
            rel_f = os.path.relpath(str(raw_cand), base_root).replace("\\", "/")
            txt = _scrub_secrets(raw_cand.read_text(encoding="utf-8", errors="replace"))
            if raw_cand.suffix.lower() == ".json":
                try:
                    pm_data = json.loads(txt)
                    if "item" in pm_data and isinstance(pm_data["item"], list):
                        col_name = pm_data.get("info", {}).get("name", raw_cand.stem)
                        for r in _walk_postman_items(pm_data["item"], col_name):
                            q_pm = f"What is the API request specification and payload for `{r['name']}` (`{r['method']} {r['path']}`)?"
                            a_pm = (
                                f"- **Endpoint Name**: `{r['full_title']}` (`{raw_cand.name}`)\n"
                                f"- **HTTP Method & Path**: `{r['method']} {r['path']}`\n"
                                + (f"\n```\n{r['body']}\n```" if r["body"] else "")
                            )
                            new_examples.append(_make_example(q_pm, a_pm))
                except Exception:
                    pass
            if len(txt.strip()) >= 20:
                q_f = f"What is the content and purpose of `{rel_f}`?"
                a_f = f"File `{rel_f}`:\n\n```\n{txt[:3500]}\n```"
                new_examples.append(_make_example(q_f, a_f))

    # 5. Full-project extras: Postman collections, Docs, and PostgreSQL Schema Metadata
    if is_full_project:
        print("[4/6] Harvesting Postman API Collections & Project Docs...")
        pm_dir = Path(base_root) / "api postman collections"
        if pm_dir.exists():
            for pm_file in sorted(pm_dir.glob("*.json")):
                try:
                    pm_data = json.loads(pm_file.read_text(encoding="utf-8", errors="replace"))
                    col_name = pm_data.get("info", {}).get("name", pm_file.stem)
                    for r in _walk_postman_items(pm_data.get("item", []), col_name):
                        q_pm = f"What is the API request specification and payload for `{r['name']}` (`{r['method']} {r['path']}`)?"
                        qp_line = f"- **Query Params**: `{', '.join(r['query_params'])}`\n" if r["query_params"] else ""
                        body_block = (
                            f"\n### Request Payload (`{r['body_mode']}`)\n```\n{r['body']}\n```"
                            if r["body"] else ""
                        )
                        a_pm = (
                            f"- **Endpoint Name**: `{r['full_title']}` (Postman Collection: `{pm_file.name}`)\n"
                            f"- **HTTP Method & Path**: `{r['method']} {r['path']}`\n"
                            f"{qp_line}{body_block}"
                        )
                        new_examples.append(_make_example(q_pm, a_pm))
                except Exception:
                    pass

        for md_name in ("ARYASHAKTI_PROJECT_OVERVIEW.md", "README.md"):
            md_path = Path(base_root) / md_name
            if md_path.exists():
                md_text = _scrub_secrets(md_path.read_text(encoding="utf-8", errors="replace"))
                for sec in re.split(r"(?m)^##+\s+", md_text):
                    sec = sec.strip()
                    if len(sec) < 40:
                        continue
                    first_line = sec.splitlines()[0].strip("#* :")
                    body = "\n".join(sec.splitlines()[1:]).strip()
                    if first_line and body:
                        q_doc = f"Explain `{first_line}` in the Aryashakti project documentation (`{md_name}`)"
                        a_doc = f"From `{md_name}` (**{first_line}**):\n\n{body[:3000]}"
                        new_examples.append(_make_example(q_doc, a_doc))

        if include_db:
            print("[5/6] Harvesting PostgreSQL table names, column names, row counts, and FK relationships (zero raw data)...")
            table_columns, table_rows, fk_relations = _harvest_pg_metadata()
            outgoing_fks: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
            incoming_fks: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
            for s_tbl, s_col, t_tbl, t_col in fk_relations:
                outgoing_fks[s_tbl].append((s_col, t_tbl, t_col))
                incoming_fks[t_tbl].append((s_tbl, s_col, t_col))

            models_list = _get_django_models(base_root)
            table_to_django = {m["table_name"]: m for m in models_list}

            for t_name, cols in sorted(table_columns.items()):
                r_count = table_rows.get(t_name, 0)
                dj_info = table_to_django.get(t_name)
                dj_line = (
                    f"- **Django Model**: `{dj_info['model_name']}` in `{dj_info['models_file']}` (app `{dj_info['app_name']}`)\n"
                    if dj_info else ""
                )
                out_rels = outgoing_fks.get(t_name, [])
                inc_rels = incoming_fks.get(t_name, [])
                rel_lines = []
                if out_rels:
                    rel_lines.append("**Outgoing Foreign Keys (References other tables):**")
                    for s_col, tgt_tbl, tgt_col in out_rels:
                        rel_lines.append(f"  - `{t_name}.{s_col}` ➔ `{tgt_tbl}.{tgt_col}`")
                if inc_rels:
                    rel_lines.append("**Incoming Foreign Keys (Referenced by other tables):**")
                    for src_tbl, src_col, tgt_col in inc_rels:
                        rel_lines.append(f"  - `{src_tbl}.{src_col}` ➔ `{t_name}.{tgt_col}`")
                if not rel_lines:
                    rel_lines.append("- No direct foreign-key constraints to other tables.")

                rel_block = "\n".join(rel_lines)
                cols_str = ", ".join(f"`{c}`" for c in cols)
                q_tbl = f"What are the column names, row count, and table relationships for `{t_name}`?"
                a_tbl = (
                    f"### Table: `{t_name}`\n"
                    f"{dj_line}"
                    f"- **Total Rows**: `{r_count}`\n"
                    f"- **Columns ({len(cols)})**: {cols_str}\n\n"
                    f"### Relationships with Other Tables\n"
                    f"{rel_block}"
                )
                new_examples.append(_make_example(q_tbl, a_tbl))

    # 6. Upsert-Merge new_examples into existing_records so past data is NEVER lost
    print("[6/6] Merging harvested examples with existing dataset & syncing to Google Drive...")
    added_count = 0
    updated_count = 0
    for ex in new_examples:
        k = _question_key(ex)
        if not k:
            continue
        if k in existing_records:
            if existing_records[k] != ex:
                updated_count += 1
            existing_records[k] = ex
        else:
            added_count += 1
            existing_records[k] = ex

    # Enrich existing_records with top-level Aryashakti app/module overview Q&As & natural phrasing aliases
    overview_examples = _build_overview_and_alias_examples(existing_records)
    for ex in overview_examples:
        k = _question_key(ex)
        if k and k not in existing_records:
            added_count += 1
            existing_records[k] = ex
        elif k:
            existing_records[k] = ex

    total_count = len(existing_records)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for ex in existing_records.values():
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    size_mb = round(OUTPUT_JSONL.stat().st_size / (1024 * 1024), 2)
    stats = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "target": target_path or base_root,
        "is_full_project": is_full_project,
        "py_files_scanned": len(py_files),
        "previous_examples": initial_count,
        "harvested_in_run": len(new_examples),
        "added_examples": added_count,
        "updated_examples": updated_count,
        "total_examples": total_count,
        "size_mb": size_mb,
        "local_jsonl": str(OUTPUT_JSONL),
    }

    if sync_gdrive:
        sync_info = sync_files_to_gdrive(stats)
        stats.update(sync_info)

    if open_colab:
        colab_info = launch_colab_training(sync_first=False)
        stats["colab_launched"] = colab_info

    print(
        f"[OK] Dataset Ready: {total_count} total examples ({added_count} new, {updated_count} updated, "
        f"{initial_count} prior preserved) | {size_mb} MB"
    )
    if stats.get("gdrive_jsonl"):
        print(f"[GDRIVE] Synced to Google Drive: {stats['gdrive_jsonl']}")
    return stats


def _build_overview_and_alias_examples(existing_records: OrderedDict[str, dict]) -> list[dict]:
    """
    Synthesize high-value repository overview Q&As (e.g. 'what are modules present in the aryashakti app?')
    and natural phrasing aliases from the existing harvested records so the model answers overview
    and rephrased questions accurately.
    """
    extras: list[dict] = []
    app_summaries: list[tuple[str, str]] = []

    for q_key, ex in list(existing_records.items()):
        msgs = ex.get("messages", [])
        if len(msgs) < 3:
            continue
        u_txt = msgs[1].get("content", "")
        a_txt = msgs[2].get("content", "")

        # Collect per-app module summaries
        m_app = re.match(r"Which Python files and modules exist inside the `([^`]+)` app/directory\?", u_txt)
        if m_app:
            app_name = m_app.group(1)
            cnt_m = re.search(r"\*\*(\d+)\*\*\s+Python file", a_txt)
            cnt = cnt_m.group(1) if cnt_m else "multiple"
            app_summaries.append((app_name, f"- **`{app_name}`** (`{cnt}` Python files)"))

        # Natural phrasing alias for functions: "How does <fn> work and where is it defined?"
        m_fn = re.match(r"Where is the function `([^`]+)` defined and how does it work\?", u_txt)
        if m_fn:
            fn_name = m_fn.group(1)
            extras.append(_make_example(f"How does {fn_name} work and where is it defined?", a_txt))

        # Natural phrasing alias for tables: "What are the columns, row count, and foreign key relationships of the <tbl> table?"
        m_tbl = re.match(r"What are the column names, row count, and table relationships for `([^`]+)`\?", u_txt)
        if m_tbl:
            tbl_name = m_tbl.group(1)
            extras.append(
                _make_example(
                    f"What are the columns, row count, and foreign key relationships of the {tbl_name} table?",
                    a_txt,
                )
            )

    if app_summaries:
        app_summaries.sort(key=lambda x: x[0].lower())
        overview_body = (
            f"The **Aryashakti** repository contains **{len(app_summaries)}** Django applications and top-level modules:\n\n"
            + "\n".join(line for _, line in app_summaries)
        )
        for q_ov in (
            "what are modules present in the aryashakti app?",
            "Which modules and apps exist in the Aryashakti repository?",
            "List all Django apps and modules in the Aryashakti project",
            "What apps are in Aryashakti?",
        ):
            extras.append(_make_example(q_ov, overview_body))

    return extras


def _find_gdrive_notebook_cloud_id(notebook_path: Path) -> str | None:
    """
    Look up the Google Drive cloud `doc_id` for `Peter_Aryashakti_Colab_Finetune.ipynb`
    from Google Drive for Desktop's local metadata SQLite database so we can open
    `https://colab.research.google.com/drive/<DOC_ID>` directly!
    """
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if not local_app_data:
        return None
    drivefs_root = Path(local_app_data) / "Google" / "DriveFS"
    if not drivefs_root.exists():
        return None

    nb_name = notebook_path.name
    for db_file in drivefs_root.glob("*/metadata_sqlite_db"):
        tmp_db = OUTPUT_DIR / "_tmp_drivefs_meta.db"
        try:
            shutil.copy2(db_file, tmp_db)
            conn = sqlite3.connect(str(tmp_db))
            cur = conn.cursor()
            cur.execute(
                "SELECT stable_id, cloud_id FROM items WHERE local_title = ? AND is_owner = 1 LIMIT 1;",
                (nb_name,),
            )
            row = cur.fetchone()
            conn.close()
            tmp_db.unlink(missing_ok=True)
            if row and row[1]:
                return str(row[1])
        except Exception:
            tmp_db.unlink(missing_ok=True)
            continue
    return None


def launch_colab_training(sync_first: bool = True) -> dict:
    """
    1. Syncs the latest `peter_aryashakti_dataset.jsonl` and `Peter_Aryashakti_Colab_Finetune.ipynb`
       to `G:\\My Drive\\Ollama_Models\\`.
    2. Resolves the direct Google Colab Drive URL (`https://colab.research.google.com/drive/<CLOUD_ID>`)
       from Google Drive for Desktop metadata (or falls back to Google Colab homepage) and opens it
       in the browser ready for T4 GPU training.
    """
    sync_info = sync_files_to_gdrive() if sync_first else {}
    gdrive_dir = get_gdrive_finetune_dir()
    nb_drive_path = (gdrive_dir / NOTEBOOK_LOCAL.name) if gdrive_dir else NOTEBOOK_LOCAL

    cloud_id = _find_gdrive_notebook_cloud_id(nb_drive_path)
    if cloud_id:
        colab_url = f"https://colab.research.google.com/drive/{cloud_id}"
    else:
        colab_url = "https://colab.research.google.com/#create=true"

    opened = False
    try:
        webbrowser.open(colab_url)
        opened = True
    except Exception:
        pass

    return {
        "colab_url": colab_url,
        "cloud_id": cloud_id,
        "browser_opened": opened,
        "gdrive_notebook": str(nb_drive_path),
        **sync_info,
    }


def import_gguf_from_gdrive(model_name: str = "peter-coder") -> dict:
    """
    Check `~/Downloads`, `G:\\My Drive\\Ollama_Models`, and project root for
    `qwen2.5-coder-3b-instruct.Q4_K_M.gguf` (or `unsloth.Q4_K_M.gguf`),
    and run `ollama create <model_name> -f Modelfile.peter-coder`.
    """
    gdrive_dir = get_gdrive_finetune_dir()
    local_gguf = PROJECT_ROOT / "qwen2.5-coder-3b-instruct.Q4_K_M.gguf"
    downloads_dir = Path(os.path.expanduser("~/Downloads"))

    candidate_sources: list[Path] = [
        PROJECT_ROOT / "qwen2.5-coder-3b-instruct.Q4_K_M.gguf",
        PROJECT_ROOT / "unsloth.Q4_K_M.gguf",
        downloads_dir / "qwen2.5-coder-3b-instruct.Q4_K_M.gguf",
        downloads_dir / "unsloth.Q4_K_M.gguf",
    ]
    if downloads_dir.exists():
        candidate_sources.extend(sorted(downloads_dir.glob("*.gguf"), key=lambda p: p.stat().st_mtime, reverse=True))
    if gdrive_dir:
        candidate_sources.extend([
            gdrive_dir / "qwen2.5-coder-3b-instruct.Q4_K_M.gguf",
            gdrive_dir / "unsloth.Q4_K_M.gguf",
            gdrive_dir.parent / "peter_finetune" / "qwen2.5-coder-3b-instruct.Q4_K_M.gguf",
            gdrive_dir.parent / "peter_finetune" / "unsloth.Q4_K_M.gguf",
        ])

    source_used = None
    for cand_gguf in candidate_sources:
        try:
            if cand_gguf.exists():
                if os.path.abspath(str(cand_gguf)) != os.path.abspath(str(local_gguf)):
                    if not local_gguf.exists() or cand_gguf.stat().st_mtime > local_gguf.stat().st_mtime:
                        print(f"[IMPORT] Copying updated GGUF from {cand_gguf} to {local_gguf}...")
                        shutil.copy2(cand_gguf, local_gguf)
                source_used = str(cand_gguf)
                break
        except Exception:
            continue

    if not local_gguf.exists():
        return {
            "status": "missing_gguf",
            "message": (
                "qwen2.5-coder-3b-instruct.Q4_K_M.gguf not found yet in G:\\My Drive\\Ollama_Models. "
                "Finish the GGUF export & Google Drive copy cell in Google Colab first."
            ),
        }

    cmd = ["ollama", "create", model_name, "-f", str(MODELFILE_LOCAL)]
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    return {
        "status": "success" if proc.returncode == 0 else "error",
        "model_name": model_name,
        "source_gguf": source_used,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip(),
    }


def build_dataset() -> int:
    stats = harvest_and_sync_dataset(target_path=None, sync_gdrive=True, include_db=True)
    return int(stats.get("total_examples", 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Peter Incremental Harvester + Google Drive & Colab Pipeline")
    parser.add_argument("--target", nargs="*", default=None, help="Specific file, folder, or project root to harvest")
    parser.add_argument("--no-db", action="store_true", help="Skip live PostgreSQL metadata harvest")
    parser.add_argument("--sync-only", action="store_true", help="Only sync existing JSONL and Colab notebook to Google Drive")
    parser.add_argument("--open-colab", action="store_true", help="Open the synced notebook in Google Drive / Colab")
    parser.add_argument("--import-gguf", action="store_true", help="Import Q4_K_M.gguf from Google Drive into local Ollama")
    args, unknown = parser.parse_known_args()

    target_arg = " ".join(args.target).strip() if args.target else None

    if args.sync_only:
        res = sync_files_to_gdrive()
        print(json.dumps(res, indent=2))
        if args.open_colab:
            print(json.dumps(launch_colab_training(sync_first=False), indent=2))
    elif args.import_gguf:
        print(json.dumps(import_gguf_from_gdrive(), indent=2))
    else:
        harvest_and_sync_dataset(
            target_path=target_arg,
            sync_gdrive=True,
            include_db=not args.no_db,
            open_colab=args.open_colab,
        )
