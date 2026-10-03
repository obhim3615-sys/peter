"""
Export Peter's offline knowledge base (AST symbols + Senior Architect Walkthroughs +
3,955 Q&As + API URL routes + Postman specs + PostgreSQL schemas & FKs) into a compact,
indexed SQLite database inside the Android Studio project:
  C:\\Users\\Abhishek.Pandey\\Videos\\peterApp\\app\\src\\main\\assets\\peter_offline.db
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.code_intelligence import SYMBOL_DB_PATH, _explain_python_symbol_ast

DATASET_JSONL = PROJECT_ROOT / "data" / "finetune" / "peter_aryashakti_dataset.jsonl"
ANDROID_ASSETS_DIR = Path(r"C:\Users\Abhishek.Pandey\Videos\peterApp\app\src\main\assets")
OUTPUT_DB = ANDROID_ASSETS_DIR / "peter_offline.db"


def build_mobile_sqlite_db() -> dict:
    ANDROID_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_DB.exists():
        OUTPUT_DB.unlink()

    conn = sqlite3.connect(str(OUTPUT_DB))
    conn.execute("PRAGMA journal_mode=OFF;")
    conn.execute("PRAGMA synchronous=OFF;")

    # 1. Create tables optimized for Android SQLite lookup + FTS
    conn.executescript(
        """
        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE symbols (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rel_path TEXT NOT NULL,
            symbol_name TEXT NOT NULL,
            symbol_lower TEXT NOT NULL,
            symbol_type TEXT NOT NULL,
            parent_class TEXT DEFAULT '',
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            walkthrough TEXT DEFAULT '',
            source_code TEXT DEFAULT ''
        );
        CREATE INDEX idx_symbols_lower ON symbols(symbol_lower);
        CREATE INDEX idx_symbols_path ON symbols(rel_path);

        CREATE TABLE qa_knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            title TEXT NOT NULL,
            question TEXT NOT NULL,
            question_lower TEXT NOT NULL,
            keywords TEXT NOT NULL,
            answer TEXT NOT NULL
        );
        CREATE INDEX idx_qa_lower ON qa_knowledge(question_lower);
        CREATE INDEX idx_qa_category ON qa_knowledge(category);

        CREATE TABLE db_tables (
            table_name TEXT PRIMARY KEY,
            django_model TEXT DEFAULT '',
            row_count TEXT DEFAULT '0',
            col_count INTEGER DEFAULT 0,
            columns_text TEXT DEFAULT '',
            relationships_text TEXT DEFAULT '',
            full_markdown TEXT DEFAULT ''
        );

        CREATE TABLE api_routes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            method TEXT NOT NULL,
            route_path TEXT NOT NULL,
            handler TEXT NOT NULL,
            source_info TEXT NOT NULL,
            full_markdown TEXT NOT NULL
        );
        CREATE INDEX idx_api_route ON api_routes(route_path);

        CREATE TABLE project_files (
            rel_path TEXT PRIMARY KEY,
            app_module TEXT NOT NULL,
            file_ext TEXT NOT NULL,
            line_count INTEGER NOT NULL,
            is_modified INTEGER DEFAULT 0,
            content TEXT NOT NULL
        );
        CREATE INDEX idx_project_files_app ON project_files(app_module);
        """
    )

    # 2. Export all AST symbols + pre-computed Senior Architect Walkthroughs
    symbol_count = 0
    if SYMBOL_DB_PATH.exists():
        src_conn = sqlite3.connect(str(SYMBOL_DB_PATH))
        rows = src_conn.execute(
            """
            SELECT rel_path, symbol_name, symbol_lower, symbol_type,
                   COALESCE(parent_class, ''), start_line, end_line, COALESCE(source_code, '')
            FROM symbols
            WHERE rel_path NOT LIKE '.venv/%'
            ORDER BY rel_path, start_line;
            """
        ).fetchall()
        src_conn.close()

        batch = []
        for rel_path, sym_name, sym_lower, sym_type, parent_cls, s_line, e_line, src_code in rows:
            wt = _explain_python_symbol_ast(src_code) if src_code else ""
            batch.append((
                rel_path,
                sym_name,
                sym_lower,
                sym_type,
                parent_cls,
                int(s_line),
                int(e_line),
                wt,
                src_code[:4000],
            ))
        conn.executemany(
            """
            INSERT INTO symbols (
                rel_path, symbol_name, symbol_lower, symbol_type,
                parent_class, start_line, end_line, walkthrough, source_code
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            batch,
        )
        symbol_count = len(batch)

    # 3. Export all 3,955 Q&A pairs + parse out structured db_tables and api_routes
    qa_count = 0
    table_count = 0
    route_count = 0

    if DATASET_JSONL.exists():
        qa_batch = []
        table_batch = []
        route_batch = []

        with DATASET_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                msgs = obj.get("messages", [])
                if len(msgs) < 3:
                    continue
                q_txt = msgs[1].get("content", "").strip()
                a_txt = msgs[2].get("content", "").strip()
                if not q_txt or not a_txt:
                    continue

                q_low = q_txt.lower()
                # Categorize
                category = "code"
                title = q_txt
                if "table relationships for `" in q_txt or "foreign key relationships of the" in q_low:
                    category = "database"
                    m_t = re.search(r"`([^`]+)`", q_txt)
                    tbl_name = m_t.group(1) if m_t else ""
                    if tbl_name:
                        title = f"Table: {tbl_name}"
                        # Parse row count, columns, relationships
                        m_rows = re.search(r"\*\*Total Rows\*\*:\s*`([^`]+)`", a_txt)
                        row_cnt = m_rows.group(1) if m_rows else "0"
                        m_dj = re.search(r"\*\*Django Model\*\*:\s*([^\n]+)", a_txt)
                        dj_mod = m_dj.group(1).strip() if m_dj else ""
                        m_cols = re.search(r"\*\*Columns \((\d+)\)\*\*:\s*([^\n]+)", a_txt)
                        col_cnt = int(m_cols.group(1)) if m_cols else 0
                        cols_str = m_cols.group(2).strip() if m_cols else ""
                        m_rels = re.split(r"### Relationships with Other Tables\s*", a_txt)
                        rels_str = m_rels[1].strip() if len(m_rels) > 1 else ""
                        table_batch.append((tbl_name, dj_mod, row_cnt, col_cnt, cols_str, rels_str, a_txt))
                elif (
                    "postman" in q_low
                    or "url route" in q_low
                    or "api endpoint" in q_low
                    or "api request specification" in q_low
                    or "**HTTP Method & Path**:" in a_txt
                ):
                    category = "api"
                    m_ep = re.search(r"\*\*HTTP Method & Path\*\*:\s*`(\w+)\s+([^`]+)`", a_txt) or re.search(
                        r"### API Endpoint:\s*`(\w+)\s+([^`]+)`", a_txt
                    )
                    if m_ep:
                        method, r_path = m_ep.group(1), m_ep.group(2)
                        m_name = re.search(r"\*\*Endpoint Name\*\*:\s*`([^`]+)`", a_txt)
                        ep_title = m_name.group(1) if m_name else f"{method} {r_path}"
                        title = f"{method} {r_path}"
                        route_batch.append((method, r_path, ep_title, "Postman Spec", a_txt))
                    else:
                        for rm in re.finditer(r"-\s*`([^`]+)`\s*(?:➔|->|→)\s*(?:\*\*)?`?([^`\*\s]+)`?(?:\*\*)?\s*\(`([^`]+)`\)", a_txt):
                            r_path, handler, src_info = rm.group(1), rm.group(2), rm.group(3)
                            route_batch.append(("ROUTE", r_path, handler, src_info, a_txt))
                elif "modules" in q_low or "apps" in q_low:
                    category = "architecture"

                tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", q_txt) if len(t) >= 3]
                keywords = " ".join(sorted(set(tokens)))
                qa_batch.append((category, title[:140], q_txt, q_low, keywords, a_txt))

        # Also extract Django urls.py routes directly from C:/work/aryashakti if present
        arya_root = Path("C:/work/aryashakti")
        if arya_root.exists():
            for url_file in sorted(arya_root.rglob("urls.py")):
                rel_u = os.path.relpath(str(url_file), str(arya_root)).replace("\\", "/")
                if ".venv" in rel_u:
                    continue
                try:
                    u_lines = url_file.read_text(encoding="utf-8", errors="replace").splitlines()
                    for idx, uline in enumerate(u_lines, 1):
                        s_ul = uline.strip()
                        if not s_ul or s_ul.startswith("#"):
                            continue
                        pm = re.search(r"""(?:re_)?path\(\s*['"]([^'"]*)['"]\s*,\s*([A-Za-z0-9_.]+)""", s_ul)
                        if pm and pm.group(2) != "include":
                            r_p, h_n = pm.group(1), pm.group(2)
                            h_clean = h_n.replace(".as_view", "")
                            md = (
                                f"### 🌐 Django URL Route: `{r_p}`\n"
                                f"- **Handler / View**: `{h_clean}`\n"
                                f"- **Defined In**: `{rel_u}:{idx}`\n"
                                f"- **Status**: Active route\n\n"
                                f"```python\n{s_ul}\n```"
                            )
                            route_batch.append(("URL", f"/{r_p.lstrip('/')}", h_clean, f"{rel_u}:{idx}", md))
                except Exception:
                    pass

        conn.executemany(
            """
            INSERT INTO qa_knowledge (category, title, question, question_lower, keywords, answer)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            qa_batch,
        )
        qa_count = len(qa_batch)

        # Deduplicate table_batch by table_name
        seen_tables = set()
        dedup_tables = []
        for row in table_batch:
            if row[0] not in seen_tables:
                seen_tables.add(row[0])
                dedup_tables.append(row)
        conn.executemany(
            """
            INSERT OR REPLACE INTO db_tables (
                table_name, django_model, row_count, col_count, columns_text, relationships_text, full_markdown
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            dedup_tables,
        )
        table_count = len(dedup_tables)

        conn.executemany(
            """
            INSERT INTO api_routes (method, route_path, handler, source_info, full_markdown)
            VALUES (?, ?, ?, ?, ?);
            """,
            route_batch,
        )
        route_count = len(route_batch)

    # 4. Export ALL Django project files from C:/work/aryashakti for full offline file tree, grep search & code editing
    #    AND mirror the exact folder/file hierarchy into assets/aryashakti/<rel_path>!
    files_count = 0
    arya_root = Path("C:/work/aryashakti")
    mirrored_root = ANDROID_ASSETS_DIR / "aryashakti"
    mirrored_root.mkdir(parents=True, exist_ok=True)
    allowed_exts = {".py", ".html", ".json", ".md", ".sql", ".js", ".css", ".txt", ".sh", ".yml", ".yaml"}
    excluded_dirs = {".venv", "venv", ".git", "__pycache__", "node_modules", "media", "static", "logs", ".idea"}
    if arya_root.exists():
        file_rows = []
        for root_dir, dirs, files in os.walk(str(arya_root)):
            dirs[:] = [d for d in dirs if d not in excluded_dirs and not d.startswith(".venv")]
            for fname in sorted(files):
                low_f = fname.lower()
                if low_f == ".env" or low_f.startswith(".env.") or low_f.endswith(".env"):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext not in allowed_exts:
                    continue
                full_p = os.path.join(root_dir, fname)
                rel_p = os.path.relpath(full_p, str(arya_root)).replace("\\", "/")
                try:
                    if os.path.getsize(full_p) > 500_000:
                        continue
                    content = Path(full_p).read_text(encoding="utf-8", errors="replace")
                    line_cnt = len(content.splitlines())
                    app_mod = rel_p.split("/")[0] if "/" in rel_p else "root"
                    file_rows.append((rel_p, app_mod, ext, line_cnt, 0, content))

                    # Write to physical mirrored folder structure inside Android assets/aryashakti/
                    dest_file = mirrored_root / Path(rel_p)
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    dest_file.write_text(content, encoding="utf-8")
                except Exception:
                    continue

        conn.executemany(
            """
            INSERT OR REPLACE INTO project_files (
                rel_path, app_module, file_ext, line_count, is_modified, content
            ) VALUES (?, ?, ?, ?, ?, ?);
            """,
            file_rows,
        )
        files_count = len(file_rows)

    conn.executemany(
        "INSERT INTO meta (key, value) VALUES (?, ?);",
        [
            ("version", "1.1.0"),
            ("project", "aryashakti"),
            ("symbols_count", str(symbol_count)),
            ("qa_count", str(qa_count)),
            ("tables_count", str(table_count)),
            ("routes_count", str(route_count)),
            ("files_count", str(files_count)),
        ],
    )
    conn.commit()
    conn.execute("VACUUM;")
    conn.close()

    size_mb = round(OUTPUT_DB.stat().st_size / (1024 * 1024), 2)
    res = {
        "output_db": str(OUTPUT_DB),
        "size_mb": size_mb,
        "files_count": files_count,
        "symbols_count": symbol_count,
        "qa_count": qa_count,
        "tables_count": table_count,
        "routes_count": route_count,
    }
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    build_mobile_sqlite_db()
