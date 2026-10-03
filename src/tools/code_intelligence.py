import os
import ast
import re
import json
import sqlite3
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from src.tools.file_tools import PROJECT_ROOT, get_project_root

APP_ROOT = Path(__file__).resolve().parents[2]
SYMBOL_DB_PATH = APP_ROOT / "data" / "sqlite" / "code_symbol_index.db"

# In-memory L1 caches keyed by normalized absolute path -> (mtime_ns, size_bytes, payload)
_MEM_FILE_CACHE: dict[str, tuple[int, int, str]] = {}
_MEM_AST_CACHE: dict[str, tuple[int, int, ast.AST]] = {}
_MEM_INDEXED_STAMP: dict[str, tuple[int, int]] = {}
_SYMBOL_DB_INITIALIZED = False

# Directories excluded from code scanning (build artifacts, caches, dependencies)
EXCLUDED_DIRS = {
    '.venv', '__pycache__', '.git', 'node_modules', 'media', 'static', 'logs', 'data',
    '.gradle', 'build', 'bin', 'obj', '.idea', 'target', 'out', '.vs',
    'cmake-build-debug', 'cmake-build-release', '.dart_tool',
}

# Supported multi-language source file extensions
SUPPORTED_CODE_EXTENSIONS = {
    # Python
    ".py",
    # Kotlin & Java (Android, Spring, Backend)
    ".kt", ".kts", ".java",
    # C & C++
    ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp",
    # C# (.NET)
    ".cs",
    # JavaScript & TypeScript (Frontend, Node, React)
    ".js", ".jsx", ".ts", ".tsx", ".mjs",
    # Go & Rust
    ".go", ".rs",
    # Mobile (Flutter / Dart, Swift)
    ".dart", ".swift",
    # Shell & SQL
    ".sh", ".bash", ".sql",
}

CONTROL_FLOW_KEYWORDS = {
    "if", "for", "while", "switch", "catch", "return", "sizeof",
    "typeof", "super", "this", "new", "delete", "throw", "import", "package"
}


def _get_symbol_db() -> sqlite3.Connection:
    """Return a SQLite connection to the persistent code symbol index."""
    global _SYMBOL_DB_INITIALIZED
    SYMBOL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(SYMBOL_DB_PATH), timeout=5.0)
    if not _SYMBOL_DB_INITIALIZED:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS indexed_files (
                project_root TEXT NOT NULL,
                rel_path TEXT NOT NULL,
                mtime_ns INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL,
                PRIMARY KEY (project_root, rel_path)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS symbols (
                project_root TEXT NOT NULL,
                rel_path TEXT NOT NULL,
                symbol_name TEXT NOT NULL,
                symbol_lower TEXT NOT NULL,
                symbol_type TEXT NOT NULL,
                parent_class TEXT,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                adjacent_classes TEXT,
                source_code TEXT
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_lookup ON symbols(project_root, symbol_lower);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(project_root, rel_path);")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS keyword_cache (
                project_root TEXT NOT NULL,
                keyword_lower TEXT NOT NULL,
                max_mtime_ns INTEGER NOT NULL,
                file_count INTEGER NOT NULL,
                report TEXT NOT NULL,
                PRIMARY KEY (project_root, keyword_lower)
            );
            """
        )
        conn.commit()
        _SYMBOL_DB_INITIALIZED = True
    return conn


def _get_project_code_stamp(root: str) -> tuple[int, int]:
    """Compute (max_mtime_ns, file_count) across all supported code files in root via lightweight os.stat()."""
    max_mtime = 0
    count = 0
    for fp in _get_code_files():
        try:
            mt = os.stat(fp).st_mtime_ns
            if mt > max_mtime:
                max_mtime = mt
            count += 1
        except OSError:
            pass
    return max_mtime, count


def _sync_py_file_symbols(abs_path: str, mtime_ns: int, size_bytes: int, content: str, tree: ast.AST) -> None:
    """Harvest all classes, functions, methods, and constants from a parsed Python file into SQLite if changed."""
    if _MEM_INDEXED_STAMP.get(abs_path) == (mtime_ns, size_bytes):
        return
    try:
        root = str(Path(get_project_root()).resolve()).replace("\\", "/")
        abs_norm = str(Path(abs_path).resolve()).replace("\\", "/")
        if not abs_norm.lower().startswith(root.lower()):
            return
        rel_path = os.path.relpath(abs_path, root).replace("\\", "/")
        if "/migrations/" in rel_path:
            _MEM_INDEXED_STAMP[abs_path] = (mtime_ns, size_bytes)
            return

        conn = _get_symbol_db()
        row = conn.execute(
            "SELECT mtime_ns, size_bytes FROM indexed_files WHERE project_root = ? AND rel_path = ?;",
            (root, rel_path),
        ).fetchone()
        if row and row[0] == mtime_ns and row[1] == size_bytes:
            _MEM_INDEXED_STAMP[abs_path] = (mtime_ns, size_bytes)
            conn.close()
            return

        lines = content.splitlines()
        top_classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        symbol_rows = []

        for idx, node in enumerate(tree.body):
            if isinstance(node, ast.ClassDef):
                s_line = getattr(node, "lineno", 1)
                e_line = getattr(node, "end_lineno", s_line)
                src = "\n".join(lines[s_line - 1 : e_line])
                adj = []
                cls_indices = [i for i, c in enumerate(top_classes) if c.name == node.name]
                if cls_indices:
                    ci = cls_indices[0]
                    if ci > 0:
                        adj.append(top_classes[ci - 1].name)
                    if ci + 1 < len(top_classes):
                        adj.append(top_classes[ci + 1].name)
                symbol_rows.append((
                    root, rel_path, node.name, node.name.lower(), "class", None,
                    s_line, e_line, json.dumps(adj), src
                ))
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        ms_line = getattr(item, "lineno", s_line)
                        me_line = getattr(item, "end_lineno", ms_line)
                        msrc = "\n".join(lines[ms_line - 1 : me_line])
                        symbol_rows.append((
                            root, rel_path, item.name, item.name.lower(), "method", node.name,
                            ms_line, me_line, "[]", msrc
                        ))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                s_line = getattr(node, "lineno", 1)
                e_line = getattr(node, "end_lineno", s_line)
                src = "\n".join(lines[s_line - 1 : e_line])
                symbol_rows.append((
                    root, rel_path, node.name, node.name.lower(), "function", None,
                    s_line, e_line, "[]", src
                ))
            elif isinstance(node, ast.Assign):
                s_line = getattr(node, "lineno", 1)
                e_line = getattr(node, "end_lineno", s_line)
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id not in {"urlpatterns", "app_name", "dependencies", "operations"}:
                        src = "\n".join(lines[s_line - 1 : e_line])
                        symbol_rows.append((
                            root, rel_path, t.id, t.id.lower(), "constant", None,
                            s_line, e_line, "[]", src[:1000]
                        ))

        with conn:
            conn.execute("DELETE FROM symbols WHERE project_root = ? AND rel_path = ?;", (root, rel_path))
            conn.execute(
                "INSERT OR REPLACE INTO indexed_files (project_root, rel_path, mtime_ns, size_bytes) VALUES (?, ?, ?, ?);",
                (root, rel_path, mtime_ns, size_bytes),
            )
            if symbol_rows:
                conn.executemany(
                    """
                    INSERT INTO symbols (
                        project_root, rel_path, symbol_name, symbol_lower, symbol_type,
                        parent_class, start_line, end_line, adjacent_classes, source_code
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    symbol_rows,
                )
        conn.close()
        _MEM_INDEXED_STAMP[abs_path] = (mtime_ns, size_bytes)
    except Exception:
        pass


def _is_env_file(path_or_name: str) -> bool:
    """Return True if the filename is a .env file that must be skipped."""
    base = os.path.basename(path_or_name).lower()
    return base == ".env" or base.startswith(".env.") or base.endswith(".env")


def _get_code_files(file_path: str = ""):
    """Yield all supported code files in the active project root or target subfolder (strictly skipping .env files)."""
    root = get_project_root()
    walk_root = root
    if file_path:
        if _is_env_file(file_path):
            return
        full_path = file_path if os.path.isabs(file_path) else os.path.join(root, file_path)
        if not os.path.exists(full_path):
            return
        if os.path.isfile(full_path):
            yield full_path
            return
        walk_root = full_path

    for dirpath, dirnames, filenames in os.walk(walk_root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS and not d.startswith("venv")]
        for f in filenames:
            if _is_env_file(f):
                continue
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_CODE_EXTENSIONS:
                yield os.path.join(dirpath, f)


def _read_file(path: str) -> str:
    """Read a file with automatic mtime_ns + size L1 caching and opportunistic AST symbol harvesting (skips .env)."""
    if _is_env_file(path):
        return ""
    try:
        abs_p = os.path.abspath(path)
        st = os.stat(abs_p)
        mtime_ns, size_bytes = st.st_mtime_ns, st.st_size
        cached = _MEM_FILE_CACHE.get(abs_p)
        if cached and cached[0] == mtime_ns and cached[1] == size_bytes:
            return cached[2]

        with open(abs_p, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        _MEM_FILE_CACHE[abs_p] = (mtime_ns, size_bytes, content)
        # If Python file, parse AST once, cache it, and incrementally update persistent symbol index
        if abs_p.lower().endswith(".py") and content:
            try:
                tree = ast.parse(content)
                _MEM_AST_CACHE[abs_p] = (mtime_ns, size_bytes, tree)
                _sync_py_file_symbols(abs_p, mtime_ns, size_bytes, content, tree)
            except Exception:
                pass
        return content
    except Exception:
        return ""


def _get_cached_ast(content: str, filepath: str = "") -> ast.AST | None:
    """Return a cached AST for `filepath` if unchanged on disk, otherwise parse once and cache."""
    if filepath:
        try:
            abs_p = os.path.abspath(filepath)
            st = os.stat(abs_p)
            cached = _MEM_AST_CACHE.get(abs_p)
            if cached and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
                return cached[2]
            tree = ast.parse(content)
            _MEM_AST_CACHE[abs_p] = (st.st_mtime_ns, st.st_size, tree)
            _sync_py_file_symbols(abs_p, st.st_mtime_ns, st.st_size, content, tree)
            return tree
        except Exception:
            pass
    try:
        return ast.parse(content)
    except Exception:
        return None


def _format_function_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = []
    if hasattr(node, 'args') and getattr(node.args, 'args', None):
        args.extend([a.arg for a in node.args.args])
    return f"({', '.join(args)})"


def _extract_balanced_block(lines: list[str], start_idx: int) -> tuple[str, int, int]:
    """Extract a curly-brace balanced code block { ... } starting at start_idx."""
    depth = 0
    found_open = False
    end_idx = start_idx
    max_search = min(len(lines), start_idx + 500)

    for j in range(start_idx, max_search):
        line = lines[j]
        # Ignore comments
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("*"):
            continue

        for ch in line:
            if ch == '{':
                depth += 1
                found_open = True
            elif ch == '}':
                depth -= 1

        if found_open and depth <= 0:
            end_idx = j
            break

    # If no braces opened within 5 lines, look for semicolon (e.g. prototype / abstract method)
    if not found_open:
        for j in range(start_idx, min(len(lines), start_idx + 5)):
            if ";" in lines[j]:
                end_idx = j
                break

    source = "\n".join(lines[start_idx:end_idx + 1])
    return source, start_idx + 1, end_idx + 1


def parse_polyglot_structure(content: str, ext: str) -> dict:
    """Extract imports, classes, and functions from any supported programming language."""
    ext = ext.lower()
    imports = []
    classes = []
    functions = []

    lines = content.splitlines()

    for line in lines:
        line_s = line.strip()
        if not line_s or line_s.startswith("//") or line_s.startswith("/*") or line_s.startswith("*"):
            continue

        # 1. Imports / Includes / Using
        if ext in {".c", ".cpp", ".cc", ".cxx", ".h", ".hpp"}:
            m = re.match(r'^\s*#include\s+([<"][^>"]+[>"])', line)
            if m:
                imports.append(f"#include {m.group(1)}")
        elif ext in {".java", ".kt", ".kts"}:
            m = re.match(r'^\s*import\s+([\w.*]+)', line)
            if m:
                imports.append(f"import {m.group(1)}")
        elif ext in {".cs"}:
            m = re.match(r'^\s*using\s+([\w.]+);', line)
            if m:
                imports.append(f"using {m.group(1)}")
        elif ext in {".js", ".jsx", ".ts", ".tsx", ".mjs"}:
            m = re.match(r'^\s*(?:import\s+.+?\s+from\s+[\'"]([^\'"]+)[\'"]|require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\))', line)
            if m:
                imports.append(m.group(1) or m.group(2))
        elif ext in {".go"}:
            m = re.match(r'^\s*(?:import\s+)?[\'"]([^\'"]+)[\'"]', line)
            if m:
                imports.append(m.group(1))
        elif ext in {".rs"}:
            m = re.match(r'^\s*use\s+([\w:]+)', line)
            if m:
                imports.append(m.group(1))

        # 2. Classes / Interfaces / Structs / Objects
        cls_m = re.search(
            r'\b(?:class|interface|struct|enum\s+class|data\s+class|object|record|trait|type\s+\w+\s+struct)\s+([A-Za-z0-9_]+)',
            line
        )
        if cls_m:
            cname = cls_m.group(1)
            if cname not in {"class", "struct", "interface", "object"}:
                classes.append(cname)

        # 3. Functions / Methods
        if ext in {".kt", ".kts"}:
            fn_m = re.search(r'\bfun\s+(?:<[^>]+>\s*)?([A-Za-z0-9_]+)\s*\(', line)
            if fn_m:
                functions.append(fn_m.group(1))
        elif ext in {".java", ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".cs", ".dart"}:
            fn_m = re.search(
                r'\b([A-Za-z0-9_]+)\s*\([^)]*\)\s*(?:const|noexcept|override|throws\s+[\w,\s]+)?\s*[{;]',
                line
            )
            if fn_m:
                name = fn_m.group(1)
                if name not in CONTROL_FLOW_KEYWORDS and not name.startswith("~"):
                    functions.append(name)
        elif ext in {".js", ".jsx", ".ts", ".tsx", ".mjs"}:
            fn_m = re.search(r'\b(?:function\s+([A-Za-z0-9_]+)|([A-Za-z0-9_]+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)', line)
            if fn_m:
                functions.append(fn_m.group(1) or fn_m.group(2))
        elif ext in {".go"}:
            fn_m = re.search(r'\bfunc\s+(?:\([^)]+\)\s+)?([A-Za-z0-9_]+)\s*\(', line)
            if fn_m:
                functions.append(fn_m.group(1))
        elif ext in {".rs"}:
            fn_m = re.search(r'\bfn\s+([A-Za-z0-9_]+)\s*\(', line)
            if fn_m:
                functions.append(fn_m.group(1))
        elif ext in {".swift"}:
            fn_m = re.search(r'\bfunc\s+([A-Za-z0-9_]+)\s*\(', line)
            if fn_m:
                functions.append(fn_m.group(1))

    return {
        "imports": list(dict.fromkeys(imports)),
        "classes": list(dict.fromkeys(classes)),
        "functions": list(dict.fromkeys(functions)),
    }


def _normalize_code_for_dedup(code: str) -> str:
    """Normalize Python source code by stripping comments, docstrings, and whitespace to detect duplicate definitions."""
    cleaned = []
    for line in (code or "").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        cleaned.append(re.sub(r"\s+", " ", s))
    return "\n".join(cleaned)


def is_exact_code_symbol(name: str) -> bool:
    """Return True if `name` is an exact class, function, or method name in the SQLite symbol index."""
    clean = (name or "").strip().strip("'\"`()")
    if not clean or len(clean) < 3 or " " in clean:
        return False
    root = get_project_root()
    root_norm = str(Path(root).resolve()).replace("\\", "/")
    try:
        conn = _get_symbol_db()
        row = conn.execute(
            """
            SELECT 1 FROM symbols
            WHERE project_root = ? AND symbol_lower = ? AND symbol_type IN ('class', 'function', 'method')
            LIMIT 1;
            """,
            (root_norm, clean.lower()),
        ).fetchone()
        conn.close()
        return row is not None
    except Exception:
        return False


def _explain_python_symbol_ast(source_code: str) -> str:
    """
    Generate a clean, structured, human-readable engineering summary from Python source code:
      - Purpose (from docstring or inferred from AST)
      - Signature & Required Inputs
      - Authentication & Access Control (for DRF Views)
      - Validation & Error Handling
      - Core Operations (External API / Service / ORM calls)
      - Return / Response Values (multiline expressions cleanly collapsed)
    """
    if not source_code or not source_code.strip():
        return ""

    import textwrap
    dedented = textwrap.dedent(source_code)
    lines = [l.strip() for l in dedented.splitlines() if l.strip() and not l.strip().startswith("#")]
    bullets: list[str] = []

    # Parse AST if possible for clean docstring, signature, required_fields, and multiline return extraction
    docstring = ""
    required_keys: list[str] = []
    ast_returns: list[str] = []
    try:
        tree = ast.parse(dedented)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                raw_doc = ast.get_docstring(node) or ""
                if raw_doc:
                    doc_lines = [dl.strip() for dl in raw_doc.splitlines() if dl.strip()]
                    if doc_lines:
                        docstring = doc_lines[0]
                        for dl in doc_lines[1:]:
                            if dl.startswith("-") and "required" in raw_doc.lower():
                                required_keys.extend(
                                    [k.strip() for k in dl.lstrip("- ").split(",") if k.strip()]
                                )
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id in ("required_fields", "mandatory_fields") and isinstance(node.value, (ast.List, ast.Tuple)):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                if elt.value not in required_keys:
                                    required_keys.append(elt.value)
            elif isinstance(node, ast.Return) and node.value is not None:
                try:
                    unparsed = re.sub(r"\s+", " ", ast.unparse(node.value)).strip()
                    if len(unparsed) > 120:
                        unparsed = unparsed[:117] + "..."
                    ast_returns.append(f"`return {unparsed}`")
                except Exception:
                    pass
    except Exception:
        pass

    # 1. Purpose / Overview
    if docstring:
        bullets.append(f"- **Purpose**: {docstring}")

    # 2. Authentication & Permissions
    auth_m = re.findall(r"authentication_classes\s*=\s*\[([^\]]+)\]", source_code)
    perm_m = re.findall(r"permission_classes\s*=\s*\[([^\]]+)\]", source_code)
    if auth_m or perm_m:
        parts = []
        if auth_m:
            parts.append(f"Auth: `{auth_m[0].strip()}`")
        if perm_m:
            parts.append(f"Permissions: `{perm_m[0].strip()}`")
        bullets.append(f"- **Authentication & Access Control**: {', '.join(parts)}")

    # 3. Signatures & Required Inputs
    sigs = [l.rstrip(":") for l in lines if re.match(r"^(?:async\s+)?def\s+\w+\s*\(", l)]
    req_gets = re.findall(r"""(?:request\.data|request\.GET|request\.query_params|data|payload)\.get\(\s*['"]([^'"]+)['"]""", source_code)
    for rg in req_gets:
        if rg not in required_keys:
            required_keys.append(rg)
    if sigs:
        sig_str = ", ".join(f"`{s}`" for s in sigs[:4])
        if required_keys:
            sig_str += f" *(Fields used: {', '.join(f'`{k}`' for k in required_keys[:8])})*"
        bullets.append(f"- **Signature & Inputs**: {sig_str}")
    elif required_keys:
        bullets.append("- **Inputs / Fields Read**: " + ", ".join(f"`{k}`" for k in required_keys[:8]))

    # 4. Validation & Error Handling
    guards = []
    if "missing_fields" in source_code:
        guards.append("Validates `required_fields` in payload and returns error dict if any are missing")
    raw_lines = dedented.splitlines()
    for idx, rl in enumerate(raw_lines):
        s = rl.strip()
        if s.startswith(("if not ", "if 'error'", 'if "error"', "if count", "if response.status_code")) and "missing_fields" not in s:
            guards.append(f"Checks `{s.rstrip(':')}`")
    except_clauses = re.findall(r"except\s+([A-Za-z0-9_.,\s]+?)(?:\s+as\s+\w+)?\s*:", source_code)
    if except_clauses:
        clean_exc = list(dict.fromkeys(e.strip() for e in except_clauses))
        guards.append("Catches exceptions: " + ", ".join(f"`{e}`" for e in clean_exc[:4]))
    if guards:
        bullets.append("- **Validation & Error Handling**: " + "; ".join(dict.fromkeys(guards[:4])))

    # 5. External API / Service / Serializer Calls
    service_calls = []
    ep_assigns = re.findall(r"""(?:endpoint|url)\s*=\s*(?:f?['"]([^'"]+)['"])""", source_code)
    if ep_assigns:
        service_calls.append("Target endpoint: " + ", ".join(f"`{ep}`" for ep in dict.fromkeys(ep_assigns[:2])))
    for l in lines:
        if any(k in l for k in ("requests.post", "requests.get", "requests.put", "requests.delete", "self.post(", "self.get(", "openai_integration.", "Serializer(", "api.")):
            if not l.startswith("def "):
                service_calls.append(f"`{l[:110]}`")
    if service_calls:
        bullets.append("- **External API / Service Calls**: " + "; ".join(dict.fromkeys(service_calls[:4])))

    # 6. Database / ORM Operations
    orm_calls = []
    for l in lines:
        if re.search(r"\b\w+\.objects\.(?:filter|get|create|update_or_create|get_or_create|all|exclude|values|count|first|last)\b|\.save\(|\.delete\(", l):
            orm_calls.append(f"`{l[:110]}`")
    if orm_calls:
        bullets.append("- **Database / ORM Operations**: " + "; ".join(dict.fromkeys(orm_calls[:4])))

    # 7. Return / API Responses
    if ast_returns:
        bullets.append("- **Returns**: " + "; ".join(dict.fromkeys(ast_returns[:4])))
    else:
        returns = [f"`{l[:110]}`" for l in lines if l.startswith("return ") and l != "return {"]
        if returns:
            bullets.append("- **Returns**: " + "; ".join(dict.fromkeys(returns[:4])))

    if not bullets:
        return ""
    return "**📋 Structured Summary:**\n" + "\n".join(bullets)


def _find_call_sites_summary(symbol_name: str, max_callers: int = 5) -> str:
    """Find concise 1-line call sites where `symbol_name` is invoked across the codebase."""
    if not symbol_name or len(symbol_name) < 3:
        return ""
    root = get_project_root()
    call_pat = re.compile(rf"\b{re.escape(symbol_name)}\s*\(")
    callers: list[str] = []
    seen_locs: set[tuple[str, str]] = set()
    for filepath in _get_code_files():
        rel_p = os.path.relpath(filepath, root).replace("\\", "/")
        if "/migrations/" in rel_p or "/tests/" in rel_p or rel_p.startswith("tests/") or rel_p == "build_project_overview.py":
            continue
        content = _read_file(filepath)
        if not content or symbol_name not in content:
            continue
        for lineno, line in enumerate(content.splitlines(), start=1):
            s = line.strip()
            if s.startswith(("def ", "class ", "#")) or not call_pat.search(s):
                continue
            ast_ctx = _find_enclosing_ast_symbol(content, lineno, filepath) if rel_p.endswith(".py") else {}
            enc = (
                f"{ast_ctx['class_name']}.{ast_ctx['func_name']}"
                if ast_ctx.get("class_name") and ast_ctx.get("func_name")
                else (ast_ctx.get("class_name") or ast_ctx.get("func_name") or "module-level")
            )
            loc_key = (rel_p, enc)
            if loc_key in seen_locs:
                continue
            seen_locs.add(loc_key)
            callers.append(f"- `{rel_p}:{lineno}` (inside `{enc}`): `{s[:110]}`")
            if len(callers) >= max_callers:
                break
        if len(callers) >= max_callers:
            break
    if not callers:
        return ""
    return "**🔗 Where It Is Called:**\n" + "\n".join(callers)


def extract_function(name: str, file_path: str = "", include_callers: bool = True) -> str:
    """
    Extract the structured summary, call sites, and source code of a function, method, or class by name.
    Automatically deduplicates identical definitions in the same file.
    """
    names_to_find = [n.strip() for n in re.split(r"[,;]+|\band\b", name) if n.strip()]
    if not names_to_find:
        names_to_find = [name.strip()]

    all_results = []
    root = get_project_root()
    root_norm = str(Path(root).resolve()).replace("\\", "/")

    for single_name in names_to_find:
        results = []
        target_lower = single_name.lower()
        caller_block = _find_call_sites_summary(single_name) if include_callers else ""

        # 0. Fast-path: Check Incremental SQLite Symbol Index first (zero-walk lookup for previously scanned files)
        try:
            conn = _get_symbol_db()
            path_filter_sql = ""
            params: list = [root_norm, target_lower]
            if file_path:
                clean_fp = file_path.replace("\\", "/").strip("/")
                path_filter_sql = " AND s.rel_path LIKE ?"
                params.append(f"%{clean_fp}%")
            cur = conn.execute(
                f"""
                SELECT s.rel_path, s.symbol_type, s.parent_class, s.start_line, s.end_line, s.source_code,
                       f.mtime_ns, f.size_bytes
                FROM symbols s
                JOIN indexed_files f
                  ON s.project_root = f.project_root AND s.rel_path = f.rel_path
                WHERE s.project_root = ?
                  AND s.symbol_lower = ?
                  AND s.symbol_type IN ('class', 'function', 'method')
                  {path_filter_sql}
                ORDER BY s.rel_path, s.start_line
                LIMIT 6
                """,
                tuple(params),
            )
            indexed_rows = cur.fetchall()
            conn.close()
            stale_files_detected = False
            dedup_map: dict[tuple[str, str], dict] = {}
            for r_rel, r_type, r_parent, r_start, r_end, r_src, idx_mtime, idx_size in indexed_rows:
                full_p = os.path.join(root, r_rel)
                try:
                    st = os.stat(full_p)
                    if st.st_mtime_ns != idx_mtime or st.st_size != idx_size or not r_src:
                        stale_files_detected = True
                        _read_file(full_p)
                        continue
                except OSError:
                    if not r_src:
                        stale_files_detected = True
                        continue

                norm_key = (r_rel, _normalize_code_for_dedup(r_src))
                if norm_key in dedup_map:
                    dedup_map[norm_key]["dup_ranges"].append(f"{r_start}-{r_end}")
                else:
                    dedup_map[norm_key] = {
                        "rel": r_rel,
                        "type": r_type,
                        "parent": r_parent,
                        "start": r_start,
                        "end": r_end,
                        "src": r_src,
                        "dup_ranges": [],
                    }

            if dedup_map and not stale_files_detected:
                for item in list(dedup_map.values())[:3]:
                    sym_label = f"`{item['parent']}.{single_name}`" if item["parent"] else f"`{single_name}`"
                    dup_note = (
                        f" *(duplicate definition at lines {', '.join(item['dup_ranges'])} omitted)*"
                        if item["dup_ranges"]
                        else ""
                    )
                    walkthrough = _explain_python_symbol_ast(item["src"])
                    wt_block = f"\n{walkthrough}\n" if walkthrough else ""
                    results.append(
                        f"📄 File: `{item['rel']}` (lines {item['start']}-{item['end']}) — {sym_label}{dup_note}"
                        f"{wt_block}\n```python\n{item['src'].strip()}\n```"
                    )
                if caller_block:
                    results.append(caller_block)
                all_results.extend(results)
                continue
            results = []
        except Exception:
            pass

        seen_norm_bodies: set[tuple[str, str]] = set()
        for filepath in _get_code_files(file_path):
            content = _read_file(filepath)
            if not content or target_lower not in content.lower():
                continue

            rel_path = os.path.relpath(filepath, root).replace('\\', '/')
            ext = os.path.splitext(filepath)[1].lower()

            # 1. Python AST parsing
            if ext == ".py":
                try:
                    tree = _get_cached_ast(content, filepath)
                    if tree is None:
                        raise ValueError("AST parse failed")
                    lines = content.splitlines()
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            if node.name.lower() == target_lower:
                                start_line = node.lineno - 1
                                end_line = node.end_lineno
                                source = '\n'.join(lines[start_line:end_line])
                                norm_key = (rel_path, _normalize_code_for_dedup(source))
                                if norm_key in seen_norm_bodies:
                                    continue
                                seen_norm_bodies.add(norm_key)
                                walkthrough = _explain_python_symbol_ast(source)
                                wt_block = f"\n{walkthrough}\n" if walkthrough else ""
                                results.append(
                                    f"📄 File: `{rel_path}` (lines {node.lineno}-{node.end_lineno}) — `{node.name}`"
                                    f"{wt_block}\n```python\n{source.strip()}\n```"
                                )
                                if len(results) >= 3:
                                    break
                    if len(results) >= 3:
                        break
                    continue
                except Exception:
                    pass  # Fallback to balanced block extraction

            # 2. Multi-language (Kotlin, Java, C++, C#, JS, TS, Go, Rust, etc.)
            lines = content.splitlines()
            for i, line in enumerate(lines):
                is_class_def = re.search(
                    rf'\b(?:class|interface|struct|enum\s+class|data\s+class|object|record|trait)\s+{re.escape(single_name)}\b',
                    line,
                    re.IGNORECASE
                )
                if ext in {".kt", ".kts"}:
                    is_func_def = re.search(rf'\bfun\s+(?:<[^>]+>\s*)?{re.escape(single_name)}\s*\(', line, re.IGNORECASE)
                elif ext == ".py":
                    is_func_def = re.search(rf'\bdef\s+{re.escape(single_name)}\s*\(', line, re.IGNORECASE)
                elif ext == ".go":
                    is_func_def = re.search(rf'\bfunc\s+(?:\([^)]+\)\s+)?{re.escape(single_name)}\s*\(', line, re.IGNORECASE)
                elif ext == ".rs":
                    is_func_def = re.search(rf'\bfn\s+{re.escape(single_name)}\s*\(', line, re.IGNORECASE)
                elif ext == ".swift":
                    is_func_def = re.search(rf'\bfunc\s+{re.escape(single_name)}\s*\(', line, re.IGNORECASE)
                elif ext in {".js", ".jsx", ".ts", ".tsx", ".mjs"}:
                    is_func_def = re.search(
                        rf'(?:\bfunction\s+{re.escape(single_name)}\s*\(|\b{re.escape(single_name)}\s*=\s*(?:async\s*)?\([^)]*\)\s*=>|\b(?:public|private|protected|static|async|override|\s)*\s*{re.escape(single_name)}\s*\([^)]*\)\s*(?::\s*[\w<>\[\]]+)?\s*\{{)',
                        line,
                        re.IGNORECASE
                    )
                else:
                    is_func_def = re.search(
                        rf'^\s*(?:@\w+(?:\([^)]*\))?\s*)*(?:public|protected|private|static|virtual|inline|explicit|final|override|abstract|async|\s)*\s*[\w<>\[\],*&:]+\s+{re.escape(single_name)}\s*\([^)]*\)\s*(?:const|noexcept|throws\s+[\w,\s]+)?\s*\{{?',
                        line,
                        re.IGNORECASE
                    )

                if is_class_def or is_func_def:
                    source, start_line, end_line = _extract_balanced_block(lines, i)
                    results.append(f"📄 File: {rel_path} (lines {start_line}-{end_line})\n\n{source}")
                    if len(results) >= 3:
                        break

            if len(results) >= 3:
                break

        if results:
            if caller_block:
                results.append(caller_block)
            all_results.extend(results)

    if not all_results:
        return f"No function or class named '{name}' found across project code files."

    return "\n\n".join(all_results)


def analyze_module(file_path: str) -> str:
    """
    Analyze any code module (Python, Kotlin, Java, C++, C, TypeScript, etc.)
    and return its complete architectural structure (imports, classes, functions).
    """
    root = get_project_root()
    full_path = os.path.join(root, file_path) if not os.path.isabs(file_path) else file_path

    content = _read_file(full_path)
    if not content:
        return f"Error: Could not read file '{file_path}'"

    rel_path = os.path.relpath(full_path, root).replace('\\', '/') if os.path.exists(full_path) else file_path
    ext = os.path.splitext(full_path)[1].lower()

    # 1. Python AST structured analysis
    if ext == ".py":
        try:
            tree = _get_cached_ast(content, full_path)
            if tree is None:
                raise ValueError("AST parse failed")
            imports = []
            classes = []
            functions = []

            for node in tree.body:
                if isinstance(node, ast.Import):
                    for n in node.names:
                        imports.append(f"import {n.name}")
                elif isinstance(node, ast.ImportFrom):
                    names = ", ".join([n.name for n in node.names])
                    module = node.module or ""
                    imports.append(f"from {module} import {names}")
                elif isinstance(node, ast.ClassDef):
                    doc = ast.get_docstring(node)
                    doc_str = f'"""{doc[:100]}..."""' if doc else ""
                    methods = [
                        f"{item.name}{_format_function_args(item)}"
                        for item in node.body
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ]
                    bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
                    base_str = f"({', '.join(bases)})" if bases else ""
                    methods_str = ", ".join(methods)
                    class_info = f"class {node.name}{base_str}:\n"
                    if doc_str:
                        class_info += f"  {doc_str}\n"
                    if methods_str:
                        class_info += f"  Methods: {methods_str}"
                    classes.append(class_info.strip())
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    doc = ast.get_docstring(node)
                    doc_str = f'"""{doc[:100]}..."""' if doc else ""
                    func_info = f"def {node.name}{_format_function_args(node)}:\n"
                    if doc_str:
                        func_info += f"  {doc_str}"
                    functions.append(func_info.strip())

            report = [f"📦 Module: {rel_path} (Python)\n"]
            if imports:
                report.append("📥 Imports:\n" + "\n".join(f"  {i}" for i in imports) + "\n")
            if classes:
                report.append("🏗️ Classes:\n" + "\n".join(f"  {c}" for c in classes) + "\n")
            if functions:
                report.append("⚡ Functions:\n" + "\n".join(f"  {f}" for f in functions))
            return "\n".join(report).strip()
        except Exception:
            pass

    # 2. Universal Polyglot Parser (Kotlin, Java, C++, C, C#, TypeScript, Go, etc.)
    struct = parse_polyglot_structure(content, ext)
    lang_name = {
        ".kt": "Kotlin", ".kts": "Kotlin Script", ".java": "Java",
        ".cpp": "C++", ".c": "C", ".hpp": "C++ Header", ".h": "C/C++ Header",
        ".cs": "C#", ".ts": "TypeScript", ".tsx": "TypeScript React",
        ".js": "JavaScript", ".jsx": "React JSX", ".go": "Go",
        ".rs": "Rust", ".dart": "Dart/Flutter", ".swift": "Swift",
    }.get(ext, ext[1:].upper() if ext else "Code")

    report = [f"📦 Module: {rel_path} ({lang_name})\n"]
    if struct["imports"]:
        report.append("📥 Imports / Includes:\n" + "\n".join(f"  {i}" for i in struct["imports"][:30]) + "\n")
    if struct["classes"]:
        report.append("🏗️ Classes / Structs:\n" + "\n".join(f"  class {c}" for c in struct["classes"][:50]) + "\n")
    if struct["functions"]:
        report.append("⚡ Functions / Methods:\n" + "\n".join(f"  {f}()" for f in struct["functions"][:60]))

    return "\n".join(report).strip()


def find_related_code(name: str) -> str:
    """
    Find where a function, class, or symbol is defined, imported, called, or referenced
    across all files in the project (Python, Kotlin, Java, C++, TypeScript, etc.).
    """
    root = get_project_root()
    target_lower = name.lower()

    defined_in = []
    imported_by = []
    called_from = []
    referenced_in = []

    total_results = 0
    max_results = 25

    for filepath in _get_code_files():
        if total_results >= max_results:
            break

        content = _read_file(filepath)
        if not content or target_lower not in content.lower():
            continue

        rel_path = os.path.relpath(filepath, root).replace('\\', '/')
        ext = os.path.splitext(filepath)[1].lower()

        # 1. Python AST inspection
        if ext == ".py":
            try:
                tree = _get_cached_ast(content, filepath)
                if tree is None:
                    raise ValueError("AST parse failed")
                for node in ast.walk(tree):
                    if total_results >= max_results:
                        break
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if node.name == name:
                            defined_in.append(f"{rel_path}:{node.lineno}")
                            total_results += 1
                    elif isinstance(node, ast.ImportFrom):
                        if any(n.name == name for n in node.names):
                            imported_by.append(f"{rel_path}:{node.lineno} (from {node.module or '.'} import {name})")
                            total_results += 1
                    elif isinstance(node, ast.Import):
                        if any(n.name == name for n in node.names):
                            imported_by.append(f"{rel_path}:{node.lineno} (import {name})")
                            total_results += 1
                    elif isinstance(node, ast.Call):
                        if (isinstance(node.func, ast.Name) and node.func.id == name) or \
                           (isinstance(node.func, ast.Attribute) and node.func.attr == name):
                            called_from.append(f"{rel_path}:{node.lineno}")
                            total_results += 1
                    elif isinstance(node, ast.Attribute) and node.attr == name:
                        ref = f"{rel_path}:{node.lineno}"
                        if ref not in called_from and ref not in defined_in:
                            referenced_in.append(ref)
                            total_results += 1
                continue
            except Exception:
                pass

        # 2. Polyglot text/regex analysis (Kotlin, Java, C++, C#, JS, TS, etc.)
        lines = content.splitlines()
        for lineno, line in enumerate(lines, start=1):
            if total_results >= max_results:
                break
            if target_lower not in line.lower():
                continue

            # Check if line is an import / include
            if re.search(rf'^\s*(?:import|#include|using|from)\s+.*?\b{re.escape(name)}\b', line):
                imported_by.append(f"{rel_path}:{lineno} ({line.strip()})")
                total_results += 1
            # Check if line defines the class/function
            elif re.search(rf'\b(?:class|interface|struct|fun|def|fn|func|object|record)\s+{re.escape(name)}\b', line):
                defined_in.append(f"{rel_path}:{lineno}")
                total_results += 1
            # Check if called
            elif re.search(rf'\b{re.escape(name)}\s*\(', line):
                called_from.append(f"{rel_path}:{lineno}")
                total_results += 1
            else:
                referenced_in.append(f"{rel_path}:{lineno}")
                total_results += 1

    if total_results == 0:
        return f"No references to '{name}' found across code files in the project."

    report = [f"🔍 Related code for '{name}':\n"]
    if defined_in:
        report.append("📍 Defined in:\n" + "\n".join(f"  {d}" for d in sorted(set(defined_in))) + "\n")
    if imported_by:
        report.append("📥 Imported by:\n" + "\n".join(f"  {i}" for i in sorted(set(imported_by))) + "\n")
    if called_from:
        report.append("📞 Called from:\n" + "\n".join(f"  {c}" for c in sorted(set(called_from))) + "\n")
    if referenced_in:
        report.append("📎 Referenced in:\n" + "\n".join(f"  {r}" for r in sorted(set(referenced_in))))

    return "\n".join(report).strip()


def index_project_code(project: str = "") -> str:
    """
    Walk the entire project, parse all code files (Python, Kotlin, Java, C++, TypeScript, etc.),
    and store structured code architecture summaries in ChromaDB.
    """
    try:
        from src.tools.rag_tools import get_vector_store
        from langchain_core.documents import Document
    except ImportError:
        return "Error: Could not import rag_tools or langchain_core."

    root = get_project_root()
    if not project:
        project = os.path.basename(root)

    vector_store = get_vector_store()

    # Delete old code structure records for this project
    try:
        existing = vector_store.get(where={"$and": [{"record_type": "code_structure"}, {"project": project}]})
        old_ids = existing.get("ids", [])
        if old_ids:
            vector_store.delete(ids=old_ids)
    except Exception:
        pass

    modules = 0
    total_classes = 0
    total_functions = 0
    languages_seen: set[str] = set()
    documents = []

    for filepath in _get_code_files():
        content = _read_file(filepath)
        if not content:
            continue

        rel_path = os.path.relpath(filepath, root).replace('\\', '/')
        ext = os.path.splitext(filepath)[1].lower()
        languages_seen.add(ext)

        # Parse classes, functions, imports
        if ext == ".py":
            try:
                tree = _get_cached_ast(content, filepath)
                if tree is None:
                    raise ValueError("AST parse failed")
                imports = []
                classes = []
                functions = []
                for node in tree.body:
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            imports.append(name.name)
                    elif isinstance(node, ast.ImportFrom):
                        imports.append(f"{node.module}")
                    elif isinstance(node, ast.ClassDef):
                        classes.append(node.name)
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        functions.append(node.name)
            except Exception:
                struct = parse_polyglot_structure(content, ext)
                imports = struct["imports"]
                classes = struct["classes"]
                functions = struct["functions"]
        else:
            struct = parse_polyglot_structure(content, ext)
            imports = struct["imports"]
            classes = struct["classes"]
            functions = struct["functions"]

        total_classes += len(classes)
        total_functions += len(functions)

        summary = (
            f"Code Architecture and Structure\n"
            f"Project: {project}\n"
            f"File: {rel_path} ({ext})\n"
            f"Classes ({len(classes)}): {', '.join(classes[:50])}\n"
            f"Functions ({len(functions)}): {', '.join(functions[:60])}\n"
            f"Imports / Includes: {', '.join(imports[:30])}"
        )

        doc_id = hashlib.sha256(f"code-{project}-{rel_path}".encode()).hexdigest()

        doc = Document(
            page_content=summary,
            metadata={
                "record_type": "code_structure",
                "source": rel_path,
                "project": project,
                "source_file": rel_path,
                "extension": ext,
                "stored_at": datetime.now(timezone.utc).isoformat()
            }
        )
        documents.append((doc_id, doc))
        modules += 1

    if documents:
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            ids = [item[0] for item in batch]
            docs = [item[1] for item in batch]
            try:
                vector_store.add_documents(documents=docs, ids=ids)
            except Exception as e:
                print(f"Error adding batch to ChromaDB: {e}")

    lang_summary = ", ".join(sorted(languages_seen))
    return (
        f"✅ Indexed multi-language code structure: {modules} files across languages [{lang_summary}], "
        f"{total_classes} classes, {total_functions} functions for project '{project}'"
    )


_DJANGO_MODELS_CACHE: dict[str, list[dict]] = {}
_DJANGO_MODELS_MTIME: dict[str, int] = {}
_PG_TABLES_CACHE: list[str] | None = None

_KNOWN_APP_PREFIXES = (
    "subscription_service_",
    "certification_card_",
    "recordactivity_",
    "farm_category_",
    "all_requests_",
    "verification_",
    "procurement_",
    "pop_manage_",
    "authtoken_",
    "broadcast_",
    "insurance_",
    "app_farm_",
    "onemoney_",
    "payment_",
    "survey_",
    "django_",
    "auth_",
    "core_",
    "cms_",
)


def _get_live_pg_tables() -> list[str]:
    """Fetch and cache exact table names from the live PostgreSQL database (if configured)."""
    global _PG_TABLES_CACHE
    if _PG_TABLES_CACHE is not None:
        return _PG_TABLES_CACHE
    try:
        from src.tools.database_tools import _get_connection
        conn, db_type = _get_connection()
        if db_type in ("postgres", "postgresql"):
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' ORDER BY table_name;"
                )
                _PG_TABLES_CACHE = [r[0] for r in cur.fetchall()]
            conn.close()
            return _PG_TABLES_CACHE
        conn.close()
    except Exception:
        pass
    _PG_TABLES_CACHE = []
    return _PG_TABLES_CACHE


def _get_django_models(root: str) -> list[dict]:
    """Scan and cache all Django model definitions in the project root, auto-refreshing if any models.py changes."""
    model_files = []
    max_mtime = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for f in filenames:
            if f == "models.py":
                full_p = os.path.join(dirpath, f)
                rel_p = os.path.relpath(full_p, root).replace("\\", "/")
                app_name_exact = os.path.basename(dirpath)
                model_files.append((app_name_exact, rel_p, full_p))
                try:
                    mt = os.stat(full_p).st_mtime_ns
                    if mt > max_mtime:
                        max_mtime = mt
                except OSError:
                    pass

    if root in _DJANGO_MODELS_CACHE and _DJANGO_MODELS_MTIME.get(root) == max_mtime:
        return _DJANGO_MODELS_CACHE[root]

    models_list: list[dict] = []
    for app_name_exact, rel_p, full_p in model_files:
        app_name = app_name_exact.lower()
        content = _read_file(full_p)
        if not content:
            continue
        tree = _get_cached_ast(content, full_p)
        if tree is None:
            continue

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                cls_name = node.name
                cls_lower = cls_name.lower()
                cls_no_sep = cls_lower.replace("_", "")

                db_table_explicit = None
                db_table_exact = None
                for item in node.body:
                    if isinstance(item, ast.ClassDef) and item.name == "Meta":
                        for meta_stmt in item.body:
                            if isinstance(meta_stmt, ast.Assign):
                                for t in meta_stmt.targets:
                                    if isinstance(t, ast.Name) and t.id == "db_table":
                                        if isinstance(meta_stmt.value, ast.Constant) and isinstance(meta_stmt.value.value, str):
                                            db_table_exact = meta_stmt.value.value
                                            db_table_explicit = db_table_exact.lower()

                default_table_lower = f"{app_name}_{cls_lower}"
                default_table_exact = f"{app_name_exact}_{cls_lower}"
                app_stripped = app_name[4:] if app_name.startswith("app_") else app_name
                alt_table = f"{app_stripped}_{cls_lower}"

                models_list.append({
                    "app_name": app_name_exact,
                    "app_lower": app_name,
                    "model_name": cls_name,
                    "cls_lower": cls_lower,
                    "cls_no_sep": cls_no_sep,
                    "models_file": rel_p,
                    "db_table_explicit": db_table_explicit,
                    "default_table_lower": default_table_lower,
                    "alt_table": alt_table,
                    "table_name": db_table_exact or default_table_exact,
                })

    _DJANGO_MODELS_CACHE[root] = models_list
    _DJANGO_MODELS_MTIME[root] = max_mtime
    return models_list


def resolve_django_table(identifier: str) -> dict | None:
    """
    Resolve a database table name (e.g. 'pop_manage_popgroup' or 'user_roles')
    or natural-language phrase (e.g. 'buyer seller detail', 'crop master', 'weather data')
    to its Django app, Model class name, models.py file path, and exact PostgreSQL table name.
    """
    if not identifier or not isinstance(identifier, str):
        return None
    clean = identifier.strip().strip("`'\"").lower()
    if not clean or len(clean) < 3:
        return None

    def _singularize(s: str) -> str:
        if s.endswith("ies") and len(s) > 4:
            return s[:-3] + "y"
        if s.endswith("ses") and len(s) > 4:
            return s[:-2]
        if s.endswith("s") and not s.endswith(("ss", "us", "is")) and len(s) > 3:
            return s[:-1]
        return s

    clean_underscore = re.sub(r"[\s-]+", "_", clean)
    clean_no_sep = re.sub(r"[\s_-]+", "", clean)
    clean_sing = _singularize(clean_underscore)
    clean_sing_no_sep = _singularize(clean_no_sep)
    candidates_set = {clean, clean_underscore, clean_no_sep, clean_sing, clean_sing_no_sep}

    root = get_project_root()
    models_list = _get_django_models(root)
    pg_tables = _get_live_pg_tables()
    pg_lower_map = {t.lower(): t for t in pg_tables}
    pg_no_sep_map = {t.lower().replace("_", ""): t for t in pg_tables}

    # 1. Exact match against Django models in models.py
    for m in models_list:
        if (
            m["cls_lower"] in candidates_set
            or m["cls_no_sep"] in candidates_set
            or m["default_table_lower"] in candidates_set
            or m["alt_table"] in candidates_set
            or (m["db_table_explicit"] and m["db_table_explicit"] in candidates_set)
        ):
            tbl_exact = m["table_name"]
            if tbl_exact.lower() in pg_lower_map:
                tbl_exact = pg_lower_map[tbl_exact.lower()]
            elif tbl_exact.lower().replace("_", "") in pg_no_sep_map:
                tbl_exact = pg_no_sep_map[tbl_exact.lower().replace("_", "")]
            return {
                "app_name": m["app_name"],
                "model_name": m["model_name"],
                "models_file": m["models_file"],
                "table_name": tbl_exact,
            }

    # 2. Exact match against live PostgreSQL tables (covers tables not in models.py like pop_manage_weatherdata, django_admin_log)
    for t_exact in pg_tables:
        t_lower = t_exact.lower()
        t_no_sep = t_lower.replace("_", "")
        suffix_lower = t_lower
        app_prefix = ""
        for pref in _KNOWN_APP_PREFIXES:
            if t_lower.startswith(pref):
                suffix_lower = t_lower[len(pref):]
                app_prefix = t_exact[:len(pref) - 1]
                break
        suffix_no_sep = suffix_lower.replace("_", "")
        if (
            t_lower in candidates_set
            or t_no_sep in candidates_set
            or suffix_lower in candidates_set
            or suffix_no_sep in candidates_set
        ):
            return {
                "app_name": app_prefix or "database",
                "model_name": None,
                "models_file": None,
                "table_name": t_exact,
            }

    # 3. Unambiguous prefix/substring match for specific multi-word phrases (len >= 8, e.g. "buyer_seller" -> BuyerSellerDetail)
    if len(clean_sing_no_sep) >= 8:
        partial_models = [
            m for m in models_list
            if clean_sing_no_sep in m["cls_no_sep"] or m["cls_no_sep"].startswith(clean_sing_no_sep)
        ]
        if len(partial_models) == 1:
            m = partial_models[0]
            tbl_exact = pg_lower_map.get(m["table_name"].lower(), m["table_name"])
            return {
                "app_name": m["app_name"],
                "model_name": m["model_name"],
                "models_file": m["models_file"],
                "table_name": tbl_exact,
            }

        partial_pg = []
        for t_exact in pg_tables:
            t_lower = t_exact.lower()
            suffix_lower = t_lower
            app_prefix = ""
            for pref in _KNOWN_APP_PREFIXES:
                if t_lower.startswith(pref):
                    suffix_lower = t_lower[len(pref):]
                    app_prefix = t_exact[:len(pref) - 1]
                    break
            suffix_no_sep = suffix_lower.replace("_", "")
            if clean_sing_no_sep in suffix_no_sep or suffix_no_sep.startswith(clean_sing_no_sep):
                partial_pg.append((app_prefix or "database", t_exact))
        if len(partial_pg) == 1:
            return {
                "app_name": partial_pg[0][0],
                "model_name": None,
                "models_file": None,
                "table_name": partial_pg[0][1],
            }

    return None


def resolve_django_column(column_name: str, app_hint: str = "") -> str:
    """
    Search all Django models.py files for a specific field/column name and return
    the exact Model class(es) and field definitions where it is declared.
    """
    if not column_name or not isinstance(column_name, str):
        return f"Invalid column name: {column_name}"

    target = column_name.strip().lower()
    root = get_project_root()
    matches = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for f in filenames:
            if f != "models.py":
                continue
            full_p = os.path.join(dirpath, f)
            rel_p = os.path.relpath(full_p, root).replace("\\", "/")
            if app_hint and app_hint.lower() not in rel_p.lower():
                continue
            content = _read_file(full_p)
            if not content or target not in content.lower():
                continue
            try:
                tree = ast.parse(content)
                lines = content.splitlines()
                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        for stmt in node.body:
                            if isinstance(stmt, ast.Assign):
                                for t in stmt.targets:
                                    if isinstance(t, ast.Name) and t.id.lower() == target:
                                        line_str = lines[stmt.lineno - 1].strip()
                                        matches.append(
                                            f"📄 {rel_p}:{stmt.lineno} (Model `{node.name}`):\n  `{line_str}`"
                                        )
            except Exception:
                continue

    if not matches:
        return f"No model field named '{column_name}' found in any models.py file."
    return f"🔍 Model field matches for '{column_name}':\n\n" + "\n\n".join(matches)


def resolve_django_route(route_query: str) -> str:
    """
    Search all urls.py and routes.py files for a route pattern or view name,
    resolve its full URL prefix from Project/urls.py, and extract the linked view.
    """
    root = get_project_root()
    q = route_query.strip().strip("/").lower()
    if not q:
        return "Please provide a non-empty route or view query."

    # 1. Parse Project/urls.py for root prefixes
    app_prefixes: dict[str, list[str]] = {}
    for root_url_cand in ("Project/urls.py", "urls.py", "config/urls.py"):
        p_full = os.path.join(root, root_url_cand)
        if os.path.exists(p_full):
            c = _read_file(p_full)
            for line in c.splitlines():
                inc_m = re.search(
                    r"""(?:path|re_path|url)\s*\(\s*r?['"]([^'"]*)['"]\s*,\s*include\(\s*['"]([^'"]+)['"]""",
                    line
                )
                if inc_m:
                    prefix, inc_mod = inc_m.group(1), inc_m.group(2)
                    app_part = inc_mod.split(".")[0]
                    app_prefixes.setdefault(app_part, []).append(prefix)

    hits = []
    view_names = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for f in filenames:
            if f not in ("urls.py", "routes.py"):
                continue
            full_p = os.path.join(dirpath, f)
            rel_p = os.path.relpath(full_p, root).replace("\\", "/")
            content = _read_file(full_p)
            if not content or q not in content.lower():
                continue
            app_folder = rel_p.split("/")[0]
            prefixes = app_prefixes.get(app_folder, [""])
            for lineno, line in enumerate(content.splitlines(), start=1):
                if q in line.lower():
                    hits.append((rel_p, lineno, line.strip(), prefixes))
                    vm = re.search(r",\s*(?:views\.)?([A-Za-z0-9_]+)(?:\.as_view\(\))?", line)
                    if vm and vm.group(1) not in ("include", "path", "re_path", "url"):
                        view_names.append((vm.group(1), app_folder))

    if not hits:
        return f"No route matching '{route_query}' found in urls.py files."

    out = [f"🌐 Route resolution for `{route_query}`:\n"]
    for rel_p, lineno, line_s, prefixes in hits[:10]:
        pref_str = ", ".join(f"`/{p.lstrip('/')}`" for p in prefixes if p) or "`/`"
        out.append(f"- **{rel_p}:{lineno}** (Root prefix: {pref_str})\n  `{line_s}`")

    for vname, app_f in view_names[:3]:
        v_src = extract_function(vname, f"{app_f}/views.py")
        if "No function or class" in v_src:
            v_src = extract_function(vname)
        if "No function or class" not in v_src:
            out.append(f"\n### Linked View Implementation (`{vname}`)\n{v_src}")

    return "\n".join(out)


def _expand_token_stems(token: str) -> list[str]:
    """
    Return safe spelling and stem variants for a single lowercase token:
      - Phonetic Aadhaar variants: 'aadhar' / 'aadhaar' / 'adhar' / 'addhar'
      - Safe suffix stripping (-ing, -ed, -es, -s) only when resulting stem is >= 4 chars
        and not a false-positive word ('string', 'thing', 'status', 'class', 'seed', etc.).
    """
    out = [token]
    if re.fullmatch(r"a+d+h?a+r", token):
        for v in ("aadhar", "aadhaar", "adhar", "addhar"):
            if v not in out:
                out.append(v)
    elif "aadhar" in token and "aadhaar" not in token:
        out.append(token.replace("aadhar", "aadhaar"))
    elif "aadhaar" in token:
        out.append(token.replace("aadhaar", "aadhar"))

    for base in list(out):
        # 1. Safe -ing stripping (requires stem >= 4 chars, so 'string'/'thing'/'bring' are untouched)
        if base.endswith("ing") and len(base) - 3 >= 4:
            stem = base[:-3]
            if len(stem) >= 4 and stem[-1] == stem[-2] and stem[-1] not in "aeioulsz":
                stem_dedup = stem[:-1]
                if len(stem_dedup) >= 4 and stem_dedup not in out:
                    out.append(stem_dedup)
            if stem not in out:
                out.append(stem)
            if (stem + "e") not in out:
                out.append(stem + "e")
        # 2. Safe -ed stripping (requires stem >= 4 chars and not ending in 'eed')
        elif base.endswith("ed") and not base.endswith("eed") and len(base) - 2 >= 4:
            stem = base[:-2]
            if stem not in out:
                out.append(stem)
            if (base[:-1]) not in out:
                out.append(base[:-1])
        # 3. Safe -es / -s stripping (requires stem >= 4 chars and not ending in ss/us/is/as/os)
        elif base.endswith("es") and not base.endswith(("sses", "uses", "ises")) and len(base) - 2 >= 4:
            stem = base[:-2]
            if stem not in out:
                out.append(stem)
            if base[:-1] not in out:
                out.append(base[:-1])
        elif base.endswith("s") and not base.endswith(("ss", "us", "is", "as", "os")) and len(base) - 1 >= 4:
            stem = base[:-1]
            if stem not in out:
                out.append(stem)
    return out


def _build_stem_group_regex(part: str) -> re.Pattern:
    """Build a case-insensitive regex matching any valid stem/phonetic variant of a single token part."""
    if re.fullmatch(r"a+d+h?a+r", part):
        return re.compile(r"a+d+h?a+r", re.IGNORECASE)
    stems = _expand_token_stems(part)
    # Sort longest first so regex alternation prefers longer matches
    stems_sorted = sorted(set(stems), key=len, reverse=True)
    return re.compile("|".join(re.escape(s) for s in stems_sorted), re.IGNORECASE)


def _line_has_close_stem_cooccurrence(line: str, group_regexes: list[re.Pattern], max_span_chars: int = 55) -> bool:
    """
    Return True if EVERY stem group regex matches on `line` AND the matches occur
    within `max_span_chars` of each other (equivalent to bounded (?=.*stem1)(?=.*stem2) in any order).
    """
    if len(group_regexes) < 2:
        return False
    match_positions: list[list[tuple[int, int]]] = []
    for rgx in group_regexes:
        spans = [m.span() for m in rgx.finditer(line)]
        if not spans:
            return False
        match_positions.append(spans)

    # Check if there is at least one combination of matches across groups within max_span_chars
    for s0, e0 in match_positions[0]:
        all_close = True
        min_start, max_end = s0, e0
        for other_spans in match_positions[1:]:
            best = min(other_spans, key=lambda sp: max(sp[1], max_end) - min(sp[0], min_start))
            min_start = min(min_start, best[0])
            max_end = max(max_end, best[1])
            if (max_end - min_start) > max_span_chars:
                all_close = False
                break
        if all_close:
            return True
    return False


def _generate_keyword_variants(keyword: str) -> tuple[list[str], list[str]]:
    """
    Generate case/delimiter variants for a project keyword plus root sub-tokens.
    Handles spelling aliases (aadhar <-> aadhaar <-> adhar <-> addhar),
    safe verb/plural stems (masking -> mask), and reversed word order (aadhar_masking -> mask_aadhaar).
    """
    clean = keyword.strip().strip("'\"`")
    if not clean:
        return [], []

    # Split CamelCase (e.g. AryaMitra -> ['Arya', 'Mitra']) and snake/kebab/space
    camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", clean)
    raw_parts = [p.lower() for p in re.split(r"[\s_\-]+", camel_split) if p.strip()]

    # Strip conversational filler tokens if accidentally attached (e.g. 'cvrp_case' -> ['cvrp'])
    filler_tokens = {
        "case", "cases", "schema", "schemas", "flow", "flows", "function", "functions",
        "detail", "details", "info", "called", "named", "call", "calling", "calls", "name", "names",
    }
    filtered_parts = [p for p in raw_parts if p not in filler_tokens]
    parts = filtered_parts if filtered_parts else raw_parts

    primary = []
    for cand in (
        clean.lower(),
        "".join(parts),
        "_".join(parts),
        "-".join(parts),
        " ".join(parts),
    ):
        if cand and cand not in primary:
            primary.append(cand)

    # Expand stems and spelling variants (e.g. aadhar <-> aadhaar, masking -> mask) and reversed 2-word order
    if len(parts) == 1:
        for v in _expand_token_stems(parts[0]):
            if v not in primary:
                primary.append(v)
    elif len(parts) == 2:
        p0_vars = _expand_token_stems(parts[0])
        p1_vars = _expand_token_stems(parts[1])
        for a in p0_vars:
            for b in p1_vars:
                for cand in (f"{a}_{b}", f"{b}_{a}", f"{a}{b}", f"{b}{a}", f"{a} {b}", f"{b} {a}"):
                    if cand not in primary:
                        primary.append(cand)

    # Generic project words that would produce too much noise as standalone sub-tokens
    generic_subtokens = {
        "arya", "aryashakti", "shakti", "farm", "user", "data", "model",
        "view", "test", "base", "main", "core", "util", "utils", "info",
        "list", "type", "item", "code", "name", "file", "post", "create",
        "plot", "plots", "report", "reports", "record", "records", "activity",
        "activities", "manage", "group", "groups", "status", "token", "detail",
        "details", "update", "delete", "fetch", "check", "get", "add",
    }
    sub_tokens = []
    if len(parts) >= 2:
        for p in reversed(parts):  # prioritize distinctive suffix token like 'mitra'
            for stem in _expand_token_stems(p):
                if len(stem) >= 4 and stem not in generic_subtokens and stem not in sub_tokens:
                    sub_tokens.append(stem)
    elif len(parts) == 1:
        for stem in _expand_token_stems(parts[0]):
            if len(stem) >= 4 and stem not in generic_subtokens and stem not in sub_tokens:
                sub_tokens.append(stem)

    return primary, sub_tokens


def _find_enclosing_ast_symbol(py_content: str, target_lineno: int, filepath: str = "") -> dict:
    """
    Given Python file content and a 1-indexed line number, find:
      - enclosing class name and method/function name (if inside a class/def)
      - top-level constant/variable assignment name (if on a module-level assignment)
      - adjacent classes in the same file (immediately preceding or following class)
    """
    info = {
        "class_name": None,
        "func_name": None,
        "assigned_name": None,
        "adjacent_classes": [],
    }
    tree = _get_cached_ast(py_content, filepath)
    if tree is None:
        return info

    top_classes: list[ast.ClassDef] = [n for n in tree.body if isinstance(n, ast.ClassDef)]

    for idx, node in enumerate(tree.body):
        start = getattr(node, "lineno", 0)
        end = getattr(node, "end_lineno", start)
        if start <= target_lineno <= end:
            if isinstance(node, ast.ClassDef):
                info["class_name"] = node.name
                for item in node.body:
                    istart = getattr(item, "lineno", 0)
                    iend = getattr(item, "end_lineno", istart)
                    if istart <= target_lineno <= iend and isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        info["func_name"] = item.name
                        break
                # Find adjacent top-level classes in the same module (e.g. CreateLoanLead next to VerifyLogic)
                cls_indices = [i for i, c in enumerate(top_classes) if c.name == node.name]
                if cls_indices:
                    ci = cls_indices[0]
                    if ci > 0:
                        info["adjacent_classes"].append(top_classes[ci - 1].name)
                    if ci + 1 < len(top_classes):
                        info["adjacent_classes"].append(top_classes[ci + 1].name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                info["func_name"] = node.name
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        info["assigned_name"] = t.id
                        break
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                info["assigned_name"] = node.target.id
            break

    return info


def investigate_keyword(keyword: str) -> str:
    """
    Perform a structured, summarized codebase investigation of a keyword, feature, or API endpoint:
      1. API Endpoints & URL Routes mapped to the feature
      2. Key Classes, Views, Functions & Methods (with concise AST step-by-step summaries)
      3. Models, Serializers & Constants
    """
    clean_kw = (keyword or "").strip().strip("'\"`")
    if not clean_kw:
        return "Error: Please provide a non-empty keyword to investigate."
    if "/" in clean_kw and not clean_kw.endswith(".py"):
        segs = [s for s in clean_kw.strip("/").split("/") if s and not re.match(r"^(?:api|v\d+|\d+)$", s, re.IGNORECASE)]
        if segs:
            clean_kw = segs[-1]

    root = get_project_root()
    root_norm = str(Path(root).resolve()).replace("\\", "/")
    primary_variants, sub_tokens = _generate_keyword_variants(clean_kw)

    # Build per-part stem group regexes for Tier 2 co-occurrence & Tier 3 rarity fallback
    camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", clean_kw)
    raw_parts = [p.lower() for p in re.split(r"[\s_\-]+", camel_split) if p.strip()]
    filler_tokens = {"case", "cases", "schema", "schemas", "flow", "flows", "function", "functions", "detail", "details", "info"}
    clean_parts = [p for p in raw_parts if p not in filler_tokens] or raw_parts
    kw_key = "_".join(clean_parts) if clean_parts else (primary_variants[0] if primary_variants else clean_kw.lower())
    stem_group_regexes = [_build_stem_group_regex(p) for p in clean_parts if len(p) >= 3]

    # Fast-path: Check persistent SQLite keyword_cache (ignore legacy un-summarized reports containing 'Searched variants:')
    live_max_mtime, live_file_count = _get_project_code_stamp(root)
    try:
        conn = _get_symbol_db()
        row = conn.execute(
            "SELECT max_mtime_ns, file_count, report FROM keyword_cache WHERE project_root = ? AND keyword_lower = ?;",
            (root_norm, kw_key),
        ).fetchone()
        if (
            row
            and row[0] == live_max_mtime
            and row[1] == live_file_count
            and row[2]
            and "No occurrences of" not in row[2]
            and "Searched variants:" not in row[2]
            and len(row[2]) <= 5000
        ):
            conn.close()
            return row[2]

        # Query SQLite symbols index for Tier 2 (all stem groups co-occur in symbol) & Tier 3 (rarest stem <= 10 symbols)
        sym_rows = conn.execute(
            "SELECT DISTINCT symbol_lower, rel_path FROM symbols WHERE project_root = ?;",
            (root_norm,),
        ).fetchall()
        conn.close()

        cooccurring_files: set[str] = set()
        for sym_l, sym_rel in sym_rows:
            if any(v in sym_l for v in primary_variants) or (
                len(stem_group_regexes) >= 2 and all(rgx.search(sym_l) for rgx in stem_group_regexes)
            ):
                cooccurring_files.add(sym_rel)
                if sym_l not in primary_variants:
                    primary_variants.append(sym_l)

        # Tier 3 Rarest-Stem Fallback: rank each stem group by how many distinct symbols match it
        if len(stem_group_regexes) >= 2:
            group_matches: list[tuple[int, list[tuple[str, str]]]] = []
            for rgx in stem_group_regexes:
                matched_syms = [(sym_l, sym_rel) for sym_l, sym_rel in sym_rows if rgx.search(sym_l)]
                unique_sym_names = {s[0] for s in matched_syms}
                if 1 <= len(unique_sym_names) <= 10:
                    group_matches.append((len(unique_sym_names), matched_syms))

            group_matches.sort(key=lambda x: x[0])
            if group_matches:
                _, rarest_syms = group_matches[0]
                for sym_l, sym_rel in rarest_syms:
                    if not cooccurring_files or sym_rel in cooccurring_files:
                        if sym_l not in primary_variants:
                            primary_variants.append(sym_l)
    except Exception:
        pass

    # Parse root Project/urls.py upfront so we can resolve full URL mount prefixes
    app_prefixes: dict[str, list[str]] = {}
    root_urls_rel = None
    for root_url_cand in ("Project/urls.py", "urls.py", "config/urls.py"):
        p_full = os.path.join(root, root_url_cand)
        if os.path.exists(p_full):
            root_urls_rel = root_url_cand
            c = _read_file(p_full)
            for line in c.splitlines():
                inc_m = re.search(
                    r"""(?:path|re_path|url)\s*\(\s*r?['"]([^'"]*)['"]\s*,\s*include\(\s*['"]([^'"]+)['"]""",
                    line
                )
                if inc_m:
                    prefix, inc_mod = inc_m.group(1), inc_m.group(2)
                    app_part = inc_mod.split(".")[0]
                    app_prefixes.setdefault(app_part, []).append(prefix)
            break

    # Collect all urls.py files in the project
    url_files: list[tuple[str, str, str]] = []  # (app_folder, rel_path, content)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for f in filenames:
            if f in ("urls.py", "routes.py"):
                full_p = os.path.join(dirpath, f)
                rel_p = os.path.relpath(full_p, root).replace("\\", "/")
                app_folder = rel_p.split("/")[0] if "/" in rel_p else ""
                url_files.append((app_folder, rel_p, _read_file(full_p)))

    # ── STEP 1: Search across code files (deduplicated per enclosing symbol) ──
    constant_model_hits = []
    view_service_hits = []
    url_direct_hits = []
    other_hits = []

    discovered_constants: list[tuple[str, str]] = []  # (const_name, rel_path)
    discovered_views: list[tuple[str, str, list[str]]] = []  # (class_or_func, rel_path, adjacent_classes)
    seen_symbol_hits: set[tuple[str, str]] = set()

    for filepath in _get_code_files():
        content = _read_file(filepath)
        if not content:
            continue
        content_lower = content.lower()
        has_primary = any(v in content_lower for v in primary_variants)
        has_stem_co = len(stem_group_regexes) >= 2 and all(rgx.search(content_lower) for rgx in stem_group_regexes)
        if not has_primary and not has_stem_co:
            continue

        rel_path = os.path.relpath(filepath, root).replace("\\", "/")
        if "/migrations/" in rel_path or "\\migrations\\" in rel_path or "/tests/" in rel_path or rel_path.startswith("tests/"):
            continue
        ext = os.path.splitext(filepath)[1].lower()
        lines = content.splitlines()

        for lineno, line in enumerate(lines, start=1):
            line_lower = line.lower()
            if not any(v in line_lower for v in primary_variants) and not _line_has_close_stem_cooccurrence(line_lower, stem_group_regexes):
                continue

            ast_ctx = _find_enclosing_ast_symbol(content, lineno, filepath) if ext == ".py" else {}
            enclosing_label = ""
            enclosing_key = f"line_{lineno}"
            is_model = rel_path.endswith("models.py")
            if ast_ctx.get("class_name") and ast_ctx.get("func_name"):
                enclosing_label = f" (inside `{ast_ctx['class_name']}.{ast_ctx['func_name']}`)"
                enclosing_key = f"{ast_ctx['class_name']}.{ast_ctx['func_name']}"
                if not is_model:
                    discovered_views.append((ast_ctx["class_name"], rel_path, []))
            elif ast_ctx.get("class_name"):
                enclosing_label = f" (inside class `{ast_ctx['class_name']}`)"
                enclosing_key = ast_ctx["class_name"]
                if not is_model:
                    discovered_views.append((ast_ctx["class_name"], rel_path, []))
            elif ast_ctx.get("func_name"):
                enclosing_label = f" (inside function `{ast_ctx['func_name']}`)"
                enclosing_key = ast_ctx["func_name"]
                if not is_model:
                    discovered_views.append((ast_ctx["func_name"], rel_path, []))
            elif ast_ctx.get("assigned_name"):
                assigned = ast_ctx["assigned_name"]
                if assigned not in {"urlpatterns", "app_name", "dependencies", "operations", "fields", "model", "exclude"}:
                    enclosing_label = f" (constant `{assigned}`)"
                    enclosing_key = assigned
                    discovered_constants.append((assigned, rel_path))

            # Deduplicate multiple line matches inside the exact same function/class
            if not rel_path.endswith(("urls.py", "routes.py")):
                if (rel_path, enclosing_key) in seen_symbol_hits:
                    continue
                seen_symbol_hits.add((rel_path, enclosing_key))

            entry = f"- `{rel_path}:{lineno}`{enclosing_label}: `{line.strip()[:120]}`"

            if rel_path.endswith(("urls.py", "routes.py")):
                url_direct_hits.append(entry)
            elif any(k in rel_path.lower() for k in ("constants.py", "models.py", "serializers.py", "choices")):
                constant_model_hits.append(entry)
            elif any(k in rel_path.lower() for k in ("views.py", "services", "tasks.py", "utils", "controllers")):
                view_service_hits.append(entry)
            else:
                other_hits.append(entry)

    # Trace discovered Serializers to Views that use them
    serializer_classes = [
        (cls_name, rel_p)
        for cls_name, rel_p, _ in discovered_views
        if "serializer" in cls_name.lower() or rel_p.endswith("serializers.py")
    ]
    for ser_name, ser_rel_p in serializer_classes:
        app_dir = ser_rel_p.split("/")[0] if "/" in ser_rel_p else ""
        cand_views_rel = f"{app_dir}/views.py" if app_dir else "views.py"
        cand_views_full = os.path.join(root, cand_views_rel)
        if os.path.exists(cand_views_full):
            v_text = _read_file(cand_views_full)
            if v_text and ser_name in v_text:
                for lno, lstr in enumerate(v_text.splitlines(), start=1):
                    if re.search(rf"\b{re.escape(ser_name)}\b", lstr):
                        ast_v = _find_enclosing_ast_symbol(v_text, lno, cand_views_full)
                        v_cls = ast_v.get("class_name") or ast_v.get("func_name")
                        if v_cls and (cand_views_rel, v_cls) not in seen_symbol_hits:
                            seen_symbol_hits.add((cand_views_rel, v_cls))
                            discovered_views.append((v_cls, cand_views_rel, []))
                            entry = f"- `{cand_views_rel}:{lno}` (inside `{v_cls}`, uses `{ser_name}`)"
                            view_service_hits.append(entry)

    # ── STEP 2 & 3: Follow ONLY primary discovered views to their URL routes (never unrelated adjacent views) ──
    route_mappings = []
    views_to_extract: list[tuple[str, str]] = []  # (view_name, rel_views_file)
    seen_views = set()

    for view_name, rel_views_file, _ in discovered_views:
        if view_name not in seen_views:
            seen_views.add(view_name)
            views_to_extract.append((view_name, rel_views_file))

    for view_name, rel_views_file in views_to_extract:
        if rel_views_file == "build_project_overview.py" or "/tests/" in rel_views_file or rel_views_file.startswith("tests/"):
            continue
        for u_app, u_rel, u_content in url_files:
            if not u_content or view_name not in u_content:
                continue
            prefixes = app_prefixes.get(u_app, [""])
            for lno, lstr in enumerate(u_content.splitlines(), start=1):
                if re.search(rf"(?:views\.|\b){re.escape(view_name)}\b(?:\.as_view|\s*,|\s*\))", lstr) and not lstr.strip().startswith("#"):
                    route_m = re.search(r"""(?:path|re_path|url)\s*\(\s*r?['"]([^'"]*)['"]""", lstr)
                    sub_route = route_m.group(1) if route_m else ""
                    full_endpoints = []
                    for pref in prefixes:
                        clean_pref = pref.strip("/")
                        clean_sub = sub_route.lstrip("^").rstrip("$").lstrip("/")
                        if clean_pref and clean_sub:
                            full_endpoints.append(f"/{clean_pref}/{clean_sub}")
                        elif clean_pref:
                            full_endpoints.append(f"/{clean_pref}/")
                        elif clean_sub:
                            full_endpoints.append(f"/{clean_sub}")
                    ep_str = ", ".join(f"`{ep}`" for ep in full_endpoints) if full_endpoints else f"`{sub_route}`"
                    route_mappings.append(
                        f"- {ep_str} → **`{view_name}`** (`{u_rel}:{lno}`)"
                    )

    # Query SQLite AST index for functions/methods/classes matching clean_kw (deduplicated by body)
    module_symbol_summary: list[str] = []
    component_summaries: list[str] = []
    seen_sym_dedup: set[tuple[str, str]] = set()
    primary_code_snippet = ""
    try:
        conn = _get_symbol_db()
        kw_like = f"%{kw_key}%"
        mod_syms = conn.execute(
            """
            SELECT rel_path, symbol_name, symbol_type, parent_class, start_line, end_line, source_code
            FROM symbols
            WHERE project_root = ?
              AND (LOWER(rel_path) LIKE ? OR LOWER(symbol_name) LIKE ? OR LOWER(parent_class) LIKE ?)
            ORDER BY rel_path, start_line;
            """,
            (root_norm, kw_like, kw_like, kw_like),
        ).fetchall()
        conn.close()
        for rp, sname, stype, pcls, sline, eline, scode in mod_syms:
            if "/tests/" in rp or rp.startswith("tests/") or rp == "build_project_overview.py" or rp.endswith("admin.py"):
                continue
            norm_k = (rp, _normalize_code_for_dedup(scode or sname))
            if norm_k in seen_sym_dedup:
                continue
            seen_sym_dedup.add(norm_k)

            full_sym = f"{pcls}.{sname}" if (stype == "method" and pcls) else sname
            kind_lbl = stype.capitalize()
            module_symbol_summary.append(f"- `{rp}:{sline}-{eline}` — {kind_lbl} **`{full_sym}`**")

            # Skip dunder/private methods in Section 3 summaries unless explicitly searched
            if sname.startswith("_") and sname.lower() != kw_key:
                continue
            # If the parent class is already summarized, only summarize methods that directly match kw_key
            if stype == "method" and pcls and any(f"`{pcls}`" in cs for cs in component_summaries) and kw_key not in sname.lower():
                continue

            if scode and stype in ("function", "method", "class") and len(component_summaries) < 2:
                wt = _explain_python_symbol_ast(scode)
                if wt:
                    component_summaries.append(f"#### `{full_sym}` (`{rp}:{sline}-{eline}`)\n{wt}")
                if not primary_code_snippet and url_direct_hits:
                    code_lines = scode.strip().splitlines()
                    trimmed = "\n".join(code_lines[:45]) + (f"\n    # ... ({len(code_lines) - 45} more lines)" if len(code_lines) > 45 else "")
                    primary_code_snippet = f"#### Source Code: `{full_sym}` (`{rp}:{sline}-{eline}`)\n```python\n{trimmed}\n```"
    except Exception:
        pass

    # Also summarize primary enclosing views that weren't already covered in component_summaries
    ordered_to_extract = sorted(
        views_to_extract,
        key=lambda item: (
            0 if (kw_key in item[0].lower() or kw_key in item[1].lower()) else (1 if item[1].endswith("views.py") else 2)
        ),
    )
    for view_name, rel_views_file in ordered_to_extract:
        if len(component_summaries) >= 4:
            break
        if rel_views_file.endswith(("admin.py", "serializers.py")) or rel_views_file == "build_project_overview.py":
            continue
        if any(f"`{view_name}`" in cs for cs in component_summaries):
            continue
        try:
            conn = _get_symbol_db()
            v_row = conn.execute(
                "SELECT start_line, end_line, source_code FROM symbols WHERE project_root = ? AND rel_path = ? AND symbol_name = ? LIMIT 1;",
                (root_norm, rel_views_file, view_name),
            ).fetchone()
            conn.close()
            if v_row and v_row[2]:
                v_start, v_end, v_src = v_row
                wt = _explain_python_symbol_ast(v_src)
                if wt:
                    component_summaries.append(f"#### `{view_name}` (`{rel_views_file}:{v_start}-{v_end}`)\n{wt}")
                if not primary_code_snippet and url_direct_hits:
                    code_lines = v_src.strip().splitlines()
                    trimmed = "\n".join(code_lines[:45]) + (f"\n    # ... ({len(code_lines) - 45} more lines)" if len(code_lines) > 45 else "")
                    primary_code_snippet = f"#### Source Code: `{view_name}` (`{rel_views_file}:{v_start}-{v_end}`)\n```python\n{trimmed}\n```"
        except Exception:
            pass

    # ── Build Concise, Structured Investigation Report ──
    total_matches = (
        len(constant_model_hits)
        + len(view_service_hits)
        + len(url_direct_hits)
        + len(other_hits)
        + len(module_symbol_summary)
    )

    report = [
        f"🔎 **5-Step End-to-End Codebase Investigation for `{clean_kw}`**",
    ]

    if total_matches == 0:
        report.append(f"- No occurrences of `{clean_kw}` were found in the project codebase.")
        return "\n".join(report)

    # 1. API Routes & Endpoints
    report.append("\n### 1. 🌐 API Endpoints & URL Routes")
    if route_mappings:
        report.extend(list(dict.fromkeys(route_mappings))[:6])
    elif url_direct_hits:
        report.extend(url_direct_hits[:6])
    else:
        report.append(f"- No dedicated URL route is directly mapped to `{clean_kw}` (internal library/model/helper).")

    # 2. Key Symbols Inventory
    report.append("\n### 2. 🧩 Matched Classes, Functions & Call Sites")
    if module_symbol_summary:
        report.append("**Defined Symbols:**\n" + "\n".join(module_symbol_summary[:10]))
    if view_service_hits:
        report.append("**Used in Views & Services:**\n" + "\n".join(view_service_hits[:6]))
    if constant_model_hits:
        report.append("**Models, Serializers & Constants:**\n" + "\n".join(constant_model_hits[:6]))

    # 3. Structured Component Walkthroughs (Summarized!)
    if component_summaries:
        report.append("\n### 3. ⚙️ How It Works (Structured Component Summary)")
        report.extend(component_summaries[:4])

    # Only attach raw code block when investigating a specific direct endpoint/symbol
    if primary_code_snippet:
        report.append("\n" + primary_code_snippet)

    report.append(
        "\n💡 *Tip: Ask `explain function <name>` (e.g., `explain function add_soil_test_plot`) to view the full source code of any specific function or class above.*"
    )

    final_report = "\n".join(report)
    try:
        conn = _get_symbol_db()
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO keyword_cache (project_root, keyword_lower, max_mtime_ns, file_count, report)
                VALUES (?, ?, ?, ?, ?);
                """,
                (root_norm, kw_key, live_max_mtime, live_file_count, final_report),
            )
        conn.close()
    except Exception:
        pass
    return final_report


def _format_numbered_source(content: str, max_chars: int = 15000) -> str:
    """Format file content with 1-indexed line numbers, capped at max_chars."""
    lines = content.splitlines()
    numbered = [f"{i}: {line}" for i, line in enumerate(lines, start=1)]
    joined = "\n".join(numbered)
    if len(joined) > max_chars:
        joined = joined[:max_chars] + f"\n... (truncated at {max_chars} chars)"
    return joined


def trace_request_flow(feature: str = "farm_category") -> str:
    """
    Trace an end-to-end client request through the Django project:
    from Project/urls.py -> <app>/urls.py -> <app>/views.py -> <app>/models.py (and managers) -> <app>/serializers.py.
    Automatically detects any Django app referenced in `feature` (by app name or URL prefix like /api/v1/survey/)
    and returns the complete, line-numbered source code across all layers so multi-endpoint, authentication,
    model/manager, and submission-flow questions can be answered with 100% verified citations.
    If `feature` is a non-app keyword (e.g. 'AryaMitra'), delegates to `investigate_keyword`.
    """
    root = get_project_root()
    feat = (feature or "farm_category").strip()
    feat_lower = feat.lower()

    # 1. Discover all Django app directories in the project root
    django_apps: dict[str, str] = {}  # lower_name -> actual folder name
    try:
        for entry in os.listdir(root):
            full_d = os.path.join(root, entry)
            if os.path.isdir(full_d) and entry not in EXCLUDED_DIRS:
                if any(os.path.exists(os.path.join(full_d, f)) for f in ("urls.py", "views.py", "models.py")):
                    django_apps[entry.lower()] = entry
    except Exception:
        pass

    # 2. Parse Project/urls.py to map URL prefixes to apps
    proj_urls_rel = "Project/urls.py"
    proj_urls_path = os.path.join(root, "Project", "urls.py")
    if not os.path.exists(proj_urls_path) and os.path.exists(os.path.join(root, "urls.py")):
        proj_urls_rel = "urls.py"
        proj_urls_path = os.path.join(root, "urls.py")

    proj_urls_content = _read_file(proj_urls_path) if os.path.exists(proj_urls_path) else ""
    prefix_to_app: list[tuple[str, str, int, str]] = []  # (prefix, app_folder, lineno, line_text)
    if proj_urls_content:
        for lineno, line in enumerate(proj_urls_content.splitlines(), start=1):
            inc_m = re.search(
                r"""(?:path|re_path|url)\s*\(\s*r?['"]([^'"]*)['"]\s*,\s*include\(\s*['"]([^'"]+)['"]""",
                line,
            )
            if inc_m:
                prefix, inc_mod = inc_m.group(1), inc_m.group(2)
                app_part = inc_mod.split(".")[0]
                prefix_to_app.append((prefix.strip("/").lower(), app_part, lineno, line.strip()))

    # 3. Determine which Django app(s) are targeted by `feature`
    matched_apps: list[str] = []

    # Check if any URL prefix from Project/urls.py appears in `feature` (e.g. 'api/v1/survey')
    for pref, app_part, _, _ in prefix_to_app:
        if pref and pref in feat_lower and app_part.lower() in django_apps:
            actual = django_apps[app_part.lower()]
            if actual not in matched_apps:
                matched_apps.append(actual)

    # Check if any Django app folder name appears as a word token in `feature`
    tokens = re.findall(r"[a-zA-Z0-9_]+", feat_lower)
    for tok in tokens:
        if tok in django_apps:
            actual = django_apps[tok]
            if actual not in matched_apps:
                matched_apps.append(actual)
        elif f"app_{tok}" in django_apps:
            actual = django_apps[f"app_{tok}"]
            if actual not in matched_apps:
                matched_apps.append(actual)

    # Special case for default 'farm_category' demo trace
    if not matched_apps and ("farm_category" in feat_lower or feat_lower in {"client", "request", "django", "project"}):
        if "app_farm" in django_apps:
            matched_apps.append(django_apps["app_farm"])

    # If no Django app matched, delegate to 5-step keyword investigation (e.g. 'AryaMitra')
    if not matched_apps:
        return investigate_keyword(feat)

    # 4. Build full end-to-end multi-layer dossier for the matched Django app(s)
    target_app = matched_apps[0]
    report = [
        f"🌐 **End-to-End Request Flow & Code Architecture Dossier (App: `{target_app}`)**\n"
    ]

    # Layer 1: Project/urls.py mount
    if proj_urls_content:
        report.append(f"### 1. Project-Level URL Mount (`{proj_urls_rel}`)")
        app_mount_lines = [
            f"  Line {lno}: `{ltxt}`"
            for pref, app_part, lno, ltxt in prefix_to_app
            if app_part.lower() == target_app.lower()
        ]
        if app_mount_lines:
            report.extend(app_mount_lines)
        else:
            # Show general lines matching target_app
            fallback_lines = [
                f"  Line {i}: `{l.strip()}`"
                for i, l in enumerate(proj_urls_content.splitlines(), 1)
                if target_app.lower() in l.lower()
            ]
            report.extend(fallback_lines[:10] if fallback_lines else [f"  (No direct include for `{target_app}` found in `{proj_urls_rel}`)"])

    # Layer 2: <app>/urls.py
    app_urls_path = os.path.join(root, target_app, "urls.py")
    if os.path.exists(app_urls_path):
        u_content = _read_file(app_urls_path)
        report.append(f"\n### 2. App URL Configuration (`{target_app}/urls.py`)")
        report.append("```python\n" + _format_numbered_source(u_content, max_chars=5000) + "\n```")

    # Layer 3: <app>/views.py
    app_views_path = os.path.join(root, target_app, "views.py")
    v_content = ""
    if os.path.exists(app_views_path):
        v_content = _read_file(app_views_path)
        report.append(f"\n### 3. App Views & Authentication (`{target_app}/views.py`)")
        report.append("```python\n" + _format_numbered_source(v_content, max_chars=15000) + "\n```")

    # Layer 4: <app>/models.py (Models & Custom Managers)
    app_models_path = os.path.join(root, target_app, "models.py")
    m_content = ""
    if os.path.exists(app_models_path):
        m_content = _read_file(app_models_path)
        report.append(f"\n### 4. App Models & Managers (`{target_app}/models.py`)")
        report.append("```python\n" + _format_numbered_source(m_content, max_chars=14000) + "\n```")

    # Layer 5: <app>/serializers.py
    app_ser_path = os.path.join(root, target_app, "serializers.py")
    if os.path.exists(app_ser_path):
        s_content = _read_file(app_ser_path)
        if s_content.strip():
            report.append(f"\n### 5. App Serializers (`{target_app}/serializers.py`)")
            report.append("```python\n" + _format_numbered_source(s_content, max_chars=6000) + "\n```")

    # Layer 6: Resolve imported Constants (e.g. SURVEY_STATUS, SURVEY_STATUS_USER) & Custom Auth Classes
    combined_vm = v_content + "\n" + m_content
    const_imports = re.findall(r"from\s+([\w.]+constants)\s+import\s+([^\n]+)", combined_vm)
    if const_imports:
        const_notes = []
        for mod_path, imported_names in const_imports:
            rel_const_file = mod_path.replace(".", "/") + ".py"
            full_const_file = os.path.join(root, rel_const_file)
            if os.path.exists(full_const_file):
                c_text = _read_file(full_const_file)
                names = [n.strip() for n in imported_names.split(",") if n.strip()]
                for nm in names:
                    for lno, lstr in enumerate(c_text.splitlines(), start=1):
                        if re.match(rf"^\s*{re.escape(nm)}\s*=", lstr):
                            snippet = "\n".join(
                                f"  { k }: {c_text.splitlines()[k-1]}"
                                for k in range(lno, min(len(c_text.splitlines()) + 1, lno + 8))
                            )
                            const_notes.append(f"- `{nm}` in `{rel_const_file}:{lno}`:\n```python\n{snippet}\n```")
        if const_notes:
            report.append("\n### 6. Imported Constants & Choices")
            report.extend(const_notes)

    auth_imports = re.findall(r"from\s+([\w.]+authentication)\s+import\s+([^\n]+)", v_content)
    if auth_imports:
        for mod_path, imported_names in auth_imports:
            rel_auth_file = mod_path.replace(".", "/") + ".py"
            full_auth_file = os.path.join(root, rel_auth_file)
            if os.path.exists(full_auth_file):
                a_text = _read_file(full_auth_file)
                report.append(f"\n### 7. Referenced Authentication Module (`{rel_auth_file}`)")
                report.append("```python\n" + _format_numbered_source(a_text, max_chars=4000) + "\n```")

    return "\n".join(report)


def harvest_for_finetune(
    target_path: str = "",
    open_colab: bool = False,
    import_gguf: bool = False,
) -> str:
    """Harvest a file, folder, or project into the cumulative fine-tuning dataset (without losing past data), sync to Google Drive (G:\\My Drive\\peter_finetune), and optionally open Google Colab or import the trained GGUF into Ollama."""
    from scripts.build_finetune_dataset import (
        harvest_and_sync_dataset,
        import_gguf_from_gdrive,
        launch_colab_training,
    )

    if import_gguf:
        res = import_gguf_from_gdrive()
        return (
            f"### Ollama GGUF Model Import Result\n"
            f"- **Status**: `{res.get('status')}`\n"
            f"- **Model**: `{res.get('model_name', 'peter-coder')}`\n"
            f"- **Source GGUF**: `{res.get('source_gguf')}`\n"
            f"- **Details**: {res.get('message') or res.get('stdout') or res.get('stderr')}"
        )

    clean_target = (target_path or "").strip()
    if clean_target.lower() in {"", "project", "this project", "all", "aryashakti", "."}:
        clean_target = None

    stats = harvest_and_sync_dataset(
        target_path=clean_target,
        sync_gdrive=True,
        include_db=(clean_target is None),
        open_colab=open_colab,
    )

    lines = [
        "### Incremental Fine-Tune Harvest & Google Drive Sync Complete",
        f"- **Target Harvested**: `{stats.get('target')}` (`{stats.get('py_files_scanned', 0)}` Python files scanned; `.env` & raw DB rows strictly skipped)",
        f"- **Previous Examples Retained**: `{stats.get('previous_examples', 0)}` *(zero past data forgotten)*",
        f"- **New Examples Added**: `{stats.get('added_examples', 0)}`",
        f"- **Existing Examples Updated**: `{stats.get('updated_examples', 0)}`",
        f"- **Total Cumulative Dataset**: **`{stats.get('total_examples', 0)}` examples (`{stats.get('size_mb', 0)} MB`)**",
        f"- **Local Copy**: `{stats.get('local_jsonl')}`",
    ]
    if stats.get("gdrive_connected") or stats.get("gdrive_jsonl"):
        lines.append(f"- **Google Drive Copy**: `{stats.get('gdrive_jsonl')}`")
        lines.append(f"- **Google Drive Colab Notebook**: `{stats.get('gdrive_notebook')}`")
        if stats.get("gdrive_note"):
            lines.append(f"- **Sync Status**: {stats.get('gdrive_note')}")
    else:
        lines.append("- **Google Drive Copy**: Not mounted (`G:\\My Drive` not detected)")

    if stats.get("colab_launched"):
        c_info = stats["colab_launched"]
        lines.append(f"- **Colab T4 Notebook Launched**: `{c_info.get('colab_url')}`")

    return "\n".join(lines)
