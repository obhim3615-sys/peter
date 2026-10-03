"""Database tools for querying live PostgreSQL (UAT/Production) or local SQLite."""

import os
import re
import sqlite3
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Fallback SQLite DB path
APP_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = str(APP_ROOT / "data" / "sqlite" / "project_data.db")


def _get_connection():
    """Returns a read-only database connection based on environment configuration."""
    db_type = os.getenv("LIVE_DB_TYPE", "postgresql").strip().lower()

    if db_type == "postgresql":
        import psycopg2

        host = os.getenv("LIVE_DB_HOST", "localhost")
        port = int(os.getenv("LIVE_DB_PORT", "5432"))
        dbname = os.getenv("LIVE_DB_NAME", "arya")
        user = os.getenv("LIVE_DB_USER", "postgres")
        password = os.getenv("LIVE_DB_PASSWORD", "root")
        sslmode = os.getenv("LIVE_DB_SSLMODE", "disable")

        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            sslmode=sslmode,
            connect_timeout=10,
        )
        # Enforce read-only at database engine level
        conn.set_session(readonly=True, autocommit=True)
        return conn, "postgresql"

    # Fallback to local SQLite file
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    return conn, "sqlite"


def get_database_schema(table_name: Optional[str] = None) -> str:
    """Returns the list of tables or the detailed column structure of a specific table.

    Args:
        table_name: Optional name of the table to inspect (e.g. 'app_farm_farm', 'auth_user').
                   If omitted, returns a list of all available tables.
    """
    try:
        conn, db_type = _get_connection()
    except Exception as e:
        return f"Database connection failed: {e}"

    cursor = conn.cursor()
    try:
        if db_type == "postgresql":
            if not table_name or not table_name.strip():
                cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    ORDER BY table_name;
                """)
                tables = [r[0] for r in cursor.fetchall()]
                if not tables:
                    return "Connected to PostgreSQL database, but 0 tables found in schema 'public'."
                return f"Total Tables: {len(tables)}\nAvailable Database Tables:\n" + "\n".join(f"- {t}" for t in tables)

            clean_table = table_name.strip().strip("'\"`")
            # Resolve exact case-sensitive table name in PostgreSQL
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' AND lower(table_name) = lower(%s)
                LIMIT 1;
            """, (clean_table,))
            exact_row = cursor.fetchone()
            if exact_row:
                clean_table = exact_row[0]

            cursor.execute("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position;
            """, (clean_table,))
            columns = cursor.fetchall()

            if not columns:
                return f"Table '{clean_table}' was not found in schema 'public'."

            # Fetch primary keys
            cursor.execute("""
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema = kcu.table_schema
                WHERE tc.constraint_type = 'PRIMARY KEY'
                  AND tc.table_name = %s;
            """, (clean_table,))
            pk_cols = {r[0] for r in cursor.fetchall()}

            result = f"Schema for table \"{clean_table}\" ({len(columns)} columns):\n"
            for col_name, dtype, nullable, default_val in columns:
                pk_flag = " [PK]" if col_name in pk_cols else ""
                null_flag = "NULL" if nullable == "YES" else "NOT NULL"
                result += f"  - {col_name}{pk_flag}: {dtype} ({null_flag})\n"

            return result

        else:
            # SQLite fallback
            if not table_name or not table_name.strip():
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
                tables = [r[0] for r in cursor.fetchall()]
                return f"Available SQLite Tables ({len(tables)}):\n" + "\n".join(f"- {t}" for t in tables)

            clean_table = table_name.strip().strip("'\"`")
            cursor.execute(f"PRAGMA table_info({clean_table});")
            columns = cursor.fetchall()
            if not columns:
                return f"Table '{clean_table}' not found in SQLite database."

            result = f"Schema for table '{clean_table}':\n"
            for c in columns:
                pk_flag = " [PK]" if c[5] else ""
                result += f"  - {c[1]}{pk_flag}: {c[2]}\n"
            return result

    except Exception as e:
        return f"Error retrieving schema: {e}"
    finally:
        conn.close()


def search_database_tables(keyword: str) -> str:
    """Searches for table names in the database matching one or more keywords or phrases.

    Args:
        keyword: Search term or comma/or-separated terms (e.g. 'drone', 'task', 'drone or task', 'farm, user').
    """
    if not keyword or not keyword.strip():
        return "Please provide a keyword to search tables."

    try:
        conn, db_type = _get_connection()
    except Exception as e:
        return f"Database connection failed: {e}"

    cursor = conn.cursor()

    # Split keywords by commas, slashes, or 'or' / 'and'
    raw_tokens = re.split(r"[,;/]|\s+(?:or|and)\s+", keyword.strip())
    tokens = [t.strip().lower() for t in raw_tokens if t.strip() and len(t.strip()) > 1]
    if not tokens:
        tokens = [keyword.strip().lower()]

    try:
        results = {}
        for token in tokens:
            if db_type == "postgresql":
                cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                      AND lower(table_name) LIKE %s
                    ORDER BY table_name;
                """, (f"%{token}%",))
            else:
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND lower(name) LIKE ? 
                    ORDER BY name;
                """, (f"%{token}%",))

            results[token] = [r[0] for r in cursor.fetchall()]

        if len(tokens) == 1:
            tok = tokens[0]
            matches = results[tok]
            if not matches:
                return f"No tables found matching '{tok}'."
            return f"Found {len(matches)} table(s) matching '{tok}':\n" + "\n".join(f"- {m}" for m in matches)

        # Multi-keyword output
        output_parts = []
        for tok, matches in results.items():
            if matches:
                output_parts.append(f"Tables matching '{tok}' ({len(matches)}):\n" + "\n".join(f"  - {m}" for m in matches))
            else:
                output_parts.append(f"Tables matching '{tok}': None found.")
        return "\n\n".join(output_parts)
    except Exception as e:
        return f"Error searching tables: {e}"
    finally:
        conn.close()


def safe_query_db(query: str) -> str:
    """Executes a strictly read-only SQL query and returns formatted tabular results.

    Args:
        query: SQL SELECT query (e.g. 'SELECT id, name, created_at FROM app_farm_farm LIMIT 10;').
    """
    if not query or not query.strip():
        return "Error: SQL query cannot be empty."

    clean_query = query.strip()

    # 1. Security Check: Reject mutating statements
    dangerous_patterns = [
        r"\bDROP\b",
        r"\bDELETE\b",
        r"\bUPDATE\b",
        r"\bINSERT\b",
        r"\bALTER\b",
        r"\bTRUNCATE\b",
        r"\bGRANT\b",
        r"\bREVOKE\b",
        r"\bCREATE\b",
        r"\bEXEC\b",
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, clean_query, re.IGNORECASE):
            return f"SECURITY REJECTION: Modifying the database via '{pattern}' is strictly forbidden. Only read-only queries are allowed."

    # Must be a SELECT, WITH, or EXPLAIN query
    if not re.match(r"^\s*(SELECT|WITH|EXPLAIN)\b", clean_query, re.IGNORECASE):
        return "SECURITY REJECTION: Query must begin with SELECT, WITH, or EXPLAIN."

    # 2. Automatically cap query limit to prevent memory/token exhaustion
    if not re.search(r"\bLIMIT\s+\d+\b", clean_query, re.IGNORECASE):
        clean_query = clean_query.rstrip("; \t\n") + " LIMIT 50;"

    try:
        conn, db_type = _get_connection()
    except Exception as e:
        return f"Database connection failed: {e}"

    cursor = conn.cursor()
    try:
        # Auto-quote mixed-case PostgreSQL table names (e.g. recordactivity_projecttask -> "recordActivity_projecttask")
        if db_type == "postgresql":
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public';
            """)
            mixed_tables = [r[0] for r in cursor.fetchall() if r[0] != r[0].lower()]
            for tbl in sorted(mixed_tables, key=len, reverse=True):
                clean_query = re.sub(
                    rf'(?<!")\b{re.escape(tbl)}\b(?!")',
                    f'"{tbl}"',
                    clean_query,
                    flags=re.IGNORECASE,
                )

        cursor.execute(clean_query)
        if not cursor.description:
            return "Query executed successfully. 0 rows returned."

        column_names = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        if not rows:
            return f"Query returned 0 rows.\nColumns: {', '.join(column_names)}"

        # Format output cleanly
        result = f"Columns: {', '.join(column_names)}\n"
        result += "-" * 60 + "\n"
        for row in rows[:50]:
            # Convert tuples/values safely to string
            result += f"{row}\n"

        if len(rows) >= 50:
            result += f"\n(Result capped at 50 rows. Total retrieved: {len(rows)})"

        return result

    except Exception as e:
        return f"SQL Execution Error: {e}"
    finally:
        conn.close()