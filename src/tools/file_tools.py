import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd  # <-- THIS WAS MISSING!

# Load environment variables
load_dotenv()

# Read the path from .env. If it's missing, fall back to current directory.
env_path = os.getenv("SANDBOX_ROOT_PATH")

APP_ROOT = Path(__file__).resolve().parents[2]

if env_path:
    PROJECT_ROOT = Path(env_path).resolve()
else:
    PROJECT_ROOT = Path(os.getcwd()).resolve()


def set_project_root(new_path: str) -> str:
    """Dynamically switch the active project root directory for all file operations."""
    global PROJECT_ROOT
    try:
        cleaned = new_path.strip().strip('"').strip("'")
        p = Path(cleaned).resolve()
        if not p.exists():
            return f"Error: Directory does not exist: {p}"
        if not p.is_dir():
            return f"Error: Path is not a directory: {p}"
        PROJECT_ROOT = p
        return f"Successfully switched project root to: {PROJECT_ROOT}"
    except Exception as e:
        return f"Failed to switch project root: {str(e)}"


def get_project_root() -> str:
    """Return the currently active project root path."""
    return str(PROJECT_ROOT)


def _is_safe_path(requested_file: str) -> Path:
    """Security check: Ensures the file is strictly inside the active project folder (or Peter's own APP_ROOT).
    Also auto-resolves files located in subdirectories (e.g. 'file_tools.py' -> 'src/tools/file_tools.py').
    """
    raw = Path(requested_file.strip().strip('"').strip("'"))
    if raw.is_absolute():
        target_path = raw.resolve()
    else:
        target_path = (PROJECT_ROOT / requested_file).resolve()
        # Fallback: if not in PROJECT_ROOT, check Peter's own APP_ROOT (e.g. src/tools/network_tools.py)
        if not target_path.exists() and (APP_ROOT / requested_file).resolve().exists():
            target_path = (APP_ROOT / requested_file).resolve()

    # If the file doesn't exist directly at the requested path, search subdirectories
    if not target_path.exists() and not raw.is_absolute():
        clean_name = raw.name
        excluded = {".git", ".venv", "__pycache__", "chroma_db", "node_modules", ".pytest_cache"}
        # Try exact filename match in PROJECT_ROOT, then APP_ROOT
        for search_root in (PROJECT_ROOT, APP_ROOT):
            for match in search_root.rglob(clean_name):
                if match.is_file() and not any(p in excluded for p in match.parts):
                    target_path = match.resolve()
                    break
            if target_path.exists():
                break
        # If still not found, try partial match (e.g. 'pop_manage_activity' -> 'pop_manage_activity_202609260142.csv')
        if not target_path.exists():
            stem_query = clean_name.replace(".csv", "").replace(".xlsx", "")
            for match in PROJECT_ROOT.rglob(f"*{stem_query}*"):
                if match.is_file() and not any(p in excluded for p in match.parts):
                    target_path = match.resolve()
                    break

    if not (target_path.is_relative_to(PROJECT_ROOT) or target_path.is_relative_to(APP_ROOT)):
        raise PermissionError(
            f"Access denied: '{target_path}' is outside the active project root '{PROJECT_ROOT}'. "
            f"To access it, type: 'switch project to {target_path.parent}'"
        )

    return target_path

# --- THE TOOLS THE AI WILL ACTUALLY USE ---

def safe_read_file(filename: str) -> str:
    """Reads a file ONLY if it is inside the project directory."""
    try:
        safe_path = _is_safe_path(filename)
        if not safe_path.exists():
            return f"Error: File '{filename}' does not exist."
            
        # NEW: Intercept spreadsheets and route them to Pandas automatically
        if safe_path.suffix.lower() in ['.xlsx', '.csv', '.xlsm']:
            return safe_read_spreadsheet(str(safe_path))
            
        # Otherwise, read as normal plain text
        with open(safe_path, "r", encoding="utf-8") as f:
            return f.read()
            
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Failed to read file: {str(e)}"
def safe_write_file(filename: str, content: str) -> str:
    """Writes to a file ONLY if it is inside the project directory."""
    try:
        safe_path = _is_safe_path(filename)
        
        # Ensure the subfolder exists (e.g., if AI tries to write to data/csv_files/new.csv)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Success: Wrote data securely to {safe_path.relative_to(PROJECT_ROOT)}"
        
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Failed to write file: {str(e)}"
def safe_read_spreadsheet(filename: str, rows: int = 50) -> str:
    """Reads rows from an Excel or CSV file securely."""
    try:
        # Force 'rows' to be an integer
        rows = int(rows)
        target_path = _is_safe_path(filename)
        if not target_path.exists():
            return f"Error: File not found: {filename}"
            
        # Read the file based on extension
        if str(target_path).lower().endswith('.csv'):
            df = pd.read_csv(target_path)
        else:
            # Forces pandas to handle weird headers gracefully
            df = pd.read_excel(target_path)
            
        # Convert to a clean string, ignoring any weird pandas indexes
        data_string = df.head(rows).to_string(index=False)
        data_string = data_string.encode("ascii", errors="replace").decode("ascii")
        return f"SUCCESSFULLY READ FILE:\n{data_string}"
        
    except Exception as e:
        return f"Error reading file: {str(e)}"


def scan_directory(directory_path: str = ".") -> str:
    """Lists all files and folders in the given directory path. Use '.' for the root sandbox."""
    try:
        raw = Path(directory_path.strip().strip('"').strip("'"))
        target_path = raw.resolve() if raw.is_absolute() else (PROJECT_ROOT / directory_path).resolve()
        
        # Sandbox security check
        if not target_path.is_relative_to(PROJECT_ROOT):
            return f"Error: Access denied. Path '{target_path}' is outside the active project root '{PROJECT_ROOT}'. Use 'switch project to {target_path}' to switch projects first."

            
        if not target_path.exists() or not target_path.is_dir():
            return f"Error: Directory not found at {target_path}"
            
        items = os.listdir(target_path)
        if not items:
            return f"The directory '{directory_path}' is empty."
            
        return f"Contents of '{directory_path}':\n" + "\n".join(items)
        
    except Exception as e:
        return f"Error scanning directory: {str(e)}"


def search_project_files(query: str, directory_path: str = ".", max_results: int = 100, file_filter: str = "") -> str:
    """Search text files recursively inside the project sandbox.

    Args:
        query: The text to search for (case-insensitive).
        directory_path: Starting directory for the search (relative to project root or absolute).
        max_results: Maximum number of matches to return.
        file_filter: Optional. If set, only files whose relative path contains this string are searched.
                     E.g. "models.py" searches only models.py files, "pop_manage/models.py" scopes to one module.
    """
    try:
        raw = Path(directory_path.strip().strip('"').strip("'"))
        target_path = raw.resolve() if raw.is_absolute() else (PROJECT_ROOT / directory_path).resolve()
        if not target_path.is_relative_to(PROJECT_ROOT):
            return f"Error: Access denied. Path '{target_path}' is outside the active project root '{PROJECT_ROOT}'. Use 'switch project to {target_path}' to switch projects first."
        if not target_path.exists() or not target_path.is_dir():
            return f"Error: Directory not found at {target_path}"
        if not query.strip():
            return "Error: Search query cannot be empty."

        excluded_directories = {
            ".git", ".venv", "__pycache__", "chroma_db",
            "node_modules", ".pytest_cache", "logs", "data", "media", "static",
        }
        excluded_extensions = {".log", ".sqlite", ".db", ".bin", ".pickle", ".onnx", ".pyc", ".png", ".jpg", ".jpeg", ".zip", ".tar", ".gz"}

        matches: list[str] = []
        query_lower = query.lower()
        file_filter_lower = file_filter.strip().lower().replace("\\", "/") if file_filter else ""

        words = query_lower.split()
        query_regex = re.compile(r"[-_ ]*".join(re.escape(w) for w in words), re.IGNORECASE) if len(words) >= 2 else None

        for root, dirs, files in os.walk(target_path):
            dirs[:] = [d for d in dirs if d not in excluded_directories]
            for file_name in files:
                if len(matches) >= max_results:
                    break
                path = Path(root) / file_name
                if path.suffix.lower() in excluded_extensions:
                    continue

                relative_path = path.relative_to(PROJECT_ROOT)
                relative_str = str(relative_path).replace("\\", "/").lower()

                # Apply file filter if specified
                if file_filter_lower and file_filter_lower not in relative_str:
                    continue

                # Match filename itself
                file_matched = query_regex.search(file_name) if query_regex else (query_lower in file_name.lower())
                if file_matched:
                    matches.append(f"[FILE] {relative_path}")
                    if len(matches) >= max_results:
                        break

                try:
                    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                except (OSError, UnicodeDecodeError):
                    continue

                for line_number, line in enumerate(lines, start=1):
                    line_matched = query_regex.search(line) if query_regex else (query_lower in line.lower())
                    if line_matched:
                        matches.append(f"{relative_path}:{line_number}: {line.strip()}")
                        if len(matches) >= max_results:
                            break
            if len(matches) >= max_results:
                break

        if not matches:
            return f"No matches found for '{query}' under '{directory_path}'."
        return f"Found {len(matches)} matches for '{query}':\n" + "\n".join(matches)
    except Exception as e:
        return f"Error searching project files: {str(e)}"


def find_files(query: str, directory_path: str = ".") -> str:
    """Find files by name or pattern recursively in the active project directory."""
    try:
        raw = Path(directory_path.strip().strip('"').strip("'"))
        target_path = raw.resolve() if raw.is_absolute() else (PROJECT_ROOT / directory_path).resolve()
        if not target_path.is_relative_to(PROJECT_ROOT):
            return f"Error: Access denied. Path is outside active root '{PROJECT_ROOT}'."
        if not target_path.exists() or not target_path.is_dir():
            return f"Error: Directory not found at {target_path}"

        excluded = {".git", ".venv", "__pycache__", "chroma_db", "node_modules", ".pytest_cache", "logs", "media", "static"}
        q = query.strip().lower()
        matches: list[str] = []

        for root, dirs, files in os.walk(target_path):
            dirs[:] = [d for d in dirs if d not in excluded]
            for file_name in files:
                if len(matches) >= 100:
                    break
                if q in file_name.lower():
                    rel = (Path(root) / file_name).relative_to(PROJECT_ROOT)
                    matches.append(str(rel).replace("\\", "/"))
            if len(matches) >= 100:
                break

        if not matches:
            return f"No files matching '{query}' found under '{directory_path}'."
        return f"Found {len(matches)} file(s) matching '{query}':\n" + "\n".join(matches)
    except Exception as e:
        return f"Error finding files: {str(e)}"