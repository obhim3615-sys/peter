import os
import re
import shlex
import subprocess
from pathlib import Path
from src.tools.file_tools import get_project_root

DANGEROUS_PATTERNS = [
    r"\brm\b", r"\brmdir\b", r"\bdel\b", r"\berase\b", r"\bformat\b",
    r"\bdrop\b", r"\btruncate\b", r"\bdelete\b", r"\bremove-item\b",
    r"\bshutdown\b", r"\breboot\b", r"\btaskkill\b",
    r"\bgit\s+reset\b", r"\bgit\s+clean\b", r"\bgit\s+checkout\s+\.",
    r"\bgit\s+restore\s+\.", r"\bgit\s+push\s+(?:-f|--force)\b",
    r">\s*[^&]",  # Redirect that truncates/overwrites a file (excluding 2>&1)
]

SAFE_READ_ONLY_PREFIXES = [
    "git grep", "git status", "git log", "git show", "git diff", "git branch", "git tag", "git ls-files",
    "findstr", "grep", "rg", "dir", "ls", "type", "cat", "head", "tail", "wc", "where", "which",
    "Get-ChildItem", "Select-String", "Get-Content",
    "python manage.py showmigrations", "python manage.py check", "python manage.py inspectdb",
]


def is_safe_read_only_command(command: str) -> bool:
    """Check if a shell command is safe and strictly read-only."""
    cmd_clean = command.strip().lower()

    # Reject dangerous keywords or overwrite redirections
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, cmd_clean):
            return False

    # Check for safe read-only prefixes
    for prefix in SAFE_READ_ONLY_PREFIXES:
        if cmd_clean.startswith(prefix.lower()):
            return True

    # Check for read-only python one-liners
    if cmd_clean.startswith("python -c") or cmd_clean.startswith("python.exe -c"):
        mutating_terms = ["open(", "write", "remove", "unlink", "rmdir", "delete", "drop", "shutil", "truncate"]
        if not any(term in cmd_clean for term in mutating_terms):
            return True

    return False


def run_terminal_command(command: str, timeout: int = 30) -> str:
    """Execute a shell command inside the active project root directory and return its output.

    Args:
        command: The terminal command to execute (e.g. 'git grep -n "drone"', 'findstr /s /i "booking" *.py').
        timeout: Maximum execution time in seconds (default 30s).
    """
    if not command or not command.strip():
        return "Error: Command cannot be empty."

    project_root = get_project_root()
    cmd = command.strip()

    # Choose shell: use PowerShell on Windows if available, otherwise cmd/system shell
    shell_executable = "powershell.exe" if os.name == "nt" else "/bin/bash"

    try:
        if os.name == "nt":
            # Execute command through PowerShell with list args and shell=False to avoid cmd.exe quote corruption
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd],
                shell=False,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )
        else:
            result = subprocess.run(
                cmd,
                shell=True,
                executable="/bin/bash",
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )

        stdout = result.stdout.strip() if result.stdout else ""
        stderr = result.stderr.strip() if result.stderr else ""

        # Limit output length to prevent overloading context window (max 5000 chars)
        max_output_len = 5000
        truncated = False

        output_parts = []
        if stdout:
            if len(stdout) > max_output_len:
                stdout = stdout[:max_output_len] + f"\n... [Output truncated; showing first {max_output_len} characters]"
                truncated = True
            output_parts.append(stdout)

        if stderr:
            if len(stderr) > 1000:
                stderr = stderr[:1000] + "\n... [Stderr truncated]"
            output_parts.append(f"⚠️ Stderr:\n{stderr}")

        combined_output = "\n\n".join(output_parts) if output_parts else "[Command completed with no output]"

        return f"💻 Terminal Command: `{cmd}`\n📂 Working Directory: `{project_root}`\n⚙️ Exit Code: {result.returncode}\n\n{combined_output}"

    except subprocess.TimeoutExpired:
        return f"⚠️ Command timed out after {timeout} seconds: `{cmd}`"
    except Exception as e:
        return f"❌ Failed to execute terminal command: {str(e)}"
