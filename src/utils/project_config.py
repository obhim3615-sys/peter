"""
Centralised project configuration and detection.

Add new projects to the KNOWN_PROJECTS list below.  Each entry maps a set of
trigger keywords (all lower-case) to a canonical project name.  The first
entry whose keyword appears in the text wins; if no keyword matches the
fallback "general" is returned.

Environment variable: PROJECT_KEYWORDS (optional)
  Comma-separated list of  "<keyword>:<project>"  pairs that are loaded at
  import time and merged on top of the defaults, e.g.
      PROJECT_KEYWORDS="skynet:skynet,hal9000:hal"
"""

from __future__ import annotations

import os
from typing import Sequence

# ---------------------------------------------------------------------------
# Configuration – edit this list to add / remove projects
# ---------------------------------------------------------------------------
KNOWN_PROJECTS: list[dict] = [
    {"keywords": ["aryaq"],       "name": "aryaq"},
    {"keywords": ["aryashakti"],  "name": "aryashakti"},
]

# Allow extending via environment variable without touching source code
_env_keywords = os.getenv("PROJECT_KEYWORDS", "")
if _env_keywords:
    for _pair in _env_keywords.split(","):
        _pair = _pair.strip()
        if ":" in _pair:
            _kw, _proj = _pair.split(":", 1)
            KNOWN_PROJECTS.append({"keywords": [_kw.strip().lower()], "name": _proj.strip()})

FALLBACK_PROJECT = "general"


def infer_project(*texts: str) -> str:
    """
    Return the canonical project name that best matches the supplied text(s).

    Parameters
    ----------
    *texts:
        One or more strings to search (e.g. user_request, assistant_response).
        They are joined and lower-cased before matching.

    Returns
    -------
    str
        The matched project name, or ``FALLBACK_PROJECT`` ("general") when
        nothing matches.

    Examples
    --------
    >>> infer_project("Check the AryaQ dashboard")
    'aryaq'
    >>> infer_project("unknown topic")
    'general'
    """
    combined = " ".join(str(t) for t in texts).lower()
    for entry in KNOWN_PROJECTS:
        for kw in entry["keywords"]:
            if kw in combined:
                return entry["name"]
    # Fallback to active sandbox folder name if configured
    env_sandbox = os.getenv("SANDBOX_ROOT_PATH", "").strip().rstrip("/\\")
    if env_sandbox:
        sandbox_name = os.path.basename(env_sandbox).lower()
        if any(entry["name"] == sandbox_name for entry in KNOWN_PROJECTS):
            return sandbox_name
    return FALLBACK_PROJECT


def all_project_names() -> list[str]:
    """Return a de-duplicated list of all known project names (excluding fallback)."""
    seen: set[str] = set()
    result: list[str] = []
    for entry in KNOWN_PROJECTS:
        name = entry["name"]
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result
