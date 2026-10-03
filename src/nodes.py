import os
import re
from pathlib import Path
from dotenv import load_dotenv
from langgraph.types import interrupt
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import Literal

from src.state import AgentState
from src.utils.logger import setup_logger
from src.utils.project_config import infer_project

# Import tools
from src.tools.file_tools import (
    PROJECT_ROOT, safe_read_file, safe_write_file, safe_read_spreadsheet,
    scan_directory, search_project_files, set_project_root, get_project_root, find_files
)
from src.tools.database_tools import safe_query_db, get_database_schema, search_database_tables
from src.tools.network_tools import check_website_health, fetch_dashboard_information

from src.tools.vision_tools import analyze_image
from src.tools.rag_tools import search_internal_knowledge, save_to_knowledge_base
from src.tools.code_intelligence import (
    extract_function, analyze_module, find_related_code, index_project_code,
    trace_request_flow, resolve_django_table, resolve_django_column, resolve_django_route,
    investigate_keyword, harvest_for_finetune, is_exact_code_symbol,
)
from src.tools.terminal_tools import run_terminal_command, is_safe_read_only_command

load_dotenv()
logger = setup_logger("orchestrator-nodes")

_env_provider = os.getenv("MODEL_PROVIDER", "").strip().lower()
USE_ONLINE_MODEL = _env_provider != "ollama"
USE_GEMINI_ONLINE_MODEL = _env_provider == "gemini"

offline_llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "peter-coder"),
    temperature=0,
    num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "8192")),
)

if USE_GEMINI_ONLINE_MODEL:
    PRIMARY_MODEL_PROVIDER = "gemini"
    FALLBACK_MODEL_PROVIDER = "ollama-fallback-online"
    logger.info("Initializing Gemini Online Mode (Gemini -> Ollama fallback)")
    gemini_llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )
    base_llm = gemini_llm.with_fallbacks([offline_llm])
elif USE_ONLINE_MODEL:
    PRIMARY_MODEL_PROVIDER = "openrouter"
    FALLBACK_MODEL_PROVIDER = "ollama"
    logger.info("Initializing Online Mode (OpenRouter -> Ollama fallback)")
    openrouter_model = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
    primary_llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model=openrouter_model,
        temperature=0
    )
    base_llm = primary_llm.with_fallbacks([offline_llm])
else:
    PRIMARY_MODEL_PROVIDER = "ollama"
    FALLBACK_MODEL_PROVIDER = None
    logger.info("Initializing Offline Mode (Local Ollama, num_ctx=16384)")
    base_llm = offline_llm

# Shared comprehensive toolsets — neither worker is ever stuck without code or docs
SHARED_READ_TOOLS = [
    # Knowledge Base & Project Documentation
    search_internal_knowledge, save_to_knowledge_base, check_website_health, fetch_dashboard_information,
    # Codebase, Filesystem & Code Intelligence
    safe_read_file, safe_read_spreadsheet, safe_query_db, get_database_schema, search_database_tables, scan_directory,
    search_project_files, find_files, analyze_image, extract_function,
    analyze_module, find_related_code, index_project_code, trace_request_flow,
    resolve_django_column, resolve_django_route, investigate_keyword, harvest_for_finetune,
    run_terminal_command,
]

# Researcher has full access to docs AND codebase inspection tools
research_tools = list(SHARED_READ_TOOLS)

# Coder has full access to docs AND codebase inspection tools, plus safe_write_file
coder_tools = list(SHARED_READ_TOOLS) + [safe_write_file]

TYPO_MAP = {
    # where / search / find
    "wheir": "where", "wher": "where", "whr": "where", "whre": "where", "whare": "where", "wer": "where",
    "serch": "search", "seach": "search", "srch": "search",
    "fnd": "find", "fidn": "find",
    # code concepts
    "classs": "class", "clasess": "classes", "clss": "class", "clas": "class",
    "modle": "model", "modles": "models", "modell": "model",
    "functon": "function", "fucntion": "function", "funciton": "function", "funtion": "function",
    "retrive": "retrieve", "retreive": "retrieve",
    "summrise": "summarize", "summerize": "summarize", "summery": "summary",
    "endpont": "endpoint", "endpiont": "endpoint", "endpint": "endpoint",
    "defination": "definition", "defin": "define",
    "metod": "method", "methd": "method",
    "qurey": "query", "querry": "query",
    "respons": "response", "respose": "response",
    "serializr": "serializer", "serailizer": "serializer",
    "templete": "template", "templte": "template",
    "spary": "spray", "spraying": "spray",
    "dashbord": "dashboard", "dashbaord": "dashboard", "dashbored": "dashboard",
    "rleated": "related", "releted": "related", "realted": "related", "relted": "related",
}


def normalize_text_typos(text: str) -> str:
    """Normalize common conversational typos across workers and routing."""
    result = text
    for typo, fix in TYPO_MAP.items():
        result = re.sub(r"\b" + typo + r"\b", fix, result, flags=re.IGNORECASE)
    return result


def _requested_directory_path(user_request: str) -> str | None:
    """Resolve a folder mentioned in a request to a project-relative path."""
    normalized_request = user_request.lower().replace("\\", "/")

    # If the user is asking about the whole project / repository / root, return "."
    whole_project_phrases = (
        "this project", "the project", "my project", "whole project", "entire project",
        "this codebase", "the codebase", "whole codebase", "this repo", "the repo",
        "root folder", "root directory", "root sandbox", "project root",
    )
    if any(phrase in normalized_request for phrase in whole_project_phrases):
        return "."

    # Check if request has an explicit path segment matching project root
    active_root_str = str(PROJECT_ROOT).lower().replace("\\", "/")
    if active_root_str in normalized_request:
        return "."

    # Directories that should never be considered as scan targets
    excluded = {
        ".git", ".venv", "__pycache__", "chroma_db",
        "node_modules", ".pytest_cache", ".mypy_cache",
        ".tox", "dist", "build", ".eggs",
    }

    # Generic English terms that happen to be folder names but shouldn't match plain text
    generic_stopwords = {
        "project", "folder", "directory", "dir", "workspace", "codebase",
        "repo", "repository", "test", "tests", "file", "files", "app", "apps",
        "main", "core", "data", "manage", "admin", "build",
    }

    candidates = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_dir():
            continue
        parts = path.relative_to(PROJECT_ROOT).parts
        if any(part in excluded for part in parts):
            continue
        relative_path = path.relative_to(PROJECT_ROOT).as_posix()
        candidates.append((relative_path.lower(), relative_path))

    # Pass 1: Exact full relative path match (e.g., "src/tools", "data/csv_files")
    for relative_path, display_path in sorted(candidates, key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(relative_path)}\b", normalized_request):
            return display_path

    # Pass 2: Exact folder name match (e.g., "csv_files" / "csv files", "libraries")
    for relative_path, display_path in sorted(candidates, key=lambda item: len(item[0]), reverse=True):
        folder_name = relative_path.rsplit("/", 1)[-1]
        folder_lower = folder_name.lower()

        # If it's a generic word (like 'project' or 'test'), only match if explicitly requested as a folder
        if folder_lower in generic_stopwords:
            explicit_pattern = rf"\b(?:folder|dir|directory|in|inside)\s+{re.escape(folder_lower)}\b|\b{re.escape(folder_lower)}\s+(?:folder|dir|directory)\b"
            if re.search(explicit_pattern, normalized_request):
                return display_path
            continue

        if re.search(rf"\b{re.escape(folder_lower)}\b", normalized_request):
            return display_path
        folder_spaced = folder_lower.replace("_", " ")
        if re.search(rf"\b{re.escape(folder_spaced)}\b", normalized_request):
            return display_path

    # Pass 3: Distinct token match for composite folder names (e.g., "csv" -> "data/csv_files")
    for relative_path, display_path in sorted(candidates, key=lambda item: len(item[0]), reverse=True):
        folder_name = relative_path.rsplit("/", 1)[-1]
        tokens = [t for t in re.split(r"[_\s/-]+", folder_name) if len(t) >= 3 and t.lower() not in generic_stopwords]
        for token in tokens:
            if re.search(rf"\b{re.escape(token.lower())}\b", normalized_request):
                return display_path

    return None



READ_ONLY_TOOLS = {
    "search_internal_knowledge",
    "check_website_health",
    "fetch_dashboard_information",
    "safe_read_file",
    "safe_read_spreadsheet",
    "safe_query_db",
    "get_database_schema",
    "search_database_tables",
    "scan_directory",
    "search_project_files",
    "analyze_image",
    "switch_project",
    "get_project_root",
    "find_files",
    "extract_function",
    "analyze_module",
    "find_related_code",
    "index_project_code",
    "trace_request_flow",
    "resolve_django_column",
    "resolve_django_route",
    "investigate_keyword",
    "harvest_for_finetune",
}

researcher_llm = base_llm.bind_tools(research_tools)
coder_llm = base_llm.bind_tools(coder_tools)

# -----------------------------------------------------------------
# Fix #2 – Tool Registry
# Map every tool name → its callable once at startup.
# To add a new tool: import it above, then add ONE line here.
# execute_action calls TOOL_REGISTRY[tool_name](**args) — no elif chain needed.
# -----------------------------------------------------------------
TOOL_REGISTRY: dict = {
    "safe_read_file":               safe_read_file,
    "safe_write_file":              safe_write_file,
    "safe_read_spreadsheet":        safe_read_spreadsheet,
    "safe_query_db":                safe_query_db,
    "get_database_schema":          get_database_schema,
    "search_database_tables":       search_database_tables,
    "scan_directory":               scan_directory,
    "search_project_files":         search_project_files,
    "find_files":                   find_files,
    "analyze_image":                analyze_image,
    "check_website_health":         check_website_health,
    "fetch_dashboard_information":  fetch_dashboard_information,
    "search_internal_knowledge":    search_internal_knowledge,
    "save_to_knowledge_base":       save_to_knowledge_base,
    "switch_project":               set_project_root,
    "get_project_root":             get_project_root,
    "extract_function":             extract_function,
    "analyze_module":               analyze_module,
    "find_related_code":            find_related_code,
    "index_project_code":           index_project_code,
    "trace_request_flow":           trace_request_flow,
    "resolve_django_column":        resolve_django_column,
    "resolve_django_route":         resolve_django_route,
    "investigate_keyword":          investigate_keyword,
    "harvest_for_finetune":         harvest_for_finetune,
    "run_terminal_command":         run_terminal_command,
}

# Rich terminal log labels for each tool (no elif needed in execute_action)
_TOOL_LOG: dict = {
    "safe_read_file":               lambda a: f"📄 READING FILE: {a.get('filename')}",
    "safe_write_file":              lambda a: f"💾 UPDATING FILE: {a.get('filename')}",
    "safe_read_spreadsheet":        lambda a: f"📊 READING SPREADSHEET: {a.get('filename')} (First {a.get('rows', 5)} rows)",
    "scan_directory":               lambda a: f"📂 EXPLORING FOLDER: {a.get('directory_path', 'Root Sandbox')}",
    "search_project_files":         lambda a: f"🔎 SEARCHING PROJECT FILES FOR: {a.get('query')}",
    "find_files":                   lambda a: f"🔍 LOCATING FILES: {a.get('query')}",
    "switch_project":               lambda a: f"🔄 SWITCHING PROJECT ROOT TO: {a.get('new_path')}",
    "get_project_root":             lambda a: "[INFO] CHECKING ACTIVE PROJECT ROOT",
    "safe_query_db":                lambda a: f"🗄️ QUERYING DATABASE: {a.get('query')}",
    "get_database_schema":          lambda a: f"📋 INSPECTING DATABASE SCHEMA: {a.get('table_name') or 'ALL TABLES'}",
    "search_database_tables":       lambda a: f"🔍 SEARCHING DATABASE TABLES FOR: {a.get('keyword')}",
    "search_internal_knowledge":    lambda a: f"🧠 SEARCHING RAG DOCS FOR: {a.get('query')}",
    "save_to_knowledge_base":       lambda a: f"📝 SAVING NEW MEMORY: {a.get('topic')}",
    "fetch_dashboard_information":  lambda _: "🌐 FETCHING LIVE DASHBOARD INFORMATION",
    "check_website_health":         lambda a: f"🌐 CHECKING WEBSITE HEALTH: {a.get('url')}",
    "analyze_image":                lambda a: f"🖼️ ANALYZING IMAGE: {a.get('image_path')}",
    "extract_function":             lambda a: f"🔬 EXTRACTING FUNCTION/CLASS: {a.get('name')}",
    "analyze_module":               lambda a: f"📦 ANALYZING MODULE: {a.get('file_path')}",
    "find_related_code":            lambda a: f"🔍 TRACING RELATED CODE FOR: {a.get('name')}",
    "index_project_code":           lambda a: f"🧠 INDEXING CODEBASE STRUCTURE: {a.get('project', 'active')}",
    "trace_request_flow":           lambda a: f"🌐 TRACING REQUEST FLOW FOR FEATURE: {a.get('feature', 'farm_category')}",
    "resolve_django_column":        lambda a: f"🏛️ RESOLVING DJANGO COLUMN: {a.get('column_name')}",
    "resolve_django_route":         lambda a: f"🛣️ RESOLVING DJANGO ROUTE: {a.get('route_query')}",
    "investigate_keyword":          lambda a: f"🕵️‍♂️ 5-STEP KEYWORD INVESTIGATION: {a.get('keyword')}",
    "harvest_for_finetune":         lambda a: f"🌾 HARVESTING & SYNCING FINETUNE DATASET: {a.get('target_path') or 'active project'}",
    "run_terminal_command":         lambda a: f"💻 RUNNING TERMINAL COMMAND: {a.get('command')}",
}


def _log_model_provider(operation: str) -> None:
    fallback = f", fallback={FALLBACK_MODEL_PROVIDER}" if FALLBACK_MODEL_PROVIDER else ""
    logger.info(f"Model provider for {operation}: primary={PRIMARY_MODEL_PROVIDER}{fallback}")


_target_project = infer_project  # backward-compat alias (removed duplicated body above)


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return str(content)


# -------------------------------------------------------------
# 1. SUPERVISOR NODE
# -------------------------------------------------------------
class SupervisorDecision(BaseModel):
    next_step: Literal["researcher", "coder", "FINISH"] = Field(
        description="Choose 'researcher' for docs/web, 'coder' for files/database/code, or 'FINISH' if task is done."
    )
    task: str = Field(description="Instructions for the chosen worker, or the final answer if FINISH.")

supervisor_chain = base_llm.with_structured_output(SupervisorDecision)


def _detect_intent(user_request: str) -> tuple[str, str] | None:
    """
    Route unambiguous requests to a worker without invoking the LLM supervisor.

    Strategy (Fix #3 – replaces flat keyword matching):
    ────────────────────────────────────────────────────
    Each candidate worker accumulates a *score* — one point per matching
    keyword — across all its signal groups.  A worker is chosen only when
    its score exceeds the rival by at least CONFIDENCE_GAP points.  When
    the signals are tied or too close, we fall through to the LLM supervisor
    so it can reason about the ambiguous phrasing.

    Adding / tuning keywords:
      • Add keywords to the relevant tuple below.
      • Raise CONFIDENCE_GAP for stricter routing (fewer LLM bypasses).
      • Lower it for more aggressive fast-path routing.
    """
    CONFIDENCE_GAP = 1          # minimum lead before we commit to a worker

    request = normalize_text_typos(user_request).lower()

    # ── CODER signals ──────────────────────────────────────────────────────
    CODER_SIGNALS: list[tuple[str, ...]] = [
        # filesystem & data files
        ("file", "folder", "directory", "path", "read", "write", "open", "scan", "list files"),
        # spreadsheets
        ("spreadsheet", "csv", "xlsx", "excel", "sheet"),
        # binary / static assets
        ("pdf", "txt", "image", "photo", "picture", "screenshot"),
        # databases
        ("database", "db", "sql", "sqlite", "table", "query", "record", "row"),
        # code navigation & intelligence
        ("where", "find every", "all occurrences", "all occurrence", "defined",
         "class", "function", "method", "variable", "import", "endpoint", "endpoints",
         "who calls", "callers of", "related code", "analyze module", "extract function",
         "difference between", "compare", "trace", "request reaches", "passes through",
         "url configuration", "route", "view", "serializer", "symbols", "client request"),
        # terminal & shell execution
        ("terminal", "powershell", "shell", "bash", "run command", "execute command",
         "terminal command", "git grep", "grep", "findstr", "git log", "git status", "git diff"),
        # error / traceback / debugging
        ("traceback", "exception", "error", "stack trace", "went wrong", "debug", "bug",
         "crash", "crashed", "failing", "failed", "keyerror", "typeerror", "valueerror",
         "attributeerror", "importerror", "indexerror", "operationalerror", "integrityerror",
         "doesnotexist", "syntaxerror", "nameerror"),
    ]

    # ── RESEARCHER signals ──────────────────────────────────────────────────
    RESEARCHER_SIGNALS: list[tuple[str, ...]] = [
        # live service / monitoring
        ("dashboard", "live status", "website health", "uptime", "ping", "monitor"),
        # project-specific name (Aryaq treated as a live resource by default)
        ("aryaq",),
        # knowledge-base / documentation / architecture
        ("documentation", "knowledge base", "project information", "project overview",
         "spec", "requirement", "readme", "wiki", "architecture", "end-to-end", "design"),
        # general research intent
        ("search", "look up", "find out", "explain", "describe", "summarise",
         "summarize", "summrise", "summerize", "summary", "summery", "overview",
         "what is", "what are", "who is", "how does", "how are", "how is",
         "details of", "tell me about", "tell me details", "information about",
         "show details", "show me", "give me details", "give details", "list details",
         "which api", "what api", "api used", "api for", "endpoint for", "how to add", "how to retrieve"),
    ]



    def _score(signals: list[tuple[str, ...]]) -> int:
        total = 0
        for group in signals:
            total += sum(1 for kw in group if kw in request)
        return total

    coder_score      = _score(CODER_SIGNALS)
    researcher_score = _score(RESEARCHER_SIGNALS)

    # Architectural overview and API lookup questions belong to Researcher RAG search
    if any(term in request for term in ("architecture", "explain the", "overview of", "how does", "end-to-end")):
        researcher_score += 4

    if any(term in request for term in ("which api", "what api", "api to", "api used", "api for", "apis for", "api that", "api is used", "endpoint for", "endpoint to")):
        researcher_score += 4

    # Code intelligence signals: if user asks to inspect/explain/find/show a function, class, method, or module, or trace a request
    if any(kind in request for kind in ("function", "functions", "class", "classes", "method", "methods", "def ", "module", "endpoint", "url configuration", "request", "route")) and any(act in request for act in ("what does", "how does", "what is", "explain", "inspect", "extract", "show", "list", "find", "related", "tell me", "who calls", "callers of", "look into", "read function", "analyze module", "module structure", "where is defined", "difference between", "compare", "trace", "passes through", "reaches")):
        coder_score += 6

    # End-to-end request tracing / URL to view flow
    if any(term in request for term in ("trace it from", "request reaches", "passes through this", "url configuration to", "route, view", "serializer, service", "trace the request", "trace it")):
        coder_score += 8

    # CamelCase comparison or investigation detection: e.g. "AddSoilTestPlotAPIView", "AryaMitra"
    camels = [c for c in re.findall(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b", user_request) if sum(1 for ch in c if ch.isupper()) >= 2 and c.lower() not in {"apiview", "view", "model", "serializer"}]
    if len(camels) >= 2:
        coder_score += 4
    elif len(camels) == 1 and any(term in request for term in ("tell me about", "what is", "trace", "where is", "explain", "investigate", "how is", "check", "find")):
        coder_score += 5

    # Terminal & CLI command execution belongs to Coder
    if any(term in request for term in ("terminal", "run command", "execute command", "powershell", "git grep", "terminal command", "run in terminal")):
        coder_score += 4

    # Traceback / Exception / Error debugging belongs strongly to Coder
    if (
        "traceback (most recent call last)" in request
        or re.search(r'file\s+"[^"]+\.py",\s+line\s+\d+', request)
        or any(term in request for term in ("traceback", "stack trace", "what went wrong", "why did this fail", "debug this", "fix this error", "error in terminal"))
    ):
        coder_score += 10

    # Explicit source/data file path or API URL route (/api/v1/...) mentioned in request belongs strongly to Coder
    if re.search(r"\b[A-Za-z0-9_./\\-]+\.(?:py|kt|kts|java|cpp|c|h|hpp|cs|ts|tsx|js|jsx|go|rs|dart|swift|xlsx|csv|xls|json|yaml|yml|sql|sh)\b", user_request, re.IGNORECASE) or re.search(r"/(?:api|v\d+)/[A-Za-z0-9_/-]+", user_request, re.IGNORECASE):
        coder_score += 8

    # Where-is / location / code-search queries belong to Coder
    if any(term in request for term in ("where is", "where are", "where", "which file", "which route", "which endpoint")):
        coder_score += 4

    logger.debug(
        f"Intent scores — coder: {coder_score}, researcher: {researcher_score} "
        f"(gap needed: {CONFIDENCE_GAP})"
    )


    if coder_score >= researcher_score + CONFIDENCE_GAP:
        logger.info(f"Fast-path intent → CODER (score {coder_score} vs researcher {researcher_score})")
        return "coder", user_request

    if researcher_score >= coder_score + CONFIDENCE_GAP:
        logger.info(f"Fast-path intent → RESEARCHER (score {researcher_score} vs coder {coder_score})")
        return "researcher", user_request

    # Scores are tied or too close — let the LLM supervisor decide
    logger.info(
        f"Intent ambiguous (coder={coder_score}, researcher={researcher_score}); "
        "delegating to LLM supervisor."
    )
    return None



def _validate_tool_args(tool_name: str, args: dict) -> dict:
    """Validate and normalize arguments before a tool is executed."""
    normalized = dict(args or {})

    required_text = {
        "safe_read_file": ("filename",),
        "safe_write_file": ("filename",),
        "safe_read_spreadsheet": ("filename",),
        "search_internal_knowledge": ("query",),
        "search_project_files": ("query",),
        "safe_query_db": ("query",),
        "search_database_tables": ("keyword",),
        "analyze_image": ("image_path",),
        "extract_function": ("name",),
        "analyze_module": ("file_path",),
        "find_related_code": ("name",),
        "resolve_django_column": ("column_name",),
        "resolve_django_route": ("route_query",),
        "investigate_keyword": ("keyword",),
        "run_terminal_command": ("command",),
    }
    for field in required_text.get(tool_name, ()):
        value = normalized.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"'{field}' must be a non-empty string for {tool_name}.")
        normalized[field] = value.strip()

    if tool_name == "safe_write_file":
        content = normalized.get("content")
        if content is None:
            normalized["content"] = ""
        elif not isinstance(content, str):
            raise ValueError("'content' must be text for safe_write_file.")

    if tool_name == "scan_directory":
        directory_path = normalized.get("directory_path", ".")
        if not isinstance(directory_path, str) or not directory_path.strip():
            raise ValueError("'directory_path' must be a non-empty string for scan_directory.")
        normalized["directory_path"] = directory_path.strip()

    for field in ("rows", "max_results"):
        if field in normalized:
            try:
                normalized[field] = max(1, min(int(normalized[field]), 1000))
            except (TypeError, ValueError) as error:
                raise ValueError(f"'{field}' must be an integer between 1 and 1000.") from error

    return normalized



def _extract_search_terms(user_request: str) -> str:
    """Reduce a natural-language code-search request to clean literal terms."""
    # Step 0: Normalize common typos
    corrected = normalize_text_typos(user_request)
    # Strip common module prefixes like 'views.' or 'models.' when extracting target identifier
    corrected = re.sub(r'\b(?:views|models|serializers|services)\.([A-Za-z0-9_]+)\b', r'\1', corrected)

    # Priority 0: Look for explicit function/method/class/view name following trigger words
    func_match = re.search(r"\b(?:function|method|def|class|view|code\s+of|implementation\s+of|definition\s+of)\s+(?:(?:called|named)\s+)?([A-Za-z0-9_]+)\b", corrected, re.IGNORECASE)
    _func_Match_stop = {
        "name", "names", "named", "call", "calls", "called", "calling",
        "in", "from", "the", "a", "an", "this", "that",
        "and", "or", "for", "to", "with", "by", "on", "at", "of", "is", "are",
        "which", "who", "what", "where", "when", "how", "why", "used", "uses", "using",
        "related", "present", "defined", "available", "inside", "under",
        "behavior", "behaviour", "code", "logic", "flow", "classes", "methods", "functions", "views", "models",
    }
    if func_match and func_match.group(1).lower() not in _func_Match_stop:
        return func_match.group(1)

    # Priority 1: Look for explicit API paths like /api/v1/record/create_assistant/ -> extract leaf endpoint segment ('create_assistant')
    api_match = re.search(r"/(?:api|v\d+)/[A-Za-z0-9_/-]+", corrected)
    if api_match:
        segs = [s for s in api_match.group(0).strip("/").split("/") if s and not re.match(r"^(?:api|v\d+|\d+)$", s, re.IGNORECASE)]
        if segs:
            return segs[-1]
        return api_match.group(0).rstrip("/")

    # Priority 2: Look for CamelCase class/type/function identifiers (e.g. OTPLogin, SignUp, AddSoilTestPlotAPIView)
    camel_cases = re.findall(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b", corrected)
    if camel_cases:
        clean_camel = [c for c in camel_cases if c.lower() not in {"apiview", "view", "model", "serializer", "models"}]
        if clean_camel:
            return clean_camel[0]

    # Priority 2.5: Check if there is an explicit identifier with an underscore (table/model/function name)
    identifiers = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*(?:_[a-zA-Z0-9]+)+\b", corrected)
    if identifiers:
        clean_ids = [i for i in identifiers if i.lower() not in {"api_key", "secret_key"}]
        if clean_ids:
            wants_tbl_or_model = any(
                w in corrected.lower() for w in ("table", "tables", "schema", "column", "columns", "database", "db", "model", "models")
            )
            for cid in clean_ids:
                if not wants_tbl_or_model and is_exact_code_symbol(cid):
                    return cid
                if wants_tbl_or_model:
                    tbl_info = resolve_django_table(cid)
                    if tbl_info:
                        return tbl_info["model_name"]
            return clean_ids[0]

    stop_words = {
        "a", "all", "an", "and", "are", "at", "can", "find", "for", "from", "get",
        "how", "in", "is", "me", "of", "on", "please", "show", "tell", "the", "to",
        "use", "used", "where", "which", "with", "every", "occurrence", "occurrences",
        "present", "located", "location", "place", "does", "this", "that", "api", "apis",
        "uses", "path", "its", "endpoint", "url", "route", "function", "functions", "class", "method", "defined", "definition",
        "u", "you", "your", "ur", "add", "data", "retrive", "retrieve", "table", "tables", "row", "rows", "column", "columns",
        "give", "list", "what", "who", "when", "why", "need", "want", "help", "about", "related",
        "look", "into", "check", "inspect", "search", "see", "view", "code", "explain", "open", "read",
        "model", "models", "classes",
        "case", "cases", "schema", "schemas", "passing", "pass", "passed", "other", "others", "also",
        "detail", "details", "info", "information", "flow", "flows", "work", "works", "working",
        # Question/relationship words that are NOT code identifiers
        "mapped", "mapping", "purpose", "belong", "belongs", "belonging", "linked", "link", "links",
        "relationship", "relationships", "many", "much", "count", "total", "number", "numbers",
        "associated", "associate", "association", "connect", "connected", "connection",
        "foreign", "key", "field", "fields", "relation", "relations",
        "application", "applications", "module", "modules", "project", "repository", "codebase", "system",
        "has", "have", "having", "its", "their", "there", "been", "being",
        "used", "using", "uses", "using", "hold", "holds", "holding", "store", "stores", "storing",
        "aryashakti", "arya", "aryaq", "peter", "django", "python", "backend", "server",
        "called", "named", "call", "calling", "calls", "name", "names",
    }
    terms = re.findall(r"[A-Za-z0-9_./-]+", corrected)
    _ALLOWED_SHORT = {"bd", "id", "ui", "db", "ct", "ip", "qr", "ai", "s3"}
    useful_terms = [
        term for term in terms
        if term.lower() not in stop_words and (len(term) >= 3 or term.lower() in _ALLOWED_SHORT)
    ]
    if not useful_terms:
        return ""
    # Join the top terms with underscore so they form a probable Django field/model name
    # e.g. ["user", "role"] → "user_role"  (matches user_role, user_roles in code)
    # But if only one term or the term already has underscores, return as-is
    if len(useful_terms) == 1:
        return useful_terms[0]
    return "_".join(t.lower() for t in useful_terms[:2])


def _extract_prior_subject(execution_result: str) -> str | None:
    """Extract the most likely module/app/table/keyword name discussed in the prior turn's result."""
    if not execution_result:
        return None

    # 0. Check if prior turn was a 5-Step Keyword Investigation header
    inv_m = re.search(r"5-Step End-to-End Codebase Investigation for `([^`]+)`", execution_result)
    if inv_m:
        kw = inv_m.group(1).strip()
        if kw:
            return re.sub(r"[\s\-]+", "_", kw)

    # 0.5 Check if prior turn was a single Function/Class extraction header
    file_sym_m = re.search(r"📄 File: `[^`]+` \(lines [0-9\-]+\) — `(?:[A-Za-z0-9_]+\.)?([A-Za-z0-9_]+)`", execution_result)
    if file_sym_m:
        return file_sym_m.group(1).strip()

    # 1. Look for Django app/module names like "pop_manage", "app_farm", "farm_category"
    bold_match = re.findall(r"\*\*([a-zA-Z][a-zA-Z0-9_]+)\*\*", execution_result)
    module_candidates = [
        m for m in bold_match
        if "_" in m or m in _KNOWN_DJANGO_APPS_CACHE
    ]
    if module_candidates:
        return module_candidates[0]

    # 2. Look for paths like "pop_manage/models.py" or "app_farm/views.py"
    path_match = re.findall(r"\b([a-zA-Z][a-zA-Z0-9_]+)/(?:models|views|urls|admin|serializers|tests)\.py\b", execution_result)
    if path_match:
        return path_match[0]

    # 3. Look for "module named X" or "application named X"
    named_match = re.search(r"(?:module|application|app)\s+(?:named|called)\s+\**([a-zA-Z][a-zA-Z0-9_]+)\**", execution_result, re.IGNORECASE)
    if named_match:
        return named_match.group(1)

    # 4. Fall back to the most frequent snake_case identifier (min 2 segments)
    all_ids = re.findall(r"\b([a-zA-Z][a-zA-Z0-9]*(?:_[a-zA-Z0-9]+)+)\b", execution_result)
    excluded = {
        "api_key", "secret_key", "created_at", "updated_at", "type_user",
        "record_type", "source_file", "order_value", "is_active", "is_deleted",
        "date_created", "date_updated", "base_url", "api_v1", "content_type",
    }
    filtered = [i for i in all_ids if i.lower() not in excluded]
    if filtered:
        from collections import Counter
        most_common = Counter(filtered).most_common(1)
        if most_common and most_common[0][1] >= 2:
            return most_common[0][0]
        return filtered[0]

    # 5. If the prior answer was a short prose summary starting with "<Subject> is used for...", extract <Subject>
    lead_m = re.match(r"^\s*\**`?([A-Za-z][A-Za-z0-9_-]{2,})`?\**\s+(?:is|are|was|handles|manages|provides|processes)\b", execution_result.strip())
    if lead_m and lead_m.group(1).lower() not in {"this", "that", "there", "here", "the", "aryashakti", "project"}:
        return lead_m.group(1).lower()

    return None


def _extract_prior_functions(execution_result: str) -> list[str]:
    """Extract the top function/method/view names from the previous turn's output for follow-up queries like 'show me function related to it'."""
    if not execution_result:
        return []
    found: list[str] = []
    skip_names = {"__init__", "__str__", "_headers", "post", "get", "put", "delete", "patch", "get_queryset", "as_view"}

    # 1. If the prior output was a single function extraction with 'Where It Is Called', prioritize its caller functions!
    if "🔗 Where It Is Called:" in execution_result and "5-Step End-to-End Codebase Investigation" not in execution_result:
        caller_section = execution_result.split("🔗 Where It Is Called:", 1)[1]
        for cm in re.findall(r"\(inside `([A-Za-z0-9_.]+)`\)", caller_section):
            leaf = cm.split(".")[0] if cm.endswith((".post", ".get", ".put", ".delete")) else cm.split(".")[-1]
            if leaf and leaf not in skip_names and leaf not in found and is_exact_code_symbol(leaf):
                found.append(leaf)
        if found:
            return found[:2]

    # 2. Extract Function / Method symbols from structured investigation reports
    for fm in re.findall(r"(?:Function|Method)\s+\*\*`(?:[A-Za-z0-9_]+\.)?([A-Za-z0-9_]+)`\*\*", execution_result):
        if fm not in skip_names and fm not in found and is_exact_code_symbol(fm):
            found.append(fm)

    # 3. Extract '(inside function `X`)' or '#### `X`' component headers
    for im in re.findall(r"\(inside function `([A-Za-z0-9_]+)`\)", execution_result):
        if im not in skip_names and im not in found and is_exact_code_symbol(im):
            found.append(im)
    for hm in re.findall(r"#### `(?:[A-Za-z0-9_]+\.)?([A-Za-z0-9_]+)`", execution_result):
        if hm not in skip_names and hm not in found and is_exact_code_symbol(hm):
            found.append(hm)

    return found[:2]


# Cache of app names seen in prior turns (populated dynamically)
_KNOWN_DJANGO_APPS_CACHE: set[str] = set()


_GENERIC_FOLLOWUP_WORDS = {
    "can", "you", "please", "show", "list", "check", "what", "how", "where", "which",
    "get", "tell", "give", "display", "view", "views", "see", "look", "find", "search",
    "also", "now", "and", "then", "next", "ok", "all", "the", "a", "an", "in", "of",
    "for", "from", "to", "with", "on", "at", "by", "about", "is", "are", "was", "were",
    "do", "does", "it", "its", "this", "that", "these", "those", "their", "them", "they",
    "me", "my", "model", "models", "class", "classes", "url", "urls", "route", "routes",
    "api", "apis", "endpoint", "endpoints", "table", "tables", "column", "columns",
    "field", "fields", "function", "functions", "method", "methods", "file", "files",
    "serializer", "serializers", "code", "detail", "details", "used", "use", "uses",
    "explain", "more", "deep", "deeply", "elaborate", "full", "step", "line", "work", "works",
    "related", "relating", "relation", "relations", "belongs", "belonging", "associated",
    "connected", "inside", "within", "there", "here",
    "called", "named", "call", "calling", "calls", "name", "names",
}


def _is_implicit_followup(user_request: str) -> bool:
    """Detect if a request is likely an implicit follow-up (short, no explicit subject).

    Examples of implicit follow-ups:
      - "can check all model classes"
      - "show me the urls"
      - "list all apis"
      - "what about the views"
      - "explain more"
      - "explain this code"
    """
    lowered = user_request.lower().strip()
    word_tokens = re.findall(r"[a-z0-9_]+", lowered)

    # If the user introduces ANY specific domain/code keyword (e.g. "cvrp", "aryamitra", "payment"),
    # it is a new topic, NOT an implicit follow-up to the prior turn!
    specific_keywords = [w for w in word_tokens if w not in _GENERIC_FOLLOWUP_WORDS]
    if specific_keywords:
        return False

    followup_starters = (
        "can", "show", "list", "check", "what", "how", "get", "tell",
        "give", "display", "view", "see", "look", "find", "search",
        "also", "now", "and", "then", "next", "ok", "explain", "elaborate", "more",
    )

    if len(word_tokens) <= 10 and word_tokens and word_tokens[0] in followup_starters:
        has_explicit_subject = bool(re.search(
            r"\b[a-zA-Z][a-zA-Z0-9]*(?:_[a-zA-Z0-9]+)+\b", user_request
        ))
        has_path = bool(re.search(r"[/\\]", user_request))
        if not has_explicit_subject and not has_path:
            return True

    # "what about ..." pattern (only when no new specific keyword is introduced)
    if lowered.startswith("what about") or lowered.startswith("how about"):
        return True

    return False


def _resolve_conversational_references(user_request: str, state: AgentState) -> str:
    """Resolve conversational context: explicit markers ('this table') AND implicit follow-ups.

    Two modes:
    1. EXPLICIT: User says "this table", "this file", etc. -> extract entity from prior result.
    2. IMPLICIT: User asks a short follow-up like "can check all model classes" or "explain more"
       after discussing a specific module/file -> infer the subject from the prior turn's execution_result.

    Falls back to prior_execution_summary when execution_result is empty (new turn reset).
    Also resolves explicit .py filenames mentioned in the request against the prior context.
    """
    lowered = user_request.lower()
    # Use execution_result if present (mid-turn), else fall back to prior_execution_summary (post-reset)
    prior_context = state.get("execution_result") or state.get("prior_execution_summary") or ""

    # -- SPECIAL: Resolve file in subfolder pattern: e.g. "urls.py in app_farm", "models.py from pop_manage" --
    in_folder_match = re.search(r"\b([A-Za-z0-9_.-]+\.[A-Za-z0-9]+)\s+(?:in|from|inside|under)\s+(?:the\s+)?([A-Za-z0-9_.-]+)\b", user_request, re.IGNORECASE)
    if in_folder_match:
        fn = in_folder_match.group(1)
        folder = in_folder_match.group(2)
        root = Path(get_project_root())
        if (root / folder / fn).exists():
            resolved = f"{folder}/{fn}"
            enriched = user_request.replace(in_folder_match.group(0), resolved)
            logger.info(f"Resolved file in subfolder: '{in_folder_match.group(0)}' -> '{resolved}'")
            return enriched
        elif (root / f"app_{folder}" / fn).exists():
            resolved = f"app_{folder}/{fn}"
            enriched = user_request.replace(in_folder_match.group(0), resolved)
            logger.info(f"Resolved file in subfolder: '{in_folder_match.group(0)}' -> '{resolved}'")
            return enriched

    # -- SPECIAL: If user mentions a bare .py filename explicitly (without directory) and it appears in prior context,
    #    resolve its full relative path so the coder can read the correct file. --
    has_explicit_path = bool(re.search(r"[A-Za-z0-9_.-]+[/\\][A-Za-z0-9_.-]+\.py", user_request))
    if not has_explicit_path:
        py_name_match = re.search(r"\b([\w]+(?:_[\w]+)*\.py)\b", user_request, re.IGNORECASE)
        if py_name_match and prior_context:
            py_name = py_name_match.group(1).lower()
            # Look for a relative path containing that filename in prior context
            path_candidates = re.findall(r"\b[\w/\\]+/" + re.escape(py_name), prior_context, re.IGNORECASE)
            if path_candidates:
                resolved_path = path_candidates[0].replace("\\", "/")
                # Only inject if the candidate actually exists on disk
                try:
                    if (Path(get_project_root()) / resolved_path).exists():
                        enriched = user_request.replace(py_name_match.group(1), resolved_path)
                        if enriched != user_request:
                            logger.info(f"Resolved .py filename '{py_name}' -> '{resolved_path}' from prior context")
                            return enriched
                except Exception:
                    pass

    # -- MODE 1: Explicit referential markers --
    referential_markers = (
        "this table", "the table", "that table", "this model", "that model",
        "this api", "the api", "that api", "these apis", "the apis",
        "this route", "the route", "these routes", "the routes",
        "these urls", "the urls", "these endpoints", "the endpoints",
        "these views", "the views", "which route", "which api", "which one",
        "which of these", "what view", "what views", "which view",
        "from this", "related to this", "related to that", "related to it",
        "for this", "for that", "for it", "of this", "of that", "of it",
        "in it", "about it", "from it", "inside it", "with it",
        "this sheet", "the sheet", "that sheet", "this file", "the file", "that file", "in this file",
        "this spreadsheet", "the spreadsheet", "this document", "the document", "this workbook",
        "this code", "the code", "that code", "all the code", "this function", "the function",
        "these functions", "this class", "the class", "these methods", "explain more",
        "in detail", "more detail", "more details", "elaborate",
    )
    has_explicit_marker = any(marker in lowered for marker in referential_markers)
    has_own_explicit_target = bool(
        re.search(r"/(?:api|v\d+)/[A-Za-z0-9_/-]+", user_request, re.IGNORECASE)
        or re.search(r"[A-Za-z0-9_./\\-]+\.(?:xlsx|csv|xls|py|json|md|txt|pdf|html|yaml|yml)\b", user_request, re.IGNORECASE)
        or re.search(r"\b[a-zA-Z][a-zA-Z0-9]*(?:_[a-zA-Z0-9]+)+\b", user_request)
        or re.search(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b", user_request)
    )

    if has_explicit_marker and not has_own_explicit_target and (prior_context or state.get("last_accessed_file")):
        # Check if user is asking about a file, code, or spreadsheet
        wants_file = any(m in lowered for m in (
            "sheet", "file", "spreadsheet", "workbook", "document", "route", "routes",
            "url", "urls", "api", "apis", "endpoint", "endpoints", "code", "function",
            "functions", "class", "method", "methods", "explain more", "in detail", "more detail", "elaborate",
        ))
        if wants_file:
            last_file = state.get("last_accessed_file")
            if last_file:
                enriched = f"read and explain {last_file}: {user_request}"
                logger.info(f"Resolved explicit reference '{user_request}' -> '{enriched}'")
                return enriched
            file_candidates = re.findall(r"[A-Za-z0-9_./\\-]+\.(?:xlsx|csv|xls|py|json|md|txt|pdf|html|yaml|yml)", prior_context, re.IGNORECASE)
            clean_files = [f.strip() for f in file_candidates if not f.strip().startswith("~$")]
            if clean_files:
                resolved_file = clean_files[0]
                enriched = f"read and explain {resolved_file}: {user_request}"
                logger.info(f"Resolved explicit reference '{user_request}' -> '{enriched}'")
                return enriched

        # 1. Prioritize Django app names found in file paths (e.g. pop_manage/views.py)
        app_candidates = re.findall(r"\b([a-zA-Z][a-zA-Z0-9_]+)/(?:models|views|urls|admin|serializers)\.py\b", prior_context)
        if app_candidates:
            resolved_entity = app_candidates[0]
            enriched = f"{user_request} (referring to {resolved_entity})"
            logger.info(f"Resolved explicit reference '{user_request}' -> '{enriched}'")
            return enriched

        # 2. Look for table/module names in prior context
        candidates = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*(?:_[a-zA-Z0-9]+)+\b", prior_context)
        excluded = {"api_key", "secret_key", "created_at", "updated_at", "type_user", "record_type", "source_file"}
        clean_candidates = [c for c in candidates if c.lower() not in excluded]
        if clean_candidates:
            resolved_entity = clean_candidates[0]
            enriched = f"{user_request} (referring to {resolved_entity})"
            logger.info(f"Resolved explicit reference '{user_request}' -> '{enriched}'")
            return enriched

    # -- MODE 2: Implicit follow-up detection --
    if prior_context and _is_implicit_followup(user_request):
        # First check if a specific .py file was the primary focus of prior_context (e.g. libraries/openai.py)
        py_cands = re.findall(r"[A-Za-z0-9_./\\-]+\.py\b", prior_context)
        if py_cands and any(w in lowered for w in ("explain", "more", "code", "detail", "details", "function", "functions", "method", "methods")):
            primary_py = py_cands[0]
            enriched = f"read and explain {primary_py}: {user_request}"
            logger.info(f"Resolved implicit code follow-up '{user_request}' -> '{enriched}'")
            return enriched
        subject = _extract_prior_subject(prior_context)
        if subject:
            # Cache the app name for future reference
            _KNOWN_DJANGO_APPS_CACHE.add(subject)
            enriched = f"{user_request} in {subject}"
            logger.info(f"Resolved implicit follow-up '{user_request}' -> '{enriched}'")
            return enriched

    return user_request


def _handle_project_switch(user_request: str) -> str | None:
    """Detect and execute on-the-fly project directory switching from chat."""
    req = user_request.strip()
    lowered = req.lower()

    if any(q in lowered for q in ("current project root", "what is current project", "show current project", "current project path", "where is project root")):
        return f"[INFO] Active project root is: {get_project_root()}"

    patterns = [
        r"^(?:switch|change|set|cd)\s+(?:to\s+)?(?:(?:the\s+)?(?:project(?:\s+root|\s+path)?|root|workspace)\s+)?(?:to\s+)?(.+)",
        r"(?:switch|change|set)\s+(?:to\s+)?project(?:\s+root|\s+path)?\s+(?:to\s+)?(.+)",
        r"(?:switch|change|set)\s+(?:root|workspace)\s+(?:to\s+|path\s+to\s+)?(.+)",
    ]
    for pat in patterns:
        m = re.search(pat, req, re.IGNORECASE)
        if m:
            target = m.group(1).strip().strip('"').strip("'")
            if target.lower().startswith("to "):
                target = target[3:].strip().strip('"').strip("'")
            # Ignore false positives like 'switch to python 3.11' or 'switch mode'
            has_sep = any(c in target for c in ("\\", "/", ":"))
            is_valid_dir = False
            try:
                is_valid_dir = Path(target).resolve().is_dir()
            except Exception:
                pass
            if not has_sep and not is_valid_dir:
                continue
            return set_project_root(target)

    # Check if user entered an absolute directory path directly (e.g. "C:/work/aryashakti")
    if len(req) >= 3 and any(c in req for c in ("\\", "/", ":")):
        action_words = ("what", "how", "why", "who", "where", "summarise", "summarize", "summrise", "find", "search", "show", "tell", "read", "open", "scan", "list", "harvest")
        if not any(w in lowered for w in action_words):
            try:
                candidate = Path(req.strip('"').strip("'")).resolve()
                if candidate.exists() and candidate.is_dir():
                    return set_project_root(str(candidate))
            except Exception:
                pass

    return None


def _handle_finetune_pipeline(user_request: str) -> str | None:
    """Detect and execute incremental finetune harvest, Google Drive sync, Colab launch, or GGUF Ollama import."""
    req = user_request.strip()
    lowered = req.lower()

    # 1. Import GGUF from Google Drive into local Ollama
    if ("import" in lowered or "load" in lowered or "update" in lowered) and (
        "gguf" in lowered or ("ollama" in lowered and ("gdrive" in lowered or "drive" in lowered or "colab" in lowered))
    ):
        return harvest_for_finetune(import_gguf=True)

    # 2. Launch Colab notebook & sync latest JSONL to Google Drive without re-harvesting
    if any(k in lowered for k in ("open colab", "launch colab", "start colab", "train on colab", "run colab")) and "harvest" not in lowered:
        from scripts.build_finetune_dataset import launch_colab_training
        info = launch_colab_training(sync_first=True)
        return (
            f"### Google Drive Synced & Colab T4 Notebook Opened\n"
            f"- **Google Drive Dataset**: `{info.get('gdrive_jsonl')}`\n"
            f"- **Google Drive Notebook**: `{info.get('gdrive_notebook')}`\n"
            f"- **Colab URL Opened**: `{info.get('colab_url')}`\n"
            f"- **Next Step**: In the opened Colab tab, ensure **Runtime -> T4 GPU** is selected and press **`Ctrl+F9` (Run all)**."
        )

    # 2b. Answer questions about what Peter is harvesting / finetune dataset contents
    if re.search(r"\b(harvest|harvesting|harvested)\b", lowered) and (
        any(lowered.startswith(q) for q in ("what ", "how ", "why ", "which ", "explain ", "tell ", "show "))
        or "?" in req
        or any(k in lowered for k in ("what are u", "what are you", "what did u", "what did you", "status", "summary"))
    ):
        from scripts.build_finetune_dataset import OUTPUT_JSONL, get_gdrive_finetune_dir
        local_count = 0
        local_mb = 0.0
        if OUTPUT_JSONL.exists():
            with open(OUTPUT_JSONL, "r", encoding="utf-8") as fh:
                local_count = sum(1 for line in fh if line.strip())
            local_mb = round(OUTPUT_JSONL.stat().st_size / (1024 * 1024), 2)
        gdir = get_gdrive_finetune_dir()
        gdrive_jsonl = str(gdir / OUTPUT_JSONL.name) if gdir else "Not detected"
        return (
            f"### What Peter Harvests for Continual Fine-Tuning (`peter-coder`)\n"
            f"When you run `harvest`, I extract verified Q&A training pairs from the active project (`{get_project_root()}`) and upsert-merge them into your dataset without forgetting any previous data:\n\n"
            f"1. **Python AST Symbols & Implementations (`533` `.py` files)**:\n"
            f"   - Every function, class, and method definition, exact file path, line numbers, caller/callee references, and source code body.\n"
            f"2. **Module & Application Architecture Summaries**:\n"
            f"   - Repository-wide app inventory (`25` Django apps + `7` support packages) and per-module structural overviews.\n"
            f"3. **Django URL Routes, Views & DRF Serializers**:\n"
            f"   - API endpoint routes (`urls.py`), view functions/classes (`views.py`), and serializer field validations (`serializers.py`).\n"
            f"4. **Postman API Collections (`4` collections)**:\n"
            f"   - Full endpoint specs, HTTP methods, headers, query parameters, and request body schemas.\n"
            f"5. **Django ORM Models (`models.py`)**:\n"
            f"   - Model classes, field types, and mapped database table names.\n"
            f"6. **PostgreSQL Database Schema Metadata (`230` tables)**:\n"
            f"   - Strictly metadata only: `table_name`, `column_name`, `row_count`, and foreign-key `relationships` (both outgoing and incoming FKs).\n\n"
            f"### Strict Privacy & Security Boundaries (Never Harvested)\n"
            f"- **`.env` / `.env.*` files** and credentials are strictly skipped.\n"
            f"- **Raw PostgreSQL table rows** (user records, PII, tokens) are never read or stored.\n\n"
            f"### Current Dataset Status\n"
            f"- **Total Cumulative Examples**: **`{local_count}` Q&A pairs (`{local_mb} MB`)**\n"
            f"- **Local Dataset**: `{OUTPUT_JSONL}`\n"
            f"- **Google Drive Synced Copy**: `{gdrive_jsonl}`"
        )

    # 3. Incremental Harvest (file, folder, or full project) + Google Drive sync (+ optional Colab launch)
    if re.search(r"\bharvest\b", lowered):
        open_colab = any(k in lowered for k in ("colab", "train", "t4"))
        # Strip Google Drive / Colab destination paths so "harvest ... to G:\My Drive\Ollama_Models"
        # never mistakes "G:\My" for the source code target path.
        cleaned_req = re.sub(
            r"""(?:\b(?:to|into|in|on|at|sync(?:\s+to)?)\s+)?['"]?[Gg]:[\\/]My(?:\s+Drive(?:[\\/][^\s'"]*)?)?['"]?""",
            " ",
            req,
            flags=re.IGNORECASE,
        )
        cleaned_req = re.sub(r"/content/drive/[^\s'\"]*", " ", cleaned_req, flags=re.IGNORECASE)

        # Extract explicit source path/folder/file if specified
        target = ""
        quoted_m = re.search(r"""['"]([A-Za-z]:[\\/][^'"]+|[\w./\\ -]+\.(?:py|json|md|html|sql))['"]""", cleaned_req)
        path_m = re.search(
            r"([A-Za-z]:[\\/][^\s\"']+|[\w./\\-]+\.(?:py|json|md|html|sql)|\b(?:folder|file|module|app|directory)\s+([A-Za-z0-9_./\\-]+))",
            cleaned_req,
            re.IGNORECASE,
        )
        if quoted_m:
            target = quoted_m.group(1).strip()
        elif path_m:
            target = (path_m.group(2) or path_m.group(1)).strip()
        else:
            # Check if user named a known subfolder inside active project root
            root = Path(get_project_root())
            ignore_words = {
                "harvest", "this", "file", "folder", "project", "skip", "env", "and",
                "train", "colab", "gdrive", "sync", "drive", "my", "google", "ollama",
                "models", "ollama_models", "to", "into", "from", "all", "full",
            }
            for token in re.findall(r"[A-Za-z0-9_.-]+", cleaned_req):
                if token.lower() not in ignore_words:
                    if (root / token).exists():
                        target = token
                        break
        return harvest_for_finetune(target_path=target, open_colab=open_colab)

    return None


def supervisor_node(state: AgentState):
    logger.info("--- 🧭 [NODE: SUPERVISOR] Routing Task ---")
    _log_model_provider("supervisor")

    # Fast-path: Check for finetune harvest / Google Drive sync / Colab launch / GGUF import commands
    finetune_res = _handle_finetune_pipeline(state["user_request"])
    if finetune_res:
        logger.info("Finetune harvest/GDrive/Colab pipeline command handled directly.")
        return {
            "active_worker": "FINISH",
            "proposed_action": "RESPOND_ONLY",
            "execution_result": finetune_res,
            "target_project": _target_project(state["user_request"]),
            "model_provider": PRIMARY_MODEL_PROVIDER,
            "status": "completed",
        }

    # Fast-path: Check if user passed a leading project path before an instruction
    # e.g. "C:/work/aryashakti summrise this project" -> switch root to C:/work/aryashakti and run "summrise this project"
    path_prefix_match = re.match(r"^([A-Za-z]:[\\/][^\s]+)\s+(.*)$", state["user_request"].strip())
    if path_prefix_match:
        target_dir = path_prefix_match.group(1).strip().strip('"').strip("'")
        remaining_task = path_prefix_match.group(2).strip()
        try:
            candidate = Path(target_dir).resolve()
            if candidate.exists() and candidate.is_dir():
                switch_msg = set_project_root(str(candidate))
                logger.info(f"Leading path auto-switched project root: {switch_msg}")
                state["user_request"] = remaining_task
        except Exception as e:
            logger.warning(f"Leading path check failed: {e}")

    # Fast-path: Check for direct project-switch commands (e.g. "switch project to C:\...")
    switch_res = _handle_project_switch(state["user_request"])
    if switch_res:
        logger.info(f"Project root switch handled: {switch_res}")
        return {
            "active_worker": "FINISH",
            "proposed_action": "RESPOND_ONLY",
            "execution_result": switch_res,
            "target_project": _target_project(state["user_request"]),
            "model_provider": PRIMARY_MODEL_PROVIDER,
            "status": "completed",
        }

    worker_outputs = state.get("worker_outputs", {})
    execution_result = state.get("execution_result")
    active_w = state.get("active_worker")

    # If a worker (coder/researcher) just finished in the current turn (either via tool execution or direct response),
    # route immediately to final_answer instead of re-running intent routing in an infinite loop!
    if active_w in {"researcher", "coder"}:
        worker_direct_out = worker_outputs.get(active_w)
        final_ev = execution_result or worker_direct_out
        if final_ev:
            logger.info("Routing worker output/tool result to final answer synthesis.")
            return {
                "active_worker": "final_answer",
                "proposed_action": "RESPOND_ONLY",
                "execution_result": final_ev,
                "target_project": state.get("target_project", _target_project(state["user_request"])),
                "model_provider": PRIMARY_MODEL_PROVIDER,
                "status": "running",
            }

    # Resolve anaphoric references (e.g. 'this table') from prior turn
    effective_request = _resolve_conversational_references(state["user_request"], state)
    prior_summary_to_save = state.get("execution_result") or state.get("prior_execution_summary")

    detected_intent = _detect_intent(effective_request)
    if detected_intent:
        worker, task = detected_intent
        logger.info(f"Deterministic intent routed to {worker.upper()}: {task}")
        return {
            "active_worker": worker,
            "worker_task": task,
            "intent": worker,
            "target_project": _target_project(state["user_request"]),
            "model_provider": PRIMARY_MODEL_PROVIDER,
            "status": "running",
            "prior_execution_summary": prior_summary_to_save,
            "last_accessed_file": state.get("last_accessed_file"),
            "last_file_content": state.get("last_file_content"),
            "execution_result": None,
            "worker_outputs": {},
            "tool_results": {},
            "tool_attempts": {},
            "history": [],
            "tool_queue": [],
            "retry_count": 0,
        }
    
    prompt = (
        f"Original User Request: {state['user_request']}\n\n"
        f"Completed Sub-Tasks and Findings:\n{worker_outputs}\n\n"
        "You supervise two specialized agents:\n"
        "- 'researcher': searches documentation/RAG, knowledge base, website status.\n"
        "- 'coder': inspects/edits files, databases, spreadsheets, extracts functions/classes, traces callers, analyzes modules.\n\n"
        "CRITICAL RULES:\n"
        "1. If the user asks to scan a folder, read a file, inspect/explain a function or class, trace callers/related code, or query a database, you MUST route to 'coder'.\n"
        "2. NEVER attempt to answer the user directly if it requires reading files or scanning directories. You MUST delegate.\n"
        "3. ONLY choose 'FINISH' if the workers have successfully returned the requested data in 'Completed Sub-Tasks' above.\n"
        "4. If 'Completed Sub-Tasks' is empty { }, you CANNOT choose FINISH. You must route to a worker."
    )
    
    decision = supervisor_chain.invoke([
        SystemMessage(content="You are a strict, logical AI routing supervisor. You do not do the work yourself, you delegate."),
        HumanMessage(content=prompt)
    ])
    
    if not decision:
        logger.warning("Supervisor LLM failed to return structured output.")
        return {
            "active_worker": "FINISH",
            "proposed_action": "RESPOND_ONLY",
            "execution_result": "My routing brain had a minor formatting glitch. Could you repeat that?",
            "status": "completed"
        }
    
    if decision.next_step == "FINISH":
        logger.info("Supervisor decided the task is complete.")
        return {
            "active_worker": "FINISH",
            "proposed_action": "RESPOND_ONLY",
            "execution_result": decision.task,
            "target_project": _target_project(state["user_request"]),
            "model_provider": PRIMARY_MODEL_PROVIDER,
            "status": "completed"
        }
    
    logger.info(f"Supervisor delegating to {decision.next_step.upper()}: {decision.task}")
    return {
        "active_worker": decision.next_step,
        "worker_task": decision.task,
        "target_project": _target_project(state["user_request"]),
        "model_provider": PRIMARY_MODEL_PROVIDER,
        "status": "running",
        "prior_execution_summary": prior_summary_to_save,
        "last_accessed_file": state.get("last_accessed_file"),
        "last_file_content": state.get("last_file_content"),
        "execution_result": None,
        "worker_outputs": {},
        "tool_results": {},
        "tool_attempts": {},
        "history": [],
        "tool_queue": [],
        "retry_count": 0,
    }


def final_answer_node(state: AgentState):
    """Turn tool evidence into a concise answer grounded in the original request."""
    _log_model_provider("final_answer")
    user_req = state.get("user_request", "")
    evidence = state.get("execution_result") or ""
    prior = state.get("prior_execution_summary") or ""
    last_file = state.get("last_accessed_file") or ""

    # Fast-path for Offline Local Model (`peter-coder` 3B):
    # When a deterministic Code Intelligence tool (`investigate_keyword`, `extract_function`, `trace_request_flow`,
    # `analyze_module`, or `safe_read_file` on a `.py` file) has already produced a complete, verified
    # report with AST step-by-step walkthroughs and exact code blocks, return it directly in <10ms
    # instead of letting the 3B model spend 60s on CPU and truncate 95% of the source code!
    if PRIMARY_MODEL_PROVIDER == "ollama" and evidence:
        ev_strip = evidence.strip()
        if any(
            ev_strip.startswith(prefix)
            for prefix in (
                "🔎 **5-Step End-to-End Codebase Investigation",
                "📄 File:",
                "🌐 **End-to-End Request Flow",
                "📦 Module:",
                "📊 **Live AryaQ Dashboard",
                "📊 Live AryaQ Dashboard",
            )
        ):
            logger.info("Returning verified Code Intelligence / Live Dashboard dossier directly (<10ms).")
            return {
                "active_worker": "FINISH",
                "execution_result": ev_strip,
                "model_provider": PRIMARY_MODEL_PROVIDER,
                "last_accessed_file": last_file,
                "last_file_content": state.get("last_file_content"),
                "status": "completed",
            }
        if last_file and str(last_file).endswith(".py") and ("def " in ev_strip or "class " in ev_strip):
            from src.tools.code_intelligence import _explain_python_symbol_ast
            wt = _explain_python_symbol_ast(ev_strip)
            wt_block = f"{wt}\n\n" if wt else ""
            formatted = f"### 📄 File Walkthrough: `{last_file}`\n{wt_block}```python\n{ev_strip[:6000]}\n```"
            return {
                "active_worker": "FINISH",
                "execution_result": formatted,
                "model_provider": PRIMARY_MODEL_PROVIDER,
                "last_accessed_file": last_file,
                "last_file_content": state.get("last_file_content"),
                "status": "completed",
            }

    context_addon = ""
    # CRITICAL: Only include previous-turn context if the current turn produced no fresh tool evidence
    # OR the user explicitly asked a follow-up referring to the previous turn ("this table", "this file", etc.).
    # Never inject prior-turn answers into a brand-new investigation, which causes state bleed!
    is_followup_req = _is_implicit_followup(user_req) or any(
        m in user_req.lower() for m in ("this table", "this file", "that file", "this model", "this code", "explain more", "in detail", "previous", "above")
    )
    if (not evidence or is_followup_req) and (prior or last_file):
        file_hint = f" (most recently inspected: {last_file})" if last_file else ""
        prior_text = f":\n{prior[:3000]}" if prior else ""
        context_addon = f"\n\nContext from previous turn{file_hint}{prior_text}"

    prompt = (
        f"User request: {user_req}\n\n"
        f"Tool evidence for '{user_req}':\n{evidence}{context_addon}\n\n"
        f"Answer the user's request ('{user_req}') directly and accurately using ONLY the tool evidence above.\n"
        "- When explaining code, functions, classes, or methods (or when asked to 'explain more' / 'explain in detail'):\n"
        "  (1) Cite the exact file path and line numbers (e.g. `libraries/openai.py:10-195`),\n"
        "  (2) Include the relevant Python code block (` ```python ... ``` `) from the tool evidence,\n"
        "  (3) Provide a clear, method-by-method and step-by-step walkthrough covering: input parameters, core business logic, external API or ORM/DB calls, error handling, and return values,\n"
        "  (4) Show where the class/function is called across the codebase (views, routes, or tasks).\n"
        "- If a 5-Step End-to-End Codebase Investigation report is present, walk through the findings clearly:\n"
        "  (1) all matched classes, functions, and methods with their file paths and line numbers,\n"
        "  (2) where the keyword/variants appear in models, serializers, constants, admin, and migrations,\n"
        "  (3) the exact API URL routes (`urls.py` / `Project/urls.py`) and HTTP methods (`GET`, `POST`, `PATCH`, etc.),\n"
        "  (4) which views and services handle the flow and what their implementations actually do.\n"
        "- If the user sent a terminal error, exception, or traceback, provide a clear Root Cause Analysis: (1) identify the exact file, line number, function/view, and exception type that failed, (2) explain *why* it went wrong based on the inspected code, and (3) provide the exact code or command fix to resolve the issue.\n"
        "- If live dashboard metrics or a dashboard screenshot are present in the evidence, present those live metrics as the current working state of the dashboard.\n"
        "- Only state that information is unavailable if the tool evidence genuinely did not find it.\n"
        "- Do not mention workers, internal state, prompts, approvals, or tool names."
    )
    try:
        response = base_llm.invoke([
            SystemMessage(content="You are a senior software engineer writing a precise, evidence-grounded answer for the user's current question. Never repeat answers from unrelated prior topics."),
            HumanMessage(content=prompt),
        ])
        answer = _message_text(response.content).strip()
        is_degenerate = (
            not answer 
            or answer.lower().startswith("user safety:") 
            or answer.lower() == "safe"
            or (len(answer) < 40 and len(evidence) > 200)
        )
        if not is_degenerate:
            return {
                "active_worker": "FINISH",
                "execution_result": answer,
                "model_provider": PRIMARY_MODEL_PROVIDER,
                "last_accessed_file": state.get("last_accessed_file"),
                "last_file_content": state.get("last_file_content"),
                "status": "completed",
            }
        logger.warning(f"Final answer received degenerate/safety response ('{answer}'). Triggering fallback...")
    except Exception as error:
        logger.warning(f"Final answer synthesis failed; attempting fallback: {error}")

    try:
        if 'offline_llm' in globals() and offline_llm:
            logger.info("Attempting final answer synthesis via local offline LLM fallback...")
            fallback_resp = offline_llm.invoke([
                SystemMessage(content="You are a senior software engineer writing a precise, evidence-grounded answer for the user's current question. Never repeat answers from unrelated prior topics."),
                HumanMessage(content=prompt),
            ])
            fallback_ans = _message_text(fallback_resp.content).strip()
            if fallback_ans and not fallback_ans.lower().startswith("user safety:") and len(fallback_ans) >= 40:
                return {
                    "active_worker": "FINISH",
                    "execution_result": fallback_ans,
                    "model_provider": "ollama",
                    "status": "completed",
                }
    except Exception as fe:
        logger.warning(f"Offline fallback also failed: {fe}")

    return {
        "active_worker": "FINISH",
        "execution_result": evidence,
        "model_provider": PRIMARY_MODEL_PROVIDER,
        "status": "completed",
    }

def _should_stop_repeating_tool(state: AgentState, tool_name: str | None = None) -> bool:
    if not state.get("execution_result"):
        return False

    history = state.get("history", [])
    tool_history = [item for item in history if item.startswith("tool:")]
    if len(tool_history) >= 6:
        logger.warning("⚠️ Tool-call limit reached for this request; stopping the workflow loop.")
        return True

    last_tool = history[-1] if history else None
    if tool_name and last_tool == f"tool:{tool_name}":
        return True

    if tool_name and history.count(f"tool:{tool_name}") >= 2:
        logger.warning(f"⚠️ Tool {tool_name} has already been attempted twice; stopping repeated execution.")
        return True

    result = state.get("execution_result", "")
    lowered = result.lower()
    if "aborted by user" in lowered or "error" in lowered or "no relevant information found" in lowered:
        return True

    if tool_name and state.get("active_worker") == "researcher":
        return "search_internal_knowledge" in (tool_name or "") and "no relevant information found" in lowered

    return False

# -------------------------------------------------------------
# 2. RESEARCHER AGENT NODE
# -------------------------------------------------------------
def researcher_node(state: AgentState):
    logger.info("--- 🔍 [WORKER: RESEARCHER] Executing ---")
    _log_model_provider("researcher")
    subtask = state.get("worker_task", state["user_request"])

    normalized_request = state["user_request"].lower().replace("dashbord", "dashboard")
    has_code_file = bool(re.search(r"\b[A-Za-z0-9_./\\-]+\.py\b", state["user_request"], re.IGNORECASE))
    live_dashboard_request = (
        not has_code_file
        and ("dashboard" in normalized_request or "aryaq" in normalized_request)
        and any(term in normalized_request for term in ("check", "current", "status", "information", "tell me", "report", "reports", "new"))
        and not any(code_term in normalized_request for code_term in ("explain why", "give me the solution", "disconnect", "function", "code"))
    )
    if live_dashboard_request and not state.get("execution_result"):
        logger.info("Routing live dashboard request to fetch_dashboard_information.")
        return {
            "proposed_action": "TOOL_CALL: fetch_dashboard_information",
            "tool_payload": {
                "name": "fetch_dashboard_information",
                "args": {"query": state["user_request"]},
            },
            "status": "running",
        }
    
    # Fast path: If the task is a request-flow or app-flow tracing query
    trace_triggers = (
        "trace it from", "trace request", "request reaches", "passes through this",
        "client request reaches", "project-level url", "request flow", "flow in this project",
        "survey flow", "user flow", "trace a ", "trace the ",
    )
    if any(t in normalized_request for t in trace_triggers) and not state.get("execution_result"):
        feature_arg = state["user_request"]
        if "other than soil" in normalized_request:
            feature_arg = "farm_category"
        logger.info(f"Researcher utilizing shared code intelligence: tracing request flow for '{feature_arg[:60]}'")
        return {
            "proposed_action": "TOOL_CALL: trace_request_flow",
            "tool_payload": {"name": "trace_request_flow", "args": {"feature": feature_arg}},
            "status": "running",
        }

    # Fast path: Project summary / overview
    is_project_summary = (
        any(w in normalized_request for w in ("summarise", "summarize", "summrise", "summerize", "overview", "what feature", "what does this project", "explain this project"))
        and any(p in normalized_request for p in ("project", "codebase", "repo", "repository", "this app", "this system", "it gives"))
    )
    if is_project_summary and not state.get("execution_result"):
        root = Path(get_project_root())
        readme_candidates = ["README.md", "readme.md", "Readme.md", "README.txt", "readme.txt"]
        found_readme = None
        for r in readme_candidates:
            if (root / r).exists():
                found_readme = r
                break

        if found_readme:
            logger.info(f"Researcher fast-path: Reading project README '{found_readme}' for project summary")
            return {
                "proposed_action": f"TOOL_CALL: safe_read_file",
                "tool_payload": {"name": "safe_read_file", "args": {"filename": found_readme}},
                "status": "running",
            }
        else:
            logger.info("Researcher fast-path: Scanning root directory for project summary")
            return {
                "proposed_action": "TOOL_CALL: scan_directory",
                "tool_payload": {"name": "scan_directory", "args": {"directory_path": "."}},
                "status": "running",
            }

    # Fast path: Feature / Keyword / API investigation in Researcher
    # Prevents the 3B model from hallucinating 1-line answers without running codebase tools
    if not state.get("execution_result"):
        kw_candidate = _extract_search_terms(subtask) or _extract_search_terms(state["user_request"])
        if not kw_candidate and state.get("prior_execution_summary"):
            kw_candidate = _extract_prior_subject(state.get("prior_execution_summary", "")) or ""
        if kw_candidate and len(kw_candidate) >= 2 and kw_candidate.lower() not in {"this", "that", "project", "codebase", "dashboard", "aryaq"}:
            logger.info(f"Researcher fast-path: running structured keyword investigation for '{kw_candidate}'")
            return {
                "proposed_action": "TOOL_CALL: investigate_keyword",
                "tool_payload": {"name": "investigate_keyword", "args": {"keyword": kw_candidate}},
                "search_terms": kw_candidate,
                "status": "running",
            }

    messages = [
        SystemMessage(content=(
            "You are a versatile project research and analysis agent. "
            "You have full access to both project documentation (search_internal_knowledge, spreadsheets, network tools) "
            "and the actual codebase (investigate_keyword, safe_read_file, extract_function, trace_request_flow, search_project_files, scan_directory). "
            "For documentation, specs, roles, or requirements, search internal knowledge. "
            "When asked about a specific keyword, feature, payment type, or flow (e.g. AryaMitra), use investigate_keyword to trace it end-to-end across code, URLs, and views. "
            "When concrete code implementations, class definitions, serializers, or request flows are needed, "
            "inspect the source files directly using investigate_keyword, safe_read_file, extract_function, or trace_request_flow. "
            "Base all answers on verified evidence."
        )),
        HumanMessage(content=f"Task: {subtask}")
    ]
    
    if state.get("execution_result") and state.get("active_worker") == "researcher":
        exec_res_lower = state["execution_result"].lower()
        history = state.get("history", [])
        # Automatic fallback: if RAG search returned nothing, automatically run 5-step codebase investigation!
        if (
            "no relevant information found" in exec_res_lower
            and "tool:investigate_keyword" not in history
        ):
            kw = _extract_search_terms(state["user_request"])
            if kw and len(kw) >= 2:
                logger.info(f"🔄 Researcher RAG returned empty; falling back to 5-step codebase investigation for '{kw}'")
                return {
                    "proposed_action": "TOOL_CALL: investigate_keyword",
                    "tool_payload": {"name": "investigate_keyword", "args": {"keyword": kw}},
                    "status": "running",
                }

        # 🛑 Strict stop condition to break infinite loops
        feedback = (
            f"Tool Output / Execution Result:\n{state['execution_result']}\n\n"
            "CRITICAL STOP CONDITION:\n"
            "1. DO NOT call any more tools.\n"
            "2. If the result says 'Aborted by user' or 'Error' or 'No relevant information found', just state that it failed and STOP.\n"
            "3. Otherwise, summarize the data above in plain text so the Supervisor can proceed."
        )
        messages.append(HumanMessage(content=feedback))
        
    response = researcher_llm.invoke(messages)

    if response.tool_calls:
        # Fix #1 – process ALL tool calls, not just [0]
        # Enrich the first call, check stop condition, then queue the rest.
        enriched_calls = []
        for raw_call in response.tool_calls:
            t_name = raw_call.get("name")
            t_args = dict(raw_call.get("args") or {})
            if t_name == "search_internal_knowledge":
                if not t_args.get("query"):
                    t_args["query"] = _extract_search_terms(state["user_request"]) or state["user_request"]
            enriched_calls.append({"name": t_name, "args": t_args})

        first_call = enriched_calls[0]
        tool_name = first_call["name"]
        tool_args = first_call["args"]

        if _should_stop_repeating_tool(state, tool_name):
            logger.warning(
                f"⚠️ Worker attempted a repeated tool call ({tool_name}) after a prior result; "
                "forcing a final response instead of looping."
            )
            outputs = state.get("worker_outputs", {})
            outputs["researcher"] = state.get("execution_result", "No data returned from the last tool call.")
            return {"worker_outputs": outputs, "proposed_action": "RESPOND_ONLY", "status": "running",
                    "tool_queue": []}

        logger.info(
            f"Researcher queuing {len(enriched_calls)} tool call(s): "
            + ", ".join(c["name"] for c in enriched_calls)
        )
        return {
            "proposed_action": f"TOOL_CALL: {tool_name}",
            "tool_payload": {"name": tool_name, "args": tool_args},
            "tool_queue": enriched_calls[1:],   # remaining calls for later passes
            "status": "running",
        }

    outputs = state.get("worker_outputs", {})
    outputs["researcher"] = response.content
    return {"worker_outputs": outputs, "proposed_action": "RESPOND_ONLY", "tool_queue": []}


# -------------------------------------------------------------
# 3. CODER AGENT NODE
# -------------------------------------------------------------
def _extract_traceback_info(text: str) -> dict | None:
    """
    Parse a Python/Django traceback or terminal error snippet from the user's message.
    Filters out framework/library paths (site-packages, .venv, lib/python) and returns
    the innermost project-relative file, line number, function name, and exception details.
    """
    frame_pattern = re.compile(
        r'File\s+"([^"]+\.py)",\s+line\s+(\d+)(?:,\s+in\s+([A-Za-z0-9_<>]+))?',
        re.IGNORECASE,
    )
    frames = frame_pattern.findall(text)

    exc_pattern = re.compile(
        r'^([A-Za-z0-9_.]*(?:Error|Exception|DoesNotExist|MultipleObjectsReturned|Warning|Fault))\s*:\s*(.+)$',
        re.MULTILINE,
    )
    exc_matches = exc_pattern.findall(text)
    exc_type = exc_matches[-1][0] if exc_matches else None
    exc_msg = exc_matches[-1][1].strip() if exc_matches else None

    if not frames and not exc_type:
        return None

    root = Path(get_project_root()).resolve()
    library_markers = (
        "site-packages", "dist-packages", ".venv", "venv", "env\\", "env/",
        "anaconda", "miniconda", "\\lib\\", "/lib/", "<frozen", "<string>",
    )

    project_frames = []
    for raw_path, line_str, func_name in frames:
        norm_p = raw_path.replace("\\", "/")
        if any(m in norm_p.lower() for m in library_markers):
            continue
        rel_candidate = None
        try:
            p_obj = Path(raw_path)
            if p_obj.is_absolute():
                try:
                    rel_candidate = p_obj.resolve().relative_to(root).as_posix()
                except ValueError:
                    parts = norm_p.split("/")
                    for i in range(len(parts)):
                        sub = "/".join(parts[i:])
                        if (root / sub).exists():
                            rel_candidate = sub
                            break
            else:
                if (root / norm_p).exists():
                    rel_candidate = norm_p
        except Exception:
            pass

        if rel_candidate:
            project_frames.append({
                "file": rel_candidate,
                "line": int(line_str),
                "func": func_name or "",
            })

    target_frame = project_frames[-1] if project_frames else None
    return {
        "target_frame": target_frame,
        "all_project_frames": project_frames,
        "exc_type": exc_type,
        "exc_msg": exc_msg,
    }


def coder_node(state: AgentState):
    logger.info("--- 💻 [WORKER: CODER] Executing ---")
    _log_model_provider("coder")
    subtask = state.get("worker_task", state["user_request"])
    
    messages = [
        SystemMessage(content=(
            "You are a code, data, and filesystem expert. Use your secure tools to inspect, read, and manipulate files or databases. "
            "You also have full access to search_internal_knowledge to consult project documentation, architecture guides, and requirements. "
            "To investigate any project keyword, concept, payment type, or feature end-to-end (searching variants across constants/models/views/migrations, checking URLs, mapping views and prefixes, and reading view implementations), use investigate_keyword. "
            "To trace a full client request from Project/urls.py to views/models, use trace_request_flow. "
            "To inspect, read, or explain the code of a specific function or class, use extract_function. "
            "To find where a function/class is imported, called, or referenced, use find_related_code. "
            "To inspect the overall architecture of a Python file, use analyze_module. "
            "You have direct access to the live UAT PostgreSQL database (`arya`) via `search_database_tables(keyword)`, `get_database_schema(table_name)`, and `safe_query_db(query)`. "
            "When asked questions about live database tables, records, users, counts, or data, use `search_database_tables` or `get_database_schema` to inspect the table columns, then execute read-only SQL queries with `safe_query_db`. "
            "You have run_terminal_command to execute shell commands autonomously (e.g. `git grep -n -i '<query>'`, `git log`, `git status`, `findstr`). "
            "You can freely use terminal commands whenever needed without waiting for the user to ask for them. "
            "For code searches across repository files, `git grep` via run_terminal_command is preferred for speed and accuracy."
        )),
        HumanMessage(content=f"Task: {subtask}")
    ]
    
    # Multi-turn context injection: if previous turn inspected a file or produced findings
    prior_summary = state.get("prior_execution_summary")
    last_file = state.get("last_accessed_file")
    last_content = state.get("last_file_content")
    if prior_summary or last_file:
        context_parts = ["--- Multi-Turn Conversation Context ---"]
        if last_file:
            context_parts.append(f"Most recently read/inspected file: {last_file}")
        if last_content and len(last_content) < 4000:
            context_parts.append(f"Content of {last_file}:\n{last_content}")
        elif prior_summary:
            context_parts.append(f"Previous answer / tool findings:\n{prior_summary[:2000]}")
        context_parts.append("Use this prior context if the user's request asks follow-up questions about it.")
        messages.append(SystemMessage(content="\n\n".join(context_parts)))

    is_retry = bool(state.get("retry_count", 0) > 0)
    if state.get("execution_result") and state.get("active_worker") == "coder":
        if is_retry:
            history = state.get("history", [])
            exec_res_str = str(state.get("execution_result", ""))
            if (
                "no function or class named" in exec_res_str.lower()
                and "tool:investigate_keyword" not in history
            ):
                kw = _extract_search_terms(state["user_request"])
                if kw:
                    logger.info(f"🔄 extract_function found no class/func named '{kw}'; falling back to investigate_keyword('{kw}')")
                    return {
                        "proposed_action": "TOOL_CALL: investigate_keyword",
                        "tool_payload": {"name": "investigate_keyword", "args": {"keyword": kw}},
                        "status": "running",
                    }
            logger.info(f"🔄 Coder self-correction active (retry {state['retry_count']}/2). Informing LLM to choose alternative.")
            feedback = (
                f"Previous Tool Execution Result:\n{state['execution_result']}\n\n"
                "SELF-CORRECTION INSTRUCTION:\n"
                "The previous tool call returned empty results or failed to find what was requested.\n"
                "1. DO NOT repeat the exact same search query or tool call.\n"
                "2. Analyze what failed and choose an ALTERNATIVE tool or query:\n"
                "   - If a code search/grep or extract_function failed, use investigate_keyword to search variants across constants, models, views, and URLs.\n"
                "   - Or inspect relevant files using safe_read_file, or list project folders using scan_directory.\n"
                "3. Call an alternative tool now to fulfill the user's request."
            )
            messages.append(HumanMessage(content=feedback))
        else:
            feedback = (
                f"Tool Output / Execution Result:\n{state['execution_result']}\n\n"
                "Summarize the data above in plain text so the final answer can be produced."
            )
            messages.append(HumanMessage(content=feedback))

    normalized_request = normalize_text_typos(state["user_request"]).lower()
    effective_task = state.get("worker_task", state["user_request"])

    # ── Fast path 0: Traceback / Exception / Terminal Error Debugging ────────
    if not is_retry and not state.get("execution_result"):
        tb_info = _extract_traceback_info(state["user_request"])
        if tb_info:
            target_frame = tb_info.get("target_frame")
            if target_frame:
                tb_file = target_frame["file"]
                tb_func = target_frame["func"]
                tb_line = target_frame["line"]
                if tb_func and tb_func not in ("<module>", "<lambda>", "<listcomp>", "<dictcomp>"):
                    logger.info(
                        f"🐞 Traceback debugger fast-path: extracting '{tb_func}' from '{tb_file}' (line {tb_line}) "
                        f"plus reading '{tb_file}'"
                    )
                    return {
                        "proposed_action": "TOOL_CALL: extract_function",
                        "tool_payload": {"name": "extract_function", "args": {"name": tb_func, "file_path": tb_file}},
                        "tool_queue": [{"name": "safe_read_file", "args": {"filename": tb_file}}],
                        "status": "running",
                    }
                else:
                    logger.info(f"🐞 Traceback debugger fast-path: reading '{tb_file}' (error at line {tb_line})")
                    return {
                        "proposed_action": "TOOL_CALL: safe_read_file",
                        "tool_payload": {"name": "safe_read_file", "args": {"filename": tb_file}},
                        "status": "running",
                    }
            elif tb_info.get("exc_msg"):
                exc_type = tb_info.get("exc_type", "Error")
                exc_msg = tb_info["exc_msg"]
                ident_m = re.search(r"'([A-Za-z0-9_]+)'", exc_msg)
                search_t = ident_m.group(1) if ident_m else _extract_search_terms(exc_msg)
                if search_t:
                    logger.info(f"🐞 Exception debugger fast-path: searching codebase for '{search_t}' ({exc_type})")
                    return {
                        "proposed_action": "TOOL_CALL: search_project_files",
                        "tool_payload": {"name": "search_project_files", "args": {"query": search_t}},
                        "status": "running",
                    }

    # If this is a self-correction retry, skip one-shot regex fast-paths and let LLM reason
    if not is_retry:
        pass  # allow fast-paths to evaluate below
    else:
        # Jump directly to LLM reasoning
        wants_read = False
        wants_columns = False
        wants_code_search = False

    # Fast path: Explicit terminal command execution
    # e.g. "run in terminal: git status", "terminal command: dir", "run command git grep drone"
    terminal_explicit = re.search(
        r"(?:run(?:\s+in)?\s+terminal(?:\s+command)?|execute(?:\s+in)?\s+terminal(?:\s+command)?|terminal\s+command|run\s+command|terminal:)\s*[:`'\"]?\s*([^`'\"]+)[`'\"]?",
        state["user_request"],
        re.IGNORECASE
    )
    if terminal_explicit and not state.get("execution_result"):
        raw_cmd = terminal_explicit.group(1).strip()
        logger.info(f"Explicit terminal command fast-path: '{raw_cmd}'")
        return {
            "proposed_action": "TOOL_CALL: run_terminal_command",
            "tool_payload": {"name": "run_terminal_command", "args": {"command": raw_cmd}},
            "status": "running",
        }

    # Fast path: Terminal search request
    # e.g. "use terminal to search for drone spray", "search in terminal for drone spray"
    terminal_search = re.search(
        r"(?:use|with|in)\s+terminal\s+(?:to\s+)?(?:search|find|grep|look\s+for)\s+(?:for\s+)?['\"]?([^'\"]+)['\"]?",
        state["user_request"],
        re.IGNORECASE
    )
    if terminal_search and not state.get("execution_result"):
        raw_target = terminal_search.group(1).strip()
        target = _extract_search_terms(raw_target) or raw_target
        split_camel = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', target)
        words = split_camel.split()
        if len(words) >= 2:
            regex_pattern = r"[-_ ]*".join(re.escape(w) for w in words)
            search_cmd = f'git grep -n -i -p -E "{regex_pattern}" -- "*.py" \':^*migrations*\' \':^*test*\''
        else:
            search_cmd = f'git grep -n -i -p "{target}" -- "*.py" \':^*migrations*\' \':^*test*\''
        logger.info(f"Terminal search fast-path: '{search_cmd}'")
        return {
            "proposed_action": "TOOL_CALL: run_terminal_command",
            "tool_payload": {"name": "run_terminal_command", "args": {"command": search_cmd}},
            "status": "running",
        }

    # Fast path: If the task requests reading or inspecting a specific spreadsheet or file
    # Ignore generic Django/Python framework filenames that appear in stack traces (e.g. exception.py, base.py)
    IGNORED_FRAMEWORK_FILES = {
        "exception.py", "base.py", "handlers.py", "wsgi.py", "asgi.py",
        "__init__.py", "threading.py", "socketserver.py", "selectors.py",
    }
    all_file_matches = re.findall(r"[A-Za-z0-9_./\\-]+\.(?:xlsx|csv|xls|py|json|md|txt|pdf|html|yaml|yml)", effective_task, re.IGNORECASE)
    valid_file_matches = [
        fm for fm in all_file_matches
        if os.path.basename(fm.replace("\\", "/")).lower() not in IGNORED_FRAMEWORK_FILES
        or ("/" in fm or "\\" in fm)
    ]
    file_match_str = valid_file_matches[0] if valid_file_matches else None
    wants_read = any(action in normalized_request for action in (
        "read", "go through", "open", "check sheet", "view", "inspect sheet",
        "tell me my", "what are my", "summarise", "summarize", "latest task",
        "check the issue", "check issue", "debug", "fix issue", "inspect this",
        "look at this", "look at the", "check this", "analyze this", "analyse this",
        "show me this", "what is in", "what's in", "check ", "explain", "why", "solution", "inspect",
        # Route/URL/API queries for a specific file
        "route", "routes", "url", "urls", "api", "apis", "endpoint", "endpoints", "list", "show", "get", "content", "contents",
    ))
    if not is_retry and file_match_str and wants_read and not state.get("execution_result"):
        filename = file_match_str.strip().strip("`").strip('"').strip("'")
        tool_to_call = "safe_read_spreadsheet" if any(filename.lower().endswith(ext) for ext in (".xlsx", ".csv", ".xls")) else "safe_read_file"
        logger.info(f"Direct file read fast-path: calling {tool_to_call} on '{filename}'")
        return {
            "proposed_action": f"TOOL_CALL: {tool_to_call}",
            "tool_payload": {"name": tool_to_call, "args": {"filename": filename}},
            "status": "running",
        }

    # Fast path: Bare "<name> file" without extension (e.g. "check constant file", "read settings file", "inspect constants file")
    bare_file_m = re.search(r"\b([a-zA-Z][a-zA-Z0-9_]{2,})\s+files?\b", normalized_request)
    if not is_retry and bare_file_m and wants_read and not state.get("execution_result"):
        bare_target = bare_file_m.group(1).lower()
        if bare_target not in {"this", "that", "the", "all", "project", "python", "code", "source", "some", "any", "which", "what"}:
            logger.info(f"Bare file request fast-path: finding files matching '{bare_target}'")
            return {
                "proposed_action": "TOOL_CALL: find_files",
                "tool_payload": {"name": "find_files", "args": {"query": bare_target}},
                "status": "running",
            }

    # ── "List all URLs / routes for a Django app" OR "Trace full app flow" fast-path ──
    # Handles: "list all the urls present in app farm urls", "routes in pop_manage",
    # AND upgrades deep flow/multi-endpoint/model+view+manager questions (e.g. "Trace a user's survey flow...")
    # to full multi-layer `trace_request_flow(cand)`.
    url_route_triggers = ("url", "urls", "route", "routes", "api", "apis", "endpoint", "endpoints", "flow")
    wants_urls_or_routes = any(t in normalized_request for t in url_route_triggers)
    if wants_urls_or_routes and not state.get("execution_result"):
        root = Path(get_project_root())
        stop_words_app = {
            "list", "all", "the", "present", "in", "from", "of", "for", "urls", "url",
            "routes", "route", "apis", "api", "endpoints", "endpoint", "what", "are",
            "show", "get", "tell", "trace", "user", "users", "this", "project", "compare",
            "post", "flow", "each", "identify", "its", "view", "views", "and", "behavior",
        }
        tokens = [w for w in re.findall(r"\b[a-zA-Z0-9_]+\b", normalized_request) if w not in stop_words_app]
        app_candidates = []
        if len(tokens) >= 2:
            joined = f"{tokens[0]}_{tokens[1]}"
            app_candidates.append(joined)
            if not joined.startswith("app_"):
                app_candidates.append(f"app_{joined}")
            app_candidates.append(f"{tokens[1]}_{tokens[0]}")
        for t in tokens:
            app_candidates.append(t)
            if not t.startswith("app_"):
                app_candidates.append(f"app_{t}")

        deep_flow_markers = (
            "trace", "flow", "compare", "difference", "versus", " vs ",
            "authentication", "permission", "model", "models", "manager",
            "serializer", "serializers", "twice", "submit", "submission",
            "available", "availability", "explain how", "what happens",
        )
        api_paths_in_req = re.findall(r"/(?:api|v\d+)/[A-Za-z0-9_/-]+", state["user_request"])
        is_deep_flow = any(m in normalized_request for m in deep_flow_markers) or len(api_paths_in_req) >= 2

        for cand in app_candidates:
            cand_urls = root / cand / "urls.py"
            cand_routes = root / cand / "routes.py"
            if cand_urls.exists() or cand_routes.exists():
                if is_deep_flow:
                    logger.info(f"Django app deep flow fast-path: running full multi-layer trace_request_flow on '{cand}'")
                    return {
                        "proposed_action": "TOOL_CALL: trace_request_flow",
                        "tool_payload": {"name": "trace_request_flow", "args": {"feature": cand}},
                        "status": "running",
                    }
                if cand_urls.exists():
                    logger.info(f"Django app URLs fast-path: reading '{cand}/urls.py'")
                    return {
                        "proposed_action": "TOOL_CALL: safe_read_file",
                        "tool_payload": {"name": "safe_read_file", "args": {"filename": f"{cand}/urls.py"}},
                        "status": "running",
                    }
                if cand_routes.exists():
                    logger.info(f"Django app routes fast-path: reading '{cand}/routes.py'")
                    return {
                        "proposed_action": "TOOL_CALL: safe_read_file",
                        "tool_payload": {"name": "safe_read_file", "args": {"filename": f"{cand}/routes.py"}},
                        "status": "running",
                    }

    # ── "List all application modules / apps" fast-path ────────────────────
    _APP_MODULE_TRIGGERS = (
        "application module", "application modules", "all modules", "app modules",
        "project modules", "codebase modules", "list modules", "list apps",
        "list applications", "all apps", "all applications", "django apps",
        "modules of", "apps of", "modules in", "apps in", "show modules", "show apps",
        "what modules", "what apps",
    )
    wants_app_modules = any(t in normalized_request for t in _APP_MODULE_TRIGGERS)
    if wants_app_modules and not state.get("execution_result"):
        logger.info("Coder fast-path: scanning root directory for application modules")
        return {
            "proposed_action": "TOOL_CALL: scan_directory",
            "tool_payload": {"name": "scan_directory", "args": {"directory_path": "."}},
            "status": "running",
        }

    code_search_triggers = (
        "where", "everywhere", "find every", "all occurrence", "all occurrences",
        "look into function", "look at function", "inspect function", "check function",
        "find function", "search function", "show function", "look into method", "find method",
        "look into class", "find class", "inspect class", "search for function",
        "all model", "model class", "model classes", "check all",
        "show all class", "list all class", "list class",
        # Count / enumerate questions
        "how many", "how much", "list all", "list every", "show all", "show every",
        "what are all", "what are the", "count all", "enumerate", "total number of",
    )
    wants_code_search = (
        any(trigger in normalized_request for trigger in code_search_triggers)
        or (
            any(term in normalized_request for term in ("where", "find", "search", "inspect", "look into", "check", "show", "list", "what", "which", "give", "tell", "explain"))
            and any(kind in normalized_request for kind in ("function", "functions", "method", "methods", "class", "classes", "endpoint", "endpoints", "api", "apis", "route", "routes", "def "))
        )
        or bool(re.search(r"/(?:api|v\d+)/[A-Za-z0-9_/-]+", state["user_request"]))
        or (
            # "how many X" pattern — count questions about code entities
            bool(re.search(r"\bhow\s+many\b", normalized_request))
        )
    )

    # ── Direct SQL query fast-path ──────────────────────────────────────────
    # Only match genuine SQL statements (never natural language "explain more about...")
    if re.match(
        r"^\s*(?:SELECT\s+.+\s+FROM\b|WITH\s+\w+\s+AS\s*\(|EXPLAIN\s+(?:ANALYZE\s+|VERBOSE\s+|\([^)]+\)\s+)?(?:SELECT|INSERT|UPDATE|DELETE|WITH)\b)",
        state["user_request"],
        re.IGNORECASE | re.DOTALL,
    ) and not state.get("execution_result"):
        logger.info("Routing direct SQL query to safe_query_db")
        return {
            "proposed_action": "TOOL_CALL: safe_query_db",
            "tool_payload": {"name": "safe_query_db", "args": {"query": state["user_request"]}},
            "status": "running",
        }

    # ── Database Table Search fast-path ──────────────────────────────────────
    # Handles: "Find all tables related to drone or task", "tables for farm", "search tables matching drone"
    table_search_match = re.search(
        r"(?:find|search|show|get|list)\s+(?:all\s+)?tables?\s+(?:related\s+to\s+|matching\s+|for\s+|with\s+|named\s+|about\s+)(.+)$",
        normalized_request
    )
    if table_search_match and not state.get("execution_result"):
        kw = table_search_match.group(1).strip().rstrip("?.!")
        logger.info(f"Routing table search request to search_database_tables(keyword='{kw}')")
        return {
            "proposed_action": "TOOL_CALL: search_database_tables",
            "tool_payload": {"name": "search_database_tables", "args": {"keyword": kw}},
            "status": "running",
        }

    # ── Database Tables Listing / Inspection fast-path ────────────────────────
    wants_all_tables = (
        ("table" in normalized_request or "tables" in normalized_request)
        and any(p in normalized_request for p in ("list all", "show all", "what tables", "which tables", "available tables", "all tables in", "list tables", "show tables", "get tables"))
        and any(db_word in normalized_request for db_word in ("database", "db", "postgres", "sql", "uat", "live"))
    )
    if wants_all_tables and not state.get("execution_result"):
        logger.info("Routing table list request to get_database_schema(table_name=None)")
        return {
            "proposed_action": "TOOL_CALL: get_database_schema",
            "tool_payload": {"name": "get_database_schema", "args": {}},
            "status": "running",
        }

    # Database Table Name Resolution: supports both explicit underscore names (e.g. "app_farm_user")
    # AND natural-language domain terms (e.g. "payment", "warehouses", "users", "weather data", "crop master")
    wants_db_schema = bool(re.search(r"\b(schemas?|columns?|datatypes?|data\s+types?|nullable|primary\s+keys?)\b", normalized_request))
    wants_model_code = bool(re.search(r"\b(models?|codebase|classes?|django)\b", normalized_request))
    explicit_db_word = bool(re.search(r"\b(tables?|database|db|postgres|postgresql|uat|sql)\b", normalized_request))
    wants_live_data = bool(
        re.search(r"\b(records?|rows?|counts?|entries|select|samples?|latest|how\s+many|top\s+\d+)\b", normalized_request)
        or (explicit_db_word and re.search(r"\bdata\b", normalized_request))
    )
    has_db_context = wants_db_schema or wants_live_data or explicit_db_word

    table_candidates = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*(?:_[a-zA-Z0-9]+)+\b", normalized_request)
    resolved_tables = []
    for cand in table_candidates:
        if not has_db_context and not wants_model_code and is_exact_code_symbol(cand):
            continue
        tbl_info = resolve_django_table(cand)
        if tbl_info and tbl_info["table_name"] not in [t["table_name"] for t in resolved_tables]:
            resolved_tables.append(tbl_info)

    # Natural-language table resolution when the user doesn't type exact underscore table names
    if has_db_context:
        _GRAMMAR_STOP = {
            "the", "a", "an", "my", "its", "this", "that", "these", "those", "all",
            "table", "tables", "model", "models", "column", "columns", "in", "for",
            "from", "of", "at", "and", "or", "to", "with", "by", "on", "is", "are",
            "was", "were", "be", "have", "has", "do", "does", "did", "schema", "schemas",
            "structure", "list", "show", "what", "which", "who", "where", "when", "why",
            "give", "get", "find", "check", "inspect", "tell", "me", "about", "class",
            "classes", "database", "db", "postgres", "postgresql", "live", "uat", "sql",
            "query", "row", "rows", "count", "counts", "total", "how", "many", "much",
            "sample", "samples", "latest", "recent", "top", "compare", "existing", "exist",
            "registered", "system", "codebase", "corresponding", "django", "missing",
            "named", "differently", "between", "there", "any", "they", "them", "their",
            "both", "each", "also", "can", "you", "please",
        }
        _SINGLE_WORD_STOP = _GRAMMAR_STOP | {
            "detail", "details", "record", "records", "data", "field", "fields",
            "status", "type", "types", "name", "names", "value", "values", "info",
            "entry", "entries", "item", "items", "master", "number", "numbers",
            "datatype", "datatypes", "nullable", "primary", "key", "keys",
        }
        raw_words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*\b", normalized_request)
        consumed_indices: set[int] = set()

        # 1. Try 4-word contiguous phrases (e.g. "one time record field", "celery task status")
        for i in range(len(raw_words) - 3):
            chunk = raw_words[i : i + 4]
            if any(w.lower() in _GRAMMAR_STOP for w in chunk):
                continue
            tbl_info = resolve_django_table("_".join(chunk))
            if tbl_info:
                consumed_indices.update((i, i + 1, i + 2, i + 3))
                if tbl_info["table_name"] not in [t["table_name"] for t in resolved_tables]:
                    resolved_tables.append(tbl_info)

        # 2. Try 3-word contiguous phrases (e.g. "buyer seller detail", "drone spray activity", "phone number map")
        for i in range(len(raw_words) - 2):
            if any((i + k) in consumed_indices for k in range(3)):
                continue
            chunk = raw_words[i : i + 3]
            if any(w.lower() in _GRAMMAR_STOP for w in chunk):
                continue
            tbl_info = resolve_django_table("_".join(chunk))
            if tbl_info:
                consumed_indices.update((i, i + 1, i + 2))
                if tbl_info["table_name"] not in [t["table_name"] for t in resolved_tables]:
                    resolved_tables.append(tbl_info)

        # 3. Try 2-word contiguous phrases (e.g. "crop master", "weather data", "project task", "admin log")
        for i in range(len(raw_words) - 1):
            if i in consumed_indices or (i + 1) in consumed_indices:
                continue
            chunk = raw_words[i : i + 2]
            if any(w.lower() in _GRAMMAR_STOP for w in chunk):
                continue
            tbl_info = resolve_django_table("_".join(chunk))
            if tbl_info:
                consumed_indices.update((i, i + 1))
                if tbl_info["table_name"] not in [t["table_name"] for t in resolved_tables]:
                    resolved_tables.append(tbl_info)

        # 4. Try remaining single words (e.g. "payment", "warehouses", "users", "farms", "organizations")
        for i, w in enumerate(raw_words):
            if i in consumed_indices or w.lower() in _SINGLE_WORD_STOP or len(w) < 3:
                continue
            tbl_info = resolve_django_table(w)
            if tbl_info and tbl_info["table_name"] not in [t["table_name"] for t in resolved_tables]:
                resolved_tables.append(tbl_info)

    if resolved_tables and not state.get("execution_result"):
        # 1. Cross-layer: User wants BOTH Live PostgreSQL Schema AND Django Model source code
        if (wants_db_schema or "postgres" in normalized_request or "database" in normalized_request) and wants_model_code and not wants_live_data:
            calls = []
            for rt in resolved_tables:
                calls.append({"name": "get_database_schema", "args": {"table_name": rt["table_name"]}})
                if rt.get("model_name") and rt.get("models_file"):
                    calls.append({"name": "extract_function", "args": {"name": rt["model_name"], "file_path": rt["models_file"]}})
            first = calls[0]
            logger.info(f"Cross-layer DB Schema + Django Model inspection queued for {[t['table_name'] for t in resolved_tables]}")
            return {
                "proposed_action": f"TOOL_CALL: {first['name']}",
                "tool_payload": first,
                "tool_queue": calls[1:],
                "status": "running",
            }

        # 2. Combined Schema + Live Data (e.g. "what are the columns and total records in farm?")
        if wants_db_schema and wants_live_data:
            calls = []
            for rt in resolved_tables:
                t_name = rt["table_name"]
                calls.append({"name": "get_database_schema", "args": {"table_name": t_name}})
                if any(c in normalized_request for c in ("count", "how many", "total")):
                    q = f'SELECT count(*) AS total_count FROM "{t_name}";'
                else:
                    q = f'SELECT * FROM "{t_name}" LIMIT 10;'
                calls.append({"name": "safe_query_db", "args": {"query": q}})
            first = calls[0]
            logger.info(f"Combined DB Schema + Live Records queued for {[t['table_name'] for t in resolved_tables]}")
            return {
                "proposed_action": f"TOOL_CALL: {first['name']}",
                "tool_payload": first,
                "tool_queue": calls[1:],
                "status": "running",
            }

        # 3. Database Schema / Columns Inspection
        if wants_db_schema:
            calls = [{"name": "get_database_schema", "args": {"table_name": rt["table_name"]}} for rt in resolved_tables]
            first = calls[0]
            logger.info(f"Inspecting live database schema for tables: {[t['table_name'] for t in resolved_tables]}")
            return {
                "proposed_action": "TOOL_CALL: get_database_schema",
                "tool_payload": first,
                "tool_queue": calls[1:],
                "status": "running",
            }

        # 4. Live Database Data / Row Counts
        if wants_live_data:
            calls = []
            for rt in resolved_tables:
                t_name = rt["table_name"]
                if any(c in normalized_request for c in ("count", "how many", "total")):
                    q = f'SELECT count(*) AS total_count FROM "{t_name}";'
                else:
                    q = f'SELECT * FROM "{t_name}" LIMIT 10;'
                calls.append({"name": "safe_query_db", "args": {"query": q}})
            first = calls[0]
            logger.info(f"Querying live database records for tables: {[t['table_name'] for t in resolved_tables]}")
            return {
                "proposed_action": "TOOL_CALL: safe_query_db",
                "tool_payload": first,
                "tool_queue": calls[1:],
                "status": "running",
            }

        # 5. Django Model Class Inspection
        wants_classes = any(w in normalized_request for w in ("class", "classes", "model", "models", "def", "structure", "detail", "details", "explain", "what is", "show", "list"))
        if wants_classes:
            calls = [{"name": "extract_function", "args": {"name": rt["model_name"], "file_path": rt["models_file"]}} for rt in resolved_tables]
            first = calls[0]
            logger.info(f"Resolved database tables {[t['table_name'] for t in resolved_tables]} -> Django models")
            return {
                "proposed_action": "TOOL_CALL: extract_function",
                "tool_payload": first,
                "tool_queue": calls[1:],
                "status": "running",
            }

    # ── "List all columns / fields in X table/model" fast-path ────────────
    # Handles: "list all columns in app farm table", "show fields in user_roles model",
    #          "list all the columns in verification signzyauth class", "schema of user model"
    _COLUMN_PHRASES = (
        "list all column", "list all the column", "list all field", "list all the field",
        "show all column", "show all field", "show column", "show field",
        "what column", "what field", "which column", "which field",
        "all column", "all field", "column in", "field in", "columns in", "fields in",
        "structure of", "schema of",
    )
    wants_columns = any(p in normalized_request for p in _COLUMN_PHRASES)

    if wants_columns and not state.get("execution_result"):
        effective = state.get("worker_task", state["user_request"])
        root_path = Path(get_project_root())
        _MOD_STOP = {
            "the", "a", "an", "my", "its", "this", "that", "app", "all", "table",
            "tables", "model", "models", "column", "columns", "field", "fields",
            "in", "for", "from", "of", "at", "and", "or", "schema", "structure",
            "list", "show", "what", "which", "give", "get", "class", "classes",
            "database", "db", "detail", "details",
        }

        # 1. Check if user explicitly mentioned a CamelCase class name (e.g. SignzyAuth, POPGroup)
        camels = [c for c in re.findall(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b", effective)
                  if c.lower() not in {"apiview", "view", "model", "serializer"}]
        if camels:
            cls_target = camels[0]
            tbl_res = resolve_django_table(cls_target)
            file_p = tbl_res["models_file"] if tbl_res else ""
            logger.info(f"Column/field list: extracting class '{cls_target}' (file: {file_p})")
            return {
                "proposed_action": "TOOL_CALL: extract_function",
                "tool_payload": {"name": "extract_function", "args": {"name": cls_target, "file_path": file_p}},
                "status": "running",
            }

        tokens = [w for w in re.findall(r"\b[a-zA-Z0-9_]+\b", effective) if w.lower() not in _MOD_STOP]

        # 2. Check if any token or joined pair resolves to a Django model via resolve_django_table
        for tok in tokens:
            tbl_res = resolve_django_table(tok)
            if tbl_res:
                logger.info(f"Column/field list: resolved table '{tok}' -> model '{tbl_res['model_name']}' in '{tbl_res['models_file']}'")
                return {
                    "proposed_action": "TOOL_CALL: extract_function",
                    "tool_payload": {"name": "extract_function", "args": {"name": tbl_res["model_name"], "file_path": tbl_res["models_file"]}},
                    "status": "running",
                }

        if len(tokens) >= 2:
            joined_pair = f"{tokens[0]}_{tokens[1]}"
            tbl_res = resolve_django_table(joined_pair)
            if tbl_res:
                logger.info(f"Column/field list: resolved pair '{joined_pair}' -> model '{tbl_res['model_name']}' in '{tbl_res['models_file']}'")
                return {
                    "proposed_action": "TOOL_CALL: extract_function",
                    "tool_payload": {"name": "extract_function", "args": {"name": tbl_res["model_name"], "file_path": tbl_res["models_file"]}},
                    "status": "running",
                }

        # 3. Check if any token or joined pair corresponds to an actual directory on disk with models.py
        for tok in tokens:
            if (root_path / tok / "models.py").exists():
                mod_p = f"{tok}/models.py"
                logger.info(f"Column/field list: analyze_module('{mod_p}') for app directory '{tok}'")
                return {
                    "proposed_action": "TOOL_CALL: analyze_module",
                    "tool_payload": {"name": "analyze_module", "args": {"file_path": mod_p}},
                    "status": "running",
                }
            elif (root_path / f"app_{tok}" / "models.py").exists():
                mod_p = f"app_{tok}/models.py"
                logger.info(f"Column/field list: analyze_module('{mod_p}') for app directory 'app_{tok}'")
                return {
                    "proposed_action": "TOOL_CALL: analyze_module",
                    "tool_payload": {"name": "analyze_module", "args": {"file_path": mod_p}},
                    "status": "running",
                }

        if len(tokens) >= 2:
            joined_pair = f"{tokens[0]}_{tokens[1]}"
            if (root_path / joined_pair / "models.py").exists():
                mod_p = f"{joined_pair}/models.py"
                logger.info(f"Column/field list: analyze_module('{mod_p}') for app directory '{joined_pair}'")
                return {
                    "proposed_action": "TOOL_CALL: analyze_module",
                    "tool_payload": {"name": "analyze_module", "args": {"file_path": mod_p}},
                    "status": "running",
                }

        # 4. Check prior turn context subject
        prior = state.get("prior_execution_summary") or ""
        prior_sub = _extract_prior_subject(prior)
        if prior_sub:
            tbl_res = resolve_django_table(prior_sub)
            if tbl_res:
                return {
                    "proposed_action": "TOOL_CALL: extract_function",
                    "tool_payload": {"name": "extract_function", "args": {"name": tbl_res["model_name"], "file_path": tbl_res["models_file"]}},
                    "status": "running",
                }
            if (root_path / prior_sub / "models.py").exists():
                mod_p = f"{prior_sub}/models.py"
                return {
                    "proposed_action": "TOOL_CALL: analyze_module",
                    "tool_payload": {"name": "analyze_module", "args": {"file_path": mod_p}},
                    "status": "running",
                }

        # 5. Fallback: try extract_function on the first non-stopword token if present
        if tokens:
            logger.info(f"Column/field list: fallback extract_function('{tokens[0]}')")
            return {
                "proposed_action": "TOOL_CALL: extract_function",
                "tool_payload": {"name": "extract_function", "args": {"name": tokens[0]}},
                "status": "running",
            }

        # 6. Final fallback: broad Django field grep
        search_cmd = (
            'git grep -n -i -p -E '
            '"models\\.(Char|Integer|Float|Boolean|Date|Foreign|Many|One|Text|Auto|Big|Small|UUID|Email|URL|JSON|File|Image)Field" '
            '-- "*.py" \':^*migrations*\' \':^*test*\''
        )
        logger.info("Column/field list fallback: broad Django field grep")
        return {
            "proposed_action": "TOOL_CALL: run_terminal_command",
            "tool_payload": {"name": "run_terminal_command", "args": {"command": search_cmd}},
            "status": "running",
        }

    # Smart Django model class detection: "check all model classes", "find class models", "models related to this"
    is_model_class_query = (
        any(
            phrase in normalized_request
            for phrase in (
                "model class", "model classes", "class model", "class models",
                "all model", "all models", "all classes", "check all class", "list all class",
                "models related", "model related", "related model", "related models",
                "classes in model", "models in"
            )
        )
        or (
            ("model" in normalized_request or "models" in normalized_request)
            and any(w in normalized_request for w in ("class", "classes", "find", "show", "list", "check", "detail", "details"))
        )
    )
    has_api_mention = any(w in normalized_request for w in ("api", "apis", "endpoint", "endpoints", "url", "urls", "route", "routes", "view", "views"))

    if is_model_class_query and not has_api_mention and not state.get("execution_result"):
        effective = state.get("worker_task", state["user_request"])
        _MOD_STOP2 = {
            "the", "a", "an", "my", "its", "this", "that", "all", "in", "for",
            "from", "of", "at", "and", "or", "referring", "to",
        }
        module_match = re.search(r"\b(?:in|referring to|for|from)\s+([a-zA-Z][a-zA-Z0-9_]+)\b", effective)
        module_name_raw = module_match.group(1) if module_match else ""
        # Discard generic stopword matches (e.g. "in the" -> "the")
        if module_name_raw.lower() in _MOD_STOP2:
            module_name_raw = ""

        prior_sub = _extract_prior_subject(
            state.get("prior_execution_summary") or state.get("execution_result", "")
        )
        search_terms = _extract_search_terms(effective)

        # Only restrict to pure 'class ' in models.py if an app scope is known or query has no specific keyword
        if module_name_raw or prior_sub or not search_terms:
            module_name = module_name_raw or prior_sub or ""
            file_filter = f"{module_name}/models.py" if module_name else "models.py"
            search_query = "class "
            logger.info(f"Smart model class search: query='{search_query}', file_filter='{file_filter}'")
            return {
                "proposed_action": "TOOL_CALL: search_project_files",
                "tool_payload": {"name": "search_project_files", "args": {"query": search_query, "file_filter": file_filter}},
                "search_terms": search_query,
                "status": "running",
            }

    # ── Code Intelligence Fast-Paths ───────────────────────────────────────
    # 1. Index code structure into vector database
    if any(p in normalized_request for p in ("index code", "index project code", "learn code structure", "reindex code")) and not state.get("execution_result"):
        logger.info("Code intelligence: indexing codebase structure via AST")
        return {
            "proposed_action": "TOOL_CALL: index_project_code",
            "tool_payload": {"name": "index_project_code", "args": {}},
            "status": "running",
        }

    # 2. Module structure analysis: e.g. "analyze module pop_manage/models.py", "analyze MainActivity.kt"
    module_req = re.search(
        r"(?:analyze|structure of|inspect module|inspect file)\s+([A-Za-z0-9_./\\]+\.(?:py|kt|kts|java|cpp|c|cc|cxx|h|hpp|cs|ts|tsx|js|jsx|go|rs|dart|swift))\b",
        normalized_request
    )
    if module_req and not state.get("execution_result"):
        mod_path = module_req.group(1).replace("\\", "/")
        logger.info(f"Code intelligence: analyzing module structure for '{mod_path}'")
        return {
            "proposed_action": "TOOL_CALL: analyze_module",
            "tool_payload": {"name": "analyze_module", "args": {"file_path": mod_path}},
            "status": "running",
        }

    # 3. Related code & callers: e.g. "who calls AddSoilTestPlotAPIView", "related code for X"
    related_triggers = ("related code", "who calls", "callers of", "where is imported", "find callers", "find references to", "references of")
    if any(t in normalized_request for t in related_triggers) and not state.get("execution_result"):
        target_name = _extract_search_terms(state["user_request"])
        if target_name:
            first_ident = target_name.split()[0]
            logger.info(f"Code intelligence: tracing related code for '{first_ident}'")
            return {
                "proposed_action": "TOOL_CALL: find_related_code",
                "tool_payload": {"name": "find_related_code", "args": {"name": first_ident}},
                "status": "running",
            }

    # 3.4 Request Flow Tracing: e.g. "explain how a client request reaches and passes through this Django project", "trace request from url"
    trace_triggers = ("trace it from", "trace request", "request reaches", "passes through this", "client request reaches", "project-level url", "request flow", "trace the request")
    if any(t in normalized_request for t in trace_triggers) and not state.get("execution_result"):
        feature_arg = _extract_search_terms(state["user_request"]) or "farm_category"
        if "other than soil" in normalized_request or feature_arg.lower() in {"client", "request", "django", "project"}:
            feature_arg = "farm_category"
        logger.info(f"Code intelligence: tracing request flow for feature '{feature_arg}'")
        return {
            "proposed_action": "TOOL_CALL: trace_request_flow",
            "tool_payload": {"name": "trace_request_flow", "args": {"feature": feature_arg}},
            "status": "running",
        }

    # 3.45 Comparison fast-path: e.g. "difference between AddSoilTestPlotAPIView and NeoPerkWebhookReceiver"
    comparison_triggers = ("difference between", "difference in", "compare", "versus", " vs ", "how do they differ", "difference in functionality")
    wants_comparison = any(t in normalized_request for t in comparison_triggers)
    if wants_comparison and not state.get("execution_result"):
        camels = [c for c in re.findall(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b", state["user_request"])
                  if sum(1 for ch in c if ch.isupper()) >= 2 and c.lower() not in {"apiview", "view", "model", "serializer"}]
        snakes = [s for s in re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*(?:_[a-zA-Z0-9]+)+\b", state["user_request"])
                  if s.lower() not in {"api_key", "secret_key"}]
        targets = camels if len(camels) >= 2 else (snakes if len(snakes) >= 2 else (camels + snakes))
        if len(targets) >= 2:
            combined_names = ", ".join(targets[:2])
            logger.info(f"Code intelligence: extracting comparison targets '{combined_names}'")
            return {
                "proposed_action": "TOOL_CALL: extract_function",
                "tool_payload": {"name": "extract_function", "args": {"name": combined_names}},
                "status": "running",
            }

    # 3.48 Follow-up Function/Method Extraction from Prior Turn
    # Handles: "show me function related to it", "show functions in it", "explain the function for this"
    is_referential_followup = _is_implicit_followup(state["user_request"]) or any(
        p in normalized_request
        for p in ("related to it", "related to this", "related to that", "in it", "for it", "of it", "about it", "from it", "this function", "these functions")
    )
    mentions_func_kind = bool(re.search(r"\b(?:functions?|methods?|classes?|def)\b", normalized_request))
    if not is_retry and not state.get("execution_result") and is_referential_followup and prior_summary:
        if mentions_func_kind:
            prior_funcs = _extract_prior_functions(prior_summary)
            if prior_funcs:
                combined_prior = ", ".join(prior_funcs[:2])
                logger.info(f"Code intelligence: extracting prior-turn related function(s) '{combined_prior}'")
                return {
                    "proposed_action": "TOOL_CALL: extract_function",
                    "tool_payload": {"name": "extract_function", "args": {"name": combined_prior}},
                    "status": "running",
                }

    # 3.5 Specific Function/Method/Class extraction & explanation
    # Handles: "add_soil_test_plot_data explain this function", "explain function X",
    #          "what does AddSoilTestPlotAPIView do", "explain add_soil_test_plot"
    has_api_url_path = bool(re.search(r"/(?:api|v\d+)/[A-Za-z0-9_/-]+", state["user_request"]))
    is_broad_func_list = bool(
        re.search(r"\b(?:used\s+for|related\s+to|all\s+functions|functions\s+in|methods\s+in|classes\s+in)\b", normalized_request)
    )
    if not is_retry and not state.get("execution_result") and not has_api_url_path and not is_broad_func_list:
        target_name = _extract_search_terms(state["user_request"])
        first_ident = target_name.split()[0] if target_name else ""
        is_exact_sym = bool(first_ident and is_exact_code_symbol(first_ident))
        explicitly_typed = bool(first_ident and first_ident.lower() in state["user_request"].lower())

        if first_ident and (
            (mentions_func_kind and (is_exact_sym or "_" in first_ident or any(c.isupper() for c in first_ident[1:])))
            or (is_exact_sym and explicitly_typed)
        ):
            logger.info(f"Code intelligence: extracting specific symbol definition for '{first_ident}'")
            return {
                "proposed_action": "TOOL_CALL: extract_function",
                "tool_payload": {"name": "extract_function", "args": {"name": first_ident}},
                "status": "running",
            }

    # 4. End-to-End Keyword / Feature / API Investigation (e.g. "tell me about soil test", "explain more about this api /api/v1/record/create_assistant/")
    investigate_triggers = (
        "trace ", "investigate ", "tell me about ", "what is ", "where is ",
        "how is ", "explain ", "check ", "details of ", "information about ",
        "everything about ", "flow of ", "api for ", "endpoint for ", "route for ",
    )
    non_code_words = ("table", "column", "field", "sheet", "spreadsheet", "folder", "directory")
    if (
        not is_retry
        and not state.get("execution_result")
        and (any(t in normalized_request for t in investigate_triggers) or wants_code_search)
        and not any(ef in normalized_request for ef in non_code_words)
    ):
        kw_target = _extract_search_terms(state["user_request"])
        if not kw_target and prior_summary:
            kw_target = _extract_prior_subject(prior_summary) or ""
        is_count_query = bool(re.search(r"\b(?:how\s+many|how\s+much|count\s+all|total\s+number)\b", normalized_request))
        if kw_target and len(kw_target) >= 3 and not is_count_query and kw_target.lower() not in {"this", "that", "project", "codebase", "dashboard", "aryaq"}:
            logger.info(f"Code intelligence: running structured keyword investigation for '{kw_target}'")
            return {
                "proposed_action": "TOOL_CALL: investigate_keyword",
                "tool_payload": {"name": "investigate_keyword", "args": {"keyword": kw_target}},
                "search_terms": kw_target,
                "status": "running",
            }

    if wants_code_search and not state.get("execution_result"):
        search_terms = _extract_search_terms(state["user_request"])
        if not search_terms and prior_summary:
            search_terms = _extract_prior_subject(prior_summary) or ""
        if search_terms:
            # Autonomous terminal tool selection for count/enumeration queries:
            root = Path(get_project_root())
            split_camel = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', search_terms)
            words = split_camel.split()
            if len(words) >= 2:
                regex_pattern = r"[-_ ]*".join(re.escape(w) for w in words)
            else:
                regex_pattern = re.escape(search_terms)

            if (root / ".git").exists():
                cmd = f'git grep -n -i -p -E "{regex_pattern}" -- "*.py" \':^*migrations*\' \':^*test*\''
                logger.info(f"Autonomous terminal search: '{cmd}'")
                return {
                    "proposed_action": "TOOL_CALL: run_terminal_command",
                    "tool_payload": {"name": "run_terminal_command", "args": {"command": cmd}},
                    "search_terms": search_terms,
                    "status": "running",
                }
            return {
                "proposed_action": "TOOL_CALL: search_project_files",
                "tool_payload": {"name": "search_project_files", "args": {"query": search_terms}},
                "search_terms": search_terms,
                "status": "running",
            }
        
    response = coder_llm.invoke(messages)

    if response.tool_calls:
        # Fix #1 – process ALL tool calls, not just [0]
        enriched_calls = []
        for raw_call in response.tool_calls:
            t_name = raw_call.get("name")
            t_args = dict(raw_call.get("args") or {})
            if t_name == "search_project_files":
                search_terms = _extract_search_terms(state["user_request"])
                if search_terms:
                    t_args["query"] = search_terms
            if t_name == "search_internal_knowledge" and not t_args.get("query"):
                t_args["query"] = _extract_search_terms(state["user_request"]) or state["user_request"]
            if t_name == "scan_directory":
                requested_directory = _requested_directory_path(state["user_request"])
                if requested_directory:
                    t_args["directory_path"] = requested_directory
            enriched_calls.append({"name": t_name, "args": t_args})

        first_call = enriched_calls[0]
        tool_name = first_call["name"]
        tool_args = first_call["args"]

        # ── Sanity guard: if LLM chose a passive status command (git status / git log) on initial attempt,
        #    redirect to a proper git grep using the extracted search terms instead. ──
        _PASSIVE_TERMINAL_CMDS = ("git status", "git log", "git branch", "git diff", "echo")
        if not is_retry and tool_name == "run_terminal_command":
            cmd_proposed = tool_args.get("command", "").strip().lower()
            if any(cmd_proposed == uc or cmd_proposed.startswith(uc + " ") for uc in _PASSIVE_TERMINAL_CMDS):
                search_terms = _extract_search_terms(state["user_request"])
                if search_terms:
                    split_camel = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', search_terms)
                    words = split_camel.split()
                    if len(words) >= 2:
                        regex_pattern = r"[-_ ]*".join(re.escape(w) for w in words)
                    else:
                        regex_pattern = re.escape(search_terms)
                    redirect_cmd = f'git grep -n -i -p -E "{regex_pattern}" -- "*.py" \':^*migrations*\' \':^*test*\''
                    logger.warning(
                        f"⚠️ LLM proposed passive command '{cmd_proposed}'; redirecting to: {redirect_cmd}"
                    )
                    tool_args["command"] = redirect_cmd
                    first_call["args"] = tool_args

        if _should_stop_repeating_tool(state, tool_name):
            logger.warning(
                f"⚠️ Worker attempted a repeated tool call ({tool_name}) after a prior result; "
                "forcing a final response instead of looping."
            )
            outputs = state.get("worker_outputs", {})
            outputs["coder"] = state.get("execution_result", "No data returned from the last tool call.")
            return {"worker_outputs": outputs, "proposed_action": "RESPOND_ONLY", "status": "running",
                    "tool_queue": []}

        logger.info(
            f"Coder queuing {len(enriched_calls)} tool call(s): "
            + ", ".join(c["name"] for c in enriched_calls)
        )
        return {
            "proposed_action": f"TOOL_CALL: {tool_name}",
            "tool_payload": {"name": tool_name, "args": tool_args},
            "tool_queue": enriched_calls[1:],   # remaining calls for later passes
            "status": "running",
        }

    outputs = state.get("worker_outputs", {})
    outputs["coder"] = response.content
    return {"worker_outputs": outputs, "proposed_action": "RESPOND_ONLY", "tool_queue": []}

# -------------------------------------------------------------
# 4. APPROVAL & TOOL EXECUTION NODES
# -------------------------------------------------------------
def require_approval(state: AgentState):
    if state.get("proposed_action") == "RESPOND_ONLY":
        return {"status": "running"}

    tool_name = state.get("tool_payload", {}).get("name")
    if tool_name in READ_ONLY_TOOLS:
        logger.info(f"✅ Auto-approved read-only tool: {tool_name}")
        return {"human_approval": "approve", "status": "running"}

    if tool_name == "run_terminal_command":
        cmd = state.get("tool_payload", {}).get("args", {}).get("command", "")
        if is_safe_read_only_command(cmd):
            logger.info(f"✅ Auto-approved safe read-only terminal command: {cmd}")
            return {"human_approval": "approve", "status": "running"}
        
    logger.info("--- ⏸️ [NODE: GATEWAY] Pausing for Human Approval ---")
    decision = interrupt(f"Alert: AI requests to execute {state['proposed_action']}.")
    
    if str(decision).lower() == "kill":
        return {"human_approval": decision, "status": "killed"}
        
    return {"human_approval": decision, "status": "running"}

def execute_action(state: AgentState):
    if state.get("proposed_action") == "RESPOND_ONLY":
        return {"status": "running"}

    if state.get("human_approval", "").lower() != "approve":
        logger.warning("❌ Tool execution aborted by user.")
        return {"execution_result": "Aborted by user.", "status": "running", "tool_queue": [], "proposed_action": "RESPOND_ONLY", "tool_payload": {}}

    payload = state.get("tool_payload", {})
    tool_name = payload.get("name")
    tool_attempts = dict(state.get("tool_attempts", {}))
    tool_attempts[tool_name] = tool_attempts.get(tool_name, 0) + 1

    try:
        args = _validate_tool_args(tool_name, payload.get("args", {}))
    except ValueError as error:
        logger.error(f"⚠️ Invalid arguments for {tool_name}: {error}")
        history = list(state.get("history", []))
        history.append(f"tool:{tool_name}:invalid_args")
        return {
            "execution_result": f"Invalid tool arguments: {error}",
            "tool_attempts": tool_attempts,
            "tool_results": {
                **state.get("tool_results", {}),
                tool_name: f"Invalid tool arguments: {error}",
            },
            "status": "running",
            "history": history,
            "tool_queue": [],
            "proposed_action": "RESPOND_ONLY",
            "tool_payload": {},
            "retry_count": state.get("retry_count", 0) + 1,
        }

    # --- Fix #2: registry-based logging (no elif chain) ---
    logger.info("--- ⚙️ [NODE: EXECUTOR] Running Tool ---")
    log_fn = _TOOL_LOG.get(tool_name)
    logger.info(log_fn(args) if log_fn else f"🔧 EXECUTING TOOL: {tool_name} | Args: {args}")

    # --- Fix #2: registry-based dispatch (no elif chain) ---
    try:
        tool_fn = TOOL_REGISTRY.get(tool_name)
        if tool_fn is None:
            result = f"Unknown tool: {tool_name}"
        else:
            result = tool_fn(**args)
    except Exception as e:
        logger.error(f"⚠️ Tool execution failed: {str(e)}")
        result = f"Error executing tool: {str(e)}"

    history = list(state.get("history", []))
    history.append(f"tool:{tool_name}")
    tool_results = dict(state.get("tool_results", {}))
    res_text = str(result)
    tool_results[tool_name] = res_text
    logger.info(f"✅ Tool execution complete. Length of result: {len(res_text)} characters.")
    preview_flat = " | ".join(line.strip() for line in res_text.splitlines() if line.strip())[:240]
    logger.info(f"📄 Tool output preview ({tool_name}): {preview_flat}")
    try:
        dump_path = Path(__file__).resolve().parent.parent / "data" / "logs" / "last_tool_output.txt"
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(
            f"=== TOOL: {tool_name} | ARGS: {args} | LENGTH: {len(res_text)} chars ===\n\n{res_text}\n",
            encoding="utf-8",
        )
    except Exception:
        pass

    last_file = state.get("last_accessed_file")
    last_content = state.get("last_file_content")
    if tool_name in ("safe_read_file", "safe_read_spreadsheet") and isinstance(result, str) and not result.startswith("Error"):
        last_file = args.get("filename")
        last_content = str(result)[:8000]
    elif tool_name in ("analyze_module", "extract_function") and isinstance(result, str) and not result.startswith("Error"):
        if args.get("file_path"):
            last_file = args.get("file_path")

    # --- Fix #1 (drain): if more calls are queued, pop the next one and keep
    # the graph looping through approve → execute without re-invoking the LLM. ---
    prev_exec = state.get("execution_result") or ""
    if prev_exec and state.get("retry_count", 0) == 0:
        accumulated_result = f"{prev_exec}\n\n---\n\n{result}"
    else:
        accumulated_result = str(result)

    remaining_queue: list = list(state.get("tool_queue", []))
    if remaining_queue:
        next_call = remaining_queue.pop(0)
        next_tool_name = next_call["name"]
        logger.info(f"🔁 Draining tool queue — next call: {next_tool_name} ({len(remaining_queue)} remaining)")
        return {
            "execution_result": accumulated_result,
            "tool_attempts": tool_attempts,
            "tool_results": tool_results,
            "model_provider": PRIMARY_MODEL_PROVIDER,
            "status": "running",
            "history": history,
            "tool_queue": remaining_queue,
            "last_accessed_file": last_file,
            "last_file_content": last_content,
            # Set up the next tool for the approval node
            "proposed_action": f"TOOL_CALL: {next_tool_name}",
            "tool_payload": next_call,
            "human_approval": "",             # reset so approval re-runs for next tool
        }

    res_str = str(result).strip().lower()
    is_empty_or_failed = (
        not res_str
        or "0 matches" in res_str
        or "no matches found" in res_str
        or "no matching results" in res_str
        or "no occurrences of" in res_str
        or "does not exist in this repository" in res_str
        or "no function or class named" in res_str
        or "no references to" in res_str
        or "exit code: 1" in res_str
        or "not found in the project" in res_str
        or "error reading file" in res_str
        or "error executing tool" in res_str
        or "error scanning directory" in res_str
        or "directory not found" in res_str
        or "file not found" in res_str
        or "invalid tool arguments" in res_str
    )
    current_retries = state.get("retry_count", 0)
    new_retries = (current_retries + 1) if is_empty_or_failed else 0

    return {
        "execution_result": accumulated_result,
        "tool_attempts": tool_attempts,
        "tool_results": tool_results,
        "model_provider": PRIMARY_MODEL_PROVIDER,
        "status": "running",
        "history": history,
        "tool_queue": [],
        "retry_count": new_retries,
        "last_accessed_file": last_file,
        "last_file_content": last_content,
        "proposed_action": "",
        "tool_payload": {},
    }