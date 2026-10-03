import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from src.utils.logger import setup_logger

load_dotenv()
logger = setup_logger("network-tools")


def get_configured_dashboard_endpoints() -> list[dict[str, str]]:
    """Return all configured dashboard endpoints for memory syncing."""
    candidates: list[str] = []

    for key in ["DASHBOARD_URL", "DASHBOARD_SUMMARY_URL", "ARYAQ_LOCAL_URL"]:
        value = os.getenv(key)
        if value:
            candidates.append(value.strip().rstrip("/"))

    if not candidates:
        candidates.append("http://127.0.0.1:8000")

    seen: set[str] = set()
    endpoints: list[dict[str, str]] = []
    for url in candidates:
        if url in seen:
            continue
        seen.add(url)

        normalized = url if url.startswith("http") else f"https://{url}"
        summary_url = normalized
        if "/api/" not in normalized.lower() and "/dashboard" not in normalized.lower() and not normalized.lower().endswith("/summary"):
            summary_url = f"{normalized}/api/dashboard/summary"

        endpoints.append({
            "name": "Configured Dashboard",
            "url": normalized,
            "summary_url": summary_url,
        })

    return endpoints


def _pick_summary_value(data: Any, preferred_keys: list[str]) -> str | None:
    if isinstance(data, dict):
        for key in preferred_keys:
            if key in data and data[key] not in (None, ""):
                return str(data[key])
        for key in ["title", "name", "dashboard", "app_name"]:
            if key in data and data[key] not in (None, ""):
                return str(data[key])
    return None


def summarize_dashboard_response(url: str, raw_body: str, response_status: int = 200, response_reason: str | None = None) -> str:
    """Normalize dashboard payloads to a compact, search-friendly summary."""
    text = (raw_body or "").strip()
    if not text:
        return (
            f"Dashboard at {url} returned an empty body. "
            f"Status: {response_status}. "
            f"Reason: {response_reason or 'unknown'}."
        )

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            summary_fields: list[str] = [f"url: {url}"]

            status = _pick_summary_value(payload, ["status", "state", "app_status", "health", "dashboard_status"])
            if status:
                summary_fields.append(f"status: {status}")

            title = _pick_summary_value(payload, ["title", "name", "dashboard", "page_title"])
            if title:
                summary_fields.append(f"title: {title}")

            updated = _pick_summary_value(payload, ["last_updated", "updated_at", "updatedAt", "timestamp", "time", "lastUpdated"])
            if updated:
                summary_fields.append(f"last_updated: {updated}")

            error = _pick_summary_value(payload, ["error", "error_message", "warning", "loading_state", "state_message"])
            if error:
                summary_fields.append(f"error_or_loading: {error}")

            if not summary_fields[1:]:
                preview = json.dumps(payload, ensure_ascii=False, default=str)[:1500]
                summary_fields.append(f"raw_payload: {preview}")

            return "; ".join(summary_fields)

        if isinstance(payload, list):
            return f"Dashboard at {url} returned a list with {len(payload)} items. Preview: {json.dumps(payload[:3], ensure_ascii=False, default=str)[:1200]}"

    except Exception:
        pass

    if text.lower().startswith("<!doctype") or "<html" in text.lower():
        title_match = re.search(r"<title>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else "Dashboard"
        return (
            f"Dashboard at {url} served HTML instead of JSON. "
            f"Page title: {title}. "
            f"Status: {response_status}. "
            f"Note: this endpoint is reachable but returned a web page, not structured metrics."
        )

    if len(text) > 3000:
        return f"Dashboard at {url} returned a non-JSON text payload. Preview: {text[:3000]}..."

    return f"Dashboard at {url} returned a text payload. Status: {response_status}. Content: {text[:1200]}"


import asyncio
import atexit
import queue
import sys
import threading

# Keep live dashboard browser & Streamlit WebSocket session open for 10 minutes (600s) by default
DASHBOARD_SESSION_TTL = int(os.getenv("DASHBOARD_SESSION_TTL", "600"))


class _DashboardSessionManager:
    """
    Maintains a persistent Playwright Chromium browser and authenticated Streamlit page
    in a dedicated worker thread with a Windows ProactorEventLoop.
    - Fixes `NotImplementedError` in `_make_subprocess_transport` on Windows Python 3.11+.
    - Keeps the Streamlit WebSocket connection alive across turns for `DASHBOARD_SESSION_TTL`
      seconds so follow-up questions do not disconnect or re-run login authentication.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._task_queue: queue.Queue | None = None
        self._worker_thread: threading.Thread | None = None

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker_thread is None or not self._worker_thread.is_alive():
                self._task_queue = queue.Queue()
                self._worker_thread = threading.Thread(
                    target=self._worker_loop,
                    name="aryaq-dashboard-session",
                    daemon=True,
                )
                self._worker_thread.start()

    def _worker_loop(self) -> None:
        # Ensure Windows ProactorEventLoop is active in this thread before starting Playwright
        if sys.platform == "win32":
            try:
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
                loop = asyncio.ProactorEventLoop()
                asyncio.set_event_loop(loop)
            except Exception:
                pass

        pw = None
        browser = None
        page = None
        current_url = None
        last_active = 0.0

        def _close_session() -> None:
            nonlocal pw, browser, page, current_url
            for obj, close_m in ((page, "close"), (browser, "close"), (pw, "stop")):
                if obj is not None:
                    try:
                        getattr(obj, close_m)()
                    except Exception:
                        pass
            pw = browser = page = current_url = None

        while True:
            try:
                item = self._task_queue.get(timeout=15.0)
            except queue.Empty:
                if browser is not None and (time.time() - last_active) > DASHBOARD_SESSION_TTL:
                    logger.info(
                        f"🕒 AryaQ dashboard session idle for >{DASHBOARD_SESSION_TTL}s; closing persistent browser."
                    )
                    _close_session()
                continue

            if item is None:
                _close_session()
                break

            url, username, password, tab, result_q = item
            try:
                from playwright.sync_api import sync_playwright

                now = time.time()
                need_new_session = (
                    pw is None
                    or browser is None
                    or page is None
                    or page.is_closed()
                    or (now - last_active) > DASHBOARD_SESSION_TTL
                )

                if need_new_session:
                    _close_session()
                    logger.info(f"🌐 Launching persistent headless browser for dashboard: {url} (TTL={DASHBOARD_SESSION_TTL}s)")
                    pw = sync_playwright().start()
                    browser = pw.chromium.launch(headless=True)
                    page = browser.new_page()
                    page.goto(url, wait_until="networkidle", timeout=30000)
                    current_url = url
                elif current_url != url:
                    logger.info(f"🌐 Navigating existing dashboard session to: {url}")
                    page.goto(url, wait_until="networkidle", timeout=30000)
                    current_url = url
                else:
                    logger.info(f"♻️ Reusing live authenticated AryaQ dashboard session (idle {int(now - last_active)}s < {DASHBOARD_SESSION_TTL}s)")

                # Check if login form is visible (either fresh session or server reset WebSocket)
                if username and password:
                    try:
                        if need_new_session:
                            try:
                                page.wait_for_selector('input[aria-label="Username"], [data-testid="stMetric"]', timeout=10000)
                            except Exception:
                                pass
                        user_input = page.locator('input[aria-label="Username"]')
                        if user_input.count() > 0 and user_input.first.is_visible():
                            logger.info(f"Submitting credentials for user '{username}'...")
                            user_input.first.fill(username)
                            page.locator('input[aria-label="Password"]').first.fill(password)
                            login_btn = page.locator('button:has-text("Secure Login")')
                            if login_btn.count() > 0:
                                login_btn.first.click()
                            logger.info("Credentials submitted, waiting for dashboard metrics to render...")
                            try:
                                page.wait_for_selector('[data-testid="stMetric"]', timeout=15000)
                            except Exception:
                                page.wait_for_timeout(3000)
                    except Exception as e:
                        logger.info(f"Login form check skipped or already authenticated: {e}")

                # Switch tab if requested
                if tab:
                    try:
                        logger.info(f"Targeting dashboard tab: '{tab}'")
                        tab_loc = page.locator(
                            f'button:has-text("{tab}"), [role="tab"]:has-text("{tab}"), [data-baseweb="tab"]:has-text("{tab}"), p:has-text("{tab}")'
                        ).first
                        if tab_loc:
                            tab_loc.click()
                            page.wait_for_timeout(2500)
                            logger.info(f"Clicked tab '{tab}' successfully.")
                    except Exception as e:
                        logger.warning(f"Failed to click tab '{tab}': {e}")

                last_active = time.time()
                title = page.title()
                full_text = page.inner_text("body").strip()

                # Scrape Streamlit KPI metrics ([data-testid="stMetric"])
                metrics = []
                try:
                    metric_els = page.query_selector_all('[data-testid="stMetric"]')
                    for m_el in metric_els:
                        m_txt = " — ".join(line.strip() for line in m_el.inner_text().splitlines() if line.strip())
                        if m_txt:
                            metrics.append(f"  • {m_txt}")
                except Exception:
                    pass

                # Scrape tabular data (HTML tables, st.table, st.dataframe, Glide Data Grids)
                table_data_reports = []
                tables = page.query_selector_all('table')
                for tbl in tables:
                    rows = tbl.query_selector_all('tr')
                    headers = [th.inner_text().strip() for th in tbl.query_selector_all('th')]
                    tbl_lines = []
                    if headers:
                        tbl_lines.append(" | ".join(headers))
                        tbl_lines.append(" | ".join(["---"] * len(headers)))

                    for r in rows:
                        tds = r.query_selector_all('td')
                        if tds:
                            row_vals = [td.inner_text().strip().replace("\n", " ") for td in tds]
                            tbl_lines.append(" | ".join(row_vals))
                            if len(tbl_lines) >= 102:  # header + separator + 100 rows
                                break
                    if tbl_lines:
                        table_data_reports.append("\n".join(tbl_lines))

                if not table_data_reports:
                    grid_rows = page.query_selector_all('[role="row"]')
                    if grid_rows:
                        grid_lines = []
                        for r in grid_rows[:102]:
                            cells = r.query_selector_all('[role="gridcell"], [role="columnheader"]')
                            if cells:
                                c_vals = [c.inner_text().strip().replace("\n", " ") for c in cells]
                                grid_lines.append(" | ".join(c_vals))
                        if grid_lines:
                            table_data_reports.append("\n".join(grid_lines))

                # Capture visual snapshot
                snapshot_dir = Path(__file__).resolve().parents[2] / "data"
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                safe_tab_name = re.sub(r"[^\w]", "_", tab.lower()) if tab else "main"
                snapshot_file = snapshot_dir / f"aryaq_dashboard_{safe_tab_name}.png"
                page.screenshot(path=str(snapshot_file), full_page=True)
                logger.info(f"✅ Dashboard screenshot saved to: {snapshot_file} (session kept alive)")

                report = [
                    f"📊 Live AryaQ Dashboard ({url}) - Tab: '{tab or 'System Overview'}':",
                    f"Page Title: {title}",
                    f"Screenshot Saved: {snapshot_file}",
                    "",
                    "📈 Live KPI / Page Summary:",
                ]
                if metrics:
                    report.extend(metrics)

                if table_data_reports:
                    report.append("\n📋 Tabular Data (Extracted up to 100 rows):")
                    for t_idx, t_data in enumerate(table_data_reports, 1):
                        report.append(f"\n--- Table {t_idx} ---")
                        report.append(t_data)
                elif not metrics:
                    report.append("Content preview (first 1500 chars):")
                    report.append(full_text[:1500])

                sections = []
                for line in full_text.splitlines():
                    s = line.strip()
                    if s in {
                        "System Overview", "Devices", "Device Instances", "Users",
                        "Scanner Details", "Heartbeats", "Password Reset", "Reports",
                        "Logged in as: Developer"
                    }:
                        sections.append(s)
                if sections:
                    report.append(f"\nAvailable Tabs & Roles: {', '.join(dict.fromkeys(sections))}")

                result_q.put("\n".join(report))
            except Exception as e:
                logger.warning(f"Headless dashboard scrape error: {e}")
                _close_session()
                result_q.put("")

    def scrape(self, url: str, username: str | None, password: str | None, tab: str | None) -> str:
        self._ensure_worker()
        result_q: queue.Queue[str] = queue.Queue()
        self._task_queue.put((url, username, password, tab, result_q))
        try:
            return result_q.get(timeout=60.0)
        except queue.Empty:
            logger.warning("Timed out waiting for persistent dashboard browser worker.")
            return ""

    def shutdown(self) -> None:
        if self._task_queue is not None:
            try:
                self._task_queue.put_nowait(None)
            except Exception:
                pass


_DASHBOARD_SESSION = _DashboardSessionManager()
atexit.register(_DASHBOARD_SESSION.shutdown)


def scrape_live_streamlit_dashboard(url: str, username: str = None, password: str = None, tab: str = None) -> str:
    """Use persistent Playwright headless browser session to log in once, switch tabs, and scrape live Streamlit dashboard metrics without disconnecting between questions."""
    try:
        import playwright  # noqa: F401
    except ImportError:
        logger.warning("Playwright is not installed; cannot scrape live Streamlit dashboard.")
        return ""

    return _DASHBOARD_SESSION.scrape(url, username, password, tab)


def fetch_dashboard_information(url: str | None = None, query: str | None = None, tab: str | None = None) -> str:
    """Fetch live dashboard information and relevant indexed project context."""
    endpoint = url
    if not endpoint:
        configured = get_configured_dashboard_endpoints()
        endpoint = configured[0]["url"] if configured else None
    if not endpoint:
        return "No dashboard URL is configured."

    # Infer tab from query if not explicitly passed
    if not tab and query:
        q_lower = query.lower()
        known_tabs = {
            "devices": "Devices",
            "device instances": "Device Instances",
            "device instance": "Device Instances",
            "instances": "Device Instances",
            "users": "Users",
            "user": "Users",
            "scanner details": "Scanner Details",
            "scanners": "Scanner Details",
            "heartbeats": "Heartbeats",
            "heartbeat": "Heartbeats",
            "password reset": "Password Reset",
            "reports": "Reports",
            "report": "Reports",
            "system overview": "System Overview",
        }
        for key, val in known_tabs.items():
            if key in q_lower:
                tab = val
                break

    username = os.getenv("ARYAQ_USER")
    password = os.getenv("ARYAQ_PASS")

    # If it is a web/Streamlit dashboard or has login credentials, use the headless browser scraper
    if "vikshep.com" in endpoint or "streamlit" in endpoint or ":8501" in endpoint or (username and password):
        live_result = scrape_live_streamlit_dashboard(endpoint, username, password, tab=tab)
        if live_result:
            return live_result

    # Fallback to standard HTTP GET if headless scrape not applicable
    session = requests.Session()
    if username and password:
        session.auth = (username, password)

    try:
        summary_url = endpoint
        if "/api/" not in endpoint.lower() and not endpoint.lower().endswith("/summary"):
            summary_url = f"{endpoint.rstrip('/')}/api/dashboard/summary"
        response = session.get(summary_url, timeout=15)
        live_summary = summarize_dashboard_response(summary_url, response.text, response.status_code, response.reason)
        return live_summary
    except requests.RequestException as error:
        return f"Dashboard at {endpoint} could not be reached: {error}"

def check_website_health(url: str = None, requires_login: bool = False) -> str:
    """Pings a website, optionally passing secure authentication."""
    
    # 1. Auto-detect conversational aliases
    if not url or url.lower() in ["app server", "dashboard", "internal website", "aryaq"]:
        url = os.getenv("ARYAQ_LOCAL_URL", "http://127.0.0.1:8000/admin")
        
    if not url.startswith("http"):
        url = "https://" + url
        
    try:
        start_time = time.time()
        
        if requires_login:
            # Use a session to handle cookies/auth
            session = requests.Session()
            username = os.getenv("ARYAQ_USER")
            password = os.getenv("ARYAQ_PASS")
            
            # Standard Basic Auth (works for most APIs)
            session.auth = (username, password)
            response = session.get(url, timeout=10)
        else:
            response = requests.get(url, timeout=10)
            
        latency = round((time.time() - start_time) * 1000)
        
        if response.status_code == 200:
            return f"✅ ONLINE & AUTHENTICATED: {url} is up. Status: {response.status_code}. Latency: {latency}ms."
        elif response.status_code in [401, 403]:
            return f"🔒 AUTH ERROR: {url} rejected the login credentials. Status: {response.status_code}."
        else:
            return f"⚠️ WARNING: {url} returned status code {response.status_code}."
            
    except requests.exceptions.Timeout:
        return f"❌ DOWN: Connection to {url} timed out."
    except Exception as e:
        return f"❌ ERROR: Could not check {url}. Details: {str(e)}"