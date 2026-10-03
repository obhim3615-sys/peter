import pytest

from src.tools.network_tools import summarize_dashboard_response, get_configured_dashboard_endpoints


def test_summarize_dashboard_response_json():
    payload = '{"status": "online", "title": "Ops dashboard", "updated_at": "2026-09-25T10:00:00Z", "alerts": 2}'
    summary = summarize_dashboard_response(
        "https://example.com/dashboard",
        payload,
        response_status=200,
    )
    assert "https://example.com/dashboard" in summary
    assert "status: online" in summary.lower()
    assert "ops dashboard" in summary.lower()


def test_summarize_dashboard_response_html():
    html = """
    <html><head><title>Ops Dashboard</title></head>
    <body><h1>Operations Overview</h1><div>System running normally</div></body></html>
    """
    summary = summarize_dashboard_response(
        "https://example.com/dashboard",
        html,
        response_status=200,
    )
    assert "html instead of json" in summary.lower()
    assert "ops dashboard" in summary.lower()


def test_get_configured_dashboard_endpoints_prefers_env(monkeypatch):
    monkeypatch.setenv("DASHBOARD_URL", "https://dashboard.example.com")
    monkeypatch.delenv("ARYAQ_LOCAL_URL", raising=False)
    endpoints = get_configured_dashboard_endpoints()
    assert any(item["url"] == "https://dashboard.example.com" for item in endpoints)
