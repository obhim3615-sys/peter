# Peter AI Assistant — Feature & Architecture Pipeline

> **Spreadsheet Version**: [`peter_feature_pipeline.csv`](file:///c:/Users/Abhishek.Pandey/Videos/peter/peter_feature_pipeline.csv) (Open directly in Excel / Google Sheets)

---

## 1. Completed Foundation (`Phase 1`)

| ID | Category | Feature / Enhancement | Priority | Status | Target Modules |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **PIPE-001** | Stability & Runtime | Fix 10s Infinite Website Health-Check Loop (`asyncio.to_thread` + configurable interval) | P0 | ✅ Completed | `src/interfaces/telegram_bot.py` |
| **PIPE-002** | Dependencies | Sync Missing Runtime & Dev Dependencies (`psycopg2-binary`, `onnxruntime`, `pytest`, etc.) | P0 | ✅ Completed | `requirements.txt` |
| **PIPE-003** | Database Integration | Connect Live Local UAT PostgreSQL (`arya`, 230 tables) with Read-Only & Mixed-Case Quoting | P0 | ✅ Completed | `.env`, `src/tools/database_tools.py` |
| **PIPE-004** | Graph Orchestration | Multi-Tool Queue Draining Fix (`route_post_tool` final payload execution) | P0 | ✅ Completed | `src/graph.py`, `src/nodes.py` |
| **PIPE-005** | Natural Language DB | 4/3/2/1-Word Natural-Language Table Resolution + Live PG Table Fallback Cache | P0 | ✅ Completed | `src/nodes.py`, `src/tools/code_intelligence.py` |
| **PIPE-013** | Code Intelligence & Caching | Incremental File & AST Symbol Index (`code_symbol_index.db` + L1 AST cache + `mtime_ns` auto-update) | P0 | ✅ Completed | `src/tools/code_intelligence.py` |

---

## 2. Upcoming Advanced Pipeline (`Phases 2–4`)

| ID | Phase | Category | Feature / Enhancement | Priority | Status | Target Modules | Effort | Technical Scope & Deliverable |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :---: | :--- |
| **PIPE-006** | Phase 2 | Database & ORM Audit | **Schema-Drift & Migration Auditor (`audit_schema_drift`)** | **P1 - High** | 📋 In Pipeline | `src/tools/database_tools.py`, `src/tools/code_intelligence.py`, `src/nodes.py` | 3 hrs | Diff all Django `models.py` AST definitions against live PostgreSQL `information_schema.columns` across all 230 tables. Detect unapplied migrations, orphan DB tables (e.g. `pop_manage_weatherdata`), missing columns, and type mismatches. |
| **PIPE-007** | Phase 2 | Analytical AI | **Foreign-Key Aware Text-to-SQL Generator with Self-Correction** | **P1 - High** | 📋 In Pipeline | `src/tools/database_tools.py`, `src/nodes.py`, `src/graph.py` | 4 hrs | Automatically inspect `information_schema` FK constraints between resolved tables, inject minimal schema + FK join paths into LLM prompt, generate read-only `SELECT` queries (`JOIN`, `GROUP BY`, aggregations), and auto-retry once on SQL error. |
| **PIPE-008** | Phase 2 | Performance Engineering | **Automated `EXPLAIN ANALYZE` & Missing Index Advisor** | **P1 - High** | 📋 In Pipeline | `src/tools/database_tools.py`, `src/nodes.py` | 3 hrs | Inspect `pg_indexes`, `pg_stat_user_tables` (`seq_scan` vs `idx_scan`, dead tuples), and run read-only `EXPLAIN (FORMAT JSON)` to detect unindexed Foreign Keys and slow sequential scans. |
| **PIPE-009** | Phase 3 | Full-Stack Intelligence | **End-to-End API ➔ View ➔ Serializer ➔ Model ➔ Live DB Tracer** | **P2 - Medium** | 📋 In Pipeline | `src/tools/code_intelligence.py`, `src/tools/database_tools.py`, `src/nodes.py` | 4 hrs | Extend `trace_url_to_view` to follow imports into DRF Serializers and Django Models, then automatically fetch live PostgreSQL table schema and sample rows for the underlying tables. |
| **PIPE-010** | Phase 3 | Security & RBAC | **Automated Role & Permission Access Matrix Auditor** | **P2 - Medium** | 📋 In Pipeline | `src/tools/code_intelligence.py`, `src/tools/database_tools.py` | 3 hrs | Cross-reference DRF `permission_classes` / role decorators in `views.py` against live UAT role mappings (`app_farm_user_roles`, `app_farm_user_roles_permissions`, `core_permission`). |
| **PIPE-011** | Phase 4 | DevOps & Alerting | **Scheduled Telegram UAT Data-Health & Celery Digest** | **P2 - Medium** | 📋 In Pipeline | `src/interfaces/telegram_bot.py`, `src/tools/database_tools.py` | 2.5 hrs | Extend background health scheduler to run periodic read-only checks on `farm_category_celery_task_status`, `payment_payment` failures, and error logs, pushing an executive summary to Telegram. |
| **PIPE-012** | Phase 4 | Architecture & Refactor | **Modularize Monolithic `src/nodes.py` into Domain Routers** | **P3 - Planned** | 📋 In Pipeline | `src/nodes.py`, `src/routers/`, `src/prompts/` | 4 hrs | Split `coder_node` intent routing, prompt templates, and deterministic tool planners into dedicated modules (`db_router.py`, `code_router.py`, `git_router.py`) with unit test coverage. |
