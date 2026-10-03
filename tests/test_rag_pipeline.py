from src.tools.rag_tools import save_conversation_memory, save_to_knowledge_base, search_internal_knowledge


def test_manual_memory_keeps_project_and_source_metadata():
    result = save_to_knowledge_base(
        "aryaq_status",
        information="AryaQ has 18 active tasks and 3 open blockers.",
        source_file="project_dashboard.txt",
        project="aryaq",
        line_number=12,
    )
    assert "Success" in result
    hits = search_internal_knowledge("active tasks blockers aryaq", project="aryaq")
    assert "18 active tasks" in hits or "knowledge snippets" in hits


def test_conversation_memory_is_deduplicated():
    first = save_conversation_memory("What is the current status?", "It is running normally.")
    second = save_conversation_memory("What is the current status?", "It is running normally.")
    assert "saved" in first.lower()
    assert "saved" in second.lower()
    assert "duplicate" not in second.lower()
