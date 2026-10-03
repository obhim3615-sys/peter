from pathlib import Path
from langgraph.graph import StateGraph, START, END

from src.state import AgentState
from src.nodes import supervisor_node, researcher_node, coder_node, require_approval, execute_action, final_answer_node

APP_ROOT = Path(__file__).resolve().parent.parent
db_path = APP_ROOT / "data" / "sqlite" / "memory.db"
db_path.parent.mkdir(parents=True, exist_ok=True)
workflow    = StateGraph(AgentState)

# Add Nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("coder", coder_node)
workflow.add_node("require_approval", require_approval)
workflow.add_node("execute_action", execute_action)
workflow.add_node("final_answer", final_answer_node)

# Router after supervisor
def route_supervisor(state: AgentState):
    worker = state.get("active_worker")
    if worker == "FINISH":
        return END
    if worker == "final_answer":
        return "final_answer"
    return worker

workflow.add_conditional_edges("supervisor", route_supervisor, {
    "researcher": "researcher",
    "coder": "coder",
    "final_answer": "final_answer",
    END: END
})

# Router after worker reasoning
def route_worker(state: AgentState):
    if state.get("proposed_action", "").startswith("TOOL_CALL"):
        return "require_approval"
    # Worker is done with its subtask, report back to supervisor
    return "supervisor"

workflow.add_conditional_edges("researcher", route_worker, {
    "require_approval": "require_approval",
    "supervisor": "supervisor"
})

workflow.add_conditional_edges("coder", route_worker, {
    "require_approval": "require_approval",
    "supervisor": "supervisor"
})

# Tool approval -> execution
def route_approval(state: AgentState):
    if state.get("status") == "killed":
        return END
    return "execute_action"

workflow.add_conditional_edges("require_approval", route_approval, {
    "execute_action": "execute_action",
    END: END
})

# Router after tool execution: drain queue, self-correct on failure, or finalize answer
def route_post_tool(state: AgentState):
    # 1. Drain queued tool calls if execute_action loaded the next call into proposed_action & tool_payload
    if state.get("proposed_action", "").startswith("TOOL_CALL") and bool(state.get("tool_payload", {}).get("name")):
        return "require_approval"

    # 2. Self-correction loop: if result is empty or failed and retry limit not reached
    worker = state.get("active_worker")
    if worker in ("coder", "researcher"):
        res = str(state.get("execution_result", "")).strip().lower()
        retry_count = state.get("retry_count", 0)

        is_empty_or_failed = (
            not res
            or "0 matches" in res
            or "no matches found" in res
            or "no matching results" in res
            or "no occurrences of" in res
            or "does not exist in this repository" in res
            or "no relevant information found" in res
            or "no function or class named" in res
            or "no references to" in res
            or "exit code: 1" in res
            or "not found in the project" in res
            or "error reading file" in res
            or "error executing tool" in res
            or "error scanning directory" in res
            or "directory not found" in res
            or "file not found" in res
            or "invalid tool arguments" in res
        )

        if is_empty_or_failed and retry_count < 2:
            from src.utils.logger import setup_logger
            logger = setup_logger("orchestrator")
            logger.info(f"🔄 Self-correction triggered (attempt {retry_count + 1}/2). Routing back to {worker}...")
            return worker

    return "final_answer"

workflow.add_conditional_edges("execute_action", route_post_tool, {
    "require_approval": "require_approval",
    "coder": "coder",
    "researcher": "researcher",
    "final_answer": "final_answer",
})

workflow.add_edge("final_answer", END)

workflow.add_edge(START, "supervisor")

app = workflow.compile()


