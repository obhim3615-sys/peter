from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# 1. Define State (the shared memory flowing across nodes)
class AgentState(TypedDict):
    user_request: str
    proposed_action: str
    human_approval: str
    execution_result: str

# 2. Mock AI Node (simulates model decision logic)
def mock_ai_reasoning(state: AgentState):
    print("\n--- 🧠 [NODE: AI] Reasoning ---")
    print(f"User request: {state['user_request']}")
    
    if "csv" in state["user_request"].lower():
        action = "UPDATE_CSV: project_metrics.csv"
    else:
        action = "UPDATE_DB: hardware_inventory_table"
        
    print(f"AI planned action: {action}")
    return {"proposed_action": action}

# 3. Human Approval Gateway (Interrupt)
def require_approval(state: AgentState):
    print("\n--- ⏸️ [NODE: GATEWAY] Pausing for Human Approval ---")
    
    # Execution halts here, state is persisted to MemorySaver
    decision = interrupt(
        f"Alert: AI requests permission to perform '{state['proposed_action']}'. Allow? (approve/reject)"
    )
    
    print(f"\n--- ▶️ [NODE: GATEWAY] Resumed with decision: {decision} ---")
    return {"human_approval": decision}

# 4. Executor Node (runs only after permission)
def execute_action(state: AgentState):
    print("\n--- ⚙️ [NODE: EXECUTOR] Running Task ---")
    
    if state["human_approval"].lower() == "approve":
        result = f"Success: Completed {state['proposed_action']}."
    else:
        result = f"Aborted: Rejected {state['proposed_action']}."
        
    print(result)
    return {"execution_result": result}

# 5. Wire the Graph
builder = StateGraph(AgentState)
builder.add_node("ai_brain", mock_ai_reasoning)
builder.add_node("gateway", require_approval)
builder.add_node("executor", execute_action)

builder.add_edge(START, "ai_brain")
builder.add_edge("ai_brain", "gateway")
builder.add_edge("gateway", "executor")
builder.add_edge("executor", END)

memory = MemorySaver()
orchestrator = builder.compile(checkpointer=memory)

# 6. Run Simulation
if __name__ == "__main__":
    thread_config = {"configurable": {"thread_id": "session-001"}}

    print("=== STEP 1: INITIAL REQUEST ===")
    initial_payload = {"user_request": "Please parse logs and update the metrics CSV"}
    
    # Start graph until interrupt
    for _ in orchestrator.stream(initial_payload, config=thread_config):
        pass

    # Prompt user in terminal (simulating an external approval via Telegram or UI)
    print("\n=== STEP 2: USER APPROVAL GATE ===")
    decision = input("Type 'approve' or 'reject': ").strip()

    # Resume graph execution with user decision
    for _ in orchestrator.stream(Command(resume=decision), config=thread_config):
        pass

    print("\n=== FINAL GRAPH STATE ===")
    print(orchestrator.get_state(thread_config).values)