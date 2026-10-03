from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict, total=False):
    user_request: str
    intent: Optional[str]
    search_terms: Optional[str]
    target_project: Optional[str]
    tool_attempts: Dict[str, int]
    tool_results: Dict[str, str]
    model_provider: Optional[str]
    active_worker: Optional[str]        # "researcher", "coder", "final_answer", or "FINISH"
    worker_task: Optional[str]          # Specific sub-task delegated by supervisor
    proposed_action: Optional[str]      # TOOL_CALL or RESPOND_ONLY
    tool_payload: Optional[Dict[str, Any]]
    tool_queue: List[Dict[str, Any]]    # Remaining tool calls waiting to be executed (Fix #1)
    execution_result: Optional[str]
    prior_execution_summary: Optional[str]   # Snapshot of previous turn's result for multi-turn context
    last_accessed_file: Optional[str]        # Path of the most recently read or inspected file
    last_file_content: Optional[str]         # Cached content of the most recently read file for follow-ups
    retry_count: Optional[int]               # Number of self-correction attempts on failed/empty tool output
    human_approval: Optional[str]       # "approve", "reject", "kill"
    status: str                         # "running", "completed", "killed"
    history: List[str]
    worker_outputs: Dict[str, str]      # Stores outputs from sub-agents
