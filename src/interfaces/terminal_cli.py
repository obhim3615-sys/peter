import asyncio
import uuid
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from src.graph import workflow
from src.utils.logger import setup_logger
from src.tools.rag_tools import save_conversation_memory

from pathlib import Path

logger = setup_logger("terminal-cli")
DB_PATH = str(Path(__file__).resolve().parents[2] / "data" / "sqlite" / "memory.db")

async def main():
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": f"terminal-cli-{thread_id}"}}
    
    print("="*50)
    print("🤖 Terminal Orchestrator Online. (Async Mode)")
    print("Type 'exit' to quit, 'clear' to reset memory.")
    print("="*50)
    
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        orchestrator = workflow.compile(checkpointer=checkpointer)
        
        while True:
            user_input = input("\n🧑‍💻 You: ").strip()
            # Strip pasted prompt prefixes like "🧑💻 You:" or "🧑‍💻 You:"
            import re as _re
            user_input = _re.sub(r"^(?:🧑(?:\u200d)?💻\s*)?You:\s*", "", user_input, flags=_re.IGNORECASE).strip()
            if not user_input:
                continue
            # Ignore accidentally pasted Peter orchestrator log lines (e.g. "2026-09-29 09:03:37 [INFO] ---")
            if _re.match(r"^\d{2,4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+\[(?:INFO|WARNING|DEBUG)\]", user_input):
                continue
            if user_input.lower() in ['exit', 'quit']:
                print("Shutting down...")
                break
            if user_input.lower() == 'clear':
                thread_id = str(uuid.uuid4())
                config = {"configurable": {"thread_id": f"terminal-cli-{thread_id}"}}
                print("🧹 Memory cleared! Fresh session started.")
                continue
                
            try:
                # 1. Run the initial graph execution
                async for _ in orchestrator.astream({"user_request": user_input}, config=config):
                    pass 
                    
                # 2. Handle Human-in-the-Loop loops
                state = await orchestrator.aget_state(config)
                
                while state.next:
                    action = state.values.get("proposed_action", "Unknown Action")
                    print(f"\n⚠️ APPROVAL REQUIRED")
                    print(f"Orchestrator wants to execute: {action}")
                    
                    decision = input("Type 'approve', 'reject', or 'kill': ").strip().lower()
                    if decision not in ['approve', 'reject', 'kill']:
                        print("Invalid choice. Defaulting to 'reject'.")
                        decision = 'reject'
                        
                    # Resume graph with decision
                    async for _ in orchestrator.astream(Command(resume=decision), config=config):
                        pass
                        
                    state = await orchestrator.aget_state(config)
                    
                    if decision == 'kill':
                        print("🛑 Execution killed by user.")
                        break
                
                # 3. Print final result
                final_result = state.values.get("execution_result", "No result generated.")
                print(f"\n🤖 AI: {final_result}")

                # 4. Persist conversation to ChromaDB for future context
                try:
                    await asyncio.to_thread(save_conversation_memory, user_input, final_result)
                except Exception as mem_err:
                    logger.warning(f"Failed to save conversation memory: {mem_err}")
                
            except Exception as e:
                logger.error(f"Execution Error: {e}")
                print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    # Windows requires a specific event loop policy for certain async operations
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        
    asyncio.run(main())