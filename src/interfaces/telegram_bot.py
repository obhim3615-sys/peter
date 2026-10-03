import os
import uuid
import time
import json
import re
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import NetworkError
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler, ContextTypes
import requests
from langgraph.types import Command
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from src.graph import workflow  # Import the raw workflow now
from src.utils.logger import setup_logger
import asyncio
from src.tools.network_tools import check_website_health, get_configured_dashboard_endpoints, summarize_dashboard_response
from src.tools.rag_tools import save_conversation_memory, save_to_knowledge_base
from scripts.import_project_knowledge import refresh_project_knowledge
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ALLOWED_USER = os.getenv("ALLOWED_TELEGRAM_USER")
from pathlib import Path
DB_PATH = str(Path(__file__).resolve().parents[2] / "data" / "sqlite" / "memory.db")
logger = setup_logger("telegram-bot")

user_sessions = {}

def get_thread_config(chat_id: str) -> dict:
    if chat_id not in user_sessions:
        user_sessions[chat_id] = str(uuid.uuid4())
    return {"configurable": {"thread_id": f"telegram-{chat_id}-{user_sessions[chat_id]}"}}

def is_authorized(chat_id: str) -> bool:
    if not ALLOWED_USER:
        return True
    return str(chat_id) == str(ALLOWED_USER)

import html


def markdown_to_telegram_html(text: str) -> str:
    """Convert GitHub-style Markdown into clean, structured Telegram HTML."""
    if not text:
        return ""

    code_blocks: list[str] = []
    inline_codes: list[str] = []

    # 1. Extract fenced code blocks: ```lang\n...\n```
    def _replace_fenced_block(match: re.Match) -> str:
        lang = (match.group(1) or "").strip().lower()
        code = match.group(2) or ""
        escaped_code = html.escape(code.strip("\r\n"))
        if lang:
            safe_lang = re.sub(r"[^a-z0-9_+-]", "", lang)
            block_html = f'<pre><code class="language-{safe_lang}">{escaped_code}</code></pre>'
        else:
            block_html = f"<pre><code>{escaped_code}</code></pre>"
        idx = len(code_blocks)
        code_blocks.append(block_html)
        return f"\x00CODEBLOCK{idx}\x00"

    processed = re.sub(
        r"```([a-zA-Z0-9_+-]*)[ \t]*\r?\n(.*?)```",
        _replace_fenced_block,
        text,
        flags=re.DOTALL,
    )

    # 2. Extract inline code: `...`
    def _replace_inline_code(match: re.Match) -> str:
        code = match.group(1)
        escaped_code = html.escape(code)
        idx = len(inline_codes)
        inline_codes.append(f"<code>{escaped_code}</code>")
        return f"\x00INLINECODE{idx}\x00"

    processed = re.sub(r"`([^`\n]+)`", _replace_inline_code, processed)

    # 3. Escape raw HTML in remaining prose
    processed = html.escape(processed)

    # 4. Convert Markdown links: [label](url)
    def _replace_link(match: re.Match) -> str:
        label = match.group(1)
        url = match.group(2).strip()
        if url.startswith(("http://", "https://")):
            return f'<a href="{url}">{label}</a>'
        return f"<code>{label}</code>"

    processed = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _replace_link, processed)

    # 5. Format headings, dividers, and bullets line-by-line
    formatted_lines: list[str] = []
    for line in processed.splitlines():
        stripped = line.strip()
        # Horizontal rule
        if re.match(r"^[-*_]{3,}$", stripped):
            formatted_lines.append("────────────────────")
            continue

        # Headings: # / ## / ### / ####
        h12_match = re.match(r"^#{1,2}\s+(.+)$", stripped)
        if h12_match:
            title = re.sub(r"\*\*(.+?)\*\*", r"\1", h12_match.group(1))
            formatted_lines.append(f"\n<b>━━ {title} ━━</b>")
            continue

        h34_match = re.match(r"^#{3,6}\s+(.+)$", stripped)
        if h34_match:
            title = re.sub(r"\*\*(.+?)\*\*", r"\1", h34_match.group(1))
            formatted_lines.append(f"\n<b>▸ {title}</b>")
            continue

        # Sub-bullets (indented 2+ spaces)
        sub_bullet_match = re.match(r"^[ \t]{2,}[-*]\s+(.+)$", line)
        if sub_bullet_match:
            formatted_lines.append(f"   ◦ {sub_bullet_match.group(1)}")
            continue

        # Top-level bullets
        top_bullet_match = re.match(r"^[ \t]*[-*]\s+(.+)$", line)
        if top_bullet_match:
            formatted_lines.append(f"• {top_bullet_match.group(1)}")
            continue

        formatted_lines.append(line)

    processed = "\n".join(formatted_lines)

    # 6. Bold (**text**) and Italic (*text*)
    processed = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", processed, flags=re.DOTALL)
    processed = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", processed)

    # 7. Restore inline codes (closing/reopening surrounding <b> or <i> so <code> is never nested inside them)
    def _unroll_inline_codes_in_tag(tag: str, text_val: str) -> str:
        pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", flags=re.DOTALL)
        def _repl(m: re.Match) -> str:
            inner = m.group(1)
            if "\x00INLINECODE" not in inner:
                return m.group(0)
            inner = re.sub(
                r"(\x00INLINECODE\d+\x00)",
                rf"</{tag}>\1<{tag}>",
                inner,
            )
            return f"<{tag}>{inner}</{tag}>"
        res = pattern.sub(_repl, text_val)
        return res.replace(f"<{tag}></{tag}>", "")

    processed = _unroll_inline_codes_in_tag("b", processed)
    processed = _unroll_inline_codes_in_tag("i", processed)

    for idx, inline_html in enumerate(inline_codes):
        processed = processed.replace(f"\x00INLINECODE{idx}\x00", inline_html)
    for idx, block_html in enumerate(code_blocks):
        processed = processed.replace(f"\x00CODEBLOCK{idx}\x00", block_html)

    # Collapse excessive blank lines
    processed = re.sub(r"\n{3,}", "\n\n", processed).strip()
    return processed


def split_markdown_for_telegram(text: str, max_length: int = 2800) -> list[str]:
    """Split Markdown on logical line/section boundaries while keeping fenced code blocks intact across chunks."""
    if not text:
        return []
    if len(text) <= max_length:
        return [text]

    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    current_lines: list[str] = []
    current_len = 0
    in_fence = False
    fence_lang = ""

    for line in lines:
        stripped = line.strip()
        # Start a new chunk before major section headers if current chunk is already substantial
        is_section_header = stripped.startswith(("## ", "### ")) and not in_fence
        if is_section_header and current_len >= int(max_length * 0.55):
            chunks.append("".join(current_lines).strip())
            current_lines = []
            current_len = 0

        # Check if adding this line would exceed max_length
        if current_len + len(line) > max_length and current_lines:
            if in_fence:
                current_lines.append("```\n")
                chunks.append("".join(current_lines).strip())
                current_lines = [f"```{fence_lang}\n"]
                current_len = len(current_lines[0])
            else:
                chunks.append("".join(current_lines).strip())
                current_lines = []
                current_len = 0

        # Track code fence state
        if stripped.startswith("```"):
            if not in_fence:
                in_fence = True
                fence_lang = stripped[3:].strip()
            else:
                in_fence = False
                fence_lang = ""

        current_lines.append(line)
        current_len += len(line)

    if current_lines:
        if in_fence:
            current_lines.append("\n```")
        final_piece = "".join(current_lines).strip()
        if final_piece:
            chunks.append(final_piece)

    return chunks


async def send_chunked_message(bot, chat_id: str, text: str, max_length: int = 2800):
    if not text:
        await bot.send_message(chat_id=chat_id, text="(Empty response)")
        return

    markdown_chunks = split_markdown_for_telegram(text, max_length=max_length)
    for md_chunk in markdown_chunks:
        html_chunk = markdown_to_telegram_html(md_chunk)
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=html_chunk,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        except Exception as err:
            logger.warning(f"HTML parse fallback for Telegram chunk ({err}); sending plain text.")
            for i in range(0, len(md_chunk), 4000):
                await bot.send_message(chat_id=chat_id, text=md_chunk[i:i + 4000])


async def maybe_send_screenshot(bot, chat_id: str, text: str, user_request: str = ""):
    """Detect if the user explicitly requested a screenshot/snapshot and send it to the Telegram chat."""
    req_lower = (user_request or "").lower()

    explicit_screenshot_phrases = (
        "screenshot",
        "screen shot",
        "snapshot",
        "send image",
        "show image",
        "dashboard image",
        "dashboard photo",
    )
    wants_screenshot = any(w in req_lower for w in explicit_screenshot_phrases)
    if not wants_screenshot:
        return

    image_patterns = [
        r"(?:screenshot|snapshot|image|saved to)\s*[:=]?\s*([A-Za-z]:[^\s\n\"']+\.(?:png|jpg|jpeg))",
        r"([A-Za-z]:[^\s\n\"']+[\\/]data[\\/][^\s\n\"']+\.(?:png|jpg|jpeg))",
        r"data[\\/]([^\s\n\"']+\.(?:png|jpg|jpeg))",
    ]
    candidate_paths = []
    for pat in image_patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        for m in matches:
            candidate_paths.append(m.strip().strip('"').strip("'"))

    default_snapshot = Path(__file__).resolve().parents[2] / "data" / "aryaq_dashboard_snapshot.png"
    if default_snapshot.exists():
        candidate_paths.append(str(default_snapshot))

    sent = set()
    for p in candidate_paths:
        try:
            path_obj = Path(p).resolve()
            if path_obj.exists() and path_obj.is_file() and str(path_obj) not in sent:
                with open(path_obj, "rb") as photo_file:
                    await bot.send_photo(
                        chat_id=chat_id,
                        photo=photo_file,
                        caption=f"📸 Live Dashboard Snapshot ({path_obj.name})"
                    )
                sent.add(str(path_obj))
                logger.info(f"✅ Sent photo {path_obj.name} to Telegram chat {chat_id}")
        except Exception as err:
            logger.warning(f"Could not send photo {p}: {err}")


async def dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Directly send the latest dashboard screenshot via /dashboard or /screenshot."""
    chat_id = str(update.message.chat_id)
    if not is_authorized(chat_id):
        return
    snapshot_path = Path(__file__).resolve().parents[2] / "data" / "aryaq_dashboard_snapshot.png"
    if snapshot_path.exists():
        with open(snapshot_path, "rb") as f:
            await update.message.reply_photo(photo=f, caption="📸 Latest AryaQ Dashboard Snapshot")
    else:
        await update.message.reply_text("ℹ️ No dashboard snapshot found. Send 'check aryaq dashboard' to scrape the latest metrics and screenshot.")


async def save_completed_conversation(user_request: str, assistant_response: str):
    """Persist a finalized exchange without blocking Telegram's event loop."""
    if not user_request or not assistant_response:
        return
    result = await asyncio.to_thread(save_conversation_memory, user_request, assistant_response)
    logger.info("Conversation memory result: %s", result)


async def telegram_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Log Telegram transport failures without an unhandled traceback."""
    error = context.error
    if isinstance(error, NetworkError):
        logger.warning("Telegram network connection interrupted; polling will retry: %s", error)
    else:
        logger.error("Unhandled Telegram update error: %s", error, exc_info=error)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.message.chat_id)
    if not is_authorized(chat_id):
        return
    await update.message.reply_text("🤖 Orchestrator online. Send a command!")

async def clear_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.message.chat_id)
    if not is_authorized(chat_id):
        return
    user_sessions[chat_id] = str(uuid.uuid4())
    await update.message.reply_text("🧹 Memory cleared! Fresh session started.")


async def refresh_knowledge(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Refresh project CSV, XLSX, and Markdown records in the RAG store."""
    chat_id = str(update.message.chat_id)
    if not is_authorized(chat_id):
        return

    status_message = await update.message.reply_text("🔄 Checking project data files and refreshing knowledge...")
    try:
        record_count = await asyncio.to_thread(refresh_project_knowledge)
        if record_count:
            await status_message.edit_text(
                f"✅ Project knowledge refreshed. Indexed {record_count} records from data/csv_files."
            )
        else:
            await status_message.edit_text("ℹ️ No supported CSV, XLSX, XLSM, or Markdown records were found.")
    except Exception as e:
        logger.error(f"[Telegram User {chat_id}] Knowledge refresh failed: {e}")
        await status_message.edit_text(f"❌ Knowledge refresh failed: {e}")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.message.chat_id)
    if not is_authorized(chat_id):
        return

    status_message = await update.message.reply_text("📸 Receiving image...")
    try:
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        
        image_dir = os.path.join("data", "images")
        os.makedirs(image_dir, exist_ok=True)
        
        file_path = os.path.join(image_dir, f"telegram_upload_{photo.file_id}.jpg")
        await file.download_to_drive(file_path)
        
        user_text = update.message.caption or "Analyze this image and tell me what you see."
        ai_prompt = f"I just uploaded an image to {file_path}. {user_text}"
        
        await status_message.edit_text(f"✅ Image saved to {file_path}\n⏳ Passing to Orchestrator...")
        
        thread_config = get_thread_config(chat_id)
        
        # NEW: Safely open the async database
        async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
            orchestrator = workflow.compile(checkpointer=checkpointer)
            
            # Use astream instead of stream for photos
            async for _ in orchestrator.astream({"user_request": ai_prompt}, config=thread_config):
                pass
                
            state = await orchestrator.aget_state(thread_config)
            
        if state.next:
            action = html.escape(str(state.values.get("proposed_action", "Unknown Action")))
            keyboard = [[
                InlineKeyboardButton("✅ Approve", callback_data="approve"),
                InlineKeyboardButton("❌ Reject", callback_data="reject"),
                InlineKeyboardButton("🛑 Kill", callback_data="kill")
            ]]
            await status_message.edit_text(
                f"⚠️ <b>Approval Required</b>\n\nOrchestrator wants to execute:\n<code>{action}</code>",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="HTML"
            )
        else:
            result = state.values.get("execution_result", "No result returned.")
            await save_completed_conversation(ai_prompt, result)
            await status_message.delete()
            await send_chunked_message(context.bot, chat_id, result)

    except Exception as e:
        logger.error(f"[Telegram User {chat_id}] Photo Error: {e}")
        await status_message.edit_text(f"❌ Error handling photo: {e}")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Store supported project documents and refresh their RAG records."""
    chat_id = str(update.message.chat_id)
    if not is_authorized(chat_id):
        return

    document = update.message.document
    extension = os.path.splitext(document.file_name or "")[1].lower()
    supported_extensions = {".csv", ".xlsx", ".xlsm", ".md", ".txt", ".pdf"}
    if extension not in supported_extensions:
        await update.message.reply_text(
            "❌ Unsupported file type. Attach a CSV, XLSX, XLSM, Markdown, TXT, or PDF file."
        )
        return

    status_message = await update.message.reply_text("📥 Downloading file and refreshing project knowledge...")
    try:
        file_name = os.path.basename(document.file_name)
        destination = os.path.join("data", "csv_files", file_name)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        telegram_file = await context.bot.get_file(document.file_id)
        await telegram_file.download_to_drive(destination)

        record_count = await asyncio.to_thread(refresh_project_knowledge)
        await status_message.edit_text(
            f"✅ Added <code>{html.escape(file_name)}</code> and refreshed project knowledge ({record_count} indexed records).",
            parse_mode="HTML",
        )
    except Exception as e:
        logger.error(f"[Telegram User {chat_id}] Document import failed: {e}")
        await status_message.edit_text(f"❌ Document import failed: {e}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.message.chat_id)
    if not is_authorized(chat_id):
        return

    user_text = update.message.text
    thread_config = get_thread_config(chat_id)
    status_message = await update.message.reply_text("⏳ Processing request...")

    try:
        current_text = ""
        last_edit_time = time.time()
        
        # NEW: Safely open the async database
        async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
            orchestrator = workflow.compile(checkpointer=checkpointer)
            
            async for event in orchestrator.astream_events({"user_request": user_text}, config=thread_config, version="v2"):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content and isinstance(chunk.content, str):
                        current_text += chunk.content
                        now = time.time()
                        if now - last_edit_time > 3.0 and not current_text.lstrip().startswith("{"):
                            try:
                                preview_md = current_text[-3000:] if len(current_text) > 3000 else current_text
                                preview_html = markdown_to_telegram_html(preview_md) + " ✍️"
                                await status_message.edit_text(preview_html, parse_mode="HTML", disable_web_page_preview=True)
                                last_edit_time = now
                            except Exception:
                                pass

            state = await orchestrator.aget_state(thread_config)
            
        if state.next:
            action = html.escape(str(state.values.get("proposed_action", "Unknown Action")))
            keyboard = [[
                InlineKeyboardButton("✅ Approve", callback_data="approve"),
                InlineKeyboardButton("❌ Reject", callback_data="reject"),
                InlineKeyboardButton("🛑 Kill", callback_data="kill")
            ]]
            display_text = markdown_to_telegram_html(current_text) if current_text and not current_text.lstrip().startswith("{") else "Task requires execution."
            await status_message.edit_text(
                f"{display_text}\n\n⚠️ <b>Approval Required</b>\nOrchestrator wants to execute:\n<code>{action}</code>",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="HTML"
            )
        else:
            result = state.values.get("execution_result", current_text)
            await save_completed_conversation(user_text, result)
            await status_message.delete()
            await send_chunked_message(context.bot, chat_id, result)
            await maybe_send_screenshot(context.bot, chat_id, result, user_text)

    except Exception as e:
        logger.error(f"[Telegram User {chat_id}] Error: {e}")
        await status_message.edit_text(f"❌ Error: {e}")
async def sync_dashboard_to_rag():
    """Fetches dashboard data periodically and updates the AI's memory with real metrics."""
    from src.tools.network_tools import fetch_dashboard_information
    while True:
        try:
            live_data = await asyncio.to_thread(fetch_dashboard_information)
            if live_data and "Live KPI Metrics:" in live_data:
                save_to_knowledge_base(
                    topic="AryaQ Live Dashboard Status",
                    information=f"Live Dashboard Metrics:\n{live_data}"
                )
                logger.info("Synced live dashboard metrics to RAG knowledge base.")
        except Exception as e:
            logger.warning(f"Failed to sync dashboard to RAG: {e}")

        # Wait 1 hour (3600 seconds) before syncing again
        await asyncio.sleep(3600)
async def background_health_check(app):
    """Runs continuously in the background to monitor server health."""
    admin_chat_id = os.getenv("ALLOWED_TELEGRAM_USER")
    if not admin_chat_id:
        return
        
    was_up = True
    while True:
        # Check every 15 minutes (900 seconds)
        # Tip: change to 10 seconds while testing!
        await asyncio.sleep(10) 
        
        # Ping the local server
        status = check_website_health(requires_login=False)
        is_up = "✅" in status
        
        if not is_up and was_up:
            # State flipped from UP to DOWN
            await app.bot.send_message(
                chat_id=admin_chat_id, 
                text=f"🚨 ALERT: The AryaQ Dashboard is DOWN!\nDetails: {status}"
            )
            was_up = False
            
        elif is_up and not was_up:
            # State flipped from DOWN to UP
            await app.bot.send_message(
                chat_id=admin_chat_id, 
                text="✅ RECOVERY: The AryaQ Dashboard is back ONLINE!"
            )
            was_up = True

async def post_init(app):
    """Fires exactly once when the bot starts to launch background tasks."""
    asyncio.create_task(background_health_check(app))
    asyncio.create_task(sync_dashboard_to_rag())

async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    decision = query.data
    chat_id = str(query.message.chat_id)
    thread_config = get_thread_config(chat_id)

    logger.info(f"📩 Human decision received: {decision.upper()} for chat_id={chat_id}")
    if decision == "approve":
        logger.info(f"✅ User approved the orchestrator action for chat_id={chat_id}")
    elif decision == "reject":
        logger.warning(f"❌ User rejected the orchestrator action for chat_id={chat_id}")
    elif decision == "kill":
        logger.warning(f"🛑 User killed the orchestrator action for chat_id={chat_id}")
    
    await query.edit_message_text(f"<i>Executing {decision.upper()}...</i>", parse_mode="HTML")
    
    try:
        current_text = ""
        last_edit_time = time.time()
        
        # NEW: Safely open the async database
        async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
            orchestrator = workflow.compile(checkpointer=checkpointer)
            
            async for event in orchestrator.astream_events(Command(resume=decision), config=thread_config, version="v2"):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content and isinstance(chunk.content, str):
                        current_text += chunk.content
                        now = time.time()
                        if now - last_edit_time > 3.0 and not current_text.lstrip().startswith("{"):
                            try:
                                preview_md = current_text[-3000:] if len(current_text) > 3000 else current_text
                                preview_html = markdown_to_telegram_html(preview_md) + " ✍️"
                                await query.edit_message_text(preview_html, parse_mode="HTML", disable_web_page_preview=True)
                                last_edit_time = now
                            except Exception:
                                pass
                                
            state = await orchestrator.aget_state(thread_config)
            
        if decision == 'kill':
            logger.info(f"🛑 Kill action executed for chat_id={chat_id}; graph execution terminated.")
            await query.edit_message_text("🛑 Graph execution terminated by Kill Switch.")
            return
        
        if decision == 'reject':
            logger.info(f"🚫 Rejection recorded for chat_id={chat_id}; no tool execution will continue.")
        
        if state.next:
            action = html.escape(str(state.values.get("proposed_action", "Unknown Action")))
            keyboard = [[
                InlineKeyboardButton("✅ Approve", callback_data="approve"),
                InlineKeyboardButton("❌ Reject", callback_data="reject"),
                InlineKeyboardButton("🛑 Kill", callback_data="kill")
            ]]
            display_text = markdown_to_telegram_html(current_text) if current_text and not current_text.lstrip().startswith("{") else "Task requires execution."
            await query.edit_message_text(
                f"{display_text}\n\n⚠️ <b>Next Step Required</b>\nOrchestrator wants to execute:\n<code>{action}</code>",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="HTML"
            )
        else:
            final_text = state.values.get("execution_result", current_text)
            await save_completed_conversation(state.values.get("user_request", ""), final_text)
            await query.delete_message()
            await send_chunked_message(context.bot, chat_id, final_text)
            await maybe_send_screenshot(context.bot, chat_id, final_text, state.values.get("user_request", ""))
            
    except Exception as e:
        logger.error(f"[Telegram User {chat_id}] Execution error: {e}")
        await query.edit_message_text(f"❌ Error executing tool: {e}")

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).post_init(post_init).build()
    app.add_error_handler(telegram_error_handler)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clear", clear_memory)) 
    app.add_handler(CommandHandler("refresh", refresh_knowledge))
    app.add_handler(CommandHandler("dashboard", dashboard_command))
    app.add_handler(CommandHandler("screenshot", dashboard_command))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(handle_button))
    logger.info("Telegram Bot is running securely (Async streaming enabled)...")
    app.run_polling()