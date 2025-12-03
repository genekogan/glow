"""
Main agent client with full infrastructure.

Wraps the Claude Agent SDK with:
- Session management and persistence
- Structured logging with session context
- Error handling and retry logic
- Langfuse observability
- Cancellation support
- Parallel tool execution
"""

import asyncio
from typing import AsyncIterator, Any, Callable
from contextlib import asynccontextmanager

from claude_agent_sdk import query, ClaudeAgentOptions

from config import settings
from models import (
    Session, Message, Event, ToolCall, User,
    SessionStatus, MessageRole, EventType,
    utcnow,
)
from storage import Storage, MemoryStorage
from observability import get_logger, LogContext
from observability.langfuse import tracer
from errors import (
    AgentError, CancelledError, classify_anthropic_error,
    with_retry, is_retryable,
)


log = get_logger("client")


# =============================================================================
# CANCELLATION TOKEN
# =============================================================================

class CancellationToken:
    """Token for cancelling agent execution."""

    def __init__(self):
        self._cancelled = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    def cancel(self):
        """Signal cancellation."""
        self._cancelled.set()
        for task in self._tasks:
            task.cancel()

    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    async def wait(self):
        """Wait until cancelled."""
        await self._cancelled.wait()

    def register_task(self, task: asyncio.Task):
        """Register a task to be cancelled."""
        self._tasks.append(task)

    def check(self):
        """Raise CancelledError if cancelled."""
        if self.is_cancelled():
            raise CancelledError()


# =============================================================================
# AGENT CLIENT
# =============================================================================

class AgentClient:
    """
    High-level client for running Claude agents.

    Handles:
    - Session lifecycle and persistence
    - User management
    - Logging and observability
    - Error handling and retries
    - Cancellation
    """

    def __init__(
        self,
        storage: Storage | None = None,
        mcp_servers: dict | None = None,
        allowed_tools: list[str] | None = None,
        system_prompt: str | None = None,
    ):
        self.storage = storage or MemoryStorage()
        self.mcp_servers = mcp_servers or {}
        self.allowed_tools = allowed_tools
        self.system_prompt = system_prompt or "You are a helpful assistant."

    async def __aenter__(self):
        await self.storage.connect()
        return self

    async def __aexit__(self, *args):
        await self.storage.disconnect()
        tracer.flush()

    # =========================================================================
    # USER MANAGEMENT
    # =========================================================================

    async def get_or_create_user(
        self,
        external_id: str,
        name: str = "",
        avatar_url: str = "",
    ) -> User:
        """Get or create a user from external auth ID."""
        user, created = await self.storage.get_or_create_user(
            external_id=external_id,
            name=name,
            avatar_url=avatar_url,
        )
        if created:
            log.info("user_created", user_id=user.id, external_id=external_id)
        return user

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    async def create_session(
        self,
        user_id: str | None = None,
        system_prompt: str | None = None,
        cwd: str = ".",
        metadata: dict | None = None,
    ) -> Session:
        """Create a new session."""
        session = Session(
            user_id=user_id,
            system_prompt=system_prompt or self.system_prompt,
            model=settings.agent.model,
            cwd=cwd,
            metadata=metadata or {},
        )
        await self.storage.create_session(session)

        log.info(
            "session_created",
            session_id=session.id,
            user_id=user_id,
        )

        return session

    async def get_session(self, session_id: str) -> Session | None:
        """Get an existing session."""
        return await self.storage.get_session(session_id)

    async def resume_session(self, session_id: str) -> Session | None:
        """Resume an existing session."""
        session = await self.storage.get_session(session_id)
        if session and session.status == SessionStatus.COMPLETED:
            session.status = SessionStatus.ACTIVE
            await self.storage.update_session(session)
        return session

    # =========================================================================
    # RUN AGENT
    # =========================================================================

    async def run(
        self,
        prompt: str,
        session: Session | None = None,
        user_id: str | None = None,
        cancel_token: CancellationToken | None = None,
        on_event: Callable[[Event], None] | None = None,
    ) -> AsyncIterator[dict]:
        """
        Run the agent with a prompt.

        Yields events as the agent works:
            {"type": "turn_start", "turn": int}
            {"type": "text", "text": str}
            {"type": "tool_start", "name": str, "input": dict}
            {"type": "tool_end", "name": str, "result": Any}
            {"type": "error", "error": str, "category": str}
            {"type": "done", "session": Session}
        """
        # Create or resume session
        if session is None:
            session = await self.create_session(user_id=user_id)

        cancel_token = cancel_token or CancellationToken()

        # Set up logging context
        with LogContext(session_id=session.id, user_id=session.user_id):
            log.info("agent_run_started", prompt=prompt[:100])

            # Create Langfuse trace
            trace = tracer.trace(
                session_id=session.id,
                user_id=session.user_id,
                name="agent_run",
            )

            try:
                async for event in self._run_agent_loop(
                    prompt=prompt,
                    session=session,
                    cancel_token=cancel_token,
                    trace=trace,
                    on_event=on_event,
                ):
                    yield event

            except CancelledError:
                session.status = SessionStatus.CANCELLED
                session.add_event(Event(type=EventType.CANCELLED))
                await self.storage.update_session(session)
                log.info("agent_cancelled")
                yield {"type": "cancelled", "session": session}

            except AgentError as e:
                session.status = SessionStatus.FAILED
                session.last_error = str(e)
                session.add_event(Event(
                    type=EventType.ERROR,
                    error_code=e.category.value,
                    error_message=str(e),
                ))
                await self.storage.update_session(session)
                log.error("agent_error", error=str(e), category=e.category.value)
                yield {"type": "error", "error": str(e), "category": e.category.value}

            except Exception as e:
                classified = classify_anthropic_error(e)
                session.status = SessionStatus.FAILED
                session.last_error = str(classified)
                await self.storage.update_session(session)
                log.error("agent_error", error=str(e))
                yield {"type": "error", "error": str(classified), "category": classified.category.value}

            finally:
                tracer.flush()

    async def _run_agent_loop(
        self,
        prompt: str,
        session: Session,
        cancel_token: CancellationToken,
        trace: Any,
        on_event: Callable[[Event], None] | None,
    ) -> AsyncIterator[dict]:
        """Internal agent loop with SDK."""

        # Add user message
        user_msg = Message(
            role=MessageRole.USER,
            content=prompt,
            user_id=session.user_id,
        )
        session.add_message(user_msg)
        await self.storage.append_message(session.id, user_msg)

        # Configure SDK options
        options = ClaudeAgentOptions(
            system_prompt=session.system_prompt,
            permission_mode=settings.agent.permission_mode,
            cwd=session.cwd,
            max_turns=settings.agent.max_turns,
            model=settings.agent.model,
            mcp_servers=self.mcp_servers,
            allowed_tools=self.allowed_tools,
            resume=session.anthropic_session_id,  # Resume if available
        )

        turn = session.turns
        current_text = ""
        current_tool_calls = []

        # Create generation in Langfuse
        generation = tracer.generation(
            trace=trace,
            name=f"turn_{turn + 1}",
            model=settings.agent.model,
            input=prompt,
        )

        try:
            async for message in query(prompt=prompt, options=options):
                # Check for cancellation
                cancel_token.check()

                msg_type = type(message).__name__

                if msg_type == "SystemMessage":
                    # System messages: init, compact_boundary, etc.
                    subtype = getattr(message, 'subtype', None)
                    data = getattr(message, 'data', {})

                    if subtype == "init":
                        # Session initialized
                        if 'session_id' in data:
                            session.anthropic_session_id = data['session_id']
                        turn += 1
                        session.turns = turn
                        yield {"type": "turn_start", "turn": turn}

                        evt = Event(type=EventType.TURN_START, turn=turn)
                        session.add_event(evt)
                        if on_event:
                            on_event(evt)

                    elif subtype == "compact_boundary":
                        trigger = data.get("trigger", "auto")
                        evt = Event(
                            type=EventType.COMPACTION,
                            data={"trigger": trigger},
                        )
                        session.add_event(evt)
                        if on_event:
                            on_event(evt)
                        yield {"type": "compaction", "trigger": trigger}

                elif msg_type == "AssistantMessage":
                    # Assistant response with content
                    content = getattr(message, 'content', [])
                    for block in content:
                        block_type = type(block).__name__
                        if block_type == "TextBlock":
                            text = getattr(block, 'text', '')
                            current_text += text
                            yield {"type": "text", "text": text}
                        elif block_type == "ToolUseBlock":
                            tool_call = ToolCall(
                                tool_use_id=getattr(block, 'id', ''),
                                name=getattr(block, 'name', ''),
                                input=getattr(block, 'input', {}),
                            )
                            current_tool_calls.append(tool_call)
                            yield {
                                "type": "tool_start",
                                "name": tool_call.name,
                                "id": tool_call.tool_use_id,
                            }

                            evt = Event(
                                type=EventType.TOOL_START,
                                turn=turn,
                                tool_call_id=tool_call.id,
                                data={"name": tool_call.name},
                            )
                            session.add_event(evt)
                            if on_event:
                                on_event(evt)

                elif msg_type == "ResultMessage":
                    # Final result
                    result_text = getattr(message, 'result', '') or current_text
                    subtype = getattr(message, 'subtype', 'success')

                    # Get session_id if available
                    if hasattr(message, 'session_id'):
                        session.anthropic_session_id = message.session_id

                    # Create assistant message
                    assistant_msg = Message(
                        role=MessageRole.ASSISTANT,
                        content=result_text,
                        tool_calls=current_tool_calls,
                        model=settings.agent.model,
                    )
                    session.add_message(assistant_msg)
                    await self.storage.append_message(session.id, assistant_msg)

                    # Complete session
                    session.complete(SessionStatus.COMPLETED)
                    await self.storage.update_session(session)

                    log.info("agent_completed", turns=session.turns)

                    # End Langfuse generation
                    generation.end(output=result_text)

                    yield {
                        "type": "done",
                        "response": result_text,
                        "session": session,
                        "turns": session.turns,
                        "cost_usd": session.usage.cost_usd,
                    }

        except asyncio.CancelledError:
            # Save partial state before raising
            if current_text:
                partial_msg = Message(
                    role=MessageRole.ASSISTANT,
                    content=current_text,
                    tool_calls=current_tool_calls,
                )
                session.add_message(partial_msg)
                await self.storage.append_message(session.id, partial_msg)
            raise CancelledError()

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    async def run_simple(
        self,
        prompt: str,
        user_id: str | None = None,
    ) -> str:
        """Simple run that returns just the final response."""
        result = None
        async for event in self.run(prompt=prompt, user_id=user_id):
            if event["type"] == "done":
                result = event["response"]
            elif event["type"] == "error":
                raise AgentError(event["error"])
        return result or ""


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

@asynccontextmanager
async def create_client(
    storage: Storage | None = None,
    **kwargs,
):
    """Create an agent client with context manager."""
    client = AgentClient(storage=storage, **kwargs)
    async with client:
        yield client
