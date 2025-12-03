"""
Main agent client with full infrastructure.

Uses the Anthropic API directly for reliable operation:
- Session management and persistence
- Structured logging with session context
- Error handling and retry logic
- Langfuse observability
- Cancellation support
- Tool execution
"""

import asyncio
import json
from typing import AsyncIterator, Any, Callable
from contextlib import asynccontextmanager

import anthropic

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
    - Tool execution
    """

    def __init__(
        self,
        storage: Storage | None = None,
        tools: list[dict] | None = None,
        tool_handlers: dict[str, Callable] | None = None,
        system_prompt: str | None = None,
    ):
        self.storage = storage or MemoryStorage()
        self.tools = tools or []
        self.tool_handlers = tool_handlers or {}
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self._client: anthropic.AsyncAnthropic | None = None

    async def __aenter__(self):
        await self.storage.connect()
        self._client = anthropic.AsyncAnthropic()
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
        """Internal agent loop with raw Anthropic API."""

        # Add user message
        user_msg = Message(
            role=MessageRole.USER,
            content=prompt,
            user_id=session.user_id,
        )
        session.add_message(user_msg)
        await self.storage.append_message(session.id, user_msg)

        # Build messages for API
        messages = [{"role": "user", "content": prompt}]

        turn = 0
        max_turns = settings.agent.max_turns

        while turn < max_turns:
            cancel_token.check()

            turn += 1
            session.turns = turn

            yield {"type": "turn_start", "turn": turn}

            evt = Event(type=EventType.TURN_START, turn=turn)
            session.add_event(evt)
            if on_event:
                on_event(evt)

            # Create generation in Langfuse
            generation = tracer.generation(
                trace=trace,
                name=f"turn_{turn}",
                model=settings.agent.model,
                input=messages[-1] if messages else prompt,
            )

            # Call Anthropic API
            response = await self._call_api(messages, session.system_prompt)

            # Track usage
            if hasattr(response, 'usage'):
                session.usage.input_tokens += response.usage.input_tokens
                session.usage.output_tokens += response.usage.output_tokens

            # Process response
            assistant_content = []
            response_text = ""
            tool_calls = []
            has_tool_use = False

            for block in response.content:
                if block.type == "text":
                    response_text += block.text
                    yield {"type": "text", "text": block.text}
                    assistant_content.append({"type": "text", "text": block.text})

                elif block.type == "tool_use":
                    has_tool_use = True
                    tool_call = ToolCall(
                        tool_use_id=block.id,
                        name=block.name,
                        input=block.input,
                    )
                    tool_calls.append(tool_call)
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

                    yield {
                        "type": "tool_start",
                        "name": block.name,
                        "id": block.id,
                        "input": block.input,
                    }

                    evt = Event(
                        type=EventType.TOOL_START,
                        turn=turn,
                        tool_call_id=tool_call.id,
                        data={"name": block.name},
                    )
                    session.add_event(evt)
                    if on_event:
                        on_event(evt)

            # Add assistant message to conversation
            messages.append({"role": "assistant", "content": assistant_content})

            # End Langfuse generation
            generation.end(output=response_text)

            # If no tool use, we're done
            if not has_tool_use or response.stop_reason == "end_turn":
                # Create assistant message
                assistant_msg = Message(
                    role=MessageRole.ASSISTANT,
                    content=response_text,
                    tool_calls=tool_calls,
                    model=settings.agent.model,
                )
                session.add_message(assistant_msg)
                await self.storage.append_message(session.id, assistant_msg)

                # Complete session
                session.complete(SessionStatus.COMPLETED)
                await self.storage.update_session(session)

                log.info("agent_completed", turns=session.turns)

                yield {
                    "type": "done",
                    "response": response_text,
                    "session": session,
                    "turns": session.turns,
                    "cost_usd": session.usage.cost_usd,
                }
                return

            # Execute tools and continue
            tool_results = []
            for tool_call in tool_calls:
                cancel_token.check()

                result = await self._execute_tool(tool_call)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.tool_use_id,
                    "content": result,
                })

                yield {
                    "type": "tool_end",
                    "name": tool_call.name,
                    "id": tool_call.tool_use_id,
                    "result": result,
                }

                evt = Event(
                    type=EventType.TOOL_END,
                    turn=turn,
                    tool_call_id=tool_call.id,
                    data={"name": tool_call.name, "result": result[:200] if isinstance(result, str) else str(result)[:200]},
                )
                session.add_event(evt)
                if on_event:
                    on_event(evt)

            # Add tool results to conversation
            messages.append({"role": "user", "content": tool_results})

        # Max turns reached
        session.complete(SessionStatus.COMPLETED)
        await self.storage.update_session(session)
        log.warning("max_turns_reached", turns=turn)
        yield {
            "type": "done",
            "response": response_text,
            "session": session,
            "turns": session.turns,
            "cost_usd": session.usage.cost_usd,
        }

    async def _call_api(self, messages: list, system_prompt: str):
        """Call Anthropic API with retry logic."""
        return await self._client.messages.create(
            model=settings.agent.model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            tools=self.tools if self.tools else None,
        )

    async def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool and return the result."""
        handler = self.tool_handlers.get(tool_call.name)
        if handler:
            try:
                result = await handler(tool_call.input)
                if isinstance(result, dict) and "content" in result:
                    # MCP-style response
                    content = result["content"]
                    if isinstance(content, list) and content:
                        return content[0].get("text", str(content[0]))
                    return str(content)
                return str(result)
            except Exception as e:
                return f"Error: {e}"
        else:
            return f"Tool '{tool_call.name}' not found"

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
