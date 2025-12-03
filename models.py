"""
Data models mirroring Anthropic's structure with extensions.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.utcnow()


def new_id() -> str:
    return str(uuid4())


# =============================================================================
# ENUMS
# =============================================================================

class SessionStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class EventType(str, Enum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    ERROR = "error"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    TEXT_DELTA = "text_delta"
    COMPACTION = "compaction"


# =============================================================================
# USER
# =============================================================================

class User(BaseModel):
    """User model."""
    id: str = Field(default_factory=new_id)
    external_id: str | None = None  # ID from auth system
    name: str = ""
    avatar_url: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# TOOL CALL
# =============================================================================

class ToolCall(BaseModel):
    """A single tool invocation."""
    id: str = Field(default_factory=new_id)
    tool_use_id: str | None = None  # Anthropic's tool_use_id
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    error: str | None = None
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime | None = None
    duration_ms: int | None = None

    def complete(self, output: Any = None, error: str | None = None):
        self.output = output
        self.error = error
        self.ended_at = utcnow()
        if self.started_at:
            self.duration_ms = int((self.ended_at - self.started_at).total_seconds() * 1000)


# =============================================================================
# MESSAGE
# =============================================================================

class Message(BaseModel):
    """A single message in the conversation."""
    id: str = Field(default_factory=new_id)
    role: MessageRole
    content: str | list[dict[str, Any]] = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=utcnow)
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None
    model: str | None = None
    # Denormalized user info for fast frontend rendering
    user_id: str | None = None
    user_name: str | None = None
    user_avatar_url: str | None = None


# =============================================================================
# EVENT
# =============================================================================

class Event(BaseModel):
    """An event in the session timeline."""
    id: str = Field(default_factory=new_id)
    type: EventType
    timestamp: datetime = Field(default_factory=utcnow)
    data: dict[str, Any] = Field(default_factory=dict)
    turn: int | None = None
    message_id: str | None = None
    tool_call_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None


# =============================================================================
# SESSION
# =============================================================================

class SessionUsage(BaseModel):
    """Aggregated usage stats for a session."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


class Session(BaseModel):
    """A conversation session with the agent."""
    id: str = Field(default_factory=new_id)
    anthropic_session_id: str | None = None  # For SDK resumption
    user_id: str | None = None
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime | None = None
    messages: list[Message] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    turns: int = 0
    usage: SessionUsage = Field(default_factory=SessionUsage)
    metadata: dict[str, Any] = Field(default_factory=dict)
    system_prompt: str | None = None
    model: str | None = None
    cwd: str | None = None
    last_error: str | None = None

    def add_message(self, message: Message):
        self.messages.append(message)
        self.updated_at = utcnow()

    def add_event(self, event: Event):
        self.events.append(event)
        self.updated_at = utcnow()

    def complete(self, status: SessionStatus = SessionStatus.COMPLETED):
        self.status = status
        self.ended_at = utcnow()
        self.updated_at = utcnow()

    def update_usage(self, input_tokens: int = 0, output_tokens: int = 0, cost_usd: float = 0.0):
        self.usage.input_tokens += input_tokens
        self.usage.output_tokens += output_tokens
        self.usage.total_tokens = self.usage.input_tokens + self.usage.output_tokens
        self.usage.cost_usd += cost_usd
        self.updated_at = utcnow()
