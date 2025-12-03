"""Tests for data models."""

import pytest
from datetime import datetime
from models import (
    Session, Message, Event, ToolCall, User,
    SessionStatus, MessageRole, EventType,
    SessionUsage, utcnow, new_id,
)


class TestUser:
    def test_create_user(self):
        user = User(name="Test User", avatar_url="https://example.com/avatar.png")
        assert user.id
        assert user.name == "Test User"
        assert user.avatar_url == "https://example.com/avatar.png"
        assert user.created_at
        assert user.external_id is None

    def test_user_with_external_id(self):
        user = User(external_id="auth0|123", name="External User")
        assert user.external_id == "auth0|123"

    def test_user_metadata(self):
        user = User(metadata={"role": "admin", "plan": "pro"})
        assert user.metadata["role"] == "admin"
        assert user.metadata["plan"] == "pro"


class TestToolCall:
    def test_create_tool_call(self):
        tc = ToolCall(name="calculator", input={"a": 1, "b": 2})
        assert tc.name == "calculator"
        assert tc.input == {"a": 1, "b": 2}
        assert tc.output is None
        assert tc.error is None

    def test_complete_tool_call(self):
        tc = ToolCall(name="calculator", input={"a": 1, "b": 2})
        tc.complete(output={"result": 3})
        assert tc.output == {"result": 3}
        assert tc.ended_at is not None
        assert tc.duration_ms is not None
        assert tc.duration_ms >= 0

    def test_tool_call_with_error(self):
        tc = ToolCall(name="calculator", input={"a": 1, "b": 0})
        tc.complete(error="Division by zero")
        assert tc.error == "Division by zero"
        assert tc.output is None


class TestMessage:
    def test_create_user_message(self):
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.timestamp

    def test_create_assistant_message(self):
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="Hi there!",
            input_tokens=10,
            output_tokens=5,
            model="claude-sonnet-4-20250514",
        )
        assert msg.role == MessageRole.ASSISTANT
        assert msg.input_tokens == 10
        assert msg.output_tokens == 5

    def test_message_with_tool_calls(self):
        tc = ToolCall(name="calculator", input={"a": 1, "b": 2})
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="Let me calculate that.",
            tool_calls=[tc],
        )
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "calculator"

    def test_message_with_user_info(self):
        msg = Message(
            role=MessageRole.USER,
            content="Hello",
            user_id="user123",
            user_name="John",
            user_avatar_url="https://example.com/john.png",
        )
        assert msg.user_id == "user123"
        assert msg.user_name == "John"


class TestEvent:
    def test_create_event(self):
        evt = Event(type=EventType.TURN_START, turn=1)
        assert evt.type == EventType.TURN_START
        assert evt.turn == 1
        assert evt.timestamp

    def test_error_event(self):
        evt = Event(
            type=EventType.ERROR,
            error_code="rate_limit",
            error_message="Too many requests",
        )
        assert evt.type == EventType.ERROR
        assert evt.error_code == "rate_limit"

    def test_tool_event(self):
        evt = Event(
            type=EventType.TOOL_START,
            tool_call_id="tc123",
            data={"name": "calculator", "input": {"a": 1}},
        )
        assert evt.tool_call_id == "tc123"
        assert evt.data["name"] == "calculator"


class TestSession:
    def test_create_session(self):
        session = Session()
        assert session.id
        assert session.status == SessionStatus.ACTIVE
        assert session.turns == 0
        assert len(session.messages) == 0
        assert len(session.events) == 0

    def test_session_with_user(self):
        session = Session(user_id="user123", system_prompt="You are helpful.")
        assert session.user_id == "user123"
        assert session.system_prompt == "You are helpful."

    def test_add_message(self):
        session = Session()
        msg = Message(role=MessageRole.USER, content="Hello")
        session.add_message(msg)
        assert len(session.messages) == 1
        assert session.messages[0].content == "Hello"

    def test_add_event(self):
        session = Session()
        evt = Event(type=EventType.SESSION_START)
        session.add_event(evt)
        assert len(session.events) == 1
        assert session.events[0].type == EventType.SESSION_START

    def test_complete_session(self):
        session = Session()
        assert session.status == SessionStatus.ACTIVE
        session.complete()
        assert session.status == SessionStatus.COMPLETED
        assert session.ended_at is not None

    def test_complete_session_with_status(self):
        session = Session()
        session.complete(SessionStatus.CANCELLED)
        assert session.status == SessionStatus.CANCELLED

    def test_update_usage(self):
        session = Session()
        session.update_usage(input_tokens=100, output_tokens=50, cost_usd=0.01)
        assert session.usage.input_tokens == 100
        assert session.usage.output_tokens == 50
        assert session.usage.total_tokens == 150
        assert session.usage.cost_usd == 0.01

    def test_cumulative_usage(self):
        session = Session()
        session.update_usage(input_tokens=100, output_tokens=50, cost_usd=0.01)
        session.update_usage(input_tokens=100, output_tokens=50, cost_usd=0.01)
        assert session.usage.input_tokens == 200
        assert session.usage.output_tokens == 100
        assert session.usage.total_tokens == 300
        assert session.usage.cost_usd == 0.02

    def test_session_metadata(self):
        session = Session(metadata={"source": "api", "version": "1.0"})
        assert session.metadata["source"] == "api"


class TestHelpers:
    def test_utcnow(self):
        now = utcnow()
        assert isinstance(now, datetime)

    def test_new_id(self):
        id1 = new_id()
        id2 = new_id()
        assert id1 != id2
        assert len(id1) == 36  # UUID format
