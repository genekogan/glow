"""
Tests for the Claude Agent SDK kernel.

Unit tests use mocked API responses.
Integration tests use the real API (requires ANTHROPIC_API_KEY).
"""

import os
import pytest
import asyncio
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from agent import (
    run_agent,
    run_agent_streaming,
    AgentConfig,
    AgentHooks,
    ToolDefinition,
    create_calculator_tool,
    create_web_search_tool,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def calculator_tool():
    return create_calculator_tool()


@pytest.fixture
def simple_config(calculator_tool):
    return AgentConfig(
        model="claude-sonnet-4-20250514",
        max_turns=3,
        system_prompt="You are a test assistant.",
        tools=[calculator_tool],
    )


# =============================================================================
# UNIT TESTS (mocked API)
# =============================================================================

class MockContentBlock:
    """Mock a content block from Claude's response."""
    def __init__(self, block_type, text=None, name=None, input=None, id=None):
        self.type = block_type
        self.text = text
        self.name = name
        self.input = input
        self.id = id


class MockResponse:
    """Mock Claude API response."""
    def __init__(self, content):
        self.content = content


def test_tool_definition_creation():
    """Test that tools can be created with proper structure."""
    tool = create_calculator_tool()

    assert tool.name == "calculator"
    assert "arithmetic" in tool.description.lower()
    assert "properties" in tool.input_schema
    assert callable(tool.handler)


def test_calculator_tool_operations():
    """Test calculator tool performs operations correctly."""
    tool = create_calculator_tool()

    assert "= 5" in tool.handler({"operation": "add", "a": 2, "b": 3})
    assert "= 6" in tool.handler({"operation": "multiply", "a": 2, "b": 3})
    assert "= 2" in tool.handler({"operation": "subtract", "a": 5, "b": 3})
    assert "= 2.5" in tool.handler({"operation": "divide", "a": 5, "b": 2})


def test_calculator_division_by_zero():
    """Test calculator handles division by zero."""
    tool = create_calculator_tool()
    result = tool.handler({"operation": "divide", "a": 5, "b": 0})
    assert "division by zero" in result.lower()


def test_hooks_structure():
    """Test that hooks can be created and have proper defaults."""
    hooks = AgentHooks()

    # Default hooks should be callable no-ops
    hooks.on_message("test")
    hooks.on_tool_start("test", {})
    hooks.on_tool_end("test", "result")
    hooks.on_turn_start(1)
    hooks.on_turn_end(1)


def test_config_defaults():
    """Test AgentConfig has sensible defaults."""
    config = AgentConfig()

    assert config.max_turns == 10
    assert config.model == "claude-sonnet-4-20250514"
    assert len(config.tools) == 0
    assert config.hooks is not None


@pytest.mark.asyncio
async def test_agent_simple_response():
    """Test agent handles a simple text response without tools."""
    config = AgentConfig(max_turns=1, tools=[])

    mock_response = MockResponse([
        MockContentBlock("text", text="Hello! How can I help you?")
    ])

    with patch("agent.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        result = await run_agent("Hi there", config)

        assert result.response == "Hello! How can I help you?"
        assert result.turns_used == 1
        assert len(result.tool_calls) == 0


@pytest.mark.asyncio
async def test_agent_with_tool_call():
    """Test agent handles tool calls correctly."""
    tool = create_calculator_tool()
    config = AgentConfig(max_turns=3, tools=[tool])

    # First response: Claude wants to use the calculator
    response1 = MockResponse([
        MockContentBlock("text", text="Let me calculate that."),
        MockContentBlock("tool_use", name="calculator", input={"operation": "add", "a": 2, "b": 3}, id="tool_1"),
    ])

    # Second response: Claude provides final answer
    response2 = MockResponse([
        MockContentBlock("text", text="The result is 5."),
    ])

    with patch("agent.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [response1, response2]
        mock_anthropic.return_value = mock_client

        result = await run_agent("What is 2 + 3?", config)

        assert "5" in result.response
        assert result.turns_used == 2
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "calculator"


@pytest.mark.asyncio
async def test_hooks_are_called():
    """Test that hooks are called during agent execution."""
    tool = create_calculator_tool()

    # Track hook calls
    hook_calls = []

    hooks = AgentHooks(
        on_message=lambda msg: hook_calls.append(("message", msg)),
        on_tool_start=lambda name, input: hook_calls.append(("tool_start", name)),
        on_tool_end=lambda name, result: hook_calls.append(("tool_end", name)),
        on_turn_start=lambda turn: hook_calls.append(("turn_start", turn)),
        on_turn_end=lambda turn: hook_calls.append(("turn_end", turn)),
    )

    config = AgentConfig(max_turns=3, tools=[tool], hooks=hooks)

    response1 = MockResponse([
        MockContentBlock("text", text="Calculating..."),
        MockContentBlock("tool_use", name="calculator", input={"operation": "add", "a": 1, "b": 1}, id="t1"),
    ])
    response2 = MockResponse([
        MockContentBlock("text", text="Done!"),
    ])

    with patch("agent.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [response1, response2]
        mock_anthropic.return_value = mock_client

        await run_agent("Calculate 1+1", config)

    # Verify hooks were called in order
    assert ("turn_start", 1) in hook_calls
    assert ("message", "Calculating...") in hook_calls
    assert ("tool_start", "calculator") in hook_calls
    assert ("tool_end", "calculator") in hook_calls
    assert ("turn_end", 1) in hook_calls
    assert ("turn_start", 2) in hook_calls
    assert ("message", "Done!") in hook_calls


@pytest.mark.asyncio
async def test_max_turns_limit():
    """Test that agent respects max_turns limit."""
    tool = create_calculator_tool()
    config = AgentConfig(max_turns=2, tools=[tool])

    # Always return a tool call to force multiple turns
    tool_response = MockResponse([
        MockContentBlock("tool_use", name="calculator", input={"operation": "add", "a": 1, "b": 1}, id="t1"),
    ])

    with patch("agent.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = tool_response
        mock_anthropic.return_value = mock_client

        result = await run_agent("Keep calculating forever", config)

        assert result.turns_used == 2
        assert "[Max turns reached]" in result.response


# =============================================================================
# INTEGRATION TESTS (real API)
# =============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_agent_simple():
    """Integration test: Simple prompt without tools."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    config = AgentConfig(
        max_turns=1,
        system_prompt="Respond with exactly one word: 'Hello'",
        tools=[],
    )

    result = await run_agent("Say hello", config)

    assert "Hello" in result.response or "hello" in result.response.lower()
    assert result.turns_used == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_agent_with_calculator():
    """Integration test: Agent uses calculator tool."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    config = AgentConfig(
        max_turns=3,
        system_prompt="Use the calculator tool for math. Be concise.",
        tools=[create_calculator_tool()],
    )

    result = await run_agent("What is 7 * 8?", config)

    assert len(result.tool_calls) >= 1
    assert result.tool_calls[0]["name"] == "calculator"
    assert "56" in result.response or "56" in str(result.tool_calls)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_agent_streaming():
    """Integration test: Streaming agent execution."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    config = AgentConfig(
        max_turns=2,
        system_prompt="Be very brief.",
        tools=[create_calculator_tool()],
    )

    events = []
    async for event in run_agent_streaming("What is 3 + 4?", config):
        events.append(event)

    # Should have turn start, text deltas, and done
    event_types = [e["type"] for e in events]
    assert "turn_start" in event_types
    assert "done" in event_types

    done_event = next(e for e in events if e["type"] == "done")
    assert done_event["turns"] >= 1


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    # Run unit tests by default
    pytest.main([__file__, "-v", "-m", "not integration"])
