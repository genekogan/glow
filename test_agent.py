"""
Tests for Claude Agent SDK boilerplate.
"""

import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from agent import run_agent, run_agent_streaming, calculator_handler, web_search_handler


# =============================================================================
# UNIT TESTS - Tool handlers
# =============================================================================

@pytest.mark.asyncio
async def test_calculator_add():
    result = await calculator_handler({"operation": "add", "a": 2, "b": 3})
    assert "5" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_calculator_multiply():
    result = await calculator_handler({"operation": "multiply", "a": 7, "b": 8})
    assert "56" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_web_search():
    result = await web_search_handler({"query": "python async"})
    assert "python async" in result["content"][0]["text"]


# =============================================================================
# INTEGRATION TESTS - Real API
# =============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_simple_prompt():
    """Test agent with a simple prompt."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    result = await run_agent(
        prompt="Say hello in exactly one word.",
        cwd="/tmp",
    )

    assert result is not None
    assert result["turns"] >= 1
    assert "response" in result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_with_tool():
    """Test agent uses calculator tool."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    result = await run_agent(
        prompt="Use the calculator to compute 6 * 7.",
        cwd="/tmp",
    )

    assert result is not None
    assert "42" in result["response"] or result["turns"] > 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_streaming():
    """Test streaming variant yields events."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    events = []
    async for event in run_agent_streaming("Say hi.", cwd="/tmp"):
        events.append(event)

    event_types = [e["type"] for e in events]
    assert "turn_start" in event_types
    assert "done" in event_types


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not integration"])
