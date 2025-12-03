"""
Claude Agent SDK - Minimal Complete Boilerplate

Demonstrates the core agent loop with:
- Multi-turn execution
- Tool calls (built-in + custom)
- Hooks for observing activity
- Streaming updates
"""

import asyncio
from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    tool,
    create_sdk_mcp_server,
)


# =============================================================================
# CUSTOM TOOLS (via MCP)
# =============================================================================

# Tool handlers (plain functions for testability)
async def calculator_handler(args):
    """A simple calculator tool."""
    op, a, b = args["operation"], args["a"], args["b"]
    result = {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b if b else "error"}[op]
    return {"content": [{"type": "text", "text": f"{a} {op} {b} = {result}"}]}


async def web_search_handler(args):
    """Mock web search tool."""
    return {"content": [{"type": "text", "text": f"Results for '{args['query']}': [mock results]"}]}


# Wrap handlers as SDK tools
calculator = tool("calculator", "Perform arithmetic", {"operation": str, "a": float, "b": float})(calculator_handler)
web_search = tool("web_search", "Search the web", {"query": str})(web_search_handler)

# Bundle tools into an MCP server
custom_tools = create_sdk_mcp_server(
    name="custom",
    version="1.0.0",
    tools=[calculator, web_search],
)


# =============================================================================
# HOOKS - Observe agent activity
# =============================================================================

async def on_pre_tool_use(input_data: dict, tool_use_id: str | None) -> dict:
    """Called before each tool execution."""
    print(f"🔧 Tool: {input_data.get('tool_name')} | Input: {input_data.get('tool_input')}")
    return {"decision": "allow"}


async def on_post_tool_use(input_data: dict, tool_use_id: str | None) -> dict:
    """Called after each tool execution."""
    print(f"   ↳ Result: {str(input_data.get('tool_result', ''))[:100]}")
    return {}


# =============================================================================
# RUN AGENT
# =============================================================================

async def run_agent(prompt: str, cwd: str = ".") -> dict:
    """
    Run the Claude agent with a prompt.

    Streams events as the agent works through turns, tool calls, and reasoning.
    Returns the final result with usage stats.
    """
    options = ClaudeAgentOptions(
        system_prompt="You are a helpful assistant. Use tools when needed.",
        permission_mode="acceptEdits",
        cwd=cwd,
        max_turns=10,
        mcp_servers={"custom": custom_tools},
        allowed_tools=["Read", "Bash", "mcp__custom__calculator", "mcp__custom__web_search"],
        hooks={
            "PreToolUse": [{"matcher": None, "hooks": [on_pre_tool_use]}],
            "PostToolUse": [{"matcher": None, "hooks": [on_post_tool_use]}],
        },
    )

    turn = 0
    final_result = None

    async for message in query(prompt=prompt, options=options):
        # Handle different message types
        match message.type:
            case "stream_event":
                event = message.event
                if event.get("type") == "message_start":
                    turn += 1
                    print(f"\n{'='*50}\n🔄 Turn {turn}")
                elif event.get("type") == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        print(delta.get("text", ""), end="", flush=True)

            case "result":
                final_result = {
                    "response": message.result,
                    "turns": message.num_turns,
                    "cost_usd": message.total_cost_usd,
                    "input_tokens": message.usage.input_tokens,
                    "output_tokens": message.usage.output_tokens,
                    "session_id": message.session_id,
                }
                print(f"\n\n{'='*50}")
                print(f"✅ Completed in {message.num_turns} turns | ${message.total_cost_usd:.4f}")

            case "system":
                if message.subtype == "compact_boundary":
                    print(f"\n📦 Context compacted at {message.trigger}")

    return final_result


# =============================================================================
# STREAMING VARIANT - Yield events for external processing
# =============================================================================

async def run_agent_streaming(prompt: str, cwd: str = "."):
    """
    Generator that yields events as the agent works.

    Yields:
        {"type": "turn_start", "turn": int}
        {"type": "text", "text": str}
        {"type": "tool_start", "name": str, "input": dict}
        {"type": "tool_end", "name": str, "result": str}
        {"type": "done", "result": dict}
    """
    options = ClaudeAgentOptions(
        system_prompt="You are a helpful assistant.",
        permission_mode="acceptEdits",
        cwd=cwd,
        max_turns=10,
        mcp_servers={"custom": custom_tools},
    )

    turn = 0

    async for message in query(prompt=prompt, options=options):
        match message.type:
            case "stream_event":
                event = message.event
                if event.get("type") == "message_start":
                    turn += 1
                    yield {"type": "turn_start", "turn": turn}
                elif event.get("type") == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield {"type": "text", "text": delta.get("text", "")}
                    elif delta.get("type") == "input_json_delta":
                        yield {"type": "tool_input_delta", "json": delta.get("partial_json", "")}
                elif event.get("type") == "content_block_start":
                    block = event.get("content_block", {})
                    if block.get("type") == "tool_use":
                        yield {"type": "tool_start", "name": block.get("name"), "id": block.get("id")}

            case "result":
                yield {
                    "type": "done",
                    "response": message.result,
                    "turns": message.num_turns,
                    "cost_usd": message.total_cost_usd,
                    "session_id": message.session_id,
                }


# =============================================================================
# DEMO
# =============================================================================

async def main():
    print("🚀 Claude Agent SDK Demo\n")

    result = await run_agent(
        prompt="What is 42 * 17? Then read the current directory and tell me what files exist.",
        cwd="/home/user/glow",
    )

    print(f"\n📊 Final: {result}")


if __name__ == "__main__":
    asyncio.run(main())
