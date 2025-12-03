"""
Claude Agent SDK - Entry Point

Demonstrates the full infrastructure:
- Custom tools via MCP
- Session management
- Logging and observability
- Error handling
"""

import asyncio
from claude_agent_sdk import tool, create_sdk_mcp_server

from client import AgentClient, create_client, CancellationToken
from storage import MemoryStorage
from observability import get_logger, setup_logging

log = get_logger("agent")


# =============================================================================
# CUSTOM TOOLS (via MCP)
# =============================================================================

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
# RUN AGENT
# =============================================================================

async def run_agent(
    prompt: str,
    user_id: str | None = None,
    cwd: str = ".",
) -> dict:
    """
    Run the agent with full infrastructure.

    Returns the final result with session data.
    """
    async with create_client(
        storage=MemoryStorage(),
        mcp_servers={"custom": custom_tools},
        allowed_tools=["Read", "Bash", "mcp__custom__calculator", "mcp__custom__web_search"],
        system_prompt="You are a helpful assistant. Use tools when needed.",
    ) as client:
        session = await client.create_session(user_id=user_id, cwd=cwd)

        result = None
        async for event in client.run(prompt=prompt, session=session):
            match event["type"]:
                case "turn_start":
                    print(f"\n{'='*50}\n🔄 Turn {event['turn']}")
                case "text":
                    print(event["text"], end="", flush=True)
                case "tool_start":
                    print(f"\n🔧 Tool: {event['name']}")
                case "tool_end":
                    print(f"   ↳ Done")
                case "error":
                    print(f"\n❌ Error: {event['error']}")
                case "done":
                    result = event
                    print(f"\n\n{'='*50}")
                    print(f"✅ Completed in {event['turns']} turns | ${event['cost_usd']:.4f}")

        return result


async def run_agent_streaming(prompt: str, user_id: str | None = None, cwd: str = "."):
    """
    Streaming variant - yields events for external processing.

    Yields:
        {"type": "turn_start", "turn": int}
        {"type": "text", "text": str}
        {"type": "tool_start", "name": str}
        {"type": "done", "session": Session, ...}
    """
    async with create_client(
        storage=MemoryStorage(),
        mcp_servers={"custom": custom_tools},
    ) as client:
        session = await client.create_session(user_id=user_id, cwd=cwd)
        async for event in client.run(prompt=prompt, session=session):
            yield event


# =============================================================================
# DEMO
# =============================================================================

async def main():
    setup_logging(level="INFO")
    print("🚀 Claude Agent SDK Demo\n")

    result = await run_agent(
        prompt="What is 42 * 17? Then tell me what files exist in the current directory.",
        cwd="/home/user/glow",
    )

    if result:
        print(f"\n📊 Session ID: {result['session'].id}")
        print(f"📊 Total cost: ${result['session'].usage.cost_usd:.4f}")
        print(f"📊 Total tokens: {result['session'].usage.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
