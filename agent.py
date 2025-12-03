"""
Claude Agent - Entry Point

Demonstrates the full infrastructure:
- Custom tools
- Session management
- Logging and observability
- Error handling
"""

import asyncio

from client import AgentClient, create_client, CancellationToken
from storage import MemoryStorage
from observability import get_logger, setup_logging

log = get_logger("agent")


# =============================================================================
# CUSTOM TOOLS
# =============================================================================

# Tool definitions in Anthropic format
TOOLS = [
    {
        "name": "calculator",
        "description": "Perform basic arithmetic operations",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform"
                },
                "a": {"type": "number", "description": "First operand"},
                "b": {"type": "number", "description": "Second operand"},
            },
            "required": ["operation", "a", "b"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web for information",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
]


# Tool handlers
async def calculator_handler(args: dict) -> str:
    """A simple calculator tool."""
    op, a, b = args["operation"], args["a"], args["b"]
    if op == "add":
        result = a + b
    elif op == "subtract":
        result = a - b
    elif op == "multiply":
        result = a * b
    elif op == "divide":
        result = a / b if b != 0 else "error: division by zero"
    else:
        result = "error: unknown operation"
    return f"{a} {op} {b} = {result}"


async def web_search_handler(args: dict) -> str:
    """Mock web search tool."""
    return f"Results for '{args['query']}': [mock search results]"


TOOL_HANDLERS = {
    "calculator": calculator_handler,
    "web_search": web_search_handler,
}


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
        tools=TOOLS,
        tool_handlers=TOOL_HANDLERS,
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
                    print(f"   ↳ {event['result']}")
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
        tools=TOOLS,
        tool_handlers=TOOL_HANDLERS,
    ) as client:
        session = await client.create_session(user_id=user_id, cwd=cwd)
        async for event in client.run(prompt=prompt, session=session):
            yield event


# =============================================================================
# DEMO
# =============================================================================

async def main():
    setup_logging(level="INFO")
    print("🚀 Claude Agent Demo\n")

    result = await run_agent(
        prompt="What is 42 * 17?",
        cwd="/home/user/glow",
    )

    if result:
        print(f"\n📊 Session ID: {result['session'].id}")
        print(f"📊 Total cost: ${result['session'].usage.cost_usd:.4f}")
        print(f"📊 Total tokens: {result['session'].usage.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
