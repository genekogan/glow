"""
Claude Agent SDK Kernel

A minimal implementation demonstrating the core agent loop:
- Takes a prompt and runs a multi-step agent interaction
- Executes tools when Claude requests them
- Streams updates via hooks to the caller
- Handles the complete request/response cycle

This is the heart of how Claude agents work.
"""

import os
import json
import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Any

import anthropic


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class ToolDefinition:
    """A tool Claude can invoke."""
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], Any]


@dataclass
class AgentHooks:
    """Callbacks for observing agent activity."""
    on_message: Callable[[str], None] = lambda msg: None
    on_tool_start: Callable[[str, dict], None] = lambda name, input: None
    on_tool_end: Callable[[str, Any], None] = lambda name, result: None
    on_turn_start: Callable[[int], None] = lambda turn: None
    on_turn_end: Callable[[int], None] = lambda turn: None


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    model: str = "claude-sonnet-4-20250514"
    max_turns: int = 10
    system_prompt: str = "You are a helpful assistant."
    tools: list[ToolDefinition] = field(default_factory=list)
    hooks: AgentHooks = field(default_factory=AgentHooks)


@dataclass
class AgentResult:
    """Final result from agent execution."""
    response: str
    turns_used: int
    tool_calls: list[dict]


# =============================================================================
# THE AGENT KERNEL
# =============================================================================

async def run_agent(prompt: str, config: AgentConfig) -> AgentResult:
    """
    The core agent loop.

    Takes a prompt, runs Claude in a loop until it completes or hits max turns.
    Each turn: Claude thinks -> optionally calls tools -> continues or stops.

    This is the kernel of how all Claude agents work.
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Convert our tools to Anthropic's format
    tools_for_api = [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        for tool in config.tools
    ]

    # Build tool lookup for execution
    tool_handlers = {tool.name: tool.handler for tool in config.tools}

    # Initialize conversation
    messages = [{"role": "user", "content": prompt}]
    all_tool_calls = []

    # The agent loop
    for turn in range(1, config.max_turns + 1):
        config.hooks.on_turn_start(turn)

        # Call Claude
        response = client.messages.create(
            model=config.model,
            max_tokens=4096,
            system=config.system_prompt,
            tools=tools_for_api if tools_for_api else None,
            messages=messages,
        )

        # Process the response
        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        # Extract text and tool use blocks
        text_parts = []
        tool_uses = []

        for block in assistant_content:
            if block.type == "text":
                text_parts.append(block.text)
                config.hooks.on_message(block.text)
            elif block.type == "tool_use":
                tool_uses.append(block)

        # If no tool calls, we're done
        if not tool_uses:
            config.hooks.on_turn_end(turn)
            return AgentResult(
                response="\n".join(text_parts),
                turns_used=turn,
                tool_calls=all_tool_calls,
            )

        # Execute each tool and collect results
        tool_results = []

        for tool_use in tool_uses:
            tool_name = tool_use.name
            tool_input = tool_use.input
            tool_id = tool_use.id

            config.hooks.on_tool_start(tool_name, tool_input)

            # Execute the tool
            handler = tool_handlers.get(tool_name)
            if handler:
                result = handler(tool_input)
                if asyncio.iscoroutine(result):
                    result = await result
            else:
                result = f"Error: Unknown tool '{tool_name}'"

            config.hooks.on_tool_end(tool_name, result)

            # Track for return value
            all_tool_calls.append({
                "name": tool_name,
                "input": tool_input,
                "result": result,
            })

            # Format result for Claude
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": str(result),
            })

        # Add tool results to conversation
        messages.append({"role": "user", "content": tool_results})
        config.hooks.on_turn_end(turn)

    # Hit max turns
    final_text = "\n".join(
        block.text for block in messages[-1]["content"]
        if hasattr(block, "text")
    ) if messages else ""

    return AgentResult(
        response=final_text or "[Max turns reached]",
        turns_used=config.max_turns,
        tool_calls=all_tool_calls,
    )


# =============================================================================
# STREAMING VARIANT
# =============================================================================

async def run_agent_streaming(
    prompt: str,
    config: AgentConfig
) -> AsyncIterator[dict]:
    """
    Streaming version - yields events as they happen.

    Events:
        {"type": "turn_start", "turn": 1}
        {"type": "text_delta", "text": "Hello..."}
        {"type": "tool_start", "name": "calculator", "input": {...}}
        {"type": "tool_end", "name": "calculator", "result": ...}
        {"type": "turn_end", "turn": 1}
        {"type": "done", "response": "...", "turns": 3}
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    tools_for_api = [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        for tool in config.tools
    ]

    tool_handlers = {tool.name: tool.handler for tool in config.tools}
    messages = [{"role": "user", "content": prompt}]
    all_tool_calls = []

    for turn in range(1, config.max_turns + 1):
        yield {"type": "turn_start", "turn": turn}

        # Stream the response
        collected_content = []

        with client.messages.stream(
            model=config.model,
            max_tokens=4096,
            system=config.system_prompt,
            tools=tools_for_api if tools_for_api else None,
            messages=messages,
        ) as stream:
            current_tool_input = ""
            current_tool_name = None
            current_tool_id = None

            for event in stream:
                if event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        current_tool_name = event.content_block.name
                        current_tool_id = event.content_block.id
                        current_tool_input = ""

                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        yield {"type": "text_delta", "text": event.delta.text}
                    elif event.delta.type == "input_json_delta":
                        current_tool_input += event.delta.partial_json

                elif event.type == "content_block_stop":
                    if current_tool_name:
                        # Parse and execute tool
                        try:
                            tool_input = json.loads(current_tool_input) if current_tool_input else {}
                        except json.JSONDecodeError:
                            tool_input = {}

                        yield {"type": "tool_start", "name": current_tool_name, "input": tool_input}

                        handler = tool_handlers.get(current_tool_name)
                        if handler:
                            result = handler(tool_input)
                            if asyncio.iscoroutine(result):
                                result = await result
                        else:
                            result = f"Error: Unknown tool '{current_tool_name}'"

                        yield {"type": "tool_end", "name": current_tool_name, "result": result}

                        all_tool_calls.append({
                            "name": current_tool_name,
                            "input": tool_input,
                            "result": result,
                            "id": current_tool_id,
                        })

                        current_tool_name = None
                        current_tool_id = None

            # Get final message
            final_message = stream.get_final_message()
            collected_content = final_message.content

        messages.append({"role": "assistant", "content": collected_content})

        # Check if we need to continue (tool calls present)
        tool_uses = [b for b in collected_content if b.type == "tool_use"]

        if not tool_uses:
            # Done - extract final text
            final_text = "\n".join(
                block.text for block in collected_content if block.type == "text"
            )
            yield {"type": "turn_end", "turn": turn}
            yield {
                "type": "done",
                "response": final_text,
                "turns": turn,
                "tool_calls": all_tool_calls,
            }
            return

        # Add tool results and continue
        tool_results = [
            {
                "type": "tool_result",
                "tool_use_id": tc["id"],
                "content": str(tc["result"]),
            }
            for tc in all_tool_calls[-len(tool_uses):]
        ]
        messages.append({"role": "user", "content": tool_results})

        yield {"type": "turn_end", "turn": turn}

    yield {
        "type": "done",
        "response": "[Max turns reached]",
        "turns": config.max_turns,
        "tool_calls": all_tool_calls,
    }


# =============================================================================
# EXAMPLE TOOLS
# =============================================================================

def create_calculator_tool() -> ToolDefinition:
    """A simple calculator tool."""
    def calculate(input: dict) -> str:
        op = input.get("operation")
        a = input.get("a", 0)
        b = input.get("b", 0)

        result = {
            "add": a + b,
            "subtract": a - b,
            "multiply": a * b,
            "divide": a / b if b != 0 else "Error: division by zero",
        }.get(op, f"Unknown operation: {op}")

        return f"{a} {op} {b} = {result}"

    return ToolDefinition(
        name="calculator",
        description="Perform basic arithmetic operations",
        input_schema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The operation to perform",
                },
                "a": {"type": "number", "description": "First operand"},
                "b": {"type": "number", "description": "Second operand"},
            },
            "required": ["operation", "a", "b"],
        },
        handler=calculate,
    )


def create_web_search_tool() -> ToolDefinition:
    """A mock web search tool."""
    def search(input: dict) -> str:
        query = input.get("query", "")
        # In reality, this would call a search API
        return f"Search results for '{query}': [Mock result 1, Mock result 2, Mock result 3]"

    return ToolDefinition(
        name="web_search",
        description="Search the web for information",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
            },
            "required": ["query"],
        },
        handler=search,
    )


# =============================================================================
# DEMO
# =============================================================================

async def main():
    """Demonstrate the agent kernel."""

    # Create hooks to observe agent activity
    hooks = AgentHooks(
        on_message=lambda msg: print(f"\n📝 Claude: {msg[:200]}{'...' if len(msg) > 200 else ''}"),
        on_tool_start=lambda name, input: print(f"\n🔧 Tool call: {name}({input})"),
        on_tool_end=lambda name, result: print(f"   ↳ Result: {result}"),
        on_turn_start=lambda turn: print(f"\n{'='*50}\n🔄 Turn {turn}"),
        on_turn_end=lambda turn: print(f"✓ Turn {turn} complete"),
    )

    # Configure the agent
    config = AgentConfig(
        model="claude-sonnet-4-20250514",
        max_turns=5,
        system_prompt="You are a helpful assistant with access to tools. Use them when needed.",
        tools=[
            create_calculator_tool(),
            create_web_search_tool(),
        ],
        hooks=hooks,
    )

    # Run the agent
    prompt = "What is 42 * 17? Then search for 'Python async programming' and summarize."

    print(f"\n🚀 Starting agent with prompt: {prompt}\n")

    result = await run_agent(prompt, config)

    print(f"\n{'='*50}")
    print(f"✅ Agent completed in {result.turns_used} turns")
    print(f"📊 Tool calls made: {len(result.tool_calls)}")
    print(f"\n📄 Final response:\n{result.response}")


if __name__ == "__main__":
    asyncio.run(main())
