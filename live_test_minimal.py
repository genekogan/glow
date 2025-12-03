#!/usr/bin/env python
"""
Minimal direct test of Claude Agent SDK - bypasses our infrastructure.
"""

import asyncio
import os
import traceback
from dotenv import load_dotenv

load_dotenv()


async def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    print(f"✓ API key: {api_key[:20]}...")

    try:
        from claude_agent_sdk import query, ClaudeAgentOptions

        options = ClaudeAgentOptions(
            system_prompt="You are a helpful assistant.",
            permission_mode="acceptEdits",
        )

        print("\n🚀 Running minimal SDK test...\n")

        async for message in query(prompt="Say hello in one word.", options=options):
            print(f"Message type: {message.type}")
            if hasattr(message, 'event'):
                print(f"  Event: {message.event}")
            if hasattr(message, 'result'):
                print(f"  Result: {message.result}")

        print("\n✅ Done!")

    except ExceptionGroup as eg:
        print(f"\n❌ ExceptionGroup:")
        for i, exc in enumerate(eg.exceptions):
            print(f"\n--- Exception {i} ---")
            print(f"Type: {type(exc).__name__}")
            print(f"Message: {exc}")
            traceback.print_exception(type(exc), exc, exc.__traceback__)

    except Exception as e:
        print(f"\n❌ {type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
