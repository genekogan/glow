#!/usr/bin/env python
"""
Live test of the agent with real API calls.

Usage:
    python live_test.py "Your prompt here"
    python live_test.py  # Uses default prompt
"""

import asyncio
import sys
import traceback
import os
from dotenv import load_dotenv

load_dotenv()

from observability import setup_logging


async def main():
    setup_logging(level="DEBUG")

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your-api-key-here":
        print("❌ Error: ANTHROPIC_API_KEY not set in .env")
        return

    print(f"✓ API key found: {api_key[:20]}...")

    # Get prompt from command line or use default
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = "What is 6 * 7? Then list the files in the current directory."

    print(f"\n🚀 Running agent with prompt:\n   \"{prompt}\"\n")
    print("=" * 60)

    try:
        from agent import run_agent
        result = await run_agent(prompt=prompt, cwd=".")

        if result:
            print("\n" + "=" * 60)
            print("📊 Results:")
            print(f"   Session ID: {result['session'].id}")
            print(f"   Turns: {result['turns']}")
            print(f"   Cost: ${result['cost_usd']:.4f}")
            print(f"   Input tokens: {result['session'].usage.input_tokens}")
            print(f"   Output tokens: {result['session'].usage.output_tokens}")
            print(f"   Messages: {len(result['session'].messages)}")
            print(f"   Events: {len(result['session'].events)}")

    except ExceptionGroup as eg:
        print(f"\n❌ ExceptionGroup caught:")
        for i, exc in enumerate(eg.exceptions):
            print(f"   [{i}] {type(exc).__name__}: {exc}")
            traceback.print_exception(type(exc), exc, exc.__traceback__)

    except Exception as e:
        print(f"\n❌ Exception: {type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
