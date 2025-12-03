#!/usr/bin/env python
"""
Live test of the agent with real API calls.

Usage:
    python live_test.py "Your prompt here"
    python live_test.py  # Uses default prompt
"""

import asyncio
import sys
from dotenv import load_dotenv

load_dotenv()

from agent import run_agent
from observability import setup_logging


async def main():
    setup_logging(level="INFO")

    # Get prompt from command line or use default
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = "What is 6 * 7? Then list the files in the current directory."

    print(f"🚀 Running agent with prompt:\n   \"{prompt}\"\n")
    print("=" * 60)

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


if __name__ == "__main__":
    asyncio.run(main())
