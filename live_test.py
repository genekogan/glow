#!/usr/bin/env python
"""
Live test of the agent with real API calls.

Usage:
    python live_test.py "Your prompt here"
    python live_test.py  # Uses default prompt
"""

import asyncio
import sys
import os
from dotenv import load_dotenv

load_dotenv()

from agent import run_agent
from observability import setup_logging


async def main():
    setup_logging(level="INFO")

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set in .env")
        return

    # Get prompt from command line or use default
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = "What is 6 * 7?"

    print(f"\n🚀 Prompt: \"{prompt}\"\n")
    print("=" * 60)

    result = await run_agent(prompt=prompt, cwd=".")

    if result:
        print("\n" + "=" * 60)
        print(f"📊 Session: {result['session'].id}")
        print(f"   Turns: {result['turns']}")
        print(f"   Messages: {len(result['session'].messages)}")


if __name__ == "__main__":
    asyncio.run(main())
