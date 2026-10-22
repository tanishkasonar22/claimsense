"""Call the ClaimSense MCP server from Python (the agent's investigation step uses this
when MCP_TOOLS=true; otherwise it calls claimsense.tools directly in-process).

    python -m claimsense.mcp_server.client          # lists tools and runs one call
"""
from __future__ import annotations

import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = StdioServerParameters(command=sys.executable, args=["-m", "claimsense.mcp_server.server"])


async def call_tool(name: str, arguments: dict) -> object:
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
            texts = [c.text for c in result.content if getattr(c, "type", "") == "text"]
            try:
                return json.loads(texts[0]) if len(texts) == 1 else [json.loads(t) for t in texts]
            except (json.JSONDecodeError, IndexError):
                return texts


async def list_tools() -> list[str]:
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return [t.name for t in (await session.list_tools()).tools]


def call_tool_sync(name: str, arguments: dict) -> object:
    return asyncio.run(call_tool(name, arguments))


if __name__ == "__main__":
    print("tools:", asyncio.run(list_tools()))
    print(call_tool_sync("get_policy_section", {"policy_code": "CS-STD", "section": "§3.1"}))
