"""ClaimSense MCP server (booklet Phase 1.5, study guide D2).

Exposes the read-only tools from claimsense/tools.py over the Model Context Protocol, so
the agent, Claude Desktop, or the MCP Inspector can all use them.

Run it:
    python -m claimsense.mcp_server.server                    # stdio (for Claude Desktop / Inspector)
    npx @modelcontextprotocol/inspector python -m claimsense.mcp_server.server

Claude Desktop config (claude_desktop_config.json):
    {"mcpServers": {"claimsense": {"command": "python",
        "args": ["-m", "claimsense.mcp_server.server"], "cwd": "/path/to/claimsense"}}}
"""
from __future__ import annotations

try:  # mcp 2.x renamed FastMCP to MCPServer
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from .. import tools

mcp = _Server("claimsense")


@mcp.tool()
def search_policy(policy_code: str, question: str, k: int = 4) -> list[dict]:
    """Search one insurance policy (e.g. CS-STD) for the sections most relevant to a question.
    Returns section numbers and text to cite."""
    return tools.search_policy(policy_code, question, k)


@mcp.tool()
def get_policy_section(policy_code: str, section: str) -> dict:
    """Get the full text of one policy section, for example section "§3.1" of policy "CS-STD"."""
    return tools.get_policy_section(policy_code, section)


@mcp.tool()
def get_claim_history(customer_id: str) -> dict:
    """Summarize a customer's past claims (count, outcomes, incident types). Customer IDs look like C00123."""
    return tools.get_claim_history(customer_id)


@mcp.tool()
def get_repair_shop_stats(shop_name: str) -> dict:
    """How many past claims involved this repair shop and what share were confirmed fraud."""
    return tools.get_repair_shop_stats(shop_name)


if __name__ == "__main__":
    mcp.run()  # stdio transport
