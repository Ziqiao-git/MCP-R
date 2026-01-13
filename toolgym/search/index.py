"""Tool search via MCP server."""

from typing import Any
from toolgym.core.types import Tool, ToolCall, ToolResult
from toolgym.mcp import MCPClient, MCPManager
from toolgym.core.types import ServerConfig


class ToolSearchIndex:
    """Search for tools via the tool_retrieval_index MCP server."""

    def __init__(
        self,
        server_path: str = "tool_retrieval_index/server.py",
        mcp_manager: MCPManager | None = None,
    ):
        self.server_path = server_path
        self.mcp = mcp_manager
        self._client: MCPClient | None = None

    async def connect(self) -> None:
        """Connect to the search MCP server."""
        config = ServerConfig(
            name="tool_search",
            command="python",
            args=[self.server_path],
            transport="stdio",
        )

        if self.mcp:
            self.mcp.add_config(config)
            await self.mcp.connect("tool_search")
        else:
            self._client = MCPClient(config)
            await self._client.connect()

    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[dict]:
        """Search for tools matching a query.

        Returns:
            List of dicts with 'tool', 'score' keys.
        """
        tool_call = ToolCall(
            id="search_1",
            name="search_tools",
            arguments={
                "query": query,
                "top_k": top_k,
                "min_score": min_score,
            },
        )

        if self.mcp:
            result = await self.mcp.call_tool(tool_call)
        elif self._client:
            result = await self._client.call_tool(tool_call)
        else:
            raise RuntimeError("Not connected to search server")

        # Parse result into Tool objects
        return self._parse_results(result.content)

    def _parse_results(self, content: str) -> list[dict]:
        """Parse search results into Tool objects."""
        import json
        try:
            data = json.loads(content)
            results = []
            for item in data:
                tool = Tool(
                    name=item["name"],
                    description=item.get("description", ""),
                    parameters=item.get("parameters", {}),
                    server=item.get("server"),
                )
                results.append({
                    "tool": tool,
                    "score": item.get("score", 0.0),
                })
            return results
        except (json.JSONDecodeError, KeyError):
            return []

    async def disconnect(self) -> None:
        """Disconnect from search server."""
        if self._client:
            await self._client.disconnect()
            self._client = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
