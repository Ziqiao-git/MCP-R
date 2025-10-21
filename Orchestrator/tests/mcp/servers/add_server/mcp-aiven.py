import unittest
import pytest
from mcpuniverse.mcp.manager import MCPManager


class TestCalculator(unittest.IsolatedAsyncioTestCase):
    """
    Calculator MCP server: https://github.com/githejie/mcp-server-calculator
    """

    @pytest.mark.skip
    async def test_client(self):
        manager = MCPManager()
        client = await manager.build_client(server_name="mcp-aiven")
        tools = await client.list_tools()
        print(tools)
        results = await client.execute_tool("extract-web-data", arguments={"url": "https://news.ycombinator.com/", "prompt": "From https://news.ycombinator.com/, could you extract the top 5 stories with their rank, title, points, and number of comments?"})
        print(results)
        await client.cleanup()


if __name__ == "__main__":
    unittest.main()
