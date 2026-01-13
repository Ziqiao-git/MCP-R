"""Dynamic agent example with on-demand tool discovery."""

import asyncio
from toolgym import MCPManager, LLMManager, DynamicReActAgent
from toolgym.search import ToolSearchIndex


async def main():
    # Create LLM via OpenRouter
    llm = LLMManager.create("gpt-4o", temperature=0.0)

    # Create MCP manager with all available servers
    mcp = MCPManager()
    mcp.load_configs("path/to/all_servers.json")

    # Create search index (connects to tool_retrieval_index MCP server)
    search = ToolSearchIndex(
        server_path="tool_retrieval_index/server.py",
        mcp_manager=mcp,
    )

    async with mcp:
        await search.connect()

        # Create dynamic agent
        agent = DynamicReActAgent(
            llm=llm,
            mcp_manager=mcp,
            search_index=search,
        )

        # Run a query - agent will discover and load tools as needed
        result = await agent.run(
            "Find restaurants near the Golden Gate Bridge "
            "and get directions from downtown SF"
        )
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
