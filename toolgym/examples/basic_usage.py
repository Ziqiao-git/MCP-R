"""Basic usage example for ToolGym."""

import asyncio
from toolgym import MCPManager, LLMManager, ReActAgent


async def main():
    # Create LLM via OpenRouter
    llm = LLMManager.create("gpt-4o", temperature=0.0)

    # Create MCP manager and load server configs
    mcp = MCPManager()
    mcp.load_configs("path/to/mcp_servers.json")

    async with mcp:
        # Connect to specific servers
        await mcp.connect("weather")
        await mcp.connect("calculator")

        # Create agent
        agent = ReActAgent(llm=llm, mcp_manager=mcp)

        # Run a query
        result = await agent.run("What's the weather in San Francisco?")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
