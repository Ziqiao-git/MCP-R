"""DynamicReAct Agent - loads MCP servers on demand based on tool discovery."""

from dataclasses import dataclass, field
from typing import Optional

from toolgym.core.base import BaseCallback
from toolgym.core.types import Message, MessageRole, Tool, ToolCall
from toolgym.agents.react import ReActAgent, ReActConfig
from toolgym.mcp import MCPManager
from toolgym.llm import BaseLLM
from toolgym.search import ToolSearchIndex


DYNAMIC_SYSTEM_PROMPT = """You are a helpful assistant with access to a large registry of tools.

To discover tools, use the `search_tools` function with a natural language query.
The search will return relevant tools with their descriptions and which server they belong to.

When you find a useful tool from the search results, you can call it directly.
The system will automatically load the required server if needed.

Always:
1. First search for relevant tools if you're not sure what's available
2. Review the search results and pick the most appropriate tool
3. Call the tool with correct arguments
4. Analyze the result and continue or provide a final answer"""


@dataclass
class DynamicReActConfig(ReActConfig):
    """Configuration for Dynamic ReAct agent."""
    max_iterations: int = 15
    system_prompt: str = DYNAMIC_SYSTEM_PROMPT
    max_shown_tools: int = 20  # Limit tools shown to LLM to prevent context overflow


class DynamicReActAgent(ReActAgent):
    """ReAct agent that dynamically loads MCP servers based on tool discovery."""

    def __init__(
        self,
        llm: BaseLLM,
        mcp_manager: MCPManager,
        search_index: ToolSearchIndex,
        config: DynamicReActConfig | None = None,
        callbacks: list[BaseCallback] | None = None,
    ):
        super().__init__(llm, mcp_manager, config or DynamicReActConfig(), callbacks)
        self.search_index = search_index
        self.config: DynamicReActConfig = self.config
        self._discovered_tools: dict[str, Tool] = {}  # tool_name -> Tool
        self._loaded_servers: set[str] = set()

    def _get_search_tool(self) -> Tool:
        """Get the search_tools meta-tool."""
        return Tool(
            name="search_tools",
            description="Search for tools by natural language query. Returns tool names, descriptions, and which server they belong to.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query to search for tools",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
            server="meta",
        )

    async def run(self, query: str) -> str:
        """Execute the agent on a query."""
        self._notify_callbacks("on_agent_start", query)
        self.reset()
        self._discovered_tools.clear()
        self._loaded_servers.clear()

        # Build initial messages
        self.state.messages = [
            Message(role=MessageRole.SYSTEM, content=self.config.system_prompt),
            Message(role=MessageRole.USER, content=query),
        ]

        # Start with search tool + any pre-loaded tools
        self.state.tools = [self._get_search_tool()] + self.mcp.get_tools()

        # Run ReAct loop
        while not self.state.finished and self.state.iteration < self.config.max_iterations:
            should_continue = await self.step()
            if not should_continue:
                break

        result = self.state.final_answer or "I was unable to complete the task."
        self._notify_callbacks("on_agent_end", result)
        return result

    async def step(self) -> bool:
        """Execute a single step with dynamic tool loading."""
        self.state.iteration += 1

        # Limit shown tools to prevent context overflow
        shown_tools = self._get_shown_tools()

        # Call LLM
        self._notify_callbacks("on_llm_start", self.state.messages)
        response = await self.llm.generate(
            messages=self.state.messages,
            tools=shown_tools if shown_tools else None,
        )
        self._notify_callbacks("on_llm_end", Message(
            role=MessageRole.ASSISTANT,
            content=response.content,
            tool_calls=response.tool_calls,
        ))

        if response.tool_calls:
            self.state.messages.append(Message(
                role=MessageRole.ASSISTANT,
                content=response.content,
                tool_calls=response.tool_calls,
            ))

            for tool_call in response.tool_calls:
                self._notify_callbacks("on_tool_call", tool_call)

                if tool_call.name == "search_tools":
                    # Handle search tool specially
                    result = await self._handle_search(tool_call)
                else:
                    # Check if we need to load a server for this tool
                    await self._ensure_tool_loaded(tool_call.name)
                    result = await self.mcp.call_tool(tool_call)

                self._notify_callbacks("on_tool_result", result)
                self.state.messages.append(Message(
                    role=MessageRole.TOOL,
                    content=result.content,
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                ))

            return True
        else:
            self.state.finished = True
            self.state.final_answer = response.content
            return False

    async def _handle_search(self, tool_call: ToolCall) -> "ToolResult":
        """Handle search_tools call."""
        from toolgym.core.types import ToolResult
        import re

        query = tool_call.arguments.get("query", "")
        top_k = tool_call.arguments.get("top_k", 5)
        min_score = tool_call.arguments.get("min_score", 0.3)

        # Call search_tools via MCP directly (returns human-readable text)
        search_call = ToolCall(
            id=f"search_{tool_call.id}",
            name="search_tools",
            arguments={"query": query, "top_k": top_k, "min_score": min_score},
        )
        result = await self.mcp.call_tool(search_call)
        content = result.content

        # Parse the human-readable format to extract tools
        # Format: "1. **@server/name** / `tool_name`\n   Score: 0.xxx\n   Description: ...\n   Parameters: ..."
        pattern = r'\d+\.\s+\*\*([^*]+)\*\*\s*/\s*`([^`]+)`\s*\n\s*Score:\s*([\d.]+)\s*\n\s*Description:\s*([^\n]+)'
        matches = re.findall(pattern, content)

        for server, tool_name, score, description in matches:
            tool = Tool(
                name=tool_name,
                description=description.strip(),
                parameters={},  # Parameters not critical for discovery
                server=server.strip(),
            )
            self._discovered_tools[tool_name] = tool

        # Update available tools
        self._update_available_tools()

        return ToolResult(
            tool_call_id=tool_call.id,
            content=content,
        )

    async def _ensure_tool_loaded(self, tool_name: str) -> None:
        """Ensure the server for a tool is loaded."""
        # Check if tool is from discovered tools
        if tool_name in self._discovered_tools:
            tool = self._discovered_tools[tool_name]
            server = tool.server
            if server and server not in self._loaded_servers:
                if self.mcp.is_connected(server):
                    self._loaded_servers.add(server)
                elif server in self.mcp.available_servers:
                    try:
                        await self.mcp.connect(server)
                        self._loaded_servers.add(server)
                        self._update_available_tools()
                    except Exception as e:
                        # Server connection failed - mark as failed so we don't retry
                        self._loaded_servers.add(server)  # Prevent retries
                        print(f"⚠️ Failed to connect to {server}: {e}")

    def _update_available_tools(self) -> None:
        """Update the list of available tools."""
        # Combine search tool + discovered tools + MCP tools
        all_tools = {self._get_search_tool().name: self._get_search_tool()}

        # Add discovered tools
        for name, tool in self._discovered_tools.items():
            all_tools[name] = tool

        # Add tools from connected MCP servers
        for tool in self.mcp.get_tools():
            all_tools[tool.name] = tool

        self.state.tools = list(all_tools.values())

    def _get_shown_tools(self) -> list[Tool]:
        """Get tools to show to LLM (limited to prevent context overflow)."""
        # Always include search tool
        tools = [self._get_search_tool()]

        # Add recently discovered/used tools up to limit
        remaining = self.config.max_shown_tools - 1
        for tool in list(self._discovered_tools.values())[:remaining]:
            tools.append(tool)

        return tools
