"""ReAct (Reasoning + Acting) Agent implementation."""

from dataclasses import dataclass, field
from typing import Optional

from toolgym.core.base import BaseAgent, AgentConfig, BaseCallback
from toolgym.core.types import Message, MessageRole, Tool, ToolCall, ToolResult, AgentState
from toolgym.mcp import MCPManager
from toolgym.llm import BaseLLM


DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant with access to tools.

When you need to use a tool, call it with the appropriate arguments.
After receiving a tool result, analyze it and decide whether to:
1. Call another tool if needed
2. Provide a final answer to the user

Always think step by step and explain your reasoning."""


@dataclass
class ReActConfig(AgentConfig):
    """Configuration for ReAct agent."""
    max_iterations: int = 10
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    verbose: bool = False


class ReActAgent(BaseAgent):
    """ReAct agent that reasons and acts iteratively."""

    def __init__(
        self,
        llm: BaseLLM,
        mcp_manager: MCPManager,
        config: ReActConfig | None = None,
        callbacks: list[BaseCallback] | None = None,
    ):
        super().__init__(config or ReActConfig(), callbacks)
        self.llm = llm
        self.mcp = mcp_manager
        self.config: ReActConfig = self.config

    async def initialize(self) -> None:
        """Initialize by loading tools from connected servers."""
        # Tools are loaded when MCP manager connects
        pass

    async def run(self, query: str) -> str:
        """Execute the agent on a query and return the final answer."""
        self._notify_callbacks("on_agent_start", query)
        self.reset()

        # Build initial messages
        self.state.messages = [
            Message(role=MessageRole.SYSTEM, content=self.config.system_prompt),
            Message(role=MessageRole.USER, content=query),
        ]

        # Get available tools
        self.state.tools = self.mcp.get_tools()

        # Run ReAct loop
        while not self.state.finished and self.state.iteration < self.config.max_iterations:
            should_continue = await self.step()
            if not should_continue:
                break

        result = self.state.final_answer or "I was unable to complete the task."
        self._notify_callbacks("on_agent_end", result)
        return result

    async def step(self) -> bool:
        """Execute a single ReAct step. Returns True if agent should continue."""
        self.state.iteration += 1

        # Call LLM
        self._notify_callbacks("on_llm_start", self.state.messages)
        response = await self.llm.generate(
            messages=self.state.messages,
            tools=self.state.tools if self.state.tools else None,
        )
        self._notify_callbacks("on_llm_end", Message(
            role=MessageRole.ASSISTANT,
            content=response.content,
            tool_calls=response.tool_calls,
        ))

        # Check if we have tool calls
        if response.tool_calls:
            # Add assistant message with tool calls
            self.state.messages.append(Message(
                role=MessageRole.ASSISTANT,
                content=response.content,
                tool_calls=response.tool_calls,
            ))

            # Execute each tool call
            for tool_call in response.tool_calls:
                self._notify_callbacks("on_tool_call", tool_call)
                result = await self.mcp.call_tool(tool_call)
                self._notify_callbacks("on_tool_result", result)

                # Add tool result to messages
                self.state.messages.append(Message(
                    role=MessageRole.TOOL,
                    content=result.content,
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                ))

            return True  # Continue loop
        else:
            # No tool calls - we have a final answer
            self.state.finished = True
            self.state.final_answer = response.content
            return False

    async def cleanup(self) -> None:
        """Clean up resources."""
        pass
