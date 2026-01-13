"""Base classes for agents and callbacks."""

from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass, field

from toolgym.core.types import Message, Tool, ToolCall, ToolResult, AgentState


@dataclass
class AgentConfig:
    """Base configuration for agents."""
    max_iterations: int = 10
    system_prompt: Optional[str] = None
    verbose: bool = False


class BaseCallback(ABC):
    """Base class for agent callbacks."""

    def on_agent_start(self, query: str) -> None:
        """Called when agent starts processing a query."""
        pass

    def on_agent_end(self, result: str) -> None:
        """Called when agent finishes processing."""
        pass

    def on_tool_call(self, tool_call: ToolCall) -> None:
        """Called when a tool is about to be called."""
        pass

    def on_tool_result(self, result: ToolResult) -> None:
        """Called when a tool returns a result."""
        pass

    def on_llm_start(self, messages: list[Message]) -> None:
        """Called when LLM is about to be called."""
        pass

    def on_llm_end(self, response: Message) -> None:
        """Called when LLM returns a response."""
        pass

    def on_error(self, error: Exception) -> None:
        """Called when an error occurs."""
        pass


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(
        self,
        config: AgentConfig,
        callbacks: list[BaseCallback] | None = None,
    ):
        self.config = config
        self.callbacks = callbacks or []
        self.state = AgentState()

    @abstractmethod
    async def run(self, query: str) -> str:
        """Execute the agent on a query and return the final answer."""
        pass

    @abstractmethod
    async def step(self) -> bool:
        """Execute a single step. Returns True if agent should continue."""
        pass

    def reset(self) -> None:
        """Reset the agent state."""
        self.state = AgentState()

    async def initialize(self) -> None:
        """Initialize the agent (load tools, connect to servers, etc.)."""
        pass

    async def cleanup(self) -> None:
        """Clean up resources (close connections, etc.)."""
        pass

    def add_callback(self, callback: BaseCallback) -> None:
        """Add a callback to the agent."""
        self.callbacks.append(callback)

    def _notify_callbacks(self, method: str, *args, **kwargs) -> None:
        """Notify all callbacks of an event."""
        for callback in self.callbacks:
            getattr(callback, method)(*args, **kwargs)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
