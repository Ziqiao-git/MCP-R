"""Core data types for ToolGym."""

from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class MessageRole(str, Enum):
    """Role of a message in the conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A message in the conversation."""
    role: MessageRole
    content: str
    tool_calls: list["ToolCall"] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


@dataclass
class Tool:
    """A tool that can be called by an agent."""
    name: str
    description: str
    parameters: dict[str, Any]
    server: Optional[str] = None

    def to_openai_format(self) -> dict:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


@dataclass
class ToolCall:
    """A tool call made by the agent."""
    id: str
    name: str
    arguments: dict[str, Any]
    server: Optional[str] = None


@dataclass
class ToolResult:
    """Result from executing a tool."""
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class AgentState:
    """Current state of an agent."""
    messages: list[Message] = field(default_factory=list)
    tools: list[Tool] = field(default_factory=list)
    iteration: int = 0
    finished: bool = False
    final_answer: Optional[str] = None


@dataclass
class ServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: Optional[str] = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: Optional[str] = None  # For HTTP/SSE transport
    headers: dict[str, str] = field(default_factory=dict)
    transport: str = "stdio"  # "stdio", "sse", or "streamable_http"

    @classmethod
    def from_dict(cls, name: str, config: dict) -> "ServerConfig":
        """Create ServerConfig from a dictionary.

        Supports formats:
        - stdio: {"command": "python", "args": [...], "env": {...}}
        - sse: {"url": "...", "headers": {...}}
        - streamable_http: {"streamable_http": {"url": "...", "headers": {...}}}
        - Smithery format: {"streamable_http": {"url": "https://server.smithery.ai/..."}}
        """
        # Handle streamable_http format (Smithery)
        if "streamable_http" in config:
            http_config = config["streamable_http"]
            return cls(
                name=name,
                url=http_config.get("url"),
                headers=http_config.get("headers", {}),
                env=config.get("env", {}),
                transport="streamable_http",
            )

        # Handle stdio format
        if "stdio" in config:
            stdio_config = config["stdio"]
            return cls(
                name=name,
                command=stdio_config.get("command"),
                args=stdio_config.get("args", []),
                env=config.get("env", {}),
                transport="stdio",
            )

        # Handle direct format
        return cls(
            name=name,
            command=config.get("command"),
            args=config.get("args", []),
            env=config.get("env", {}),
            url=config.get("url"),
            headers=config.get("headers", {}),
            transport=config.get("transport", "stdio"),
        )

    @classmethod
    def from_smithery(cls, server_name: str) -> "ServerConfig":
        """Create config for a Smithery server by name.

        Server names can be provided with or without @ prefix.
        Example: '@user/server' or 'user/server'
        """
        # Ensure @ prefix for URL
        if not server_name.startswith('@'):
            url_name = f"@{server_name}"
        else:
            url_name = server_name
        return cls(
            name=server_name,
            url=f"https://server.smithery.ai/{url_name}",
            transport="streamable_http",
        )
