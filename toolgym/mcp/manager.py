"""MCP Manager for handling multiple MCP server connections."""

import json
import os
from pathlib import Path
from typing import Any, Optional

from toolgym.core.types import Tool, ToolCall, ToolResult, ServerConfig
from toolgym.mcp.client import MCPClient


class MCPManager:
    """Manages connections to multiple MCP servers."""

    def __init__(self):
        self._configs: dict[str, ServerConfig] = {}
        self._clients: dict[str, MCPClient] = {}
        self._tools: dict[str, Tool] = {}  # tool_name -> Tool
        self._tool_to_server: dict[str, str] = {}  # tool_name -> server_name

    def load_configs(self, config_path: str | Path) -> None:
        """Load server configurations from a JSON file.

        Supports formats:
        - Dict format: {"server_name": {...config...}, ...}
        - mcpServers format: {"mcpServers": {"server_name": {...}, ...}}
        - Smithery list format: ["@user/server", "@user/server2", ...]
        """
        config_path = Path(config_path)
        with open(config_path) as f:
            data = json.load(f)

        # Handle Smithery server list format
        if isinstance(data, list):
            for server_name in data:
                self._configs[server_name] = ServerConfig.from_smithery(server_name)
            return

        # Handle mcpServers wrapper
        servers = data.get("mcpServers", data)
        for name, config in servers.items():
            if isinstance(config, dict):
                self._configs[name] = ServerConfig.from_dict(name, config)

    def load_smithery_servers(self, server_names: list[str]) -> None:
        """Load configurations for Smithery servers by name."""
        for name in server_names:
            self._configs[name] = ServerConfig.from_smithery(name)

    def add_config(self, config: ServerConfig) -> None:
        """Add a single server configuration."""
        self._configs[config.name] = config

    def set_env(self, server_name: str, env: dict[str, str]) -> None:
        """Set environment variables for a server."""
        if server_name in self._configs:
            self._configs[server_name].env.update(env)

    async def connect(self, server_name: str) -> MCPClient:
        """Connect to a specific server."""
        if server_name in self._clients:
            return self._clients[server_name]

        if server_name not in self._configs:
            raise ValueError(f"Unknown server: {server_name}")

        client = MCPClient(self._configs[server_name])
        await client.connect()
        self._clients[server_name] = client

        # Cache tools
        tools = await client.list_tools()
        for tool in tools:
            self._tools[tool.name] = tool
            self._tool_to_server[tool.name] = server_name

        return client

    async def connect_all(self) -> None:
        """Connect to all configured servers."""
        for name in self._configs:
            await self.connect(name)

    async def disconnect(self, server_name: str) -> None:
        """Disconnect from a specific server."""
        if server_name in self._clients:
            await self._clients[server_name].disconnect()
            del self._clients[server_name]

            # Remove cached tools
            tools_to_remove = [
                name for name, server in self._tool_to_server.items()
                if server == server_name
            ]
            for tool_name in tools_to_remove:
                del self._tools[tool_name]
                del self._tool_to_server[tool_name]

    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        for name in list(self._clients.keys()):
            await self.disconnect(name)

    def get_tools(self, server_name: Optional[str] = None) -> list[Tool]:
        """Get all available tools, optionally filtered by server."""
        if server_name:
            return [
                tool for tool in self._tools.values()
                if tool.server == server_name
            ]
        return list(self._tools.values())

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get a specific tool by name."""
        return self._tools.get(tool_name)

    def get_server_for_tool(self, tool_name: str) -> Optional[str]:
        """Get the server name for a tool."""
        return self._tool_to_server.get(tool_name)

    async def call_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call, routing to the appropriate server."""
        server_name = tool_call.server or self._tool_to_server.get(tool_call.name)
        if not server_name:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Unknown tool: {tool_call.name}",
                is_error=True,
            )

        if server_name not in self._clients:
            # Try to connect if we have the config
            if server_name in self._configs:
                await self.connect(server_name)
            else:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    content=f"Server not connected: {server_name}",
                    is_error=True,
                )

        return await self._clients[server_name].call_tool(tool_call)

    @property
    def connected_servers(self) -> list[str]:
        """Get list of connected server names."""
        return list(self._clients.keys())

    @property
    def available_servers(self) -> list[str]:
        """Get list of all configured server names."""
        return list(self._configs.keys())

    def is_connected(self, server_name: str) -> bool:
        """Check if a server is connected."""
        return server_name in self._clients

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect_all()
