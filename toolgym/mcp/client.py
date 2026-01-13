"""MCP Client for communicating with MCP servers."""

import asyncio
import json
import hashlib
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional
from contextlib import AsyncExitStack
from urllib.parse import urlparse

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import OAuthClientMetadata, OAuthToken

from toolgym.core.types import Tool, ToolCall, ToolResult, ServerConfig


# Default token storage directory
DEFAULT_TOKEN_DIR = Path(__file__).parent.parent.parent / "MCP_INFO_MGR" / "mcp_data" / "working" / "smithery_tokens"


def _get_server_key(server_url: str) -> str:
    """Generate a safe filename key from server URL.

    Handles Smithery URLs like https://server.smithery.ai/@user/server
    Returns key like 'user_server' to match stored token files.
    """
    parsed = urlparse(server_url)
    path = parsed.path.strip('/')
    if path:
        # Remove @ prefix and replace / with _ for safe filenames
        return path.lstrip('@').replace('/', '_')
    return hashlib.sha256(server_url.encode()).hexdigest()[:16]


class SimpleTokenStorage:
    """Simple file-based token storage compatible with MCP OAuth."""

    def __init__(self, server_url: str, storage_dir: Path = None):
        self.storage_dir = storage_dir or DEFAULT_TOKEN_DIR
        self.server_url = server_url
        key = _get_server_key(server_url)
        self.tokens_file = self.storage_dir / f"tokens_{key}.json"
        self.client_info_file = self.storage_dir / f"client_info_{key}.json"

    async def get_tokens(self) -> Optional[OAuthToken]:
        """Get stored OAuth tokens."""
        if not self.tokens_file.exists():
            return None
        try:
            with open(self.tokens_file, 'r') as f:
                data = json.load(f)
                return OAuthToken.model_validate(data)
        except Exception:
            return None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Store OAuth tokens."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        with open(self.tokens_file, 'w') as f:
            json.dump(tokens.model_dump(mode='json'), f, indent=2)

    async def get_client_info(self):
        """Get stored client info."""
        if not self.client_info_file.exists():
            return None
        try:
            with open(self.client_info_file, 'r') as f:
                data = json.load(f)
                from mcp.shared.auth import OAuthClientInformationFull
                return OAuthClientInformationFull.model_validate(data)
        except Exception:
            return None

    async def set_client_info(self, client_info) -> None:
        """Store client info."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        with open(self.client_info_file, 'w') as f:
            json.dump(client_info.model_dump(mode='json'), f, indent=2)


class MCPClient:
    """Client for communicating with a single MCP server."""

    def __init__(self, config: ServerConfig, timeout: int = 30):
        self.config = config
        self.timeout = timeout
        self.session: Optional[ClientSession] = None
        self._tools: list[Tool] = []
        self._exit_stack = AsyncExitStack()
        self._cleanup_lock = asyncio.Lock()

    async def connect(self) -> None:
        """Connect to the MCP server."""
        if self.config.transport == "streamable_http":
            await self._connect_http()
        elif self.config.transport == "sse":
            await self._connect_sse()
        else:
            await self._connect_stdio()

    async def _connect_stdio(self) -> None:
        """Connect via stdio transport."""
        server_params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=self.config.env or None,
        )
        transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read_stream, write_stream = transport
        self.session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream, read_timeout_seconds=timedelta(seconds=self.timeout))
        )
        await self.session.initialize()

    async def _connect_sse(self) -> None:
        """Connect via SSE transport."""
        transport = await self._exit_stack.enter_async_context(
            sse_client(self.config.url)
        )
        read_stream, write_stream = transport
        self.session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream, read_timeout_seconds=timedelta(seconds=self.timeout))
        )
        await self.session.initialize()

    async def _connect_http(self) -> None:
        """Connect via streamable HTTP transport (Smithery)."""
        headers = dict(self.config.headers) if self.config.headers else {}
        storage = SimpleTokenStorage(self.config.url)

        # Build OAuth metadata from stored info or defaults
        info = await storage.get_client_info()
        client_metadata = OAuthClientMetadata(
            client_name=getattr(info, 'client_name', None) or "ToolGym Client",
            client_uri=getattr(info, 'client_uri', None) or "http://localhost/",
            redirect_uris=getattr(info, 'redirect_uris', None) or ["http://localhost:8765/oauth/callback"],
            grant_types=getattr(info, 'grant_types', None) or ["authorization_code", "refresh_token"],
            response_types=getattr(info, 'response_types', None) or ["code"],
            scope=getattr(info, 'scope', None),
            token_endpoint_auth_method=getattr(info, 'token_endpoint_auth_method', None) or "none"
        )

        # Non-interactive OAuth (uses stored tokens only)
        async def redirect_handler(url: str) -> None:
            raise RuntimeError(f"OAuth required. Authenticate at: {url}")

        async def callback_handler() -> tuple[str, Optional[str]]:
            raise RuntimeError("OAuth callback not supported")

        auth = OAuthClientProvider(
            server_url=self.config.url,
            client_metadata=client_metadata,
            storage=storage,
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
        )

        transport = await self._exit_stack.enter_async_context(
            streamablehttp_client(self.config.url, headers=headers or None, auth=auth)
        )
        read_stream, write_stream, _ = transport
        self.session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream, read_timeout_seconds=timedelta(seconds=self.timeout))
        )
        await self.session.initialize()

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        async with self._cleanup_lock:
            try:
                await self._exit_stack.aclose()
                self.session = None
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

    async def list_tools(self) -> list[Tool]:
        """Get list of available tools from the server."""
        if not self.session:
            raise RuntimeError("Not connected to server")

        result = await self.session.list_tools()
        self._tools = [
            Tool(
                name=tool.name,
                description=tool.description or "",
                parameters=tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                server=self.config.name,
            )
            for tool in result.tools
        ]
        return self._tools

    async def call_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call."""
        if not self.session:
            raise RuntimeError("Not connected to server")

        try:
            result = await self.session.call_tool(
                tool_call.name,
                arguments=tool_call.arguments,
            )

            # Format result content
            if hasattr(result, 'content') and result.content:
                content_parts = []
                for item in result.content:
                    if hasattr(item, 'text'):
                        content_parts.append(item.text)
                    else:
                        content_parts.append(str(item))
                content = "\n".join(content_parts)
            else:
                content = str(result)

            return ToolResult(
                tool_call_id=tool_call.id,
                content=content,
                is_error=getattr(result, 'isError', False),
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error calling tool: {str(e)}",
                is_error=True,
            )

    @property
    def tools(self) -> list[Tool]:
        """Get cached tools list."""
        return self._tools

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
