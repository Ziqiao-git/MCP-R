"""Entry point for Pipedream MCP server"""
import argparse
import asyncio
from mcp.server.stdio import stdio_server
from .server import server


async def main():
    parser = argparse.ArgumentParser(description="Pipedream MCP Server")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse"])
    parser.add_argument("--port", type=int, default=8000, help="Port for SSE transport")
    args = parser.parse_args()

    if args.transport == "stdio":
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
    elif args.transport == "sse":
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Route
        import uvicorn

        sse = SseServerTransport("/messages")
        
        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await server.run(
                    streams[0], streams[1], server.create_initialization_options()
                )
        
        app = Starlette(
            routes=[Route("/messages", endpoint=handle_sse)]
        )
        
        uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    asyncio.run(main())