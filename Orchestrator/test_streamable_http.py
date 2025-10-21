"""
Simple smoke test for streamable HTTP MCP servers.

Usage:
    python MCP/scripts/test_streamable_http.py \
        --server exa \
        --config scraping/remote_server_configs.json

Requirements:
    * `SMITHERY_API_KEY` (and optional profile) exported in your environment.
    * The specified server entry must be present in the JSON config.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent  # MCP/scripts
PACKAGE_DIR = SCRIPT_DIR.parent              # MCP
PROJECT_ROOT = PACKAGE_DIR.parent           # repository root
for path in (PACKAGE_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import asyncio
from dotenv import load_dotenv

from mcpuniverse.mcp.manager import MCPManager


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test streamable HTTP MCP servers.")
    parser.add_argument(
        "--server",
        action="append",
        dest="servers",
        required=True,
        help="Qualified server name (may be given multiple times).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scraping/remote_server_configs.json"),
        help="JSON file produced by generate_remote_configs.py",
    )
    parser.add_argument(
        "--transport",
        choices=["auto", "streamable_http"],
        default="auto",
        help="Transport hint passed to MCPManager.build_client (default: auto).",
    )
    return parser.parse_args(argv)


def load_config(path: Path, server_name: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if server_name not in data:
        raise KeyError(f"Server {server_name} not found in {path}")
    return data[server_name]


async def test_server(manager: MCPManager, server_name: str, server_config: dict, transport: str) -> None:
    if server_name not in manager.list_server_names():
        manager.add_server_config(server_name, server_config)
    print(f"\nConnecting to {server_name} using transport={transport!r} …")
    try:
        client = await manager.build_client(server_name, transport=transport)
    except asyncio.CancelledError as exc:
        print(f"[cancelled] {server_name}: {exc}")
        return "cancelled"
    except Exception as exc:
        print(f"[error] {server_name}: {exc}")
        return "error"

    try:
        tools = await client.list_tools()
        print(f"Connected. {len(tools)} tool(s) available:")
        for tool in tools:
            print(f"  - {tool.name}")
        return "ok"
    finally:
        await client.cleanup()


async def run_tests(server_names: list[str], config_path: Path, transport: str) -> None:
    manager = MCPManager()
    configs = load_config_file(config_path)
    summary = {"ok": 0, "error": 0, "cancelled": 0, "skipped": 0}
    for server in server_names:
        server_config = configs.get(server)
        if not server_config:
            print(f"[skip] {server} not found in {config_path}")
            summary["skipped"] += 1
            continue
        outcome = await test_server(manager, server, server_config, transport)
        summary[outcome or "error"] += 1
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def load_config_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: Optional[Iterable[str]] = None) -> int:
    load_dotenv(str(PACKAGE_DIR / ".env"))
    args = parse_args(argv)
    asyncio.run(run_tests(args.servers, args.config, args.transport))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
