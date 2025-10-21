"""
Generate MCP-Universe configuration fragments for remote Smithery MCP servers.

This script reads:
  * reachability_ok_servers.ndjson — output from extract_ok_servers / check_remote_reachability
  * smithery_metadata.ndjson — metadata fetched from the Smithery registry

For each server marked as reachable, it produces a configuration entry that
MCPManager can consume via the new `streamable_http` transport.

Usage:
    python scraping/generate_remote_configs.py \
        --ok scraping/reachability_ok_servers.ndjson \
        --metadata scraping/smithery_metadata.ndjson \
        --output scraping/remote_server_configs.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional
from urllib.parse import urlencode, urlparse, parse_qsl, urlunparse


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MCP streamable HTTP configs from Smithery metadata.")
    parser.add_argument(
        "--ok",
        type=Path,
        default=Path("scraping/reachability_ok_servers.ndjson"),
        help="NDJSON file containing servers with status == ok.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("scraping/smithery_metadata.ndjson"),
        help="Smithery registry metadata NDJSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scraping/remote_server_configs.json"),
        help="Destination JSON file for generated server configs.",
    )
    parser.add_argument(
        "--api-key-variable",
        type=str,
        default="SMITHERY_API_KEY",
        help="Environment variable placeholder to inject into URLs (default: SMITHERY_API_KEY).",
    )
    parser.add_argument(
        "--profile-variable",
        type=str,
        default="",
        help="Optional environment variable placeholder for profile query parameter.",
    )
    return parser.parse_args(argv)


def load_ok_servers(path: Path) -> Dict[str, dict]:
    servers = {}
    if not path.exists():
        raise FileNotFoundError(f"OK servers file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            qualified = record.get("qualifiedName")
            if not qualified:
                continue
            servers[qualified] = record
    return servers


def load_metadata(path: Path) -> Dict[str, dict]:
    registry: Dict[str, dict] = {}
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            qualified = record.get("qualifiedName")
            if qualified:
                registry[qualified] = record
    return registry


def inject_query_parameter(url: str, key: str, value_template: str) -> str:
    """
    Add a query parameter to a URL if it's not already present.
    """
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if key not in query:
        query[key] = value_template
    encoded = urlencode(query, doseq=True)
    return urlunparse(parsed._replace(query=encoded))


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    ok_servers = load_ok_servers(args.ok)
    metadata = load_metadata(args.metadata)

    generated: Dict[str, dict] = {}
    missing_metadata = []

    api_key_template = f"{{{{{args.api_key_variable}}}}}"
    profile_template = f"{{{{{args.profile_variable}}}}}" if args.profile_variable else ""

    for qualified, record in sorted(ok_servers.items()):
        meta = metadata.get(qualified)
        if not meta:
            missing_metadata.append(qualified)
            continue

        connections = meta.get("connections") or []
        deployment_url = None
        headers = {}

        for connection in connections:
            if connection.get("type") in {"http", "streamable-http"}:
                deployment_url = connection.get("deploymentUrl") or connection.get("url")
                connection_headers = connection.get("headers") or {}
                headers.update(connection_headers)
                break

        if not deployment_url:
            # Fallback to registry-level deploymentUrl if present
            deployment_url = meta.get("deploymentUrl")

        if not deployment_url:
            missing_metadata.append(qualified)
            continue

        url_with_key = inject_query_parameter(deployment_url, "api_key", api_key_template)
        url_with_key = url_with_key.replace("%7B%7B", "{{").replace("%7D%7D", "}}")
        if args.profile_variable:
            url_with_key = inject_query_parameter(url_with_key, "profile", profile_template)
            url_with_key = url_with_key.replace("%7B%7B", "{{").replace("%7D%7D", "}}")

        generated[qualified] = {
            "streamable_http": {
                "url": url_with_key,
                "headers": headers,
            },
            "env": {}
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(generated, handle, ensure_ascii=False, indent=2)

    print(f"Generated configs for {len(generated)} servers -> {args.output}")
    if missing_metadata:
        print(
            "Skipped servers without metadata:",
            ", ".join(missing_metadata),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
