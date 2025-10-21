"""
Fetch detailed MCP server metadata from the Smithery registry.

This script reads server identifiers from an NDJSON file (one JSON object per
line, containing at least a `qualifiedName` field), queries the Smithery
registry for each server, and stores the full metadata response locally.

Example usage:
    python fetch_metadata.py \
        --input smithery_servers.ndjson \
        --output-dir smithery_metadata \
        --ndjson-out smithery_metadata.ndjson \
        --sleep 0.2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib.parse import quote
from urllib3.util.retry import Retry


SMITHERY_REGISTRY_BASE = "https://registry.smithery.ai/servers"
ENV_VAR_API_KEY = "SMITHERY_API_KEY"


def load_local_env(env_filename: str = ".env") -> None:
    """
    Populate os.environ with key/value pairs from a sibling .env file.

    Only keys that are not already present in the environment are set.
    """
    env_path = Path(__file__).resolve().with_name(env_filename)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


def build_session(retries: int, backoff: float) -> requests.Session:
    """Create a requests.Session that performs basic retry/backoff handling."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def iter_ndjson(path: Path) -> Iterator[dict]:
    """Yield JSON objects from an NDJSON file."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc


def sanitize_filename(name: str) -> str:
    """
    Convert a server qualified name into a filesystem-safe filename.

    The mapping replaces any character outside of [A-Za-z0-9_.-] with an
    underscore to avoid creating unintended directory structures.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "server"


def encode_server_id(qualified_name: str) -> str:
    """
    Encode the server identifier for use in the Smithery REST path segment.

    We allow '@' to remain literal while percent-encoding forward slashes and
    any other reserved characters.
    """
    return quote(qualified_name, safe="@")


def fetch_server_metadata(
    session: requests.Session,
    server_id: str,
    api_key: str,
    timeout: int,
) -> dict:
    """Fetch metadata for a single server from the registry."""
    encoded_id = encode_server_id(server_id)
    url = f"{SMITHERY_REGISTRY_BASE}/{encoded_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = session.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def write_json(path: Path, payload: dict) -> None:
    """Write a JSON payload to disk with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def append_ndjson(path: Path, payload: dict) -> None:
    """Append a single JSON object to an NDJSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Smithery MCP metadata.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("smithery_servers.ndjson"),
        help="NDJSON file containing server listings (default: smithery_servers.ndjson).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("smithery_metadata"),
        help="Directory for per-server JSON outputs (default: smithery_metadata).",
    )
    parser.add_argument(
        "--ndjson-out",
        type=Path,
        default=None,
        help="Optional NDJSON file to append the fetched metadata to.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Seconds to sleep between requests (default: 0.2).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP request timeout in seconds (default: 30).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Number of automatic retries for failed requests (default: 5).",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=0.5,
        help="Backoff factor for retries (sleep = factor * (2 ** (retry-1))).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip servers whose JSON output already exists in --output-dir.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of servers to process.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    load_local_env()
    args = parse_args(argv)

    api_key = os.getenv(ENV_VAR_API_KEY)
    if not api_key:
        print(f"Error: set {ENV_VAR_API_KEY} in your environment or .env file.", file=sys.stderr)
        return 1

    if not args.input.exists():
        print(f"Error: input file {args.input} does not exist.", file=sys.stderr)
        return 1

    session = build_session(retries=args.retries, backoff=args.retry_backoff)

    processed = 0
    successes = 0
    failures = 0

    for record in iter_ndjson(args.input):
        qualified_name = record.get("qualifiedName") or record.get("id")
        if not qualified_name:
            continue

        safe_name = sanitize_filename(qualified_name)
        target_path = args.output_dir / f"{safe_name}.json"

        if args.resume and target_path.exists():
            print(f"[skip] {qualified_name} -> {target_path.name}")
            continue

        try:
            metadata = fetch_server_metadata(
                session=session,
                server_id=qualified_name,
                api_key=api_key,
                timeout=args.timeout,
            )
        except requests.HTTPError as http_error:
            failures += 1
            status = http_error.response.status_code if http_error.response else "unknown"
            print(f"[fail] {qualified_name} (HTTP {status}): {http_error}")
            continue
        except requests.RequestException as request_error:
            failures += 1
            print(f"[fail] {qualified_name}: {request_error}")
            continue

        write_json(target_path, metadata)
        if args.ndjson_out is not None:
            append_ndjson(args.ndjson_out, metadata)

        successes += 1
        processed += 1
        print(f"[ok]   {qualified_name} -> {target_path.name}")

        if args.limit is not None and successes >= args.limit:
            break

        if args.sleep > 0:
            time.sleep(args.sleep)

    print(
        f"Done. Successes: {successes}, failures: {failures}, "
        f"output directory: {args.output_dir}"
    )
    if args.ndjson_out is not None:
        print(f"NDJSON output appended to: {args.ndjson_out}")

    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
