"""
Summarize remote vs. self-hosted MCP servers from a Smithery metadata NDJSON dump.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional


def iter_ndjson(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count remote vs local MCP servers.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("smithery_metadata.ndjson"),
        help="NDJSON file produced by fetch_metadata.py (default: smithery_metadata.ndjson).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        print(f"Error: {args.input} does not exist.")
        return 1

    remote = 0
    local = 0
    missing = 0

    for record in iter_ndjson(args.input):
        flag = record.get("remote")
        if flag is True:
            remote += 1
        elif flag is False:
            local += 1
        else:
            missing += 1

    total = remote + local + missing
    print(f"Total servers: {total}")
    print(f"  Remote hosted: {remote}")
    print(f"  Self-host / stdio: {local}")
    print(f"  Missing flag: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
