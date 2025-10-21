"""
Extract full Smithery metadata entries for the MCP servers that passed
reachability tests.

Usage:
    python scraping/extract_ok_server_metadata.py \
        --ok scraping/reachability_ok_servers.ndjson \
        --metadata scraping/smithery_metadata.ndjson \
        --output scraping/remote_server_metadata.ndjson
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract metadata for reachable Smithery MCP servers.")
    parser.add_argument(
        "--ok",
        type=Path,
        default=Path("scraping/reachability_ok_servers.ndjson"),
        help="NDJSON file produced by extract_ok_servers.py",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("scraping/smithery_metadata.ndjson"),
        help="Full Smithery metadata NDJSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scraping/remote_server_metadata.ndjson"),
        help="Destination NDJSON file for the filtered metadata",
    )
    return parser.parse_args(argv)


def load_ok_identifiers(path: Path) -> set[str]:
    ok_ids: set[str] = set()
    if not path.exists():
        raise FileNotFoundError(f"OK server list not found: {path}")
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
                ok_ids.add(qualified)
    return ok_ids


def build_metadata_index(path: Path) -> Dict[str, dict]:
    index: Dict[str, dict] = {}
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
                index[qualified] = record
    return index


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    ok_ids = load_ok_identifiers(args.ok)
    metadata_index = build_metadata_index(args.metadata)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    missing = []
    written = 0
    with args.output.open("w", encoding="utf-8") as out:
        for qualified in sorted(ok_ids):
            record = metadata_index.get(qualified)
            if record is None:
                missing.append(qualified)
                continue
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Written {written} metadata entries to {args.output}")
    if missing:
        print(f"Missing {len(missing)} entries (not found in metadata file):")
        for name in missing:
            print(f"  - {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
