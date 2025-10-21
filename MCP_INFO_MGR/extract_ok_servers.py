"""
Extract entries with status == "ok" from the reachability results NDJSON.

Usage:
    python scraping/extract_ok_servers.py \
        --input scraping/reachability_results.ndjson \
        --output scraping/reachability_ok_servers.ndjson
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter reachability results to only include successful entries."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("scraping/reachability_results.ndjson"),
        help="NDJSON file produced by check_remote_reachability.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scraping/reachability_ok_servers.ndjson"),
        help="Destination NDJSON file for successful entries",
    )
    parser.add_argument(
        "--status",
        type=str,
        default="ok",
        help='Status value to filter on (default: "ok")',
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        raise SystemExit(f"Input file {args.input} not found.")

    count = 0
    with args.input.open("r", encoding="utf-8") as infile, args.output.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == args.status:
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    print(f"Written {count} entries with status == {args.status!r} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
