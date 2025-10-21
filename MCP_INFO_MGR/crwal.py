# smithery_dump.py
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def load_local_env() -> None:
    """Populate os.environ with key/value pairs from a sibling .env file."""
    env_path = Path(__file__).resolve().with_name(".env")
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


load_local_env()

SMITHERY_API = "https://registry.smithery.ai/servers"
API_KEY = os.getenv("SMITHERY_API_KEY") or "REPLACE_ME"  # put your key here or export env var

DEFAULT_PAGE_SIZE = 100
DEFAULT_SLEEP = 0.2  # seconds, be polite
DEFAULT_TIMEOUT = 30
DEFAULT_OUT_NDJSON = Path("smithery_servers.ndjson")


def build_session(retries: int, backoff: float) -> requests.Session:
    """Create a requests.Session with retry support."""
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


def fetch_page(
    session: requests.Session,
    page: int,
    page_size: int,
    timeout: int,
) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params = {"page": page, "pageSize": page_size}
    response = session.get(SMITHERY_API, headers=headers, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def dump_ndjson(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Smithery MCP registry entries.")
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE, help="Number of records per page.")
    parser.add_argument("--start-page", type=int, default=1, help="Page to start scraping from.")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional maximum number of pages to fetch (relative to start-page).",
    )
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP, help="Delay between page fetches (seconds).")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=5, help="Number of automatic retries for failed requests.")
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=0.5,
        help="Backoff factor for retries (sleep = factor * (2 ** (retry-1))).",
    )
    parser.add_argument(
        "--ndjson",
        type=Path,
        default=DEFAULT_OUT_NDJSON,
        help="Where to write the NDJSON output file.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop duplicate entries (based on qualifiedName) while scraping.",
    )
    return parser.parse_args(argv)


def process_servers(
    servers: Iterable[Dict[str, Any]],
    seen: set[str] | None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Return deduplicated records and number skipped."""
    if seen is None:
        return list(servers), 0

    deduped: List[Dict[str, Any]] = []
    skipped = 0
    for rec in servers:
        qualified_name = rec.get("qualifiedName")
        if qualified_name and qualified_name in seen:
            skipped += 1
            continue
        if qualified_name:
            seen.add(qualified_name)
        deduped.append(rec)
    return deduped, skipped


def main(argv: Iterable[str] | None = None) -> None:
    if not API_KEY or API_KEY == "REPLACE_ME":
        raise SystemExit("Set SMITHERY_API_KEY env var or put your key in the script.")

    args = parse_args(argv)
    session = build_session(retries=args.retries, backoff=args.retry_backoff)

    all_servers: List[Dict[str, Any]] = []
    seen: set[str] | None = set() if args.dedupe else None

    page = args.start_page
    pages_fetched = 0
    total_skipped = 0

    try:
        while True:
            data = fetch_page(
                session=session,
                page=page,
                page_size=args.page_size,
                timeout=args.timeout,
            )
            servers = data.get("servers", [])
            pagination = data.get("pagination", {})
            total_pages = int(pagination.get("totalPages", page))

            deduped, skipped = process_servers(servers, seen)
            total_skipped += skipped
            all_servers.extend(deduped)

            fetched_count = len(deduped)
            print(
                f"Fetched page {page}/{total_pages} — {len(servers)} servers "
                f"({fetched_count} kept, {skipped} skipped)"
            )

            pages_fetched += 1
            if not servers:
                break

            # Stop criteria: total pages, max pages, or no more results
            if page >= total_pages:
                break
            if args.max_pages is not None and pages_fetched >= args.max_pages:
                break

            page += 1
            if args.sleep > 0:
                time.sleep(args.sleep)
    finally:
        session.close()

    # Save outputs
    dump_ndjson(all_servers, args.ndjson)

    print(f"Done. Saved {len(all_servers)} servers to:")
    print(f"  - {args.ndjson}")
    if total_skipped and args.dedupe:
        print(f"Skipped {total_skipped} duplicate records.")


if __name__ == "__main__":
    main()
