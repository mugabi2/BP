from __future__ import annotations

import argparse
import logging
import sys

from src.config import load_settings
from src.db import connect, release_collection_lock, try_acquire_collection_lock
from src.scraper import collect


def configure_logging(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect betPawa virtual league results.")
    parser.add_argument(
        "command",
        choices=("collect",),
        help="collect scrapes public virtual-results data into PostgreSQL.",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Keep Chrome open at the end of collect so you can inspect what happened.",
    )
    args = parser.parse_args()

    settings = load_settings()
    configure_logging(settings.log_path)

    conn = connect(settings.database_url)
    lock_acquired = False
    try:
        lock_acquired = try_acquire_collection_lock(conn)
        if not lock_acquired:
            logging.info("Another collection run is already active. Skipping this trigger.")
            return 0

        result = collect(settings, conn, keep_open=args.keep_open)
        logging.info(
            "Done. seasons=%s skipped=%s match_changes=%s",
            result.seasons_seen,
            result.seasons_skipped,
            result.match_rows,
        )
    finally:
        if lock_acquired:
            release_collection_lock(conn)
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
