from __future__ import annotations

import argparse
import logging
import sys

from src.config import load_settings
from src.db import connect, export_csvs
from src.scraper import collect, login_once


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
        choices=("login", "collect", "export"),
        help="login saves a reusable browser session; collect scrapes and exports; export only writes CSVs.",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Keep Chrome open at the end of collect so you can inspect what happened.",
    )
    args = parser.parse_args()

    settings = load_settings()
    configure_logging(settings.log_path)

    if args.command == "login":
        login_once(settings)
        return 0

    conn = connect(settings.database_path)
    try:
        if args.command == "collect":
            result = collect(settings, conn, keep_open=args.keep_open)
            export_csvs(conn, settings.export_dir)
            logging.info(
                "Done. seasons=%s skipped=%s match_changes=%s",
                result.seasons_seen,
                result.seasons_skipped,
                result.match_rows,
            )
        elif args.command == "export":
            export_csvs(conn, settings.export_dir)
            logging.info("CSV export complete.")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
