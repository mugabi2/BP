from __future__ import annotations

import argparse
import logging
import sys

from src.config import load_settings
from src.odds_scraper import collect_odds


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect betPawa virtual league odds snapshots.")
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Keep Chrome open at the end so you can inspect what was captured.",
    )
    args = parser.parse_args()

    configure_logging()
    settings = load_settings()
    try:
        result = collect_odds(settings, keep_open=args.keep_open)
    except Exception as exc:
        message = str(exc)
        if "ProcessSingleton" in message or "Lock file can not be created" in message:
            logging.error(
                "Chrome could not open the saved profile at %s because it is already in use. "
                "Close the collector Chrome window that uses auth/browser-profile, then run this again.",
                settings.user_data_dir,
            )
            return 1
        raise

    logging.info(
        "Done. rows=%s markets=%s snapshot=%s daily=%s history=%s",
        result.rows,
        ",".join(result.markets_seen) if result.markets_seen else "-",
        result.snapshot_path or "-",
        result.daily_path or "-",
        result.history_path or "-",
    )
    logging.info("Final odds file=%s", result.final_path or "-")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
