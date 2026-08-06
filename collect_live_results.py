from __future__ import annotations

import argparse
import logging
import sys

from src.config import load_settings
from src.live_results import DEFAULT_LIVE_GAP_MATCHDAYS, collect_live_results


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Incrementally collect current-season live results for prediction.")
    parser.add_argument(
        "--season-id",
        help="Season id to collect. If omitted, the current visible season is used.",
    )
    parser.add_argument(
        "--target-matchday",
        type=int,
        help="Target matchday to support. If omitted, the visible matchday is used.",
    )
    parser.add_argument(
        "--live-gap-matchdays",
        type=int,
        default=DEFAULT_LIVE_GAP_MATCHDAYS,
        help="Completed-result gap before the target matchday. Default: 2.",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Keep Chrome open at the end so you can inspect what was captured.",
    )
    args = parser.parse_args()

    configure_logging()
    settings = load_settings()
    try:
        result = collect_live_results(
            settings,
            season_id=args.season_id,
            target_matchday=args.target_matchday,
            live_gap_matchdays=args.live_gap_matchdays,
            keep_open=args.keep_open,
        )
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
        "Done. season=%s target_md=%s cutoff_md=%s reused=%s fetched=%s rows=%s output=%s",
        result.season_id,
        result.target_matchday,
        result.feature_cutoff_matchday,
        ",".join(str(value) for value in result.reused_matchdays) or "-",
        ",".join(str(value) for value in result.fetched_matchdays) or "-",
        result.rows_written,
        result.output_path,
    )
    if result.incomplete_matchdays:
        logging.warning("Incomplete matchdays remain: %s", result.incomplete_matchdays)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
