from __future__ import annotations

import logging
import os
import subprocess
import sys
import time


def positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer: {raw}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero: {value}")
    return value


def run_collection() -> int:
    completed = subprocess.run(
        [sys.executable, "collect_virtuals.py", "collect"],
        check=False,
    )
    return completed.returncode


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    interval_seconds = positive_int("COLLECT_INTERVAL_SECONDS", 10_800)
    retry_seconds = positive_int("COLLECT_RETRY_SECONDS", 300)

    logging.info(
        "Collector scheduler started. interval=%ss retry=%ss",
        interval_seconds,
        retry_seconds,
    )

    while True:
        started_at = time.monotonic()
        logging.info("Starting betPawa results collection.")
        return_code = run_collection()
        elapsed = int(time.monotonic() - started_at)

        if return_code == 0:
            sleep_seconds = max(1, interval_seconds - elapsed)
            logging.info(
                "Collection finished successfully in %ss. Next run in %ss.",
                elapsed,
                sleep_seconds,
            )
        else:
            sleep_seconds = retry_seconds
            logging.error(
                "Collection failed with exit code %s after %ss. Retrying in %ss.",
                return_code,
                elapsed,
                sleep_seconds,
            )

        time.sleep(sleep_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
