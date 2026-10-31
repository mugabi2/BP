from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer: {raw}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero: {value}")
    return value


def collect_time() -> tuple[int, int]:
    raw = os.getenv("COLLECT_TIME", "13:00").strip()
    try:
        hour_text, minute_text = raw.split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"COLLECT_TIME must use HH:MM format: {raw}") from exc

    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError(f"COLLECT_TIME is outside a valid 24-hour time: {raw}")
    return hour, minute


def next_scheduled_run(now: datetime, hour: int, minute: int) -> datetime:
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now >= target:
        target += timedelta(days=1)
    return target


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

    timezone_name = os.getenv("COLLECT_TIMEZONE", "Africa/Kampala")
    timezone = ZoneInfo(timezone_name)
    hour, minute = collect_time()
    retry_seconds = positive_int("COLLECT_RETRY_SECONDS", 300)

    logging.info(
        "Collector scheduler started. daily_time=%02d:%02d timezone=%s retry=%ss",
        hour,
        minute,
        timezone_name,
        retry_seconds,
    )

    while True:
        now = datetime.now(timezone)
        scheduled = next_scheduled_run(now, hour, minute)
        sleep_seconds = max(1, int((scheduled - now).total_seconds()))
        logging.info("Next scheduled collection: %s", scheduled.isoformat())
        time.sleep(sleep_seconds)

        while True:
            started_at = time.monotonic()
            logging.info("Starting scheduled betPawa results collection.")
            return_code = run_collection()
            elapsed = int(time.monotonic() - started_at)

            if return_code == 0:
                logging.info(
                    "Collection finished successfully in %ss. Going quiet until the next daily schedule.",
                    elapsed,
                )
                break

            logging.error(
                "Collection failed with exit code %s after %ss. Retrying in %ss.",
                return_code,
                elapsed,
                retry_seconds,
            )
            time.sleep(retry_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
