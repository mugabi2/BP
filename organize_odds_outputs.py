from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from src.config import load_settings
from src.odds_scraper import ODDS_FIELDS, all_odds_dir, final_odds_dir, upsert_final_odds


def main() -> int:
    parser = argparse.ArgumentParser(description="Organize odds CSVs and rebuild final_odds.csv.")
    parser.add_argument(
        "--copy-only",
        action="store_true",
        help="Copy root odds CSVs into all_odds_files instead of moving them.",
    )
    args = parser.parse_args()

    settings = load_settings()
    captures_dir = all_odds_dir(settings)
    final_dir = final_odds_dir(settings)
    captures_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    arranged = arrange_root_csvs(settings.odds_dir, captures_dir, copy_only=args.copy_only)
    rows = read_all_wide_capture_rows(captures_dir)
    final_path = upsert_final_odds(final_dir / "final_odds.csv", rows) if rows else None

    print(f"Arranged files: {arranged}")
    print(f"Rows used for final odds: {len(rows)}")
    print(f"Final odds: {final_path or '-'}")
    return 0


def arrange_root_csvs(root: Path, captures_dir: Path, copy_only: bool = False) -> int:
    count = 0
    for path in sorted(root.glob("*.csv")):
        if path.parent == captures_dir:
            continue
        if path.name.startswith("final_odds"):
            continue

        destination = captures_dir / path.name
        if destination.exists():
            if copy_only:
                continue
            destination = unique_duplicate_path(captures_dir / "root_duplicates", path.name)

        try:
            if copy_only:
                shutil.copy2(path, destination)
            else:
                shutil.move(str(path), str(destination))
            count += 1
        except PermissionError:
            shutil.copy2(path, destination)
            count += 1
            print(f"Copied locked file instead of moving it: {path}")

    return count


def unique_duplicate_path(directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    counter = 2
    while True:
        candidate = directory / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def read_all_wide_capture_rows(captures_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(captures_dir.glob("*.csv")):
        if "_long_legacy_" in path.stem:
            continue

        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ODDS_FIELDS:
                continue
            rows.extend({field: row.get(field, "") for field in ODDS_FIELDS} for row in reader)

    rows.sort(
        key=lambda row: (
            row["captured_at"],
            int(row["matchday"] or 0),
            row["home_team"],
            row["away_team"],
        )
    )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
