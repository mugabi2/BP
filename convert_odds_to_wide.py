from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from src.config import load_settings
from src.odds_scraper import ODDS_COLUMNS, ODDS_FIELDS, odds_column


LONG_ODDS_FIELDS = [
    "captured_at",
    "source_url",
    "league_id",
    "league_name",
    "matchday",
    "start_label",
    "market",
    "home_team",
    "away_team",
    "selection",
    "odd",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert legacy long odds CSVs to the wide fixture format.")
    parser.add_argument(
        "--odds-dir",
        type=Path,
        default=load_settings().odds_dir,
        help="Folder containing odds CSVs.",
    )
    args = parser.parse_args()

    converted = convert_odds_folder(args.odds_dir)
    for path, rows, legacy in converted:
        print(f"{path}: {rows} rows ({legacy})")
    if not converted:
        print("No legacy long-format odds CSVs found.")
    return 0


def convert_odds_folder(odds_dir: Path) -> list[tuple[Path, int, str]]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    captures_dir = odds_dir / "all_odds_files"
    candidates = [
        path
        for path in sorted(captures_dir.glob("odds_snapshot_*.csv"))
        if "_long_legacy_" not in path.stem and "_wide_" not in path.stem
    ]
    candidates.extend(sorted(captures_dir.glob("odds_captured_on_*.csv")))
    candidates.append(captures_dir / "odds_history_master.csv")

    converted = []
    seen_paths = set()
    for path in candidates:
        if path in seen_paths or not path.exists():
            continue
        seen_paths.add(path)

        rows = read_legacy_rows(path)
        if rows is None:
            continue

        wide_rows = legacy_rows_to_wide(rows)
        output_path, legacy_label = archive_or_wide_path(path, stamp)
        write_wide_rows(output_path, wide_rows)
        converted.append((output_path, len(wide_rows), legacy_label))

    return converted


def read_legacy_rows(path: Path) -> list[dict] | None:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != LONG_ODDS_FIELDS:
            return None
        return list(reader)


def legacy_rows_to_wide(rows: list[dict]) -> list[dict]:
    fixtures = {}

    for row in rows:
        key = (
            row["captured_at"],
            row["source_url"],
            row["league_id"],
            row["league_name"],
            row["matchday"],
            row["start_label"],
            row["home_team"],
            row["away_team"],
        )
        if key not in fixtures:
            fixtures[key] = {
                "captured_at": row["captured_at"],
                "source_url": row["source_url"],
                "league_id": row["league_id"],
                "league_name": row["league_name"],
                "matchday": row["matchday"],
                "start_label": row["start_label"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                **{column: "" for column in ODDS_COLUMNS},
            }

        market = row["market"]
        selection = row["selection"]
        if market == "O/U":
            continue

        column = odds_column(market, selection)
        if column in ODDS_COLUMNS:
            fixtures[key][column] = row["odd"]

    return sorted(
        fixtures.values(),
        key=lambda row: (
            row["captured_at"],
            int(row["matchday"] or 0),
            row["home_team"],
            row["away_team"],
        ),
    )


def archive_or_wide_path(path: Path, stamp: str) -> tuple[Path, str]:
    legacy_path = path.with_name(f"{path.stem}_long_legacy_{stamp}{path.suffix}")
    try:
        path.replace(legacy_path)
        return path, str(legacy_path)
    except PermissionError:
        wide_path = path.with_name(f"{path.stem}_wide_{stamp}{path.suffix}")
        return wide_path, "locked; original left in place"


def write_wide_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ODDS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
