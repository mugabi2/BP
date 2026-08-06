from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .db import MATCH_RESULT_FIELDS
from .scraper import (
    _body_text,
    _current_matchday,
    current_season_id,
    extract_matchday_results,
    matchday_url,
    now_iso,
    open_context,
    open_results_page,
    select_season,
    wait_for_matchday_content,
)


DEFAULT_LIVE_GAP_MATCHDAYS = 2
MATCHES_PER_MATCHDAY = 10
LIVE_RESULTS_RE = re.compile(r"^season_(?P<season_id>\d+)_tgt_md(?P<target_matchday>\d+)_results\.csv$")


@dataclass(frozen=True)
class LiveResultsCaptureResult:
    season_id: str
    target_matchday: int
    feature_cutoff_matchday: int
    output_path: Path
    previous_path: Path | None
    reused_matchdays: list[int]
    fetched_matchdays: list[int]
    incomplete_matchdays: dict[int, int]
    rows_written: int


def live_results_dir(settings: Settings) -> Path:
    return settings.database_path.parent / "live_seasons"


def live_results_path(output_dir: Path, season_id: str, target_matchday: int) -> Path:
    return output_dir / f"season_{season_id}_tgt_md{target_matchday}_results.csv"


def collect_live_results(
    settings: Settings,
    season_id: str | None = None,
    target_matchday: int | None = None,
    live_gap_matchdays: int = DEFAULT_LIVE_GAP_MATCHDAYS,
    keep_open: bool = False,
) -> LiveResultsCaptureResult:
    output_dir = live_results_dir(settings)
    output_dir.mkdir(parents=True, exist_ok=True)

    settings.user_data_dir.mkdir(parents=True, exist_ok=True)
    playwright, context, page = open_context(settings)
    try:
        open_results_page(page, settings)

        selected_season_id = str(season_id) if season_id else current_season_id(page)
        if not selected_season_id:
            raise ValueError("Could not detect the current season_id from the results page.")

        if season_id:
            select_season(page, selected_season_id, settings.action_delay_ms)

        selected_target_matchday = target_matchday or detect_target_matchday(_body_text(page))
        if not selected_target_matchday:
            raise ValueError("Could not detect target matchday. Pass --target-matchday explicitly.")
        if selected_target_matchday < 1 or selected_target_matchday > settings.matchdays_per_season:
            raise ValueError(
                f"target matchday must be between 1 and {settings.matchdays_per_season}: "
                f"{selected_target_matchday}"
            )
        if live_gap_matchdays < 0:
            raise ValueError(f"live gap matchdays cannot be negative: {live_gap_matchdays}")

        feature_cutoff_matchday = int(selected_target_matchday) - int(live_gap_matchdays)
        existing_path = find_latest_live_results_file(output_dir, selected_season_id)
        existing_rows = read_live_results(existing_path, selected_season_id) if existing_path else []
        existing_rows = [row for row in existing_rows if int(row["matchday"]) <= max(feature_cutoff_matchday, 0)]
        already_complete = complete_matchdays(existing_rows)

        needed_matchdays = list(range(1, max(feature_cutoff_matchday, 0) + 1))
        missing_matchdays = [matchday for matchday in needed_matchdays if matchday not in already_complete]

        captured_at = now_iso()
        fetched_rows: list[dict] = []
        fetched_matchdays: list[int] = []
        for matchday in missing_matchdays:
            page.goto(matchday_url(settings, selected_season_id, matchday), wait_until="load", timeout=60_000)
            page.wait_for_timeout(settings.action_delay_ms)
            wait_for_matchday_content(page, matchday, settings.page_ready_timeout_ms)

            rows = extract_matchday_results(page, selected_season_id, matchday, captured_at)
            if rows:
                fetched_rows.extend(rows)
                fetched_matchdays.append(matchday)
            logging.info(
                "Live results season=%s matchday=%s rows=%s",
                selected_season_id,
                matchday,
                len(rows),
            )

        final_rows = normalize_rows(existing_rows + fetched_rows, selected_season_id)
        final_rows = [row for row in final_rows if int(row["matchday"]) <= max(feature_cutoff_matchday, 0)]
        final_rows = sort_rows(final_rows)

        output_path = live_results_path(output_dir, selected_season_id, int(selected_target_matchday))
        write_live_results(output_path, final_rows)
        remove_old_live_results_files(output_dir, selected_season_id, output_path)

        incomplete = incomplete_matchdays(final_rows, needed_matchdays)
        if incomplete:
            logging.warning("Incomplete matchdays in %s: %s", output_path, incomplete)

        return LiveResultsCaptureResult(
            season_id=selected_season_id,
            target_matchday=int(selected_target_matchday),
            feature_cutoff_matchday=feature_cutoff_matchday,
            output_path=output_path,
            previous_path=existing_path,
            reused_matchdays=sorted(already_complete),
            fetched_matchdays=fetched_matchdays,
            incomplete_matchdays=incomplete,
            rows_written=len(final_rows),
        )
    finally:
        if keep_open:
            print("Keeping browser open. Press Enter here when you want to close it.")
            input()
        context.close()
        playwright.stop()


def find_latest_live_results_file(output_dir: Path, season_id: str) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob(f"season_{season_id}_tgt_md*_results.csv"):
        match = LIVE_RESULTS_RE.match(path.name)
        if match and match.group("season_id") == str(season_id):
            candidates.append((int(match.group("target_matchday")), path))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def detect_target_matchday(text: str) -> int:
    compact = re.sub(r"\s+", " ", text)
    match = re.search(r"\bMatchday\s*(\d+)\b\s*(?:[^\w\s]\s*)?Starts in\b", compact, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"\bMatchday\s*:?\s*(\d+)\b", compact, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return _current_matchday(text)


def read_live_results(path: Path, season_id: str) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = sorted(set(MATCH_RESULT_FIELDS) - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing_columns)}")
        return normalize_rows(reader, season_id)


def normalize_rows(rows, season_id: str) -> list[dict]:
    by_key: dict[tuple[str, int, str, str], dict] = {}
    for raw_row in rows:
        row = {field: clean_value(raw_row.get(field, "")) for field in MATCH_RESULT_FIELDS}
        if str(row["season_id"]) != str(season_id):
            continue
        if not row["matchday"] or not row["home_team"] or not row["away_team"]:
            continue
        row["season_id"] = str(row["season_id"])
        row["matchday"] = int(row["matchday"])
        row["home_team"] = str(row["home_team"]).upper().strip()
        row["away_team"] = str(row["away_team"]).upper().strip()
        key = (row["season_id"], row["matchday"], row["home_team"], row["away_team"])
        by_key[key] = row
    return list(by_key.values())


def clean_value(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def complete_matchdays(rows: list[dict], matches_per_matchday: int = MATCHES_PER_MATCHDAY) -> set[int]:
    counts: dict[int, set[tuple[str, str]]] = {}
    for row in rows:
        matchday = int(row["matchday"])
        counts.setdefault(matchday, set()).add((row["home_team"], row["away_team"]))
    return {matchday for matchday, fixtures in counts.items() if len(fixtures) >= matches_per_matchday}


def incomplete_matchdays(rows: list[dict], needed_matchdays: list[int]) -> dict[int, int]:
    counts: dict[int, set[tuple[str, str]]] = {matchday: set() for matchday in needed_matchdays}
    for row in rows:
        matchday = int(row["matchday"])
        if matchday in counts:
            counts[matchday].add((row["home_team"], row["away_team"]))
    return {
        matchday: len(fixtures)
        for matchday, fixtures in counts.items()
        if len(fixtures) < MATCHES_PER_MATCHDAY
    }


def sort_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            int(row["matchday"]),
            str(row["home_team"]),
            str(row["away_team"]),
        ),
    )


def write_live_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATCH_RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MATCH_RESULT_FIELDS})
    temp_path.replace(path)


def remove_old_live_results_files(output_dir: Path, season_id: str, keep_path: Path) -> None:
    for path in output_dir.glob(f"season_{season_id}_tgt_md*_results.csv"):
        if path.resolve() == keep_path.resolve():
            continue
        match = LIVE_RESULTS_RE.match(path.name)
        if not match or match.group("season_id") != str(season_id):
            continue
        try:
            path.unlink()
        except PermissionError:
            logging.warning("Could not remove old live results file because it is open: %s", path)
