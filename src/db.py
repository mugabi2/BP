from __future__ import annotations

import csv
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable


SCHEMA = """
CREATE TABLE IF NOT EXISTS seasons (
    season_id TEXT PRIMARY KEY,
    league_id TEXT NOT NULL,
    league_name TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    capture_started_at TEXT,
    capture_finished_at TEXT,
    duration_seconds INTEGER,
    matchdays_captured INTEGER DEFAULT 0,
    matches_captured INTEGER DEFAULT 0,
    successful TEXT DEFAULT 'no',
    error TEXT
);

CREATE TABLE IF NOT EXISTS match_results (
    season_id TEXT NOT NULL,
    matchday INTEGER NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    halftime_home_goals INTEGER,
    halftime_away_goals INTEGER,
    fulltime_home_goals INTEGER NOT NULL,
    fulltime_away_goals INTEGER NOT NULL,
    captured_at TEXT NOT NULL,
    PRIMARY KEY (season_id, matchday, home_team, away_team)
);

"""


SEASONS_FIELDS = [
    "season_id",
    "league_id",
    "league_name",
    "capture_started_at",
    "capture_finished_at",
    "duration_seconds",
    "matchdays_captured",
    "matches_captured",
    "successful",
    "error",
]
MATCH_RESULT_FIELDS = [
    "season_id",
    "matchday",
    "home_team",
    "away_team",
    "halftime_home_goals",
    "halftime_away_goals",
    "fulltime_home_goals",
    "fulltime_away_goals",
    "captured_at",
]
def connect(database_path: Path) -> sqlite3.Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    ensure_season_columns(conn)
    ensure_match_result_columns(conn)
    return conn


def ensure_season_columns(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(seasons)").fetchall()}
    migrations = {
        "capture_started_at": "ALTER TABLE seasons ADD COLUMN capture_started_at TEXT",
        "capture_finished_at": "ALTER TABLE seasons ADD COLUMN capture_finished_at TEXT",
        "duration_seconds": "ALTER TABLE seasons ADD COLUMN duration_seconds INTEGER",
        "matchdays_captured": "ALTER TABLE seasons ADD COLUMN matchdays_captured INTEGER DEFAULT 0",
        "matches_captured": "ALTER TABLE seasons ADD COLUMN matches_captured INTEGER DEFAULT 0",
        "successful": "ALTER TABLE seasons ADD COLUMN successful TEXT DEFAULT 'no'",
        "error": "ALTER TABLE seasons ADD COLUMN error TEXT",
    }
    for column, sql in migrations.items():
        if column not in columns:
            conn.execute(sql)


def ensure_match_result_columns(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(match_results)").fetchall()}
    if "home_goals" in columns or "away_goals" in columns:
        conn.execute("DROP TABLE match_results")
        conn.execute(
            """
            CREATE TABLE match_results (
                season_id TEXT NOT NULL,
                matchday INTEGER NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                halftime_home_goals INTEGER,
                halftime_away_goals INTEGER,
                fulltime_home_goals INTEGER NOT NULL,
                fulltime_away_goals INTEGER NOT NULL,
                captured_at TEXT NOT NULL,
                PRIMARY KEY (season_id, matchday, home_team, away_team)
            )
            """
        )
        return

    migrations = {
        "halftime_home_goals": "ALTER TABLE match_results ADD COLUMN halftime_home_goals INTEGER",
        "halftime_away_goals": "ALTER TABLE match_results ADD COLUMN halftime_away_goals INTEGER",
        "fulltime_home_goals": "ALTER TABLE match_results ADD COLUMN fulltime_home_goals INTEGER",
        "fulltime_away_goals": "ALTER TABLE match_results ADD COLUMN fulltime_away_goals INTEGER",
    }
    for column, sql in migrations.items():
        if column not in columns:
            conn.execute(sql)


def start_season_capture(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT INTO seasons (
            season_id, league_id, league_name, first_seen_at, last_seen_at,
            capture_started_at, capture_finished_at, duration_seconds,
            matchdays_captured, matches_captured,
            successful, error
        )
        VALUES (
            :season_id, :league_id, :league_name, :captured_at, :captured_at,
            :capture_started_at, NULL, NULL,
            0, 0,
            'no', NULL
        )
        ON CONFLICT(season_id) DO UPDATE SET
            league_id = excluded.league_id,
            league_name = excluded.league_name,
            last_seen_at = excluded.last_seen_at,
            capture_started_at = excluded.capture_started_at,
            capture_finished_at = NULL,
            duration_seconds = NULL,
            matchdays_captured = 0,
            matches_captured = 0,
            successful = 'no',
            error = NULL
        """,
        row,
    )


def finish_season_capture(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        UPDATE seasons
        SET
            capture_finished_at = :capture_finished_at,
            duration_seconds = :duration_seconds,
            matchdays_captured = :matchdays_captured,
            matches_captured = :matches_captured,
            successful = :successful,
            error = :error,
            last_seen_at = :capture_finished_at
        WHERE season_id = :season_id
        """,
        row,
    )


def upsert_season(conn: sqlite3.Connection, row: dict) -> None:
    start_season_capture(
        conn,
        {
            **row,
            "capture_started_at": row["captured_at"],
        },
    )


def captured_match_count(conn: sqlite3.Connection, season_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS count FROM match_results WHERE season_id = ?",
        (season_id,),
    ).fetchone()
    return int(row["count"] if row else 0)


def captured_matchdays(conn: sqlite3.Connection, season_id: str, matches_per_matchday: int = 10) -> set[int]:
    rows = conn.execute(
        """
        SELECT matchday, COUNT(*) AS count
        FROM match_results
        WHERE season_id = ?
        GROUP BY matchday
        HAVING COUNT(*) >= ?
        """,
        (season_id, matches_per_matchday),
    ).fetchall()
    return {int(row["matchday"]) for row in rows}


def season_is_fully_captured(
    conn: sqlite3.Connection,
    season_id: str,
    expected_matchdays: int,
    matches_per_matchday: int = 10,
) -> bool:
    row = conn.execute(
        """
        SELECT successful, matches_captured
        FROM seasons
        WHERE season_id = ?
        """,
        (season_id,),
    ).fetchone()
    if row:
        return row["successful"] == "yes"
    return captured_match_count(conn, season_id) >= expected_matchdays * matches_per_matchday


def upsert_many(conn: sqlite3.Connection, table: str, rows: Iterable[dict]) -> int:
    rows = list(rows)
    if not rows:
        return 0

    if table == "match_results":
        sql = """
            INSERT INTO match_results (
                season_id, matchday, home_team, away_team,
                halftime_home_goals, halftime_away_goals,
                fulltime_home_goals, fulltime_away_goals,
                captured_at
            )
            VALUES (
                :season_id, :matchday, :home_team, :away_team,
                :halftime_home_goals, :halftime_away_goals,
                :fulltime_home_goals, :fulltime_away_goals,
                :captured_at
            )
            ON CONFLICT(season_id, matchday, home_team, away_team) DO UPDATE SET
                halftime_home_goals = excluded.halftime_home_goals,
                halftime_away_goals = excluded.halftime_away_goals,
                fulltime_home_goals = excluded.fulltime_home_goals,
                fulltime_away_goals = excluded.fulltime_away_goals,
                captured_at = excluded.captured_at
        """
    else:
        raise ValueError(f"Unsupported table: {table}")

    before = conn.total_changes
    conn.executemany(sql, rows)
    return conn.total_changes - before


def export_csvs(conn: sqlite3.Connection, export_dir: Path) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    match_results_dir = export_dir.parent / "match_day_results"
    match_results_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = {
        "seasons": SEASONS_FIELDS,
        "match_results": MATCH_RESULT_FIELDS,
    }
    for table in ("seasons", "match_results"):
        if table == "match_results":
            rows = conn.execute(
                """
                SELECT
                    season_id,
                    matchday,
                    home_team,
                    away_team,
                    halftime_home_goals,
                    halftime_away_goals,
                    fulltime_home_goals,
                    fulltime_away_goals,
                    captured_at
                FROM match_results
                ORDER BY season_id DESC, matchday ASC, home_team ASC
                """
            ).fetchall()
        else:
            if table == "seasons":
                rows = conn.execute(
                    """
                    SELECT
                        season_id,
                        league_id,
                        league_name,
                        capture_started_at,
                        capture_finished_at,
                        duration_seconds,
                        matchdays_captured,
                        matches_captured,
                        successful,
                        error
                    FROM seasons
                    ORDER BY season_id DESC
                    """
                ).fetchall()
            else:
                rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        output_dir = match_results_dir if table == "match_results" else export_dir
        filename = "match_results_master.csv" if table == "match_results" else "seasons_master.csv"
        path = output_dir / filename
        try:
            write_csv(path, fieldnames[table], rows)
        except PermissionError:
            fallback_path = output_dir / f"{path.stem}_latest.csv"
            write_csv(fallback_path, fieldnames[table], rows)
            logging.warning(
                "Could not update %s because it is open/locked. Wrote %s instead.",
                path,
                fallback_path,
            )

    export_daily_match_results(conn, match_results_dir)


def export_daily_match_results(conn: sqlite3.Connection, export_dir: Path) -> None:
    today = datetime.now().strftime("%d_%m_%Y")
    path = export_dir / f"match_results_captured_on_{today}.csv"
    rows = conn.execute(
        """
        SELECT
            season_id,
            matchday,
            home_team,
            away_team,
            halftime_home_goals,
            halftime_away_goals,
            fulltime_home_goals,
            fulltime_away_goals,
            captured_at
        FROM match_results
        ORDER BY season_id DESC, matchday ASC, home_team ASC
        """
    ).fetchall()
    try:
        write_csv(path, MATCH_RESULT_FIELDS, rows)
    except PermissionError:
        fallback_path = export_dir / f"match_results_captured_on_{today}_latest.csv"
        write_csv(fallback_path, MATCH_RESULT_FIELDS, rows)
        logging.warning(
            "Could not update %s because it is open/locked. Wrote %s instead.",
            path,
            fallback_path,
        )


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[sqlite3.Row]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict(row) for row in rows)
