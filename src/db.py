from __future__ import annotations

from typing import Iterable

import psycopg
from psycopg.rows import dict_row


COLLECTION_LOCK_ID = 7794001


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


SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS seasons (
        season_id TEXT PRIMARY KEY,
        league_id TEXT NOT NULL,
        league_name TEXT NOT NULL,
        first_seen_at TIMESTAMPTZ NOT NULL,
        last_seen_at TIMESTAMPTZ NOT NULL,
        capture_started_at TIMESTAMPTZ,
        capture_finished_at TIMESTAMPTZ,
        duration_seconds INTEGER,
        matchdays_captured INTEGER DEFAULT 0,
        matches_captured INTEGER DEFAULT 0,
        successful TEXT DEFAULT 'no',
        error TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS match_results (
        season_id TEXT NOT NULL,
        matchday INTEGER NOT NULL,
        home_team TEXT NOT NULL,
        away_team TEXT NOT NULL,
        halftime_home_goals INTEGER,
        halftime_away_goals INTEGER,
        fulltime_home_goals INTEGER NOT NULL,
        fulltime_away_goals INTEGER NOT NULL,
        captured_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (season_id, matchday, home_team, away_team)
    )
    """,
)


def connect(database_url: str):
    conn = psycopg.connect(database_url, row_factory=dict_row)
    for statement in SCHEMA_STATEMENTS:
        conn.execute(statement)
    conn.commit()
    return conn


def try_acquire_collection_lock(conn) -> bool:
    row = conn.execute(
        "SELECT pg_try_advisory_lock(%s) AS acquired",
        (COLLECTION_LOCK_ID,),
    ).fetchone()
    conn.commit()
    return bool(row and row["acquired"])


def release_collection_lock(conn) -> None:
    conn.execute(
        "SELECT pg_advisory_unlock(%s)",
        (COLLECTION_LOCK_ID,),
    )
    conn.commit()


def start_season_capture(conn, row: dict) -> None:
    conn.execute(
        """
        INSERT INTO seasons (
            season_id, league_id, league_name, first_seen_at, last_seen_at,
            capture_started_at, capture_finished_at, duration_seconds,
            matchdays_captured, matches_captured,
            successful, error
        )
        VALUES (
            %(season_id)s, %(league_id)s, %(league_name)s, %(captured_at)s, %(captured_at)s,
            %(capture_started_at)s, NULL, NULL,
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


def finish_season_capture(conn, row: dict) -> None:
    conn.execute(
        """
        UPDATE seasons
        SET
            capture_finished_at = %(capture_finished_at)s,
            duration_seconds = %(duration_seconds)s,
            matchdays_captured = %(matchdays_captured)s,
            matches_captured = %(matches_captured)s,
            successful = %(successful)s,
            error = %(error)s,
            last_seen_at = %(capture_finished_at)s
        WHERE season_id = %(season_id)s
        """,
        row,
    )


def upsert_season(conn, row: dict) -> None:
    start_season_capture(
        conn,
        {
            **row,
            "capture_started_at": row["captured_at"],
        },
    )


def captured_match_count(conn, season_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS count FROM match_results WHERE season_id = %s",
        (season_id,),
    ).fetchone()
    return int(row["count"] if row else 0)


def captured_matchdays(conn, season_id: str, matches_per_matchday: int = 10) -> set[int]:
    rows = conn.execute(
        """
        SELECT matchday, COUNT(*) AS count
        FROM match_results
        WHERE season_id = %s
        GROUP BY matchday
        HAVING COUNT(*) >= %s
        """,
        (season_id, matches_per_matchday),
    ).fetchall()
    return {int(row["matchday"]) for row in rows}


def season_is_fully_captured(
    conn,
    season_id: str,
    expected_matchdays: int,
    matches_per_matchday: int = 10,
) -> bool:
    row = conn.execute(
        """
        SELECT successful, matches_captured
        FROM seasons
        WHERE season_id = %s
        """,
        (season_id,),
    ).fetchone()
    if row:
        return row["successful"] == "yes"
    return captured_match_count(conn, season_id) >= expected_matchdays * matches_per_matchday


def upsert_many(conn, table: str, rows: Iterable[dict]) -> int:
    rows = list(rows)
    if not rows:
        return 0

    if table != "match_results":
        raise ValueError(f"Unsupported table: {table}")

    sql = """
        INSERT INTO match_results (
            season_id, matchday, home_team, away_team,
            halftime_home_goals, halftime_away_goals,
            fulltime_home_goals, fulltime_away_goals,
            captured_at
        )
        VALUES (
            %(season_id)s, %(matchday)s, %(home_team)s, %(away_team)s,
            %(halftime_home_goals)s, %(halftime_away_goals)s,
            %(fulltime_home_goals)s, %(fulltime_away_goals)s,
            %(captured_at)s
        )
        ON CONFLICT(season_id, matchday, home_team, away_team) DO UPDATE SET
            halftime_home_goals = excluded.halftime_home_goals,
            halftime_away_goals = excluded.halftime_away_goals,
            fulltime_home_goals = excluded.fulltime_home_goals,
            fulltime_away_goals = excluded.fulltime_away_goals,
            captured_at = excluded.captured_at
    """
    with conn.cursor() as cursor:
        cursor.executemany(sql, rows)
    return len(rows)
