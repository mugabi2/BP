from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_CSV_PATH = ROOT / "Match day results" / "match_results.csv"

MATCH_RESULT_COLUMNS = [
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

REQUIRED_RESULT_COLUMNS = {
    "season_id",
    "matchday",
    "home_team",
    "away_team",
    "fulltime_home_goals",
    "fulltime_away_goals",
}

NUMERIC_RESULT_COLUMNS = [
    "matchday",
    "halftime_home_goals",
    "halftime_away_goals",
    "fulltime_home_goals",
    "fulltime_away_goals",
]


@dataclass(frozen=True)
class TrainingDataConfig:
    source: str
    csv_path: Path
    database_url: str | None = None


def resolve_training_data_config(
    csv_path: Path | str | None = None,
    data_source: str | None = None,
    database_url: str | None = None,
) -> TrainingDataConfig:
    load_dotenv(ROOT / ".env", override=False)

    raw_source = (data_source or os.getenv("TRAINING_DATA_SOURCE") or "csv").strip().lower()
    if raw_source in {"postgres", "postgresql", "pg"}:
        normalized_source = "postgres"
    elif raw_source == "csv":
        normalized_source = "csv"
    else:
        raise ValueError(
            f"Unsupported TRAINING_DATA_SOURCE={raw_source!r}. Expected 'csv' or 'postgres'."
        )

    env_csv_path = os.getenv("TRAINING_RESULTS_CSV_PATH")
    resolved_csv_path = Path(csv_path) if csv_path is not None else (
        Path(env_csv_path) if env_csv_path else DEFAULT_RESULTS_CSV_PATH
    )
    if not resolved_csv_path.is_absolute():
        resolved_csv_path = (ROOT / resolved_csv_path).resolve()

    resolved_db_url = (
        database_url
        or os.getenv("TRAINING_DATABASE_URL")
        or os.getenv("DATABASE_URL")
        or None
    )

    if normalized_source == "postgres" and not resolved_db_url:
        raise ValueError(
            "DATABASE_URL is required when TRAINING_DATA_SOURCE=postgres. "
            "Set DATABASE_URL in your environment or .env, or set TRAINING_DATA_SOURCE=csv."
        )

    return TrainingDataConfig(
        source=normalized_source,
        csv_path=resolved_csv_path,
        database_url=resolved_db_url,
    )


def normalize_results_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing = sorted(REQUIRED_RESULT_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Training results dataset is missing required columns: {', '.join(missing)}")

    for col in MATCH_RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    if df.empty:
        empty = pd.DataFrame(columns=MATCH_RESULT_COLUMNS + ["target_btts"])
        return empty

    df["season_id"] = df["season_id"].astype(str).str.strip()
    df["home_team"] = df["home_team"].astype(str).str.upper().str.strip()
    df["away_team"] = df["away_team"].astype(str).str.upper().str.strip()

    for col in NUMERIC_RESULT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["matchday", "fulltime_home_goals", "fulltime_away_goals"])
    df["matchday"] = df["matchday"].astype(int)
    df["fulltime_home_goals"] = df["fulltime_home_goals"].astype(int)
    df["fulltime_away_goals"] = df["fulltime_away_goals"].astype(int)

    if "captured_at" in df.columns:
        df["captured_at"] = df["captured_at"].map(
            lambda value: value.isoformat() if hasattr(value, "isoformat") else value
        )

    df = df.drop_duplicates(["season_id", "matchday", "home_team", "away_team"])
    df = df.sort_values(["season_id", "matchday", "home_team", "away_team"]).reset_index(drop=True)
    df["target_btts"] = ((df["fulltime_home_goals"] > 0) & (df["fulltime_away_goals"] > 0)).astype(int)

    ordered_cols = [col for col in MATCH_RESULT_COLUMNS if col in df.columns]
    extra_cols = [col for col in df.columns if col not in ordered_cols and col != "target_btts"]
    return df[ordered_cols + extra_cols + ["target_btts"]]


def load_results_from_csv(csv_path: Path | str = DEFAULT_RESULTS_CSV_PATH) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Training results CSV file not found: {path}")
    raw_df = pd.read_csv(path)
    return normalize_results_dataframe(raw_df)


def load_results_from_postgres(database_url: str) -> pd.DataFrame:
    if not database_url:
        raise ValueError("A PostgreSQL connection string (DATABASE_URL) is required.")

    query = """
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
        ORDER BY season_id, matchday, home_team, away_team
    """

    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError(
            "psycopg is required for TRAINING_DATA_SOURCE=postgres. "
            "Install it with: pip install 'psycopg[binary]>=3.1,<4'"
        ) from exc

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        rows = conn.execute(query).fetchall()

    raw_df = pd.DataFrame(rows, columns=MATCH_RESULT_COLUMNS)
    return normalize_results_dataframe(raw_df)


def load_training_results(
    csv_path: Path | str | None = None,
    data_source: str | None = None,
    database_url: str | None = None,
) -> pd.DataFrame:
    config = resolve_training_data_config(
        csv_path=csv_path,
        data_source=data_source,
        database_url=database_url,
    )
    if config.source == "postgres":
        assert config.database_url is not None
        df = load_results_from_postgres(config.database_url)
    else:
        df = load_results_from_csv(config.csv_path)

    df.attrs["training_data_source"] = config.source
    df.attrs["training_csv_path"] = str(config.csv_path)
    return df
