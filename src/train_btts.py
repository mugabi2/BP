from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.btts_training import (
    DEFAULT_CANDIDATE_OUTPUT_DIR,
    DEFAULT_ODDS_PATH,
    DEFAULT_RESULTS_CSV_PATH,
    LIVE_GAP_MATCHDAYS,
    PICK_THRESHOLD,
    TrainingConfig,
    run_training,
)


ROOT = Path(__file__).resolve().parents[1]


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the existing BTTS model candidates."
    )
    parser.add_argument(
        "--data-source",
        choices=("postgres", "csv"),
        default=os.getenv("TRAINING_DATA_SOURCE", "postgres"),
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("TRAINING_DATABASE_URL") or os.getenv("DATABASE_URL"),
        help="PostgreSQL URL. Standard PG* variables are also supported.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path(os.getenv("TRAINING_CSV_PATH", str(DEFAULT_RESULTS_CSV_PATH))),
    )
    parser.add_argument(
        "--odds-path",
        type=Path,
        default=Path(os.getenv("TRAINING_ODDS_PATH", str(DEFAULT_ODDS_PATH))),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            os.getenv("TRAINING_OUTPUT_DIR", str(DEFAULT_CANDIDATE_OUTPUT_DIR))
        ),
    )
    parser.add_argument(
        "--database-lock",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("TRAINING_USE_DATABASE_LOCK", False),
        help="Hold collector/training advisory locks for the full run.",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv(ROOT / ".env", override=False)
    args = parse_args()
    config = TrainingConfig(
        data_source=args.data_source,
        database_url=args.database_url,
        results_csv_path=args.csv_path.resolve(),
        odds_path=args.odds_path.resolve(),
        output_dir=args.output_dir.resolve(),
        live_gap_matchdays=LIVE_GAP_MATCHDAYS,
        pick_threshold=PICK_THRESHOLD,
        use_database_lock=args.database_lock,
    )
    result = run_training(config)
    print(
        "Training complete: "
        f"selected={result.selected_model_name}, "
        f"candidate_bundle={result.model_path.parent}, "
        f"mlflow_logged={result.mlflow_result.get('logged', False)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
