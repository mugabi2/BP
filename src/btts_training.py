from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from joblib import dump
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.mlflow_tracking import (
    compute_candidate_metrics,
    evaluate_thresholds_table,
    log_btts_training_to_mlflow,
)
from src.training_data import DEFAULT_RESULTS_CSV_PATH, load_training_results


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ODDS_PATH = ROOT / "odds" / "final_odds" / "final_odds.csv"
DEFAULT_CANDIDATE_OUTPUT_DIR = ROOT / "training-output" / "latest-candidate"

LIVE_GAP_MATCHDAYS = 2
PICK_THRESHOLD = 0.65
RANDOM_STATE = 42
VALIDATION_FRACTION = 0.20

# Training takes the collector lock for a consistent historical snapshot, then
# its own lock to prevent overlapping training jobs.
COLLECTION_LOCK_ID = 7_794_001
TRAINING_LOCK_ID = 7_794_002

ODDS_RAW_COLUMNS = ["odds_1", "odds_x", "odds_2", "odds_btts_yes", "odds_btts_no"]
ODDS_DERIVED_COLUMNS = [
    "implied_home_prob",
    "implied_draw_prob",
    "implied_away_prob",
    "implied_btts_yes_prob",
    "implied_btts_no_prob",
    "btts_market_margin",
    "market_btts_yes_prob_norm",
    "market_btts_no_prob_norm",
]
ODDS_FEATURE_COLUMNS = ODDS_RAW_COLUMNS + ODDS_DERIVED_COLUMNS

TEAM_STATS_COLUMNS = [
    "rank",
    "played",
    "points",
    "wins",
    "draws",
    "losses",
    "goals_for",
    "goals_against",
    "goal_difference",
    "btts_rate_so_far",
    "scored_rate_so_far",
    "conceded_rate_so_far",
    "avg_goals_for_so_far",
    "avg_goals_against_so_far",
]
FORM_COLUMNS = [
    "form_points_last_5",
    "form_wins_last_5",
    "form_draws_last_5",
    "form_losses_last_5",
]
SEASON_FEATURE_COLUMNS = [
    "season_matches_so_far",
    "season_btts_rate_so_far",
    "season_avg_goals_per_match_so_far",
    "season_home_win_rate_so_far",
    "season_draw_rate_so_far",
    "season_away_win_rate_so_far",
]
TRACKING_COLUMNS = [
    "season_id",
    "target_matchday",
    "feature_cutoff_matchday",
    "home_team",
    "away_team",
]
TARGET_COLUMN = "target_btts"


@dataclass(frozen=True)
class TrainingConfig:
    data_source: str = "postgres"
    database_url: str | None = None
    results_csv_path: Path = DEFAULT_RESULTS_CSV_PATH
    odds_path: Path = DEFAULT_ODDS_PATH
    output_dir: Path = DEFAULT_CANDIDATE_OUTPUT_DIR
    live_gap_matchdays: int = LIVE_GAP_MATCHDAYS
    pick_threshold: float = PICK_THRESHOLD
    random_state: int = RANDOM_STATE
    validation_fraction: float = VALIDATION_FRACTION
    use_database_lock: bool = False


@dataclass
class TrainingRunResult:
    selected_model_name: str
    model_path: Path
    columns_path: Path
    comparison_path: Path
    threshold_path: Path
    summary_path: Path
    training_rows: int
    validation_rows: int
    seasons: int
    features: int
    model_comparison: pd.DataFrame
    threshold_report: pd.DataFrame
    mlflow_result: dict[str, Any]


def add_implied_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ODDS_RAW_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["implied_home_prob"] = np.where(df["odds_1"] > 0, 1 / df["odds_1"], np.nan)
    df["implied_draw_prob"] = np.where(df["odds_x"] > 0, 1 / df["odds_x"], np.nan)
    df["implied_away_prob"] = np.where(df["odds_2"] > 0, 1 / df["odds_2"], np.nan)
    df["implied_btts_yes_prob"] = np.where(
        df["odds_btts_yes"] > 0, 1 / df["odds_btts_yes"], np.nan
    )
    df["implied_btts_no_prob"] = np.where(
        df["odds_btts_no"] > 0, 1 / df["odds_btts_no"], np.nan
    )

    btts_total = df["implied_btts_yes_prob"] + df["implied_btts_no_prob"]
    df["btts_market_margin"] = btts_total - 1
    df["market_btts_yes_prob_norm"] = df["implied_btts_yes_prob"] / btts_total
    df["market_btts_no_prob_norm"] = df["implied_btts_no_prob"] / btts_total
    return df


def load_odds(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"No final odds file found at {path}; continuing with missing odds features.")
        return pd.DataFrame(
            columns=["matchday", "home_team", "away_team"] + ODDS_FEATURE_COLUMNS
        )

    raw = pd.read_csv(path)
    raw.columns = [column.strip() for column in raw.columns]
    raw["home_team"] = raw["home_team"].astype(str).str.upper().str.strip()
    raw["away_team"] = raw["away_team"].astype(str).str.upper().str.strip()
    raw["matchday"] = pd.to_numeric(raw["matchday"], errors="coerce")
    raw = raw.dropna(subset=["matchday", "home_team", "away_team"])
    raw["matchday"] = raw["matchday"].astype(int)

    wide = raw.rename(
        columns={
            "odd_1": "odds_1",
            "odd_x": "odds_x",
            "odd_2": "odds_2",
            "btts_yes": "odds_btts_yes",
            "btts_no": "odds_btts_no",
        }
    )
    if "captured_at" in wide.columns:
        wide = wide.sort_values("captured_at")
    wide = wide.drop_duplicates(["home_team", "away_team"], keep="last")
    return add_implied_odds_features(wide)


def make_team_match_rows(history: pd.DataFrame) -> pd.DataFrame:
    home = history[
        [
            "matchday",
            "home_team",
            "away_team",
            "fulltime_home_goals",
            "fulltime_away_goals",
            "target_btts",
        ]
    ].copy()
    home.columns = ["matchday", "team", "opponent", "goals_for", "goals_against", "btts"]
    home["side"] = "home"

    away = history[
        [
            "matchday",
            "away_team",
            "home_team",
            "fulltime_away_goals",
            "fulltime_home_goals",
            "target_btts",
        ]
    ].copy()
    away.columns = ["matchday", "team", "opponent", "goals_for", "goals_against", "btts"]
    away["side"] = "away"

    team_rows = pd.concat([home, away], ignore_index=True)
    team_rows["points"] = np.select(
        [
            team_rows["goals_for"] > team_rows["goals_against"],
            team_rows["goals_for"] == team_rows["goals_against"],
        ],
        [3, 1],
        default=0,
    )
    team_rows["win"] = (team_rows["points"] == 3).astype(int)
    team_rows["draw"] = (team_rows["points"] == 1).astype(int)
    team_rows["loss"] = (team_rows["points"] == 0).astype(int)
    team_rows["scored"] = (team_rows["goals_for"] > 0).astype(int)
    team_rows["conceded"] = (team_rows["goals_against"] > 0).astype(int)
    return team_rows


def build_team_stats(history: pd.DataFrame) -> pd.DataFrame:
    team_rows = make_team_match_rows(history)
    if team_rows.empty:
        return pd.DataFrame(columns=["team"] + TEAM_STATS_COLUMNS).set_index("team")

    stats = team_rows.groupby("team").agg(
        played=("team", "size"),
        points=("points", "sum"),
        wins=("win", "sum"),
        draws=("draw", "sum"),
        losses=("loss", "sum"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
        btts_rate_so_far=("btts", "mean"),
        scored_rate_so_far=("scored", "mean"),
        conceded_rate_so_far=("conceded", "mean"),
        avg_goals_for_so_far=("goals_for", "mean"),
        avg_goals_against_so_far=("goals_against", "mean"),
    ).reset_index()
    stats["goal_difference"] = stats["goals_for"] - stats["goals_against"]
    stats = stats.sort_values(
        ["points", "goal_difference", "goals_for", "team"],
        ascending=[False, False, False, True],
    )
    stats["rank"] = np.arange(1, len(stats) + 1)
    return stats.set_index("team")


def build_form_stats(history: pd.DataFrame) -> pd.DataFrame:
    team_rows = make_team_match_rows(history).sort_values(["team", "matchday"])
    if team_rows.empty:
        return pd.DataFrame(columns=["team"] + FORM_COLUMNS).set_index("team")

    last_five = team_rows.groupby("team", group_keys=False).tail(5)
    return last_five.groupby("team").agg(
        form_points_last_5=("points", "sum"),
        form_wins_last_5=("win", "sum"),
        form_draws_last_5=("draw", "sum"),
        form_losses_last_5=("loss", "sum"),
    )


def build_season_stats(history: pd.DataFrame) -> dict[str, float]:
    if history.empty:
        return {column: np.nan for column in SEASON_FEATURE_COLUMNS}

    home_win = history["fulltime_home_goals"] > history["fulltime_away_goals"]
    draw = history["fulltime_home_goals"] == history["fulltime_away_goals"]
    away_win = history["fulltime_home_goals"] < history["fulltime_away_goals"]
    total_goals = history["fulltime_home_goals"] + history["fulltime_away_goals"]
    return {
        "season_matches_so_far": float(len(history)),
        "season_btts_rate_so_far": float(history["target_btts"].mean()),
        "season_avg_goals_per_match_so_far": float(total_goals.mean()),
        "season_home_win_rate_so_far": float(home_win.mean()),
        "season_draw_rate_so_far": float(draw.mean()),
        "season_away_win_rate_so_far": float(away_win.mean()),
    }


def get_team_values(
    table: pd.DataFrame,
    team: str,
    columns: list[str],
    prefix: str,
) -> dict[str, Any]:
    if team in table.index:
        return {f"{prefix}_{column}": table.loc[team, column] for column in columns}
    return {f"{prefix}_{column}": np.nan for column in columns}


def make_stats_bundle(
    results_df: pd.DataFrame,
    season_id: str,
    cutoff_matchday: int,
) -> dict[str, Any]:
    history = results_df[
        (results_df["season_id"] == str(season_id))
        & (results_df["matchday"] <= cutoff_matchday)
    ].copy()
    return {
        "team_stats": build_team_stats(history),
        "form_stats": build_form_stats(history),
        "season_stats": build_season_stats(history),
    }


def add_comparison_features(record: dict[str, Any]) -> dict[str, Any]:
    record["points_diff"] = record["home_points"] - record["away_points"]
    record["rank_diff"] = record["away_rank"] - record["home_rank"]
    record["goal_difference_diff"] = (
        record["home_goal_difference"] - record["away_goal_difference"]
    )
    record["goals_for_diff"] = record["home_goals_for"] - record["away_goals_for"]
    record["goals_against_diff"] = (
        record["home_goals_against"] - record["away_goals_against"]
    )
    record["form_diff"] = record["home_form_points_last_5"] - record["away_form_points_last_5"]
    record["btts_rate_diff"] = record["home_btts_rate_so_far"] - record["away_btts_rate_so_far"]
    record["scored_rate_diff"] = record["home_scored_rate_so_far"] - record["away_scored_rate_so_far"]
    record["conceded_rate_diff"] = (
        record["home_conceded_rate_so_far"] - record["away_conceded_rate_so_far"]
    )
    return record


def build_base_feature_record(
    match: pd.Series,
    bundle: dict[str, Any],
    *,
    include_target: bool,
    live_gap_matchdays: int = LIVE_GAP_MATCHDAYS,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "season_id": str(match["season_id"]),
        "target_matchday": int(match["matchday"]),
        "feature_cutoff_matchday": int(match["matchday"]) - live_gap_matchdays,
        "home_team": match["home_team"],
        "away_team": match["away_team"],
    }
    record.update(bundle["season_stats"])
    record.update(
        get_team_values(bundle["team_stats"], match["home_team"], TEAM_STATS_COLUMNS, "home")
    )
    record.update(
        get_team_values(bundle["team_stats"], match["away_team"], TEAM_STATS_COLUMNS, "away")
    )
    record.update(
        get_team_values(bundle["form_stats"], match["home_team"], FORM_COLUMNS, "home")
    )
    record.update(
        get_team_values(bundle["form_stats"], match["away_team"], FORM_COLUMNS, "away")
    )
    record = add_comparison_features(record)
    if include_target:
        record[TARGET_COLUMN] = int(match[TARGET_COLUMN])
    return record


def make_odds_template(odds_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if odds_df.empty:
        return (
            pd.DataFrame(
                columns=["home_team", "away_team"] + ODDS_FEATURE_COLUMNS
            ),
            ["home_team", "away_team"],
        )

    keys = ["home_team", "away_team"]
    value_columns = [column for column in ODDS_FEATURE_COLUMNS if column in odds_df.columns]
    template = odds_df.groupby(keys, as_index=False)[value_columns].median(numeric_only=True)
    return template, keys


def build_training_table(
    results_df: pd.DataFrame,
    odds_template_df: pd.DataFrame,
    odds_keys: list[str],
    *,
    live_gap_matchdays: int = LIVE_GAP_MATCHDAYS,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    stats_cache: dict[tuple[str, int], dict[str, Any]] = {}

    for _, match in results_df.iterrows():
        target_matchday = int(match["matchday"])
        cutoff_matchday = target_matchday - live_gap_matchdays
        if cutoff_matchday < 1:
            continue

        cache_key = (str(match["season_id"]), cutoff_matchday)
        if cache_key not in stats_cache:
            stats_cache[cache_key] = make_stats_bundle(
                results_df,
                str(match["season_id"]),
                cutoff_matchday,
            )

        records.append(
            build_base_feature_record(
                match,
                stats_cache[cache_key],
                include_target=True,
                live_gap_matchdays=live_gap_matchdays,
            )
        )

    feature_df = pd.DataFrame(records)
    if feature_df.empty:
        raise ValueError(
            "No training rows were built. Check matchday coverage and LIVE_GAP_MATCHDAYS."
        )

    merged = feature_df.merge(
        odds_template_df,
        on=odds_keys.copy(),
        how="left",
        suffixes=("", "_odds"),
    )
    for column in ODDS_FEATURE_COLUMNS:
        if column not in merged.columns:
            merged[column] = np.nan
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    return merged


def select_model_features(
    training_df: pd.DataFrame,
) -> tuple[list[str], list[str], pd.DataFrame]:
    candidate_features = [
        column
        for column in training_df.columns
        if column not in TRACKING_COLUMNS + [TARGET_COLUMN]
        and pd.api.types.is_numeric_dtype(training_df[column])
    ]
    all_missing = [
        column for column in candidate_features if training_df[column].notna().sum() == 0
    ]
    model_features = [column for column in candidate_features if column not in all_missing]
    summary = pd.DataFrame(
        {
            "feature": candidate_features,
            "non_null_rows": [
                int(training_df[column].notna().sum()) for column in candidate_features
            ],
            "used_by_model": [column in model_features for column in candidate_features],
        }
    )
    return model_features, all_missing, summary


def chronological_split(
    training_df: pd.DataFrame,
    validation_fraction: float = VALIDATION_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    ordered = training_df.sort_values(
        ["season_id", "target_matchday", "home_team", "away_team"]
    ).reset_index(drop=True)
    split_index = int(len(ordered) * (1 - validation_fraction))
    if split_index < 1 or split_index >= len(ordered):
        raise ValueError("Not enough training rows for the requested validation split.")
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()


def build_candidate_models(random_state: int = RANDOM_STATE) -> dict[str, Any]:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            random_state=random_state
        ),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            random_state=random_state
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
        "ExtraTreesClassifier": ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
        "SoftVotingEnsemble": VotingClassifier(
            estimators=[
                (
                    "lr",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
                ("gb", GradientBoostingClassifier(random_state=random_state)),
                ("hgb", HistGradientBoostingClassifier(random_state=random_state)),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=8,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
                (
                    "et",
                    ExtraTreesClassifier(
                        n_estimators=400,
                        min_samples_leaf=8,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ],
            voting="soft",
        ),
    }


def fit_candidate_models(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    model_features: list[str],
    *,
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    x_train = train_df[model_features]
    y_train = train_df[TARGET_COLUMN]
    x_valid = valid_df[model_features]
    y_valid = valid_df[TARGET_COLUMN]

    trained_models: dict[str, dict[str, Any]] = {}
    model_results: list[dict[str, Any]] = []

    for model_name, estimator in build_candidate_models(random_state).items():
        print(f"Training {model_name}...", flush=True)
        candidate = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", estimator),
            ]
        )
        candidate.fit(x_train, y_train)
        prob_yes = candidate.predict_proba(x_valid)[:, 1]
        pred = (prob_yes >= 0.5).astype(int)
        trained_models[model_name] = {
            "model": candidate,
            "prob": prob_yes,
            "pred": pred,
        }
        metrics = {
            "model": model_name,
            "accuracy": accuracy_score(y_valid, pred),
            "precision": precision_score(y_valid, pred, zero_division=0),
            "recall": recall_score(y_valid, pred, zero_division=0),
            "f1": f1_score(y_valid, pred, zero_division=0),
            "roc_auc": (
                roc_auc_score(y_valid, prob_yes)
                if y_valid.nunique() == 2
                else np.nan
            ),
            "brier_score": brier_score_loss(y_valid, prob_yes),
        }
        model_results.append(metrics)
        print(
            f"Completed {model_name}: accuracy={metrics['accuracy']:.4f}, "
            f"brier={metrics['brier_score']:.4f}",
            flush=True,
        )

    comparison = pd.DataFrame(model_results).sort_values(
        ["accuracy", "brier_score"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return trained_models, comparison


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_dataframe(path: Path, dataframe: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        dataframe.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_dump_model(path: Path, model: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        dump(model, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_model_bundle(
    *,
    output_dir: Path,
    model: Any,
    model_features: list[str],
    all_missing_features: list[str],
    selected_model_name: str,
    model_comparison: pd.DataFrame,
    threshold_report: pd.DataFrame,
    live_gap_matchdays: int,
    pick_threshold: float,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "btts_model.joblib"
    columns_path = output_dir / "btts_feature_columns.json"
    comparison_path = output_dir / "model_comparison.csv"
    threshold_path = output_dir / "threshold_evaluation.csv"

    comparison_columns = ["model", "accuracy", "roc_auc", "brier_score"]
    columns_payload = {
        "target": TARGET_COLUMN,
        "feature_columns": model_features,
        "excluded_all_missing_features": all_missing_features,
        "selected_model_name": selected_model_name,
        "model_comparison": (
            model_comparison[comparison_columns].round(4).to_dict(orient="records")
        ),
        "live_gap_matchdays": live_gap_matchdays,
        "pick_threshold": pick_threshold,
        "notes": (
            "For target matchday N, features use completed results only through N - 2."
        ),
    }

    _atomic_dump_model(model_path, model)
    _atomic_write_text(columns_path, json.dumps(columns_payload, indent=2))
    _atomic_write_dataframe(comparison_path, model_comparison)
    _atomic_write_dataframe(threshold_path, threshold_report)
    return model_path, columns_path, comparison_path, threshold_path


@contextmanager
def postgres_training_lock(database_url: str | None) -> Iterator[None]:
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("psycopg is required for the PostgreSQL training lock.") from exc

    connection = psycopg.connect(database_url or "")
    acquired: list[int] = []
    try:
        for lock_id, label in (
            (COLLECTION_LOCK_ID, "collector"),
            (TRAINING_LOCK_ID, "training"),
        ):
            row = connection.execute(
                "SELECT pg_try_advisory_lock(%s)",
                (lock_id,),
            ).fetchone()
            if not row or not bool(row[0]):
                raise RuntimeError(
                    f"Could not acquire the {label} advisory lock; another job is active."
                )
            acquired.append(lock_id)
        yield
    finally:
        for lock_id in reversed(acquired):
            connection.execute("SELECT pg_advisory_unlock(%s)", (lock_id,))
        connection.close()


def run_training(config: TrainingConfig) -> TrainingRunResult:
    lock_context = (
        postgres_training_lock(config.database_url)
        if config.use_database_lock and config.data_source == "postgres"
        else nullcontext()
    )

    with lock_context:
        print(f"Loading training data from {config.data_source}...", flush=True)
        results = load_training_results(
            csv_path=config.results_csv_path,
            data_source=config.data_source,
            database_url=config.database_url,
        )
        training_data_source = results.attrs.get(
            "training_data_source", config.data_source
        )
        seasons = int(results["season_id"].nunique())
        print(f"Loaded {len(results)} matches across {seasons} seasons.", flush=True)

        odds = load_odds(config.odds_path)
        odds_template, odds_keys = make_odds_template(odds)
        print("Building N-2 feature table...", flush=True)
        training_df = build_training_table(
            results,
            odds_template,
            odds_keys,
            live_gap_matchdays=config.live_gap_matchdays,
        )
        model_features, all_missing_features, feature_summary = select_model_features(
            training_df
        )
        if not model_features:
            raise ValueError("No usable numeric model features were generated.")

        train_df, valid_df = chronological_split(
            training_df,
            validation_fraction=config.validation_fraction,
        )
        print(
            f"Training rows={len(train_df)}, validation rows={len(valid_df)}, "
            f"features={len(model_features)}",
            flush=True,
        )

        trained_models, model_comparison = fit_candidate_models(
            train_df,
            valid_df,
            model_features,
            random_state=config.random_state,
        )
        selected_model_name = str(model_comparison.loc[0, "model"])
        selected_bundle = trained_models[selected_model_name]
        selected_model = selected_bundle["model"]
        valid_prob = selected_bundle["prob"]
        valid_pred = selected_bundle["pred"]
        y_valid = valid_df[TARGET_COLUMN]
        threshold_report = evaluate_thresholds_table(y_valid, valid_prob)

        print(f"Selected model: {selected_model_name}", flush=True)
        print(model_comparison.to_string(index=False), flush=True)
        print("Confusion matrix:", flush=True)
        print(confusion_matrix(y_valid, valid_pred), flush=True)
        print(
            classification_report(
                y_valid,
                valid_pred,
                target_names=["BTTS No", "BTTS Yes"],
            ),
            flush=True,
        )

        model_path, columns_path, comparison_path, threshold_path = write_model_bundle(
            output_dir=config.output_dir,
            model=selected_model,
            model_features=model_features,
            all_missing_features=all_missing_features,
            selected_model_name=selected_model_name,
            model_comparison=model_comparison,
            threshold_report=threshold_report,
            live_gap_matchdays=config.live_gap_matchdays,
            pick_threshold=config.pick_threshold,
        )
        feature_summary_path = config.output_dir / "feature_summary.csv"
        _atomic_write_dataframe(feature_summary_path, feature_summary)
        print(f"Saved candidate model bundle to {config.output_dir}", flush=True)

        mlflow_result = log_btts_training_to_mlflow(
            trained_models=trained_models,
            model_comparison=model_comparison,
            best_model_name=selected_model_name,
            best_model=selected_model,
            threshold_report=threshold_report,
            y_valid=y_valid,
            train_row_count=len(train_df),
            valid_row_count=len(valid_df),
            num_seasons=seasons,
            feature_columns=model_features,
            pick_threshold=config.pick_threshold,
            live_gap_matchdays=config.live_gap_matchdays,
            training_data_source=training_data_source,
            model_path=model_path,
            columns_path=columns_path,
        )

        selected_metrics = compute_candidate_metrics(
            y_valid,
            valid_prob,
            valid_pred,
            pick_threshold=config.pick_threshold,
        )
        summary_path = config.output_dir / "training_summary.json"
        summary = {
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "training_data_source": training_data_source,
            "raw_match_rows": int(len(results)),
            "training_rows": int(len(train_df)),
            "validation_rows": int(len(valid_df)),
            "seasons": seasons,
            "features": int(len(model_features)),
            "selected_model_name": selected_model_name,
            "selected_metrics": selected_metrics,
            "model_path": str(model_path),
            "columns_path": str(columns_path),
            "mlflow": mlflow_result,
        }
        _atomic_write_text(summary_path, json.dumps(summary, indent=2))

        return TrainingRunResult(
            selected_model_name=selected_model_name,
            model_path=model_path,
            columns_path=columns_path,
            comparison_path=comparison_path,
            threshold_path=threshold_path,
            summary_path=summary_path,
            training_rows=len(train_df),
            validation_rows=len(valid_df),
            seasons=seasons,
            features=len(model_features),
            model_comparison=model_comparison,
            threshold_report=threshold_report,
            mlflow_result=mlflow_result,
        )
