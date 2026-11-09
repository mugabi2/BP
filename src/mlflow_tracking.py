from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import urllib.error
import urllib.request
import warnings

from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_MLFLOW_EXPERIMENT_NAME = "BTTS Prediction"
DEFAULT_MLFLOW_REGISTERED_MODEL_NAME = "btts-predictor"


@dataclass(frozen=True)
class MLflowConfig:
    tracking_uri: str
    experiment_name: str
    registered_model_name: str


def resolve_mlflow_config(
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    registered_model_name: str | None = None,
) -> MLflowConfig:
    load_dotenv(ROOT / ".env", override=False)

    resolved_uri = (
        tracking_uri
        or os.getenv("MLFLOW_TRACKING_URI")
        or DEFAULT_MLFLOW_TRACKING_URI
    ).strip()
    resolved_experiment = (
        experiment_name
        or os.getenv("MLFLOW_EXPERIMENT_NAME")
        or DEFAULT_MLFLOW_EXPERIMENT_NAME
    ).strip()
    resolved_model_name = (
        registered_model_name
        or os.getenv("MLFLOW_REGISTERED_MODEL_NAME")
        or DEFAULT_MLFLOW_REGISTERED_MODEL_NAME
    ).strip()

    return MLflowConfig(
        tracking_uri=resolved_uri,
        experiment_name=resolved_experiment,
        registered_model_name=resolved_model_name,
    )


def evaluate_thresholds_table(y_true: pd.Series, prob_yes: np.ndarray) -> pd.DataFrame:
    rows = []
    y_array = y_true.to_numpy()
    for threshold in np.arange(0.50, 0.81, 0.05):
        pred = (prob_yes >= 0.5).astype(int)
        confidence = np.maximum(prob_yes, 1 - prob_yes)
        picked = confidence >= threshold
        if picked.sum() == 0:
            hit_rate = np.nan
        else:
            hit_rate = (pred[picked] == y_array[picked]).mean()
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "picked_matches": int(picked.sum()),
                "coverage": round(float(picked.mean()), 3),
                "picked_hit_rate": np.nan if pd.isna(hit_rate) else round(float(hit_rate), 3),
            }
        )
    return pd.DataFrame(rows)


def compute_candidate_metrics(
    y_true: pd.Series,
    prob_yes: np.ndarray,
    pred: np.ndarray | None = None,
    pick_threshold: float = 0.65,
) -> dict[str, float]:
    if pred is None:
        pred = (prob_yes >= 0.5).astype(int)

    y_array = y_true.to_numpy()
    confidence = np.maximum(prob_yes, 1 - prob_yes)
    picked = confidence >= pick_threshold
    picked_hit_rate = (
        float((pred[picked] == y_array[picked]).mean())
        if int(picked.sum()) > 0
        else float("nan")
    )

    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob_yes)) if y_true.nunique() == 2 else float("nan"),
        "brier_score": float(brier_score_loss(y_true, prob_yes)),
        "picked_matches": float(int(picked.sum())),
        "pick_coverage": float(picked.mean()),
        "picked_hit_rate": picked_hit_rate,
    }


def check_mlflow_server_reachable(tracking_uri: str, timeout_seconds: float = 3.0) -> tuple[bool, str]:
    normalized = (tracking_uri or "").strip()
    if not normalized:
        return False, "MLFLOW_TRACKING_URI is empty."

    if not normalized.lower().startswith(("http://", "https://")):
        return True, "Non-HTTP tracking URI configured."

    base = normalized.rstrip("/")
    last_error = ""
    for endpoint in ("/health", "/version", ""):
        url = f"{base}{endpoint}"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
                if 200 <= getattr(response, "status", 200) < 500:
                    return True, f"Connected to {url}"
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                return True, f"HTTP {exc.code} from {url}"
            last_error = f"HTTP {exc.code} from {url}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

    return False, f"Cannot reach MLflow tracking server at {normalized} ({last_error})"


def _clean_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for key, value in metrics.items():
        if value is None or pd.isna(value):
            continue
        cleaned[key] = float(value)
    return cleaned


def _threshold_metrics_from_df(threshold_df: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if threshold_df is None or threshold_df.empty:
        return metrics
    for _, row in threshold_df.iterrows():
        t_label = f"{float(row['threshold']):.2f}".replace(".", "_")
        metrics[f"picked_matches_t_{t_label}"] = float(row["picked_matches"])
        metrics[f"coverage_t_{t_label}"] = float(row["coverage"])
        if not pd.isna(row["picked_hit_rate"]):
            metrics[f"picked_hit_rate_t_{t_label}"] = float(row["picked_hit_rate"])
    return metrics


def log_btts_training_to_mlflow(
    *,
    trained_models: dict[str, dict[str, Any]],
    model_comparison: pd.DataFrame,
    best_model_name: str,
    best_model: Any,
    threshold_report: pd.DataFrame,
    y_valid: pd.Series,
    train_row_count: int,
    valid_row_count: int,
    num_seasons: int,
    feature_columns: list[str],
    pick_threshold: float,
    live_gap_matchdays: int,
    training_data_source: str,
    model_path: Path,
    columns_path: Path,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    registered_model_name: str | None = None,
) -> dict[str, Any]:
    config = resolve_mlflow_config(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        registered_model_name=registered_model_name,
    )
    training_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    reachable, reachability_detail = check_mlflow_server_reachable(config.tracking_uri)
    if not reachable:
        warning_msg = (
            f"MLflow tracking server is unavailable ({reachability_detail}). "
            "Continuing locally without MLflow logging; local model files remain saved."
        )
        warnings.warn(warning_msg, RuntimeWarning, stacklevel=2)
        print(f"WARNING: {warning_msg}")
        return {
            "logged": False,
            "tracking_uri": config.tracking_uri,
            "experiment_name": config.experiment_name,
            "registered_model_name": config.registered_model_name,
            "training_timestamp": training_timestamp,
            "reason": reachability_detail,
        }

    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as exc:
        warning_msg = (
            f"MLflow Python package is not installed ({exc}). "
            "Continuing locally without MLflow logging; local model files remain saved."
        )
        warnings.warn(warning_msg, RuntimeWarning, stacklevel=2)
        print(f"WARNING: {warning_msg}")
        return {
            "logged": False,
            "tracking_uri": config.tracking_uri,
            "experiment_name": config.experiment_name,
            "registered_model_name": config.registered_model_name,
            "training_timestamp": training_timestamp,
            "reason": f"ImportError: {exc}",
        }

    try:
        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)

        common_counts = {
            "training_row_count": float(train_row_count),
            "validation_row_count": float(valid_row_count),
            "number_of_seasons": float(num_seasons),
            "number_of_features": float(len(feature_columns)),
            "pick_threshold": float(pick_threshold),
        }

        best_bundle = trained_models[best_model_name]
        best_metrics = compute_candidate_metrics(
            y_valid,
            best_bundle["prob"],
            best_bundle["pred"],
            pick_threshold=pick_threshold,
        )

        candidate_run_ids: dict[str, str] = {}
        registered_version: str | None = None

        with mlflow.start_run(run_name=f"btts-training-{best_model_name}") as parent_run:
            mlflow.set_tags(
                {
                    "selected_model_name": best_model_name,
                    "registered_model_name": config.registered_model_name,
                    "training_data_source": training_data_source,
                    "training_timestamp": training_timestamp,
                }
            )
            mlflow.log_params(
                {
                    "selected_model_name": best_model_name,
                    "model_name": best_model_name,
                    "training_data_source": training_data_source,
                    "training_row_count": int(train_row_count),
                    "validation_row_count": int(valid_row_count),
                    "number_of_seasons": int(num_seasons),
                    "number_of_features": int(len(feature_columns)),
                    "pick_threshold": float(pick_threshold),
                    "live_gap_matchdays": int(live_gap_matchdays),
                    "training_timestamp": training_timestamp,
                    "registered_model_name": config.registered_model_name,
                }
            )
            mlflow.log_metrics(
                _clean_metrics(
                    {
                        **common_counts,
                        **best_metrics,
                        **_threshold_metrics_from_df(threshold_report),
                    }
                )
            )

            with TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                comparison_csv = tmp_path / "model_comparison.csv"
                comparison_json = tmp_path / "model_comparison.json"
                threshold_csv = tmp_path / "threshold_evaluation.csv"
                threshold_json = tmp_path / "threshold_evaluation.json"
                summary_json = tmp_path / "selected_model_summary.json"

                model_comparison.to_csv(comparison_csv, index=False)
                comparison_json.write_text(
                    json.dumps(model_comparison.round(4).to_dict(orient="records"), indent=2),
                    encoding="utf-8",
                )
                threshold_report.to_csv(threshold_csv, index=False)
                threshold_json.write_text(
                    json.dumps(threshold_report.round(4).to_dict(orient="records"), indent=2),
                    encoding="utf-8",
                )
                summary_json.write_text(
                    json.dumps(
                        {
                            "selected_model_name": best_model_name,
                            "registered_model_name": config.registered_model_name,
                            "training_timestamp": training_timestamp,
                            "training_data_source": training_data_source,
                            "training_row_count": int(train_row_count),
                            "validation_row_count": int(valid_row_count),
                            "number_of_seasons": int(num_seasons),
                            "number_of_features": int(len(feature_columns)),
                            "pick_threshold": float(pick_threshold),
                            "metrics": _clean_metrics(best_metrics),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                mlflow.log_artifact(str(comparison_csv), artifact_path="evaluation")
                mlflow.log_artifact(str(comparison_json), artifact_path="evaluation")
                mlflow.log_artifact(str(threshold_csv), artifact_path="evaluation")
                mlflow.log_artifact(str(threshold_json), artifact_path="evaluation")
                mlflow.log_artifact(str(summary_json), artifact_path="evaluation")

            if columns_path.exists():
                mlflow.log_artifact(str(columns_path), artifact_path="local_model_bundle")
            if model_path.exists():
                mlflow.log_artifact(str(model_path), artifact_path="local_model_bundle")

            for candidate_name, candidate_bundle in trained_models.items():
                cand_prob = candidate_bundle["prob"]
                cand_pred = candidate_bundle["pred"]
                cand_metrics = compute_candidate_metrics(
                    y_valid,
                    cand_prob,
                    cand_pred,
                    pick_threshold=pick_threshold,
                )
                cand_thresholds = evaluate_thresholds_table(y_valid, cand_prob)

                with mlflow.start_run(run_name=f"candidate-{candidate_name}", nested=True) as child_run:
                    candidate_run_ids[candidate_name] = child_run.info.run_id
                    mlflow.set_tags(
                        {
                            "model_name": candidate_name,
                            "is_selected_best": str(candidate_name == best_model_name).lower(),
                            "training_data_source": training_data_source,
                            "training_timestamp": training_timestamp,
                        }
                    )
                    mlflow.log_params(
                        {
                            "model_name": candidate_name,
                            "training_data_source": training_data_source,
                            "training_row_count": int(train_row_count),
                            "validation_row_count": int(valid_row_count),
                            "number_of_seasons": int(num_seasons),
                            "number_of_features": int(len(feature_columns)),
                            "pick_threshold": float(pick_threshold),
                            "live_gap_matchdays": int(live_gap_matchdays),
                            "is_selected_best": candidate_name == best_model_name,
                            "training_timestamp": training_timestamp,
                        }
                    )
                    mlflow.log_metrics(
                        _clean_metrics(
                            {
                                **common_counts,
                                **cand_metrics,
                                **_threshold_metrics_from_df(cand_thresholds),
                            }
                        )
                    )
                    with TemporaryDirectory() as cand_tmp:
                        cand_thresh_path = Path(cand_tmp) / f"{candidate_name}_threshold_evaluation.csv"
                        cand_thresholds.to_csv(cand_thresh_path, index=False)
                        mlflow.log_artifact(str(cand_thresh_path), artifact_path="threshold_evaluation")

            try:
                model_info = mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path="model",
                    registered_model_name=config.registered_model_name,
                )
                registered_version = getattr(model_info, "registered_model_version", None)
            except Exception as reg_exc:
                warnings.warn(
                    f"MLflow sklearn model registration warning: {reg_exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

            print(
                f"Logged training run to MLflow ({config.tracking_uri}) "
                f"in experiment '{config.experiment_name}' (run_id={parent_run.info.run_id})."
            )
            return {
                "logged": True,
                "tracking_uri": config.tracking_uri,
                "experiment_name": config.experiment_name,
                "registered_model_name": config.registered_model_name,
                "registered_model_version": registered_version,
                "parent_run_id": parent_run.info.run_id,
                "candidate_run_ids": candidate_run_ids,
                "training_timestamp": training_timestamp,
            }
    except Exception as exc:
        warning_msg = (
            f"MLflow logging failed ({type(exc).__name__}: {exc}). "
            "Continuing locally; local model files remain saved."
        )
        warnings.warn(warning_msg, RuntimeWarning, stacklevel=2)
        print(f"WARNING: {warning_msg}")
        return {
            "logged": False,
            "tracking_uri": config.tracking_uri,
            "experiment_name": config.experiment_name,
            "registered_model_name": config.registered_model_name,
            "training_timestamp": training_timestamp,
            "reason": f"{type(exc).__name__}: {exc}",
        }
