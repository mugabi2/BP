"""MLflow tracking for BTTS model-training runs.

The training pipeline remains usable when MLflow is unavailable: candidate
artifacts are saved locally first, then this module attempts remote logging.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping
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
    return MLflowConfig(
        tracking_uri=(
            tracking_uri
            or os.getenv("MLFLOW_TRACKING_URI")
            or DEFAULT_MLFLOW_TRACKING_URI
        ).strip(),
        experiment_name=(
            experiment_name
            or os.getenv("MLFLOW_EXPERIMENT_NAME")
            or DEFAULT_MLFLOW_EXPERIMENT_NAME
        ).strip(),
        registered_model_name=(
            registered_model_name
            or os.getenv("MLFLOW_REGISTERED_MODEL_NAME")
            or DEFAULT_MLFLOW_REGISTERED_MODEL_NAME
        ).strip(),
    )


def check_mlflow_server_reachable(
    tracking_uri: str,
    timeout_seconds: float = 3.0,
) -> tuple[bool, str]:
    uri = (tracking_uri or "").strip()
    if not uri:
        return False, "MLFLOW_TRACKING_URI is empty."
    if not uri.lower().startswith(("http://", "https://")):
        return True, "Non-HTTP tracking URI configured."

    last_error = ""
    for endpoint in ("/health", "/version", ""):
        url = f"{uri.rstrip('/')}{endpoint}"
        try:
            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                if 200 <= getattr(response, "status", 200) < 500:
                    return True, f"Connected to {url}"
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                return True, f"HTTP {exc.code} from {url}"
            last_error = f"HTTP {exc.code} from {url}"
        except Exception as exc:  # network errors vary by platform
            last_error = f"{type(exc).__name__}: {exc}"
    return False, f"Cannot reach MLflow at {uri} ({last_error})"


def compute_candidate_metrics(
    y_true: pd.Series,
    prob_yes: np.ndarray,
    pred: np.ndarray | None = None,
    *,
    pick_threshold: float = 0.65,
) -> dict[str, float]:
    if pred is None:
        pred = (prob_yes >= 0.5).astype(int)
    y_array = y_true.to_numpy()
    confidence = np.maximum(prob_yes, 1 - prob_yes)
    picked = confidence >= pick_threshold
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y_true, prob_yes))
            if y_true.nunique() == 2
            else float("nan")
        ),
        "brier_score": float(brier_score_loss(y_true, prob_yes)),
        "picked_matches": float(int(picked.sum())),
        "pick_coverage": float(picked.mean()),
        "picked_hit_rate": (
            float((pred[picked] == y_array[picked]).mean())
            if int(picked.sum())
            else float("nan")
        ),
    }


def _clean_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for key, value in metrics.items():
        if value is None or pd.isna(value):
            continue
        cleaned[key] = float(value)
    return cleaned


def _threshold_metrics(threshold_report: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for _, row in threshold_report.iterrows():
        label = f"{float(row['threshold']):.2f}".replace(".", "_")
        metrics[f"picked_matches_t_{label}"] = float(row["picked_matches"])
        metrics[f"coverage_t_{label}"] = float(row["coverage"])
        if not pd.isna(row["picked_hit_rate"]):
            metrics[f"picked_hit_rate_t_{label}"] = float(row["picked_hit_rate"])
    return metrics


def _failure_result(config: MLflowConfig, timestamp: str, reason: str) -> dict[str, Any]:
    warning = f"MLflow logging skipped or failed: {reason}"
    warnings.warn(warning, RuntimeWarning, stacklevel=3)
    print(f"WARNING: {warning}")
    return {
        "logged": False,
        "tracking_uri": config.tracking_uri,
        "experiment_name": config.experiment_name,
        "registered_model_name": config.registered_model_name,
        "training_timestamp": timestamp,
        "reason": reason,
    }


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
    iteration: int = 1,
    extra_artifacts: Mapping[str, Path] | None = None,
    extra_metrics: Mapping[str, float] | None = None,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    registered_model_name: str | None = None,
) -> dict[str, Any]:
    """Log one parent training run, all candidate runs, and the selected model."""
    config = resolve_mlflow_config(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        registered_model_name=registered_model_name,
    )
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    reachable, detail = check_mlflow_server_reachable(config.tracking_uri)
    if not reachable:
        return _failure_result(config, timestamp, detail)

    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as exc:
        return _failure_result(config, timestamp, f"ImportError: {exc}")

    best_bundle = trained_models[best_model_name]
    best_metrics = compute_candidate_metrics(
        y_valid,
        best_bundle["prob"],
        best_bundle["pred"],
        pick_threshold=pick_threshold,
    )
    common_metrics = {
        "training_row_count": float(train_row_count),
        "validation_row_count": float(valid_row_count),
        "number_of_seasons": float(num_seasons),
        "number_of_features": float(len(feature_columns)),
        "pick_threshold": float(pick_threshold),
    }
    common_params = {
        "iteration": int(iteration),
        "training_data_source": training_data_source,
        "training_row_count": int(train_row_count),
        "validation_row_count": int(valid_row_count),
        "number_of_seasons": int(num_seasons),
        "number_of_features": int(len(feature_columns)),
        "pick_threshold": float(pick_threshold),
        "live_gap_matchdays": int(live_gap_matchdays),
        "training_timestamp": timestamp,
    }
    candidate_run_ids: dict[str, str] = {}
    registered_version: str | None = None

    try:
        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)
        with mlflow.start_run(
            run_name=f"btts-iteration-{iteration}-{best_model_name}"
        ) as parent_run:
            mlflow.set_tags(
                {
                    "selected_model_name": best_model_name,
                    "registered_model_name": config.registered_model_name,
                    "training_data_source": training_data_source,
                    "iteration": str(iteration),
                }
            )
            mlflow.log_params({**common_params, "selected_model_name": best_model_name})
            mlflow.log_metrics(
                _clean_metrics(
                    {
                        **common_metrics,
                        **best_metrics,
                        **_threshold_metrics(threshold_report),
                        **(extra_metrics or {}),
                    }
                )
            )

            with TemporaryDirectory() as temporary:
                temp = Path(temporary)
                comparison_path = temp / "model_comparison.csv"
                threshold_path = temp / "threshold_evaluation.csv"
                summary_path = temp / "selected_model_summary.json"
                model_comparison.to_csv(comparison_path, index=False)
                threshold_report.to_csv(threshold_path, index=False)
                summary_path.write_text(
                    json.dumps(
                        {
                            **common_params,
                            "selected_model_name": best_model_name,
                            "metrics": _clean_metrics(best_metrics),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                mlflow.log_artifact(str(comparison_path), artifact_path="evaluation")
                mlflow.log_artifact(str(threshold_path), artifact_path="evaluation")
                mlflow.log_artifact(str(summary_path), artifact_path="evaluation")

            for name, path in {
                "model_bundle/btts_model_iteration1.joblib": model_path,
                "model_bundle/btts_feature_columns_iteration1.json": columns_path,
                **(extra_artifacts or {}),
            }.items():
                if path.exists():
                    artifact_dir = str(Path(name).parent).replace("\\", "/")
                    mlflow.log_artifact(str(path), artifact_path=artifact_dir)

            for candidate_name, bundle in trained_models.items():
                candidate_metrics = compute_candidate_metrics(
                    y_valid,
                    bundle["prob"],
                    bundle["pred"],
                    pick_threshold=pick_threshold,
                )
                with mlflow.start_run(
                    run_name=f"candidate-{candidate_name}", nested=True
                ) as child_run:
                    candidate_run_ids[candidate_name] = child_run.info.run_id
                    mlflow.set_tags(
                        {
                            "model_name": candidate_name,
                            "is_selected_best": str(
                                candidate_name == best_model_name
                            ).lower(),
                            "iteration": str(iteration),
                        }
                    )
                    mlflow.log_params(
                        {
                            **common_params,
                            "model_name": candidate_name,
                            "is_selected_best": candidate_name == best_model_name,
                        }
                    )
                    mlflow.log_metrics(
                        _clean_metrics({**common_metrics, **candidate_metrics})
                    )

            try:
                model_info = mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path="model",
                    registered_model_name=config.registered_model_name,
                )
                registered_version = getattr(
                    model_info, "registered_model_version", None
                )
            except Exception as exc:
                warnings.warn(
                    f"Model registry warning: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

            return {
                "logged": True,
                "tracking_uri": config.tracking_uri,
                "experiment_name": config.experiment_name,
                "registered_model_name": config.registered_model_name,
                "registered_model_version": registered_version,
                "parent_run_id": parent_run.info.run_id,
                "candidate_run_ids": candidate_run_ids,
                "training_timestamp": timestamp,
            }
    except Exception as exc:
        return _failure_result(config, timestamp, f"{type(exc).__name__}: {exc}")
