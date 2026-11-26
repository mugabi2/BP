"""Replay a completed Iteration 1 training result into the configured MLflow."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from joblib import load
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mlflow_tracking import check_mlflow_server_reachable, resolve_mlflow_config


DIAGNOSTICS = ROOT / "docs" / "model_diagnostics"
SUMMARY_PATH = DIAGNOSTICS / "iteration1_training_summary.json"
MODEL_PATH = ROOT / "models" / "btts_model_iteration1.joblib"
COLUMNS_PATH = ROOT / "models" / "btts_feature_columns_iteration1.json"


def main() -> None:
    import mlflow
    import mlflow.sklearn

    config = resolve_mlflow_config()
    reachable, detail = check_mlflow_server_reachable(config.tracking_uri)
    if not reachable:
        raise SystemExit(detail)

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    comparison = pd.read_csv(DIAGNOSTICS / "iteration1_model_comparison.csv")
    thresholds = pd.read_csv(DIAGNOSTICS / "iteration1_threshold_evaluation.csv")
    baseline = pd.read_csv(DIAGNOSTICS / "iteration1_baseline_comparison.csv")
    selected_name = summary["selected_model_name"]
    selected = comparison.loc[comparison["model"].eq(selected_name)].iloc[0]
    market = baseline.loc[baseline["method"].eq("Market probability")]

    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    child_run_ids: dict[str, str] = {}
    with mlflow.start_run(run_name=f"btts-iteration-1-{selected_name}-replay") as parent:
        mlflow.set_tags({
            "iteration": "1",
            "selected_model_name": selected_name,
            "registered_model_name": config.registered_model_name,
            "training_data_source": "csv",
            "artifact_replay": "true",
        })
        mlflow.log_params({
            "iteration": 1,
            "selected_model_name": selected_name,
            "training_row_count": summary["training_rows"],
            "validation_row_count": summary["validation_rows"],
            "number_of_seasons": summary["seasons"],
            "number_of_features": summary["number_of_features"],
        })
        metrics = {
            "accuracy": float(selected["accuracy"]),
            "roc_auc": float(selected["roc_auc"]),
            "brier_score": float(selected["brier_score"]),
        }
        threshold_65 = thresholds.loc[thresholds["threshold"].round(2).eq(0.65)]
        if not threshold_65.empty:
            row = threshold_65.iloc[0]
            metrics.update({
                "picked_matches_t_0_65": float(row["picked_matches"]),
                "coverage_t_0_65": float(row["coverage"]),
                "picked_hit_rate_t_0_65": float(row["picked_hit_rate"]),
            })
        if not market.empty:
            row = market.iloc[0]
            metrics.update({
                "market_accuracy": float(row["accuracy"]),
                "market_roc_auc": float(row["roc_auc"]),
                "market_brier_score": float(row["brier_score"]),
            })
        mlflow.log_metrics(metrics)

        for path in sorted(DIAGNOSTICS.glob("iteration1_*")):
            if path.is_file():
                mlflow.log_artifact(str(path), artifact_path="diagnostics")
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model_bundle")
        mlflow.log_artifact(str(COLUMNS_PATH), artifact_path="model_bundle")

        for candidate in comparison.itertuples(index=False):
            with mlflow.start_run(
                run_name=f"candidate-{candidate.model}", nested=True
            ) as child:
                child_run_ids[candidate.model] = child.info.run_id
                mlflow.set_tags({
                    "model_name": candidate.model,
                    "is_selected_best": str(candidate.model == selected_name).lower(),
                    "iteration": "1",
                    "artifact_replay": "true",
                })
                mlflow.log_metrics({
                    "accuracy": float(candidate.accuracy),
                    "roc_auc": float(candidate.roc_auc),
                    "brier_score": float(candidate.brier_score),
                })

        model_info = mlflow.sklearn.log_model(
            sk_model=load(MODEL_PATH),
            artifact_path="model",
            registered_model_name=config.registered_model_name,
        )
        result = {
            "logged": True,
            "tracking_uri": config.tracking_uri,
            "experiment_name": config.experiment_name,
            "registered_model_name": config.registered_model_name,
            "registered_model_version": getattr(
                model_info, "registered_model_version", None
            ),
            "parent_run_id": parent.info.run_id,
            "candidate_run_ids": child_run_ids,
            "training_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "artifact_replay": True,
        }

    summary["mlflow"] = result
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
