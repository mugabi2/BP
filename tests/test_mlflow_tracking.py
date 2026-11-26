from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.mlflow_tracking import (
    compute_candidate_metrics,
    log_btts_training_to_mlflow,
)


class MLflowTrackingTests(unittest.TestCase):
    def test_candidate_metrics_include_confident_pick_performance(self) -> None:
        y_true = pd.Series([0, 1, 1, 0])
        probabilities = np.array([0.1, 0.8, 0.6, 0.4])
        metrics = compute_candidate_metrics(
            y_true,
            probabilities,
            pick_threshold=0.75,
        )
        self.assertEqual(metrics["picked_matches"], 2.0)
        self.assertEqual(metrics["picked_hit_rate"], 1.0)

    @patch(
        "src.mlflow_tracking.check_mlflow_server_reachable",
        return_value=(False, "test server unavailable"),
    )
    def test_unavailable_server_does_not_destroy_local_artifacts(self, _mock) -> None:
        result = log_btts_training_to_mlflow(
            trained_models={
                "test": {
                    "model": object(),
                    "prob": np.array([0.2, 0.8]),
                    "pred": np.array([0, 1]),
                }
            },
            model_comparison=pd.DataFrame(
                [{"model": "test", "accuracy": 1.0, "roc_auc": 1.0,
                  "brier_score": 0.04}]
            ),
            best_model_name="test",
            best_model=object(),
            threshold_report=pd.DataFrame(
                [{"threshold": 0.65, "picked_matches": 2, "coverage": 1.0,
                  "picked_hit_rate": 1.0}]
            ),
            y_valid=pd.Series([0, 1]),
            train_row_count=4,
            valid_row_count=2,
            num_seasons=2,
            feature_columns=["feature"],
            pick_threshold=0.65,
            live_gap_matchdays=2,
            training_data_source="csv",
            model_path=Path("candidate.joblib"),
            columns_path=Path("columns.json"),
        )
        self.assertFalse(result["logged"])
        self.assertIn("test server unavailable", result["reason"])


if __name__ == "__main__":
    unittest.main()
