from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from joblib import load
import pandas as pd

from src.btts_training import (
    build_candidate_models,
    build_training_table,
    make_odds_template,
    select_model_features,
    write_model_bundle,
)
from src.training_data import normalize_results_dataframe, resolve_training_data_config


def _sample_results() -> pd.DataFrame:
    fixtures = {
        1: [("A", "B", 1, 1), ("C", "D", 0, 1)],
        2: [("A", "C", 2, 0), ("B", "D", 1, 2)],
        3: [("A", "D", 1, 0), ("B", "C", 2, 2)],
        4: [("A", "B", 0, 1), ("C", "D", 1, 1)],
    }
    rows = []
    for season_id in ("100", "101"):
        for matchday, matches in fixtures.items():
            for home, away, home_goals, away_goals in matches:
                rows.append(
                    {
                        "season_id": season_id,
                        "matchday": matchday,
                        "home_team": home,
                        "away_team": away,
                        "halftime_home_goals": 0,
                        "halftime_away_goals": 0,
                        "fulltime_home_goals": home_goals,
                        "fulltime_away_goals": away_goals,
                        "captured_at": "2026-01-01T00:00:00Z",
                    }
                )
    return normalize_results_dataframe(pd.DataFrame(rows))


class BttsTrainingTests(unittest.TestCase):
    def test_feature_table_preserves_n_minus_two_cutoff(self) -> None:
        results = _sample_results()
        odds_template, odds_keys = make_odds_template(pd.DataFrame())

        training = build_training_table(results, odds_template, odds_keys)

        self.assertEqual(int(training["target_matchday"].min()), 3)
        self.assertTrue(
            (
                training["feature_cutoff_matchday"]
                == training["target_matchday"] - 2
            ).all()
        )
        matchday_three = training[training["target_matchday"] == 3]
        self.assertTrue((matchday_three["season_matches_so_far"] == 2).all())

    def test_candidate_model_set_is_unchanged(self) -> None:
        self.assertEqual(
            list(build_candidate_models()),
            [
                "LogisticRegression",
                "GradientBoostingClassifier",
                "HistGradientBoostingClassifier",
                "RandomForestClassifier",
                "ExtraTreesClassifier",
                "SoftVotingEnsemble",
            ],
        )

    def test_model_bundle_is_written_atomically(self) -> None:
        comparison = pd.DataFrame(
            [
                {
                    "model": "LogisticRegression",
                    "accuracy": 0.6,
                    "precision": 0.6,
                    "recall": 0.6,
                    "f1": 0.6,
                    "roc_auc": 0.7,
                    "brier_score": 0.2,
                }
            ]
        )
        thresholds = pd.DataFrame(
            [
                {
                    "threshold": 0.65,
                    "picked_matches": 2,
                    "coverage": 0.5,
                    "picked_hit_rate": 1.0,
                }
            ]
        )

        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            output_dir = Path(temporary)
            model_path, columns_path, _, _ = write_model_bundle(
                output_dir=output_dir,
                model={"model": "test"},
                model_features=["home_points"],
                all_missing_features=[],
                selected_model_name="LogisticRegression",
                model_comparison=comparison,
                threshold_report=thresholds,
                live_gap_matchdays=2,
                pick_threshold=0.65,
            )

            self.assertEqual(load(model_path), {"model": "test"})
            payload = json.loads(columns_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["feature_columns"], ["home_points"])
            self.assertEqual(payload["live_gap_matchdays"], 2)
            self.assertFalse(list(output_dir.glob(".*.tmp")))

    def test_postgres_can_use_standard_libpq_environment(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "TRAINING_DATA_SOURCE": "postgres",
                "PGHOST": "postgres",
                "PGUSER": "bp_user",
                "PGDATABASE": "bp",
            },
            clear=True,
        ):
            config = resolve_training_data_config()

        self.assertEqual(config.source, "postgres")
        self.assertIsNone(config.database_url)

    def test_feature_selection_excludes_all_missing_columns(self) -> None:
        results = _sample_results()
        odds_template, odds_keys = make_odds_template(pd.DataFrame())
        training = build_training_table(results, odds_template, odds_keys)

        features, missing, _ = select_model_features(training)

        self.assertIn("odds_btts_yes", missing)
        self.assertNotIn("odds_btts_yes", features)
        self.assertIn("home_points", features)


if __name__ == "__main__":
    unittest.main()
