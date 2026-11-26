"""Tests for the rolling base-feature builder."""
import unittest

import pandas as pd

from src.base_features import build_base_feature_rows


def sample_results() -> pd.DataFrame:
    rows = [
        ("1", 1, "ARS", "CHE", 1, 1),
        ("1", 1, "LIV", "MCI", 2, 0),
        ("1", 2, "CHE", "LIV", 1, 0),
        ("1", 2, "MCI", "ARS", 2, 2),
        ("1", 3, "ARS", "LIV", 4, 0),
    ]
    frame = pd.DataFrame(rows, columns=[
        "season_id", "matchday", "home_team", "away_team",
        "fulltime_home_goals", "fulltime_away_goals",
    ])
    frame["target_btts"] = (
        (frame["fulltime_home_goals"] > 0)
        & (frame["fulltime_away_goals"] > 0)
    ).astype(int)
    return frame


class BaseFeatureTests(unittest.TestCase):
    def test_cutoff_uses_only_completed_history(self) -> None:
        result = build_base_feature_rows(sample_results(), live_gap_matchdays=1)
        row = result[
            (result["target_matchday"] == 3)
            & (result["home_team"] == "ARS")
        ].iloc[0]
        self.assertEqual(row["feature_cutoff_matchday"], 2)
        self.assertEqual(row["season_matches_so_far"], 4)
        self.assertEqual(row["home_played"], 2)
        self.assertEqual(row["home_points"], 2)
        self.assertAlmostEqual(row["home_btts_rate_so_far"], 1.0)

    def test_target_result_does_not_leak_into_features(self) -> None:
        baseline = build_base_feature_rows(sample_results(), live_gap_matchdays=1)
        changed = sample_results()
        changed.loc[changed["matchday"].eq(3), [
            "fulltime_home_goals", "fulltime_away_goals", "target_btts"
        ]] = [0, 8, 0]
        updated = build_base_feature_rows(changed, live_gap_matchdays=1)
        feature_columns = [
            column for column in baseline.select_dtypes("number").columns
            if column != "target_btts"
        ]
        pd.testing.assert_frame_equal(
            baseline[feature_columns], updated[feature_columns]
        )


if __name__ == "__main__":
    unittest.main()
