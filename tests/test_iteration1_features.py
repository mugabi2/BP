"""Regression tests for BTTS Iteration 1 feature engineering."""
import unittest

import pandas as pd

from src.iteration1_features import add_iteration1_features


def sample_results():
    rows = [
        ("100", 1, "ARS", "CHE", 1, 1),
        ("100", 2, "CHE", "ARS", 2, 0),
        ("101", 1, "ARS", "CHE", 0, 1),
        ("101", 2, "CHE", "ARS", 2, 2),
        ("101", 3, "ARS", "CHE", 3, 3),  # Must not influence MD4.
        ("102", 1, "ARS", "CHE", 4, 4),  # Future season.
    ]
    return pd.DataFrame(rows, columns=[
        "season_id", "matchday", "home_team", "away_team",
        "fulltime_home_goals", "fulltime_away_goals"
    ])


def fixture(season="101", cutoff=2):
    return pd.DataFrame([{
        "season_id": season,
        "target_matchday": cutoff + 2,
        "feature_cutoff_matchday": cutoff,
        "home_team": "ARS",
        "away_team": "CHE",
    }])


class Iteration1FeatureTests(unittest.TestCase):
    def test_future_matchday_and_season_never_leak(self):
        results = sample_results()
        baseline = add_iteration1_features(fixture(), results)
        changed = results.copy()
        changed.loc[changed["matchday"].eq(3), [
            "fulltime_home_goals", "fulltime_away_goals"
        ]] = [8, 0]
        changed.loc[changed["season_id"].eq("102"), [
            "fulltime_home_goals", "fulltime_away_goals"
        ]] = [8, 0]
        updated = add_iteration1_features(fixture(), changed)
        numeric = baseline.select_dtypes("number").columns
        pd.testing.assert_frame_equal(baseline[numeric], updated[numeric])
        self.assertEqual(baseline.loc[0, "h2h_matches"], 4)
        self.assertAlmostEqual(baseline.loc[0, "h2h_btts_rate"], 0.5)

    def test_cross_season_btts_and_side_form_are_available(self):
        features = add_iteration1_features(fixture(), sample_results()).iloc[0]
        self.assertAlmostEqual(features["home_prev_5_seasons_btts_rate"], 0.5)
        self.assertAlmostEqual(features["away_prev_5_seasons_btts_rate"], 0.5)
        self.assertEqual(features["home_team_is_ARS"], 1)
        self.assertEqual(features["away_team_is_CHE"], 1)
        self.assertGreaterEqual(features["home_side_matches"], 1)
        self.assertGreaterEqual(features["away_recent_goals_for_5"], 0)

    def test_unseen_current_season_can_use_prior_history(self):
        features = add_iteration1_features(fixture("103", 2), sample_results()).iloc[0]
        self.assertEqual(features["h2h_matches"], 6)
        self.assertAlmostEqual(features["home_prev_5_seasons_btts_rate"], 4 / 6)

    def test_unknown_teams_do_not_crash(self):
        row = fixture()
        row["home_team"] = "XYZ"
        result = add_iteration1_features(row, sample_results())
        self.assertEqual(result.loc[0, "home_team_is_ARS"], 0)
        self.assertTrue(pd.isna(result.loc[0, "home_recent_goals_for_5"]))

    def test_iteration1_inventory_contains_all_feature_families(self):
        result = add_iteration1_features(fixture(), sample_results())
        expected = {
            "home_team_is_ARS",
            "away_team_is_CHE",
            "home_side_btts_rate",
            "away_recent_goals_for_10",
            "h2h_btts_rate",
            "home_prev_20_seasons_btts_rate",
        }
        self.assertTrue(expected.issubset(result.columns))
        self.assertGreater(len(result.columns), len(fixture().columns) + 20)


if __name__ == "__main__":
    unittest.main()
