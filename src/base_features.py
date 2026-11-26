"""Fast, leakage-safe base features for BTTS training rows."""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd


TEAM_STATS_COLUMNS = [
    "rank", "played", "points", "wins", "draws", "losses", "goals_for",
    "goals_against", "goal_difference", "btts_rate_so_far",
    "scored_rate_so_far", "conceded_rate_so_far", "avg_goals_for_so_far",
    "avg_goals_against_so_far",
]
FORM_COLUMNS = [
    "form_points_last_5", "form_wins_last_5", "form_draws_last_5",
    "form_losses_last_5",
]
SEASON_FEATURE_COLUMNS = [
    "season_matches_so_far", "season_btts_rate_so_far",
    "season_avg_goals_per_match_so_far", "season_home_win_rate_so_far",
    "season_draw_rate_so_far", "season_away_win_rate_so_far",
]


def _new_team_state() -> dict:
    return {
        "played": 0, "points": 0, "wins": 0, "draws": 0, "losses": 0,
        "goals_for": 0.0, "goals_against": 0.0, "btts": 0,
        "scored": 0, "conceded": 0, "form": deque(maxlen=5),
    }


def _add_team_match(
    teams: dict[str, dict],
    team: str,
    goals_for: float,
    goals_against: float,
) -> None:
    state = teams.setdefault(team, _new_team_state())
    win = int(goals_for > goals_against)
    draw = int(goals_for == goals_against)
    loss = int(goals_for < goals_against)
    points = 3 * win + draw
    state["played"] += 1
    state["points"] += points
    state["wins"] += win
    state["draws"] += draw
    state["losses"] += loss
    state["goals_for"] += goals_for
    state["goals_against"] += goals_against
    state["btts"] += int(goals_for > 0 and goals_against > 0)
    state["scored"] += int(goals_for > 0)
    state["conceded"] += int(goals_against > 0)
    state["form"].append((points, win, draw, loss))


def _add_match(teams: dict[str, dict], season: dict, row: object) -> None:
    home_goals = float(row.fulltime_home_goals)
    away_goals = float(row.fulltime_away_goals)
    _add_team_match(teams, str(row.home_team), home_goals, away_goals)
    _add_team_match(teams, str(row.away_team), away_goals, home_goals)
    season["matches"] += 1
    season["btts"] += int(home_goals > 0 and away_goals > 0)
    season["total_goals"] += home_goals + away_goals
    season["home_wins"] += int(home_goals > away_goals)
    season["draws"] += int(home_goals == away_goals)
    season["away_wins"] += int(home_goals < away_goals)


def _snapshot(teams: dict[str, dict], season: dict) -> dict:
    team_values: dict[str, dict] = {}
    sortable = []
    for team, state in teams.items():
        played = state["played"]
        values = {
            "played": played,
            "points": state["points"],
            "wins": state["wins"],
            "draws": state["draws"],
            "losses": state["losses"],
            "goals_for": state["goals_for"],
            "goals_against": state["goals_against"],
            "goal_difference": state["goals_for"] - state["goals_against"],
            "btts_rate_so_far": state["btts"] / played,
            "scored_rate_so_far": state["scored"] / played,
            "conceded_rate_so_far": state["conceded"] / played,
            "avg_goals_for_so_far": state["goals_for"] / played,
            "avg_goals_against_so_far": state["goals_against"] / played,
        }
        form = list(state["form"])
        values.update({
            "form_points_last_5": sum(item[0] for item in form),
            "form_wins_last_5": sum(item[1] for item in form),
            "form_draws_last_5": sum(item[2] for item in form),
            "form_losses_last_5": sum(item[3] for item in form),
        })
        team_values[team] = values
        sortable.append((team, values))
    sortable.sort(
        key=lambda item: (
            -item[1]["points"], -item[1]["goal_difference"],
            -item[1]["goals_for"], item[0],
        )
    )
    for rank, (team, _) in enumerate(sortable, start=1):
        team_values[team]["rank"] = rank

    matches = season["matches"]
    season_values = {
        "season_matches_so_far": matches,
        "season_btts_rate_so_far": season["btts"] / matches if matches else np.nan,
        "season_avg_goals_per_match_so_far": season["total_goals"] / matches if matches else np.nan,
        "season_home_win_rate_so_far": season["home_wins"] / matches if matches else np.nan,
        "season_draw_rate_so_far": season["draws"] / matches if matches else np.nan,
        "season_away_win_rate_so_far": season["away_wins"] / matches if matches else np.nan,
    }
    return {"teams": team_values, "season": season_values}


def _team_values(bundle: dict, team: str, prefix: str) -> dict:
    values = bundle["teams"].get(team)
    columns = TEAM_STATS_COLUMNS + FORM_COLUMNS
    if values is None:
        return {f"{prefix}_{column}": np.nan for column in columns}
    return {f"{prefix}_{column}": values[column] for column in columns}


def _comparison_values(record: dict) -> None:
    record["points_diff"] = record["home_points"] - record["away_points"]
    record["rank_diff"] = record["away_rank"] - record["home_rank"]
    record["goal_difference_diff"] = record["home_goal_difference"] - record["away_goal_difference"]
    record["goals_for_diff"] = record["home_goals_for"] - record["away_goals_for"]
    record["goals_against_diff"] = record["home_goals_against"] - record["away_goals_against"]
    record["form_diff"] = record["home_form_points_last_5"] - record["away_form_points_last_5"]
    record["btts_rate_diff"] = record["home_btts_rate_so_far"] - record["away_btts_rate_so_far"]
    record["scored_rate_diff"] = record["home_scored_rate_so_far"] - record["away_scored_rate_so_far"]
    record["conceded_rate_diff"] = record["home_conceded_rate_so_far"] - record["away_conceded_rate_so_far"]


def build_base_feature_rows(results: pd.DataFrame, live_gap_matchdays: int = 2) -> pd.DataFrame:
    """Build all base rows in one chronological pass per season."""
    records: list[dict] = []
    for season_id, season_frame in results.groupby("season_id", sort=False):
        season_frame = season_frame.sort_values("matchday", kind="stable")
        completed = list(season_frame.itertuples(index=False))
        cutoffs = sorted({
            int(matchday) - live_gap_matchdays
            for matchday in season_frame["matchday"]
            if int(matchday) - live_gap_matchdays >= 1
        })
        teams: dict[str, dict] = {}
        season = {
            "matches": 0, "btts": 0, "total_goals": 0.0,
            "home_wins": 0, "draws": 0, "away_wins": 0,
        }
        bundles: dict[int, dict] = {}
        position = 0
        for cutoff in cutoffs:
            while position < len(completed) and completed[position].matchday <= cutoff:
                _add_match(teams, season, completed[position])
                position += 1
            bundles[cutoff] = _snapshot(teams, season)

        for match in completed:
            cutoff = int(match.matchday) - live_gap_matchdays
            if cutoff < 1:
                continue
            bundle = bundles[cutoff]
            record = {
                "season_id": str(season_id),
                "target_matchday": int(match.matchday),
                "feature_cutoff_matchday": cutoff,
                "home_team": str(match.home_team),
                "away_team": str(match.away_team),
                **bundle["season"],
            }
            record.update(_team_values(bundle, str(match.home_team), "home"))
            record.update(_team_values(bundle, str(match.away_team), "away"))
            _comparison_values(record)
            record["target_btts"] = int(match.target_btts)
            records.append(record)
    return pd.DataFrame.from_records(records)
