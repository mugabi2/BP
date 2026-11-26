"""Iteration 1 BTTS features, shared by training and live prediction.

All historical features are as-of: previous seasons, plus the current season
strictly through feature_cutoff_matchday. No collector or schema changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HISTORY_WINDOWS = (5, 10, 20)
RECENT_WINDOWS = (5, 10)
SIDE_COLUMNS = ("goals_for", "goals_against", "btts")


def _season_key(value: object) -> tuple:
    """Sort numeric virtual season IDs chronologically when available."""
    text = str(value)
    try:
        return (0, int(text))
    except (ValueError, TypeError):
        return (1, text)


def _prepare_results(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    required = {
        "season_id", "matchday", "home_team", "away_team",
        "fulltime_home_goals", "fulltime_away_goals",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")
    df = results.copy()
    df["season_id"] = df["season_id"].astype(str)
    df["matchday"] = pd.to_numeric(df["matchday"], errors="coerce")
    df["fulltime_home_goals"] = pd.to_numeric(df["fulltime_home_goals"], errors="coerce")
    df["fulltime_away_goals"] = pd.to_numeric(df["fulltime_away_goals"], errors="coerce")
    df = df.dropna(subset=["matchday", "fulltime_home_goals", "fulltime_away_goals"])
    df["home_team"] = df["home_team"].astype(str).str.upper().str.strip()
    df["away_team"] = df["away_team"].astype(str).str.upper().str.strip()
    df = df.drop_duplicates(
        ["season_id", "matchday", "home_team", "away_team"], keep="last"
    )
    df["target_btts"] = (
        (df["fulltime_home_goals"] > 0) & (df["fulltime_away_goals"] > 0)
    ).astype(float)
    return {
        season: group.sort_values("matchday").reset_index(drop=True)
        for season, group in df.groupby("season_id", sort=False)
    }


def _team_rows(history: pd.DataFrame) -> pd.DataFrame:
    home = history[["season_id", "matchday", "home_team", "away_team",
                    "fulltime_home_goals", "fulltime_away_goals", "target_btts"]].copy()
    home.columns = ["season_id", "matchday", "team", "opponent",
                    "goals_for", "goals_against", "btts"]
    home["side"] = "home"
    away = history[["season_id", "matchday", "away_team", "home_team",
                    "fulltime_away_goals", "fulltime_home_goals", "target_btts"]].copy()
    away.columns = home.columns[:7].tolist()
    away["side"] = "away"
    return pd.concat([home, away], ignore_index=True)


def _rate(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].mean()) if not frame.empty else np.nan


def _context_values(history: pd.DataFrame, home_team: str, away_team: str,
                    previous_season_frames: dict[int, pd.DataFrame]) -> dict:
    values = {}
    # Identity is handled separately by one-hot columns.
    teams = _team_rows(history) if not history.empty else pd.DataFrame(
        columns=["season_id", "matchday", "team", "opponent", "goals_for",
                 "goals_against", "btts", "side"]
    )
    order = {season: idx for idx, season in enumerate(
        sorted(teams["season_id"].unique(), key=_season_key)
    )} if not teams.empty else {}
    if not teams.empty:
        teams = teams.assign(_season_order=teams["season_id"].map(order))
        teams = teams.sort_values(["_season_order", "matchday"], kind="stable")
    for prefix, team, side in (("home", home_team, "home"), ("away", away_team, "away")):
        t = teams[teams["team"] == team]
        side_matches = t[t["side"] == side]
        values[f"{prefix}_side_matches"] = len(side_matches)
        for column in SIDE_COLUMNS:
            values[f"{prefix}_side_{column}_rate"] = (
                _rate(side_matches, column) if column == "btts" else
                _rate(side_matches, column)
            )
        for window in RECENT_WINDOWS:
            recent = t.tail(window)
            side_recent = side_matches.tail(window)
            for name, frame in (("recent", recent), ("side_recent", side_recent)):
                values[f"{prefix}_{name}_matches_{window}"] = len(frame)
                values[f"{prefix}_{name}_goals_for_{window}"] = (
                    float(frame["goals_for"].mean()) if not frame.empty else np.nan
                )
                values[f"{prefix}_{name}_goals_against_{window}"] = (
                    float(frame["goals_against"].mean()) if not frame.empty else np.nan
                )
                values[f"{prefix}_{name}_btts_rate_{window}"] = _rate(frame, "btts")

    pair = history[
        ((history["home_team"] == home_team) & (history["away_team"] == away_team)) |
        ((history["home_team"] == away_team) & (history["away_team"] == home_team))
    ] if not history.empty else history
    values["h2h_matches"] = len(pair)
    values["h2h_btts_rate"] = _rate(pair, "target_btts")
    if not pair.empty:
        home_goals = np.where(pair["home_team"] == home_team,
                              pair["fulltime_home_goals"], pair["fulltime_away_goals"])
        away_goals = np.where(pair["home_team"] == home_team,
                              pair["fulltime_away_goals"], pair["fulltime_home_goals"])
        values["h2h_home_win_rate"] = float(np.mean(home_goals > away_goals))
        values["h2h_draw_rate"] = float(np.mean(home_goals == away_goals))
        values["h2h_away_win_rate"] = float(np.mean(home_goals < away_goals))
        values["h2h_home_goals_avg"] = float(np.mean(home_goals))
        values["h2h_away_goals_avg"] = float(np.mean(away_goals))
    else:
        for name in ("h2h_home_win_rate", "h2h_draw_rate", "h2h_away_win_rate",
                     "h2h_home_goals_avg", "h2h_away_goals_avg"):
            values[name] = np.nan
    for window in HISTORY_WINDOWS:
        historic = previous_season_frames.get(window)
        if historic is None or historic.empty:
            for prefix in ("home", "away"):
                values[f"{prefix}_prev_{window}_seasons_btts_rate"] = np.nan
                values[f"{prefix}_prev_{window}_seasons_goals_for_avg"] = np.nan
            continue
        historic_teams = _team_rows(historic)
        for prefix, team in (("home", home_team), ("away", away_team)):
            past = historic_teams[historic_teams["team"] == team]
            values[f"{prefix}_prev_{window}_seasons_btts_rate"] = _rate(past, "btts")
            values[f"{prefix}_prev_{window}_seasons_goals_for_avg"] = _rate(past, "goals_for")
    return values


def add_iteration1_features(
    feature_rows: pd.DataFrame, all_results: pd.DataFrame
) -> pd.DataFrame:
    """Return model-ready numeric Iteration 1 columns with strict temporal cutoffs.

    Current-season matches with matchday > cutoff are excluded, including live
    results that may already exist in the history file. Future seasons excluded.
    """
    if feature_rows.empty:
        return feature_rows.copy()
    seasons = _prepare_results(all_results)
    season_order = sorted(seasons, key=_season_key)
    season_positions = {s: i for i, s in enumerate(season_order)}
    result = feature_rows.copy().reset_index(drop=True)
    result["home_team"] = result["home_team"].astype(str).str.upper().str.strip()
    result["away_team"] = result["away_team"].astype(str).str.upper().str.strip()
    all_teams = sorted(set(all_results["home_team"].astype(str).str.upper().str.strip()) |
                       set(all_results["away_team"].astype(str).str.upper().str.strip()))
    # Stable one-hot identity columns. Unseen teams produce all zeros at inference.
    for team in all_teams:
        result[f"home_team_is_{team}"] = (result["home_team"] == team).astype("int8")
        result[f"away_team_is_{team}"] = (result["away_team"] == team).astype("int8")

    # One season's previous seasons are fixed; reuse instead of reassembling for
    # every matchday. At inference, unknown current season IDs are handled too.
    prior_cache = {}
    context_cache = {}
    new_records = [None] * len(result)
    group_columns = ["season_id", "feature_cutoff_matchday"]
    for (season, cutoff), group in result.groupby(group_columns, sort=False):
        season = str(season)
        cutoff = int(cutoff)
        if season not in prior_cache:
            position = season_positions.get(season)
            if position is None:
                position = sum(_season_key(s) < _season_key(season) for s in season_order)
            prior_seasons = season_order[max(0, position-20):position]
            previous = {
                n: pd.concat([seasons[s] for s in prior_seasons[-n:]], ignore_index=True)
                if prior_seasons else pd.DataFrame(columns=all_results.columns)
                for n in HISTORY_WINDOWS
            }
            prior_cache[season] = previous
        previous = prior_cache[season]
        current = seasons.get(season)
        current = (current[current["matchday"] <= cutoff] if current is not None
                   else pd.DataFrame(columns=all_results.columns))
        history = pd.concat([previous[20], current], ignore_index=True)
        for idx, row in group.iterrows():
            pair_key = (season, cutoff, row["home_team"], row["away_team"])
            if pair_key not in context_cache:
                context_cache[pair_key] = _context_values(
                    history, row["home_team"], row["away_team"], previous
                )
            new_records[idx] = context_cache[pair_key]
    added = pd.DataFrame.from_records(new_records)
    return pd.concat([result, added], axis=1)
