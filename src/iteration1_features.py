"""Iteration 1 BTTS features, shared by training and live prediction.

All historical features are as-of: previous seasons, plus the current season
strictly through feature_cutoff_matchday. No collector or schema changes.
"""
from __future__ import annotations

from collections import deque

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


def _group_by_team(team_rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if team_rows.empty:
        return {}
    return {
        str(team): group.reset_index(drop=True)
        for team, group in team_rows.groupby("team", sort=False)
    }


def _prepare_context(
    history: pd.DataFrame,
    previous_team_groups: dict[int, dict[str, pd.DataFrame]],
) -> dict:
    """Prepare reusable lookups for one season/cutoff feature snapshot."""
    teams = _team_rows(history) if not history.empty else pd.DataFrame(
        columns=["season_id", "matchday", "team", "opponent", "goals_for",
                 "goals_against", "btts", "side"]
    )
    if not teams.empty:
        order = {
            season: idx
            for idx, season in enumerate(
                sorted(teams["season_id"].unique(), key=_season_key)
            )
        }
        teams = teams.assign(_season_order=teams["season_id"].map(order))
        teams = teams.sort_values(["_season_order", "matchday"], kind="stable")

    pair_groups: dict[tuple[str, str], pd.DataFrame] = {}
    if not history.empty:
        pair_keys = [
            tuple(sorted((str(home), str(away))))
            for home, away in zip(history["home_team"], history["away_team"])
        ]
        keyed = history.assign(_pair_key=pair_keys)
        pair_groups = {
            key: group.drop(columns="_pair_key").reset_index(drop=True)
            for key, group in keyed.groupby("_pair_key", sort=False)
        }

    return {
        "team_groups": _group_by_team(teams),
        "pair_groups": pair_groups,
        "previous_team_groups": previous_team_groups,
    }


def _context_values(context: dict, home_team: str, away_team: str) -> dict:
    values = {}
    empty_team_rows = pd.DataFrame(
        columns=["season_id", "matchday", "team", "opponent", "goals_for",
                 "goals_against", "btts", "side"]
    )
    team_groups = context["team_groups"]
    for prefix, team, side in (("home", home_team, "home"), ("away", away_team, "away")):
        t = team_groups.get(team, empty_team_rows)
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

    pair = context["pair_groups"].get(
        tuple(sorted((home_team, away_team))),
        pd.DataFrame(),
    )
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
        historic_teams = context["previous_team_groups"].get(window, {})
        for prefix, team in (("home", home_team), ("away", away_team)):
            past = historic_teams.get(team, empty_team_rows)
            values[f"{prefix}_prev_{window}_seasons_btts_rate"] = _rate(past, "btts")
            values[f"{prefix}_prev_{window}_seasons_goals_for_avg"] = _rate(past, "goals_for")
    return values


def _new_rolling_state() -> dict:
    """Create compact aggregates for one chronological history stream."""
    return {
        "team_recent": {},
        "team_side_recent": {},
        "team_side_totals": {},
        "pair_totals": {},
    }


def _append_history_match(state: dict, row: object) -> None:
    """Add one completed match to the rolling feature state."""
    home = str(row.home_team)
    away = str(row.away_team)
    home_goals = float(row.fulltime_home_goals)
    away_goals = float(row.fulltime_away_goals)
    btts = float(home_goals > 0 and away_goals > 0)

    for team, side, goals_for, goals_against in (
        (home, "home", home_goals, away_goals),
        (away, "away", away_goals, home_goals),
    ):
        values = (goals_for, goals_against, btts)
        state["team_recent"].setdefault(team, deque(maxlen=max(RECENT_WINDOWS))).append(values)
        state["team_side_recent"].setdefault(
            (team, side), deque(maxlen=max(RECENT_WINDOWS))
        ).append(values)
        totals = state["team_side_totals"].setdefault(
            (team, side), [0.0, 0.0, 0.0, 0.0]
        )
        totals[0] += 1
        totals[1] += goals_for
        totals[2] += goals_against
        totals[3] += btts

    first, second = sorted((home, away))
    pair = state["pair_totals"].setdefault(
        (first, second),
        {
            "matches": 0.0,
            "btts": 0.0,
            "goals_first": 0.0,
            "goals_second": 0.0,
            "wins_first": 0.0,
            "wins_second": 0.0,
            "draws": 0.0,
        },
    )
    goals_first, goals_second = (
        (home_goals, away_goals) if home == first else (away_goals, home_goals)
    )
    pair["matches"] += 1
    pair["btts"] += btts
    pair["goals_first"] += goals_first
    pair["goals_second"] += goals_second
    if goals_first > goals_second:
        pair["wins_first"] += 1
    elif goals_second > goals_first:
        pair["wins_second"] += 1
    else:
        pair["draws"] += 1


def _recent_values(rows: deque, window: int) -> tuple[int, float, float, float]:
    selected = list(rows)[-window:]
    if not selected:
        return 0, np.nan, np.nan, np.nan
    values = np.asarray(selected, dtype=float)
    return len(selected), float(values[:, 0].mean()), float(values[:, 1].mean()), float(values[:, 2].mean())


def _previous_team_summary(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    if frame.empty:
        return {}
    team_rows = _team_rows(frame)
    grouped = team_rows.groupby("team", sort=False).agg(
        btts=("btts", "mean"),
        goals_for=("goals_for", "mean"),
    )
    return {
        str(team): (float(row.btts), float(row.goals_for))
        for team, row in grouped.iterrows()
    }


def _rolling_context_values(
    state: dict,
    previous_summaries: dict[int, dict[str, tuple[float, float]]],
    home_team: str,
    away_team: str,
) -> dict:
    """Read all Iteration 1 values from the rolling state in O(1)."""
    values: dict[str, float | int] = {}
    empty_recent: deque = deque()
    for prefix, team, side in (
        ("home", home_team, "home"),
        ("away", away_team, "away"),
    ):
        totals = state["team_side_totals"].get((team, side), [0.0] * 4)
        count = int(totals[0])
        values[f"{prefix}_side_matches"] = count
        values[f"{prefix}_side_goals_for_rate"] = totals[1] / count if count else np.nan
        values[f"{prefix}_side_goals_against_rate"] = totals[2] / count if count else np.nan
        values[f"{prefix}_side_btts_rate"] = totals[3] / count if count else np.nan

        recent_rows = state["team_recent"].get(team, empty_recent)
        side_rows = state["team_side_recent"].get((team, side), empty_recent)
        for window in RECENT_WINDOWS:
            for name, rows in (("recent", recent_rows), ("side_recent", side_rows)):
                matches, goals_for, goals_against, btts = _recent_values(rows, window)
                values[f"{prefix}_{name}_matches_{window}"] = matches
                values[f"{prefix}_{name}_goals_for_{window}"] = goals_for
                values[f"{prefix}_{name}_goals_against_{window}"] = goals_against
                values[f"{prefix}_{name}_btts_rate_{window}"] = btts

    pair = state["pair_totals"].get(tuple(sorted((home_team, away_team))))
    if pair:
        matches = pair["matches"]
        home_is_first = home_team == min(home_team, away_team)
        values["h2h_matches"] = int(matches)
        values["h2h_btts_rate"] = pair["btts"] / matches
        values["h2h_home_win_rate"] = pair["wins_first" if home_is_first else "wins_second"] / matches
        values["h2h_draw_rate"] = pair["draws"] / matches
        values["h2h_away_win_rate"] = pair["wins_second" if home_is_first else "wins_first"] / matches
        values["h2h_home_goals_avg"] = pair["goals_first" if home_is_first else "goals_second"] / matches
        values["h2h_away_goals_avg"] = pair["goals_second" if home_is_first else "goals_first"] / matches
    else:
        values["h2h_matches"] = 0
        for name in (
            "h2h_btts_rate", "h2h_home_win_rate", "h2h_draw_rate",
            "h2h_away_win_rate", "h2h_home_goals_avg", "h2h_away_goals_avg",
        ):
            values[name] = np.nan

    for window in HISTORY_WINDOWS:
        summaries = previous_summaries[window]
        for prefix, team in (("home", home_team), ("away", away_team)):
            btts, goals_for = summaries.get(team, (np.nan, np.nan))
            values[f"{prefix}_prev_{window}_seasons_btts_rate"] = btts
            values[f"{prefix}_prev_{window}_seasons_goals_for_avg"] = goals_for
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

    # One season's previous seasons are fixed. Build a rolling state once and add
    # each current-season match once as cutoffs advance. This avoids re-grouping
    # the same 20 seasons for every matchday.
    prior_cache = {}
    new_records = [None] * len(result)
    for season_value, season_group in result.groupby("season_id", sort=False):
        season = str(season_value)
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
            prior_cache[season] = {
                "frames": previous,
                "summaries": {
                    n: _previous_team_summary(frame)
                    for n, frame in previous.items()
                },
            }
        previous_bundle = prior_cache[season]
        previous = previous_bundle["frames"]
        state = _new_rolling_state()
        for history_row in previous[20].itertuples(index=False):
            _append_history_match(state, history_row)

        current = seasons.get(season)
        current_rows = list(current.itertuples(index=False)) if current is not None else []
        position = 0
        cutoff_groups = season_group.groupby("feature_cutoff_matchday", sort=True)
        for cutoff_value, cutoff_group in cutoff_groups:
            cutoff = int(cutoff_value)
            while position < len(current_rows) and current_rows[position].matchday <= cutoff:
                _append_history_match(state, current_rows[position])
                position += 1
            for idx, row in cutoff_group.iterrows():
                new_records[idx] = _rolling_context_values(
                    state,
                    previous_bundle["summaries"],
                    row["home_team"],
                    row["away_team"],
                )
    added = pd.DataFrame.from_records(new_records)
    return pd.concat([result, added], axis=1)
