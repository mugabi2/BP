from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

try:
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:  # pragma: no cover
    InconsistentVersionWarning = UserWarning

try:
    from src.config import load_settings
    from src.live_results import collect_live_results
    from src.odds_scraper import collect_odds as collect_current_odds
    COLLECTOR_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - Streamlit will show a button-level error.
    load_settings = None
    collect_live_results = None
    collect_current_odds = None
    COLLECTOR_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "btts_model.joblib"
COLUMNS_PATH = PROJECT_ROOT / "models" / "btts_feature_columns.json"
ODDS_PATH = PROJECT_ROOT / "odds" / "final_odds" / "final_odds.csv"
LIVE_RESULTS_DIR = PROJECT_ROOT / "data" / "live_seasons"

DEFAULT_LIVE_GAP_MATCHDAYS = 2
DEFAULT_PICK_THRESHOLD = 0.65

ODDS_RAW_COLUMNS = ["odds_1", "odds_x", "odds_2", "odds_btts_yes", "odds_btts_no"]
ODDS_DERIVED_COLUMNS = [
    "implied_home_prob",
    "implied_draw_prob",
    "implied_away_prob",
    "implied_btts_yes_prob",
    "implied_btts_no_prob",
    "btts_market_margin",
    "market_btts_yes_prob_norm",
    "market_btts_no_prob_norm",
]
ODDS_FEATURE_COLUMNS = ODDS_RAW_COLUMNS + ODDS_DERIVED_COLUMNS

TEAM_STATS_COLUMNS = [
    "rank",
    "played",
    "points",
    "wins",
    "draws",
    "losses",
    "goals_for",
    "goals_against",
    "goal_difference",
    "btts_rate_so_far",
    "scored_rate_so_far",
    "conceded_rate_so_far",
    "avg_goals_for_so_far",
    "avg_goals_against_so_far",
]
FORM_COLUMNS = [
    "form_points_last_5",
    "form_wins_last_5",
    "form_draws_last_5",
    "form_losses_last_5",
]
SEASON_FEATURE_COLUMNS = [
    "season_matches_so_far",
    "season_btts_rate_so_far",
    "season_avg_goals_per_match_so_far",
    "season_home_win_rate_so_far",
    "season_draw_rate_so_far",
    "season_away_win_rate_so_far",
]
OUTPUT_COLUMNS = [
    "season_id",
    "target_matchday",
    "feature_cutoff_matchday",
    "home_team",
    "away_team",
    "btts_yes_probability",
    "prediction",
    "confidence",
    "decision",
]
TRACKING_COLUMNS = ["season_id", "target_matchday", "feature_cutoff_matchday", "home_team", "away_team"]

FIXTURE_RE = re.compile(r"\b(?P<home>[A-Z]{2,4})\s*(?:-|\u2013)\s*(?P<away>[A-Z]{2,4})\b")
MATCHDAY_RE = re.compile(r"\bMatchday\s*:?\s*(?P<matchday>\d+)\b", re.IGNORECASE)
SEASON_RE = re.compile(r"\bSeason\s*#?\s*(?P<season_id>\d+)\b", re.IGNORECASE)


st.set_page_config(page_title="BTTS Predictor", layout="wide")
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


@st.cache_resource
def load_model_bundle() -> tuple[object, dict]:
    model = load(MODEL_PATH)
    payload = json.loads(COLUMNS_PATH.read_text(encoding="utf-8"))
    return model, payload


@st.cache_data
def load_results_csv(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    required = {
        "season_id",
        "matchday",
        "home_team",
        "away_team",
        "fulltime_home_goals",
        "fulltime_away_goals",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")

    df["season_id"] = df["season_id"].astype(str)
    df["home_team"] = clean_team_series(df["home_team"])
    df["away_team"] = clean_team_series(df["away_team"])
    for col in [
        "matchday",
        "halftime_home_goals",
        "halftime_away_goals",
        "fulltime_home_goals",
        "fulltime_away_goals",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["matchday", "fulltime_home_goals", "fulltime_away_goals"])
    df["matchday"] = df["matchday"].astype(int)
    df["fulltime_home_goals"] = df["fulltime_home_goals"].astype(int)
    df["fulltime_away_goals"] = df["fulltime_away_goals"].astype(int)
    df = df.drop_duplicates(["season_id", "matchday", "home_team", "away_team"], keep="last")
    df = df.sort_values(["season_id", "matchday", "home_team", "away_team"]).reset_index(drop=True)
    df["target_btts"] = ((df["fulltime_home_goals"] > 0) & (df["fulltime_away_goals"] > 0)).astype(int)
    return df


@st.cache_data
def load_odds(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame(columns=["matchday", "home_team", "away_team"] + ODDS_FEATURE_COLUMNS)

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    required = {"matchday", "home_team", "away_team"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")

    df["home_team"] = clean_team_series(df["home_team"])
    df["away_team"] = clean_team_series(df["away_team"])
    df["matchday"] = pd.to_numeric(df["matchday"], errors="coerce")
    df = df.dropna(subset=["matchday", "home_team", "away_team"])
    df["matchday"] = df["matchday"].astype(int)
    df = df.rename(
        columns={
            "odd_1": "odds_1",
            "odd_x": "odds_x",
            "odd_2": "odds_2",
            "btts_yes": "odds_btts_yes",
            "btts_no": "odds_btts_no",
        }
    )
    if "captured_at" in df.columns:
        df = df.sort_values("captured_at")
    df = df.drop_duplicates(["matchday", "home_team", "away_team"], keep="last")
    return add_implied_odds_features(df)


def clean_team_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().str.strip()


def add_implied_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ODDS_RAW_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["implied_home_prob"] = np.where(df["odds_1"] > 0, 1 / df["odds_1"], np.nan)
    df["implied_draw_prob"] = np.where(df["odds_x"] > 0, 1 / df["odds_x"], np.nan)
    df["implied_away_prob"] = np.where(df["odds_2"] > 0, 1 / df["odds_2"], np.nan)
    df["implied_btts_yes_prob"] = np.where(df["odds_btts_yes"] > 0, 1 / df["odds_btts_yes"], np.nan)
    df["implied_btts_no_prob"] = np.where(df["odds_btts_no"] > 0, 1 / df["odds_btts_no"], np.nan)
    btts_total = df["implied_btts_yes_prob"] + df["implied_btts_no_prob"]
    df["btts_market_margin"] = btts_total - 1
    df["market_btts_yes_prob_norm"] = df["implied_btts_yes_prob"] / btts_total
    df["market_btts_no_prob_norm"] = df["implied_btts_no_prob"] / btts_total
    return df


def parse_first_int(pattern: re.Pattern, text: str, group: str) -> int | None:
    match = pattern.search(text or "")
    return int(match.group(group)) if match else None


def ocr_is_available() -> bool:
    try:
        import pytesseract  # noqa: F401
    except Exception:
        return False
    return False


def parse_fixtures_and_odds(text: str, target_matchday: int) -> pd.DataFrame:
    rows = []
    lines = [line.strip().upper() for line in (text or "").splitlines() if line.strip()]
    for index, line in enumerate(lines):
        fixture_match = FIXTURE_RE.search(line)
        if not fixture_match:
            continue
        home = fixture_match.group("home")
        away = fixture_match.group("away")
        nearby = " ".join(lines[index : min(index + 5, len(lines))])
        decimals = [float(value) for value in re.findall(r"\b\d+(?:\.\d{1,3})\b", nearby)]
        row = {"matchday": int(target_matchday), "home_team": home, "away_team": away}
        if len(decimals) >= 3:
            row.update({"odds_1": decimals[0], "odds_x": decimals[1], "odds_2": decimals[2]})
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["matchday", "home_team", "away_team"] + ODDS_RAW_COLUMNS)
    df = pd.DataFrame(rows).drop_duplicates(["home_team", "away_team"], keep="last")
    return df


def parse_btts_odds_text(text: str) -> pd.DataFrame:
    rows = []
    lines = [line.strip().upper() for line in (text or "").splitlines() if line.strip()]
    for index, line in enumerate(lines):
        fixture_match = FIXTURE_RE.search(line)
        if not fixture_match:
            continue
        nearby = " ".join(lines[index : min(index + 5, len(lines))])
        decimals = [float(value) for value in re.findall(r"\b\d+(?:\.\d{1,3})\b", nearby)]
        row = {
            "home_team": fixture_match.group("home"),
            "away_team": fixture_match.group("away"),
        }
        if len(decimals) >= 2:
            row.update({"odds_btts_yes": decimals[0], "odds_btts_no": decimals[1]})
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["home_team", "away_team", "odds_btts_yes", "odds_btts_no"])
    return pd.DataFrame(rows).drop_duplicates(["home_team", "away_team"], keep="last")


def parse_standings_text(text: str) -> pd.DataFrame:
    rows = []
    line_re = re.compile(
        r"^\s*(?P<rank>\d{1,2})\s*(?P<team>[A-Z]{2,4})\s+"
        r"(?P<points>\d+)\s+(?P<wins>\d+)\s+(?P<draws>\d+)\s+(?P<losses>\d+)\s+"
        r"(?P<goals_for>\d+)\s*:\s*(?P<goals_against>\d+)"
    )
    for raw_line in (text or "").splitlines():
        line = raw_line.strip().upper()
        match = line_re.match(line)
        if not match:
            continue
        row = {key: match.group(key) for key in match.groupdict()}
        for key in ["rank", "points", "wins", "draws", "losses", "goals_for", "goals_against"]:
            row[key] = int(row[key])
        row["played"] = row["wins"] + row["draws"] + row["losses"]
        row["goal_difference"] = row["goals_for"] - row["goals_against"]
        row["avg_goals_for_so_far"] = row["goals_for"] / row["played"] if row["played"] else np.nan
        row["avg_goals_against_so_far"] = row["goals_against"] / row["played"] if row["played"] else np.nan
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["team"] + TEAM_STATS_COLUMNS)
    return pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)


def merge_fixture_odds(fixtures: pd.DataFrame, odds_file: pd.DataFrame, btts_text_odds: pd.DataFrame) -> pd.DataFrame:
    fixtures = fixtures.copy()
    if fixtures.empty:
        return fixtures

    for col in ODDS_RAW_COLUMNS:
        if col not in fixtures.columns:
            fixtures[col] = np.nan

    if not odds_file.empty:
        latest_by_fixture = odds_file.sort_values("matchday").drop_duplicates(["home_team", "away_team"], keep="last")
        fixture_keys = ["home_team", "away_team"]
        file_cols = fixture_keys + [col for col in ODDS_RAW_COLUMNS if col in latest_by_fixture.columns]
        fixtures = fixtures.merge(
            latest_by_fixture[file_cols],
            on=fixture_keys,
            how="left",
            suffixes=("", "_file"),
        )
        for col in ODDS_RAW_COLUMNS:
            file_col = f"{col}_file"
            if file_col in fixtures.columns:
                fixtures[col] = fixtures[col].combine_first(fixtures[file_col])
                fixtures = fixtures.drop(columns=[file_col])

    if not btts_text_odds.empty:
        fixtures = fixtures.merge(btts_text_odds, on=["home_team", "away_team"], how="left", suffixes=("", "_manual"))
        for col in ["odds_btts_yes", "odds_btts_no"]:
            manual_col = f"{col}_manual"
            if manual_col in fixtures.columns:
                fixtures[col] = fixtures[manual_col].combine_first(fixtures[col])
                fixtures = fixtures.drop(columns=[manual_col])

    return add_implied_odds_features(fixtures)


def make_team_match_rows(history: pd.DataFrame) -> pd.DataFrame:
    home = history[
        ["matchday", "home_team", "away_team", "fulltime_home_goals", "fulltime_away_goals", "target_btts"]
    ].copy()
    home.columns = ["matchday", "team", "opponent", "goals_for", "goals_against", "btts"]
    away = history[
        ["matchday", "away_team", "home_team", "fulltime_away_goals", "fulltime_home_goals", "target_btts"]
    ].copy()
    away.columns = ["matchday", "team", "opponent", "goals_for", "goals_against", "btts"]
    team_rows = pd.concat([home, away], ignore_index=True)
    team_rows["points"] = np.select(
        [team_rows["goals_for"] > team_rows["goals_against"], team_rows["goals_for"] == team_rows["goals_against"]],
        [3, 1],
        default=0,
    )
    team_rows["win"] = (team_rows["points"] == 3).astype(int)
    team_rows["draw"] = (team_rows["points"] == 1).astype(int)
    team_rows["loss"] = (team_rows["points"] == 0).astype(int)
    team_rows["scored"] = (team_rows["goals_for"] > 0).astype(int)
    team_rows["conceded"] = (team_rows["goals_against"] > 0).astype(int)
    return team_rows


def build_team_stats(history: pd.DataFrame) -> pd.DataFrame:
    team_rows = make_team_match_rows(history)
    if team_rows.empty:
        return pd.DataFrame(columns=["team"] + TEAM_STATS_COLUMNS).set_index("team")
    stats = team_rows.groupby("team").agg(
        played=("team", "size"),
        points=("points", "sum"),
        wins=("win", "sum"),
        draws=("draw", "sum"),
        losses=("loss", "sum"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
        btts_rate_so_far=("btts", "mean"),
        scored_rate_so_far=("scored", "mean"),
        conceded_rate_so_far=("conceded", "mean"),
        avg_goals_for_so_far=("goals_for", "mean"),
        avg_goals_against_so_far=("goals_against", "mean"),
    ).reset_index()
    stats["goal_difference"] = stats["goals_for"] - stats["goals_against"]
    stats = stats.sort_values(["points", "goal_difference", "goals_for", "team"], ascending=[False, False, False, True])
    stats["rank"] = np.arange(1, len(stats) + 1)
    return stats.set_index("team")


def build_form_stats(history: pd.DataFrame) -> pd.DataFrame:
    team_rows = make_team_match_rows(history).sort_values(["team", "matchday"])
    if team_rows.empty:
        return pd.DataFrame(columns=["team"] + FORM_COLUMNS).set_index("team")
    last5 = team_rows.groupby("team", group_keys=False).tail(5)
    return last5.groupby("team").agg(
        form_points_last_5=("points", "sum"),
        form_wins_last_5=("win", "sum"),
        form_draws_last_5=("draw", "sum"),
        form_losses_last_5=("loss", "sum"),
    )


def build_season_stats(history: pd.DataFrame) -> dict:
    if history.empty:
        return {col: np.nan for col in SEASON_FEATURE_COLUMNS}
    home_win = history["fulltime_home_goals"] > history["fulltime_away_goals"]
    draw = history["fulltime_home_goals"] == history["fulltime_away_goals"]
    away_win = history["fulltime_home_goals"] < history["fulltime_away_goals"]
    total_goals = history["fulltime_home_goals"] + history["fulltime_away_goals"]
    return {
        "season_matches_so_far": len(history),
        "season_btts_rate_so_far": history["target_btts"].mean(),
        "season_avg_goals_per_match_so_far": total_goals.mean(),
        "season_home_win_rate_so_far": home_win.mean(),
        "season_draw_rate_so_far": draw.mean(),
        "season_away_win_rate_so_far": away_win.mean(),
    }


def make_stats_bundle(results_df: pd.DataFrame, season_id: str, cutoff_matchday: int) -> dict:
    history = results_df[
        (results_df["season_id"] == str(season_id)) & (results_df["matchday"] <= cutoff_matchday)
    ].copy()
    return {
        "history": history,
        "team_stats": build_team_stats(history),
        "form_stats": build_form_stats(history),
        "season_stats": build_season_stats(history),
    }


def build_standings_bundle(standings: pd.DataFrame) -> dict:
    team_stats = pd.DataFrame(columns=["team"] + TEAM_STATS_COLUMNS).set_index("team")
    if not standings.empty:
        team_stats = standings.set_index("team").copy()
        for col in TEAM_STATS_COLUMNS:
            if col not in team_stats.columns:
                team_stats[col] = np.nan
        team_stats["btts_rate_so_far"] = np.nan
        team_stats["scored_rate_so_far"] = np.nan
        team_stats["conceded_rate_so_far"] = np.nan

    matches = float(standings["played"].sum() / 2) if not standings.empty else np.nan
    total_goals = float(standings["goals_for"].sum()) if not standings.empty else np.nan
    draws = float(standings["draws"].sum() / 2) if not standings.empty else np.nan
    season_stats = {
        "season_matches_so_far": matches,
        "season_btts_rate_so_far": np.nan,
        "season_avg_goals_per_match_so_far": total_goals / matches if matches else np.nan,
        "season_home_win_rate_so_far": np.nan,
        "season_draw_rate_so_far": draws / matches if matches else np.nan,
        "season_away_win_rate_so_far": np.nan,
    }
    form_stats = pd.DataFrame(columns=["team"] + FORM_COLUMNS).set_index("team")
    return {"history": pd.DataFrame(), "team_stats": team_stats, "form_stats": form_stats, "season_stats": season_stats}


def get_team_values(table: pd.DataFrame, team: str, columns: list[str], prefix: str) -> dict:
    if team in table.index:
        return {f"{prefix}_{col}": table.loc[team, col] for col in columns}
    return {f"{prefix}_{col}": np.nan for col in columns}


def add_comparison_features(record: dict) -> dict:
    record["points_diff"] = record["home_points"] - record["away_points"]
    record["rank_diff"] = record["away_rank"] - record["home_rank"]
    record["goal_difference_diff"] = record["home_goal_difference"] - record["away_goal_difference"]
    record["goals_for_diff"] = record["home_goals_for"] - record["away_goals_for"]
    record["goals_against_diff"] = record["home_goals_against"] - record["away_goals_against"]
    record["form_diff"] = record["home_form_points_last_5"] - record["away_form_points_last_5"]
    record["btts_rate_diff"] = record["home_btts_rate_so_far"] - record["away_btts_rate_so_far"]
    record["scored_rate_diff"] = record["home_scored_rate_so_far"] - record["away_scored_rate_so_far"]
    record["conceded_rate_diff"] = record["home_conceded_rate_so_far"] - record["away_conceded_rate_so_far"]
    return record


def build_base_feature_record(match: pd.Series, bundle: dict, live_gap_matchdays: int) -> dict:
    record = {
        "season_id": str(match["season_id"]),
        "target_matchday": int(match["matchday"]),
        "feature_cutoff_matchday": int(match["matchday"]) - live_gap_matchdays,
        "home_team": match["home_team"],
        "away_team": match["away_team"],
    }
    record.update(bundle["season_stats"])
    record.update(get_team_values(bundle["team_stats"], match["home_team"], TEAM_STATS_COLUMNS, "home"))
    record.update(get_team_values(bundle["team_stats"], match["away_team"], TEAM_STATS_COLUMNS, "away"))
    record.update(get_team_values(bundle["form_stats"], match["home_team"], FORM_COLUMNS, "home"))
    record.update(get_team_values(bundle["form_stats"], match["away_team"], FORM_COLUMNS, "away"))
    return add_comparison_features(record)


def build_prediction_features(
    fixture_odds: pd.DataFrame,
    season_id: str,
    feature_cutoff_matchday: int,
    live_gap_matchdays: int,
    bundle: dict,
) -> pd.DataFrame:
    records = []
    for _, fixture in fixture_odds.iterrows():
        match = pd.Series(
            {
                "season_id": season_id,
                "matchday": int(fixture["matchday"]),
                "home_team": fixture["home_team"],
                "away_team": fixture["away_team"],
            }
        )
        record = build_base_feature_record(match, bundle, live_gap_matchdays)
        record["feature_cutoff_matchday"] = feature_cutoff_matchday
        for col in ODDS_FEATURE_COLUMNS:
            record[col] = fixture[col] if col in fixture.index else np.nan
        records.append(record)
    return pd.DataFrame(records)


def find_live_results_file(season_id: str, target_matchday: int) -> Path | None:
    exact = LIVE_RESULTS_DIR / f"season_{season_id}_tgt_md{target_matchday}_results.csv"
    if exact.exists():
        return exact
    candidates = sorted(LIVE_RESULTS_DIR.glob(f"season_{season_id}_tgt_md*_results.csv"))
    return candidates[-1] if candidates else None


def selected_odds_for_matchday(odds: pd.DataFrame, target_matchday: int) -> pd.DataFrame:
    rows = odds[odds["matchday"] == int(target_matchday)].copy()
    if rows.empty:
        return rows
    return rows.drop_duplicates(["matchday", "home_team", "away_team"], keep="last")


def make_predictions(model: object, live_df: pd.DataFrame, feature_columns: list[str], pick_threshold: float) -> pd.DataFrame:
    live_df = live_df.copy()
    for col in feature_columns:
        if col not in live_df.columns:
            live_df[col] = np.nan
    probabilities = model.predict_proba(live_df[feature_columns])[:, 1]
    predictions = live_df[TRACKING_COLUMNS].copy()
    predictions["btts_yes_probability"] = probabilities
    predictions["prediction"] = np.where(predictions["btts_yes_probability"] >= 0.5, "Yes", "No")
    predictions["confidence"] = np.maximum(predictions["btts_yes_probability"], 1 - predictions["btts_yes_probability"])
    predictions["decision"] = np.where(predictions["confidence"] >= pick_threshold, "Pick", "Skip")
    return predictions[OUTPUT_COLUMNS].sort_values(["decision", "confidence"], ascending=[True, False]).reset_index(drop=True)


def missing_feature_table(live_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    report = []
    for col in feature_columns:
        if col not in live_df.columns:
            report.append({"feature": col, "missing_rows": len(live_df), "status": "not_created"})
            continue
        missing_rows = int(live_df[col].isna().sum())
        if missing_rows:
            status = "missing_all_rows" if missing_rows == len(live_df) else "missing_some_rows"
            report.append({"feature": col, "missing_rows": missing_rows, "status": status})
    return pd.DataFrame(report)


def format_predictions(predictions: pd.DataFrame) -> pd.io.formats.style.Styler:
    table = predictions.copy()
    table["btts_yes_probability"] = table["btts_yes_probability"].map(lambda value: f"{value:.1%}")
    table["confidence"] = table["confidence"].map(lambda value: f"{value:.1%}")
    return table.style.applymap(
        lambda value: "font-weight: 700; color: #0f7b0f" if value == "Pick" else "color: #777",
        subset=["decision"],
    )


def show_capture_result(result) -> None:
    st.session_state["live_results_path"] = str(result.output_path)
    st.success("Live results updated.")
    st.write(
        {
            "season_id": result.season_id,
            "target_matchday": result.target_matchday,
            "cutoff_matchday": result.feature_cutoff_matchday,
            "reused_matchdays": result.reused_matchdays,
            "fetched_matchdays": result.fetched_matchdays,
            "rows_written": result.rows_written,
            "output_path": str(result.output_path),
        }
    )
    if result.incomplete_matchdays:
        st.warning(f"Incomplete matchdays remain: {result.incomplete_matchdays}")


def main() -> None:
    st.title("BTTS Predictor")
    st.caption("Saved model only. Capture or provide prediction-time inputs, then press Predict BTTS.")

    try:
        model, columns_payload = load_model_bundle()
        odds = load_odds(str(ODDS_PATH))
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    feature_columns = list(columns_payload.get("feature_columns", []))
    live_gap_matchdays = int(columns_payload.get("live_gap_matchdays", DEFAULT_LIVE_GAP_MATCHDAYS))
    pick_threshold = float(columns_payload.get("pick_threshold", DEFAULT_PICK_THRESHOLD))

    with st.sidebar:
        st.header("Prediction Setup")
        mode = st.radio(
            "Input mode",
            ["Automated collectors", "Manual screenshot + standings"],
            help="Manual mode uses what you paste/upload. Automated mode calls the existing Playwright collectors.",
        )
        season_id = ""
        target_matchday = 0
        feature_cutoff_matchday = 0
        if mode == "Automated collectors":
            season_id = st.text_input("season_id", value=st.session_state.get("season_id", "138596")).strip()
            target_matchday = st.number_input(
                "target_matchday",
                min_value=1,
                max_value=38,
                value=int(st.session_state.get("target_matchday", 19)),
            )
            target_matchday = int(target_matchday)
            feature_cutoff_matchday = target_matchday - live_gap_matchdays
            st.metric("feature_cutoff_matchday", feature_cutoff_matchday)
        else:
            st.caption("Season and matchday are detected from the pasted standings/header text.")
        st.metric("pick_threshold", f"{pick_threshold:.0%}")
        st.caption(f"Saved model features: {len(feature_columns)}")

    if mode == "Automated collectors":
        st.subheader("Automated Updates")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Update live results", use_container_width=True):
                if not load_settings or not collect_live_results:
                    st.error(f"Live results collector is not importable. {COLLECTOR_IMPORT_ERROR}")
                else:
                    with st.spinner("Collecting completed match results up to cutoff..."):
                        try:
                            result = collect_live_results(
                                load_settings(),
                                season_id=season_id,
                                target_matchday=target_matchday,
                                live_gap_matchdays=live_gap_matchdays,
                                keep_open=False,
                            )
                            load_results_csv.clear()
                            show_capture_result(result)
                        except Exception as exc:
                            st.error(f"Live results capture failed: {exc}")
        with col2:
            if st.button("Update odds / fixtures", use_container_width=True):
                if not load_settings or not collect_current_odds:
                    st.error(f"Odds collector is not importable. {COLLECTOR_IMPORT_ERROR}")
                else:
                    with st.spinner("Collecting upcoming fixtures and odds..."):
                        try:
                            result = collect_current_odds(load_settings(), keep_open=False)
                            load_odds.clear()
                            odds = load_odds(str(ODDS_PATH))
                            st.success("Odds updated.")
                            st.write(
                                {
                                    "rows": result.rows,
                                    "markets_seen": result.markets_seen,
                                    "final_path": str(result.final_path) if result.final_path else "",
                                }
                            )
                        except Exception as exc:
                            st.error(f"Odds capture failed: {exc}")

        live_path = Path(st.session_state.get("live_results_path", "")) if st.session_state.get("live_results_path") else find_live_results_file(season_id, target_matchday)
        fixture_odds = selected_odds_for_matchday(odds, target_matchday)
        st.info(f"Results source: {live_path if live_path else 'run Update live results first'}")
        st.info(f"Fixture/odds rows for target matchday: {len(fixture_odds)}")

        if st.button("Predict BTTS", type="primary"):
            if not live_path or not live_path.exists():
                st.error("No live results CSV found. Run Update live results first.")
                st.stop()
            if fixture_odds.empty:
                st.error("No target fixtures/odds found. Run Update odds / fixtures first.")
                st.stop()

            results = load_results_csv(str(live_path))
            bundle = make_stats_bundle(results, season_id, feature_cutoff_matchday)
            live_df = build_prediction_features(fixture_odds, season_id, feature_cutoff_matchday, live_gap_matchdays, bundle)
            show_prediction_output(model, live_df, feature_columns, pick_threshold, target_matchday)

    else:
        st.subheader("Manual Screenshot + Standings Input")
        screenshots = st.file_uploader(
            "Upload fixture screenshot(s)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Upload the two fixture screenshots. This app will show them for reference; OCR extraction needs a local OCR engine.",
        )
        if screenshots:
            cols = st.columns(min(len(screenshots), 3))
            for idx, upload in enumerate(screenshots):
                cols[idx % len(cols)].image(upload, caption=upload.name, use_container_width=True)
            if not ocr_is_available():
                st.warning(
                    "Screenshot OCR is not available in this Python environment. "
                    "Use the automated odds/fixture collector, or paste the fixture lines below."
                )

        fixture_text = st.text_area(
            "Fixture / 1X2 odds text if OCR is unavailable",
            height=150,
            placeholder="Example:\nARS - BOU\n1 1.78\nX 3.60\n2 4.85\n\nCHE - AST\n1 2.00\nX 4.20\n2 3.25",
        )
        btts_text = st.text_area(
            "Optional BTTS odds text",
            height=120,
            placeholder="Example:\nARS - BOU\nYes 1.60\nNo 2.30",
        )
        standings_text = st.text_area(
            "Standings table text",
            height=260,
            placeholder="Paste the Season # line and the full Club Pts W D L Goals table here.",
        )

        detected_season = parse_first_int(SEASON_RE, standings_text, "season_id")
        detected_matchday = parse_first_int(MATCHDAY_RE, standings_text + "\n" + fixture_text, "matchday")
        active_season_id = str(detected_season) if detected_season else ""
        active_target_matchday = int(detected_matchday) if detected_matchday else None
        active_cutoff = active_target_matchday - live_gap_matchdays if active_target_matchday else None

        standings = parse_standings_text(standings_text)
        fixture_rows = parse_fixtures_and_odds(fixture_text, active_target_matchday or 0)
        btts_rows = parse_btts_odds_text(btts_text)
        fixture_odds = merge_fixture_odds(fixture_rows, odds, btts_rows)

        col1, col2, col3 = st.columns(3)
        col1.metric("season_id", active_season_id or "-")
        col2.metric("target_matchday", active_target_matchday if active_target_matchday else "-")
        col3.metric("feature_cutoff_matchday", active_cutoff if active_cutoff is not None else "-")

        with st.expander("Parsed inputs", expanded=False):
            st.write("Fixtures")
            st.dataframe(fixture_odds[["home_team", "away_team"] + [c for c in ODDS_RAW_COLUMNS if c in fixture_odds.columns]], use_container_width=True, hide_index=True)
            st.write("Standings")
            st.dataframe(standings, use_container_width=True, hide_index=True)

        if st.button("Predict BTTS", type="primary"):
            if not active_season_id:
                st.error("No season_id was detected. Paste standings text containing a line like Season #138605.")
                st.stop()
            if not active_target_matchday:
                st.error("No target matchday was detected. Paste text containing a line like Matchday 19.")
                st.stop()
            if fixture_odds.empty:
                st.error(
                    "No fixtures were found. Screenshot OCR is not available here, so use automated odds/fixture "
                    "capture or paste fixture lines like ARS - MCI."
                )
                st.stop()
            if standings.empty:
                st.error("No standings rows were found. Paste the full Club Pts W D L Goals table before predicting.")
                st.stop()

            bundle = build_standings_bundle(standings)
            live_df = build_prediction_features(fixture_odds, active_season_id, active_cutoff, live_gap_matchdays, bundle)
            show_prediction_output(model, live_df, feature_columns, pick_threshold, active_target_matchday)


def show_prediction_output(
    model: object,
    live_df: pd.DataFrame,
    feature_columns: list[str],
    pick_threshold: float,
    target_matchday: int,
) -> None:
    missing = missing_feature_table(live_df, feature_columns)
    with st.expander("Missing Prediction Fields", expanded=not missing.empty):
        if missing.empty:
            st.success("All saved model features are present.")
        else:
            st.warning("The model will impute the missing numeric fields below.")
            st.dataframe(missing, use_container_width=True, hide_index=True)

    try:
        predictions = make_predictions(model, live_df, feature_columns, pick_threshold)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    st.subheader("BTTS Predictions")
    st.dataframe(format_predictions(predictions), use_container_width=True, hide_index=True)
    st.download_button(
        "Download predictions CSV",
        data=predictions.to_csv(index=False).encode("utf-8"),
        file_name=f"btts_predictions_md{target_matchday}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
