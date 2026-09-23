from __future__ import annotations

import json
import os
import re
import queue
import subprocess
import threading
import time
import sys
import traceback
import warnings
from dataclasses import replace
from datetime import datetime
from io import BytesIO
from pathlib import Path
from tempfile import gettempdir
from types import SimpleNamespace

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
ODDS_PATH = PROJECT_ROOT / "data" / "odds" / "final_odds" / "final_odds_master.csv"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "outputs"
LIVE_PREDICTIONS_PATH = PREDICTIONS_DIR / "btts_predictions_live_master.csv"
LIVE_RESULTS_DIR = PROJECT_ROOT / "data" / "live_seasons"
SOUND_DIR = PROJECT_ROOT / "data" / "sounds"
THRESHOLD_BELL_PATH = SOUND_DIR / "mixkit-bell-notification-933.wav"
PREDICTION_DONE_POP_PATH = SOUND_DIR / "mixkit-long-pop-2358.wav"
LIVE_RESULTS_FILE_RE = re.compile(r"^season_(?P<season_id>\d+)_tgt_md(?P<target_matchday>\d+)_results\.csv$")
STREAMLIT_BROWSER_PROFILE_DIR = Path(gettempdir()) / "btts-streamlit-browser-profile"
TESSERACT_EXE_PATHS = [
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
]

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
LIVE_RESULTS_REQUIRED_FEATURES = [
    "season_btts_rate_so_far",
    "season_home_win_rate_so_far",
    "season_away_win_rate_so_far",
    "home_btts_rate_so_far",
    "home_scored_rate_so_far",
    "home_conceded_rate_so_far",
    "away_btts_rate_so_far",
    "away_scored_rate_so_far",
    "away_conceded_rate_so_far",
    "home_form_points_last_5",
    "home_form_wins_last_5",
    "home_form_draws_last_5",
    "home_form_losses_last_5",
    "away_form_points_last_5",
    "away_form_wins_last_5",
    "away_form_draws_last_5",
    "away_form_losses_last_5",
    "form_diff",
    "btts_rate_diff",
    "scored_rate_diff",
    "conceded_rate_diff",
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
        import pytesseract
    except Exception:
        return False

    configured_cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "")
    if configured_cmd and Path(configured_cmd).exists():
        return True
    for exe_path in TESSERACT_EXE_PATHS:
        if exe_path.exists():
            pytesseract.pytesseract.tesseract_cmd = str(exe_path)
            return True
    return False


def extract_text_from_image_upload(upload) -> str:
    if not ocr_is_available():
        return ""
    try:
        from PIL import Image
        import pytesseract
    except Exception:
        return ""

    image = Image.open(BytesIO(upload.getvalue()))
    return pytesseract.image_to_string(image)


def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    if not ocr_is_available():
        return ""
    try:
        from PIL import Image
        import pytesseract
    except Exception:
        return ""

    image = Image.open(BytesIO(image_bytes))
    return pytesseract.image_to_string(image)


def read_clipboard_image_png() -> tuple[bytes | None, str]:
    try:
        from PIL import Image, ImageGrab
    except Exception as exc:
        return None, f"Clipboard image support is unavailable: {exc}"

    clipboard = ImageGrab.grabclipboard()
    image = None
    if isinstance(clipboard, Image.Image):
        image = clipboard
    elif isinstance(clipboard, list) and clipboard:
        for item in clipboard:
            try:
                image = Image.open(item)
                break
            except Exception:
                continue

    if image is None:
        return None, "No image found on the Windows clipboard. Copy a screenshot first, then click this button."

    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue(), ""


def clean_ocr_odd(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        if "." in value:
            return float(value)
        if len(value) == 3:
            return float(value) / 100
        return float(value)
    except ValueError:
        return None


def extract_1x2_odds(text: str) -> tuple[float | None, float | None, float | None]:
    normalized = re.sub(r"[^\w.\s]", " ", text.upper())
    normalized = re.sub(r"\s+", " ", normalized)
    match = re.search(
        r"\b1\s+(?P<home>\d+(?:\.\d+)?|\d{3})\s+X\s+"
        r"(?P<draw>\d+(?:\.\d+)?|\d{3})\s+2\s+"
        r"(?P<away>\d+(?:\.\d+)?|\d{3})\b",
        normalized,
    )
    if not match:
        return None, None, None
    return (
        clean_ocr_odd(match.group("home")),
        clean_ocr_odd(match.group("draw")),
        clean_ocr_odd(match.group("away")),
    )


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
        row = {"matchday": int(target_matchday), "home_team": home, "away_team": away}
        odds_1, odds_x, odds_2 = extract_1x2_odds(nearby)
        if all(value is not None for value in [odds_1, odds_x, odds_2]):
            row.update({"odds_1": odds_1, "odds_x": odds_x, "odds_2": odds_2})
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


def find_exact_live_results_file(season_id: str, target_matchday: int) -> Path | None:
    exact = LIVE_RESULTS_DIR / f"season_{season_id}_tgt_md{target_matchday}_results.csv"
    return exact if exact.exists() else None


def session_live_results_file(season_id: str, target_matchday: int) -> Path | None:
    raw_path = st.session_state.get("live_results_path")
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_file():
        return None
    match = LIVE_RESULTS_FILE_RE.match(path.name)
    if (
        match
        and match.group("season_id") == str(season_id)
        and int(match.group("target_matchday")) == int(target_matchday)
    ):
        return path
    return None


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
    return table.style.map(
        lambda value: "font-weight: 700; color: #0f7b0f" if value == "Pick" else "color: #777",
        subset=["decision"],
    )


def play_prediction_alert(predictions: pd.DataFrame) -> None:
    has_threshold_pick = bool(predictions["decision"].eq("Pick").any())
    sound_path = THRESHOLD_BELL_PATH if has_threshold_pick else PREDICTION_DONE_POP_PATH
    if sound_path.is_file():
        st.audio(sound_path.read_bytes(), format="audio/wav", autoplay=True)


def save_predictions_live_file(predictions: pd.DataFrame) -> Path:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    rows = predictions.copy()
    rows.insert(0, "prediction_timestamp", datetime.now().astimezone().isoformat(timespec="seconds"))
    write_header = not LIVE_PREDICTIONS_PATH.exists()
    rows.to_csv(LIVE_PREDICTIONS_PATH, mode="a", header=write_header, index=False)
    try:
        load_saved_predictions.clear()
    except Exception:
        pass
    return LIVE_PREDICTIONS_PATH


@st.cache_data
def load_saved_predictions(predictions_dir_text: str) -> pd.DataFrame:
    predictions_dir = Path(predictions_dir_text)
    live_file = predictions_dir / LIVE_PREDICTIONS_PATH.name
    files = [live_file] if live_file.exists() else []
    frames = []
    required = {"season_id", "target_matchday", "home_team", "away_team", "prediction"}

    for path in files:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")
        df["prediction_file"] = path.name
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=sorted(required) + ["prediction_file", "prediction_timestamp"])

    predictions = pd.concat(frames, ignore_index=True)
    predictions["season_id"] = predictions["season_id"].astype(str)
    predictions["target_matchday"] = pd.to_numeric(predictions["target_matchday"], errors="coerce")
    predictions["home_team"] = clean_team_series(predictions["home_team"])
    predictions["away_team"] = clean_team_series(predictions["away_team"])
    predictions["prediction"] = clean_team_series(predictions["prediction"]).map({"YES": "Yes", "NO": "No"}).fillna(
        predictions["prediction"]
    )
    if "decision" not in predictions.columns:
        predictions["decision"] = "Pick"
    predictions["decision"] = predictions["decision"].fillna("Pick")
    if "prediction_timestamp" not in predictions.columns:
        predictions["prediction_timestamp"] = ""

    for col in ["btts_yes_probability", "confidence"]:
        if col not in predictions.columns:
            predictions[col] = np.nan
        predictions[col] = pd.to_numeric(predictions[col], errors="coerce")

    predictions = predictions.dropna(subset=["target_matchday", "home_team", "away_team"])
    predictions["target_matchday"] = predictions["target_matchday"].astype(int)
    predictions = predictions.drop_duplicates(
        ["season_id", "target_matchday", "home_team", "away_team"],
        keep="last",
    )
    return predictions.reset_index(drop=True)


@st.cache_data
def load_live_season_actuals(live_results_dir_text: str) -> pd.DataFrame:
    live_results_dir = Path(live_results_dir_text)
    files = sorted(live_results_dir.glob("season_*_tgt_md*_results.csv"))
    frames = []

    for path in files:
        match = LIVE_RESULTS_FILE_RE.match(path.name)
        if not match:
            continue
        df = load_results_csv(str(path))
        df["actuals_file"] = path.name
        df["actuals_source_target_matchday"] = int(match.group("target_matchday"))
        frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=[
                "season_id",
                "matchday",
                "home_team",
                "away_team",
                "fulltime_home_goals",
                "fulltime_away_goals",
                "target_btts",
                "actuals_file",
                "actuals_source_target_matchday",
            ]
        )

    actuals = pd.concat(frames, ignore_index=True)
    actuals = actuals.sort_values(["season_id", "matchday", "actuals_source_target_matchday"])
    actuals = actuals.drop_duplicates(["season_id", "matchday", "home_team", "away_team"], keep="last")
    return actuals.reset_index(drop=True)


def evaluate_predictions(predictions: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()

    result_cols = [
        "season_id",
        "matchday",
        "home_team",
        "away_team",
        "fulltime_home_goals",
        "fulltime_away_goals",
        "target_btts",
    ]
    optional_result_cols = ["actuals_file", "actuals_source_target_matchday"]
    result_cols += [col for col in optional_result_cols if col in actuals.columns]
    actuals_for_join = actuals[result_cols].rename(columns={"matchday": "target_matchday"})
    evaluated = predictions.merge(
        actuals_for_join,
        on=["season_id", "target_matchday", "home_team", "away_team"],
        how="left",
        validate="many_to_one",
    )
    evaluated["actual"] = np.where(evaluated["target_btts"] == 1, "Yes", "No")
    evaluated.loc[evaluated["target_btts"].isna(), "actual"] = np.nan
    evaluated["correct"] = evaluated["prediction"].eq(evaluated["actual"]).astype("object")
    evaluated.loc[evaluated["actual"].isna(), "correct"] = np.nan
    evaluated["picked"] = evaluated["decision"].astype(str).str.lower().eq("pick")
    return evaluated.sort_values(
        ["target_matchday", "decision", "confidence"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def _pct(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.1%}"


def summarize_evaluation(evaluated: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    completed = evaluated[evaluated["actual"].notna()].copy()
    if completed.empty:
        return pd.DataFrame()

    rows = []
    grouped = completed.groupby(group_cols, dropna=False) if group_cols else [((), completed)]
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        picked = group[group["picked"]]
        correct = int(group["correct"].eq(True).sum())
        missed = int(group["correct"].eq(False).sum())
        pick_correct = int(picked["correct"].eq(True).sum()) if not picked.empty else 0
        pick_missed = int(picked["correct"].eq(False).sum()) if not picked.empty else 0
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(
            {
                "predictions": len(group),
                "correct": correct,
                "missed": missed,
                "accuracy": correct / len(group) if len(group) else np.nan,
                "picks": len(picked),
                "pick_correct": pick_correct,
                "pick_missed": pick_missed,
                "pick_accuracy": pick_correct / len(picked) if not picked.empty else np.nan,
                "avg_confidence": group["confidence"].mean(),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def format_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary.copy()
    for col in ["accuracy", "pick_accuracy", "avg_confidence"]:
        if col in table.columns:
            table[col] = table[col].map(_pct)
    return table


def chart_percent_table(summary: pd.DataFrame, label_col: str, value_cols: list[str]) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    chart = summary[[label_col] + value_cols].copy()
    chart[label_col] = chart[label_col].astype(str)
    for col in value_cols:
        chart[col] = chart[col] * 100
    return chart.set_index(label_col)


def confidence_bucket_summary(evaluated: pd.DataFrame) -> pd.DataFrame:
    completed = evaluated[evaluated["actual"].notna() & evaluated["confidence"].notna()].copy()
    if completed.empty:
        return pd.DataFrame()
    completed["confidence_bucket"] = pd.cut(
        completed["confidence"],
        bins=[0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0],
        labels=["<=50", "50-55", "55-60", "60-65", "65-70", "70-80", "80-90", "90-100"],
        include_lowest=True,
    )
    return summarize_evaluation(completed, ["confidence_bucket"])


def show_evaluation_visuals(evaluated: pd.DataFrame, by_matchday: pd.DataFrame, by_side: pd.DataFrame) -> None:
    completed = evaluated[evaluated["actual"].notna()].copy()
    if completed.empty:
        st.info("No completed predictions yet, so the charts will appear after matching actual results are saved.")
        return

    left, right = st.columns(2)
    with left:
        st.markdown("#### Accuracy trend")
        trend = chart_percent_table(by_matchday, "target_matchday", ["accuracy", "pick_accuracy"])
        st.line_chart(trend)
    with right:
        st.markdown("#### Correct vs missed")
        counts = by_matchday[["target_matchday", "correct", "missed"]].copy()
        counts["target_matchday"] = counts["target_matchday"].astype(str)
        st.bar_chart(counts.set_index("target_matchday"))

    left, right = st.columns(2)
    with left:
        st.markdown("#### BTTS Yes vs No")
        side_chart = chart_percent_table(by_side, "prediction", ["accuracy", "pick_accuracy"])
        st.bar_chart(side_chart)
    with right:
        st.markdown("#### Confidence buckets")
        buckets = confidence_bucket_summary(evaluated)
        if buckets.empty:
            st.caption("No confidence values available yet.")
        else:
            bucket_chart = chart_percent_table(buckets, "confidence_bucket", ["accuracy", "pick_accuracy"])
            st.bar_chart(bucket_chart)


def show_evaluation_page() -> None:
    st.subheader("Prediction Evaluation")
    st.caption(
        "Compares saved predictions in data/outputs with completed actual results from data/live_seasons. "
        "This measures model understanding, not betting ROI."
    )

    include_skips = st.checkbox("Show skipped rows in detail table", value=True)

    try:
        predictions = load_saved_predictions(str(PREDICTIONS_DIR))
    except Exception as exc:
        st.error(f"Could not load saved predictions: {exc}")
        return

    if predictions.empty:
        st.warning(f"No saved prediction files found in {PREDICTIONS_DIR}.")
        return

    try:
        actuals = load_live_season_actuals(str(LIVE_RESULTS_DIR))
        evaluated = evaluate_predictions(predictions, actuals)
    except Exception as exc:
        st.error(f"Could not evaluate predictions: {exc}")
        return

    completed = evaluated[evaluated["actual"].notna()]
    pending = evaluated[evaluated["actual"].isna()]
    picked = completed[completed["picked"]]
    overall_accuracy = completed["correct"].eq(True).mean() if not completed.empty else np.nan
    pick_accuracy = picked["correct"].eq(True).mean() if not picked.empty else np.nan

    metric_cols = st.columns(5)
    metric_cols[0].metric("Saved predictions", len(predictions))
    metric_cols[1].metric("Completed", len(completed))
    metric_cols[2].metric("Overall accuracy", _pct(overall_accuracy))
    metric_cols[3].metric("Pick accuracy", _pct(pick_accuracy))
    metric_cols[4].metric("Pending", len(pending))

    by_side = summarize_evaluation(evaluated, ["prediction"])
    if not by_side.empty:
        by_side = by_side.sort_values("prediction")
    if not by_side.empty:
        best = by_side.sort_values(["pick_accuracy", "accuracy"], ascending=False).iloc[0]
        st.info(
            f"Current stronger side: BTTS {best['prediction']} "
            f"(pick accuracy {_pct(best['pick_accuracy'])}, all accuracy {_pct(best['accuracy'])})."
        )

    by_matchday = summarize_evaluation(evaluated, ["target_matchday"])
    if not by_matchday.empty:
        by_matchday = by_matchday.sort_values("target_matchday")

    tabs = st.tabs(["Visuals", "By matchday", "Yes vs No", "Details", "Suggestions"])
    with tabs[0]:
        show_evaluation_visuals(evaluated, by_matchday, by_side)
    with tabs[1]:
        st.dataframe(format_summary_table(by_matchday), use_container_width=True, hide_index=True)
    with tabs[2]:
        st.dataframe(format_summary_table(by_side), use_container_width=True, hide_index=True)
        by_matchday_side = summarize_evaluation(evaluated, ["target_matchday", "prediction"])
        if not by_matchday_side.empty:
            by_matchday_side = by_matchday_side.sort_values(["target_matchday", "prediction"])
        st.dataframe(format_summary_table(by_matchday_side), use_container_width=True, hide_index=True)
    with tabs[3]:
        detail = evaluated.copy() if include_skips else evaluated[evaluated["picked"]].copy()
        detail["score"] = np.where(
            detail["actual"].notna(),
            detail["fulltime_home_goals"].astype("Int64").astype(str)
            + "-"
            + detail["fulltime_away_goals"].astype("Int64").astype(str),
            "",
        )
        detail_cols = [
            "prediction_timestamp",
            "prediction_file",
            "season_id",
            "target_matchday",
            "home_team",
            "away_team",
            "prediction",
            "actual",
            "score",
            "decision",
            "confidence",
            "correct",
            "actuals_file",
        ]
        detail_cols = [col for col in detail_cols if col in detail.columns]
        st.dataframe(detail[detail_cols], use_container_width=True, hide_index=True)
        st.download_button(
            "Download evaluation CSV",
            data=evaluated.to_csv(index=False).encode("utf-8"),
            file_name="btts_prediction_evaluation.csv",
            mime="text/csv",
        )
    with tabs[4]:
        st.markdown(
            """
            - Track BTTS Yes and BTTS No separately. A model can have good total accuracy while losing money on one side.
            - Treat wrong Picks as losses for now. True money ROI needs odds and stake history, but this view tells you whether the model logic is improving.
            - Keep Skip rows in the evaluation. If Skips are often correct, the threshold may be too strict; if Picks miss while Skips hit, the confidence calibration needs work.
            - Review by matchday. If late matchdays perform worse, train or calibrate the model with matchday-stage features.
            - Use the confidence bucket chart to see whether higher confidence really means higher accuracy.
            - Save every prediction before results are known. Evaluation is only honest when the prediction timestamp is before the match outcome.
            """
        )


def model_input_table(live_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    table = live_df.copy()
    for col in feature_columns:
        if col not in table.columns:
            table[col] = np.nan
    return table[TRACKING_COLUMNS + feature_columns]


def show_manual_input_verification(
    live_df: pd.DataFrame,
    feature_columns: list[str],
    standings: pd.DataFrame,
    fixture_odds: pd.DataFrame,
    feature_source: str,
    show_details: bool = False,
) -> None:
    missing = missing_feature_table(live_df, feature_columns)
    feature_table = model_input_table(live_df, feature_columns)

    st.caption(
        f"Ready: {len(fixture_odds)} fixtures, {len(standings)} standings teams, "
        f"{len(feature_columns)} model features. Source: {feature_source}"
    )

    if not show_details:
        return

    tabs = st.tabs(["Fixtures", "Standings", "Model input", "Missing fields"])
    with tabs[0]:
        st.dataframe(
            fixture_odds[["matchday", "home_team", "away_team"]],
            use_container_width=True,
            hide_index=True,
        )
    with tabs[1]:
        st.dataframe(standings, use_container_width=True, hide_index=True)
    with tabs[2]:
        st.caption("These are the exact feature columns passed to the saved model.")
        st.dataframe(feature_table, use_container_width=True, hide_index=True)
    with tabs[3]:
        if missing.empty:
            st.success("All saved model features have values.")
        else:
            st.warning("Rows shown here will be handled by the saved model pipeline/imputer.")
            st.dataframe(missing, use_container_width=True, hide_index=True)


def show_capture_result(result, show_details: bool = False) -> None:
    st.session_state["live_results_path"] = str(result.output_path)
    st.success(f"Live results updated: {result.rows_written} rows written.")
    if show_details:
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


def refresh_odds_after_live_update(show_details: bool = False) -> pd.DataFrame:
    if not load_settings or not collect_current_odds:
        raise RuntimeError(f"Odds collector is not importable. {COLLECTOR_IMPORT_ERROR}")
    result = run_odds_capture_command()
    load_odds.clear()
    odds_df = load_odds(str(ODDS_PATH))
    st.success(f"Odds refreshed: {result.rows} rows.")
    if show_details:
        st.write(
            {
                "odds_rows": result.rows,
                "markets_seen": result.markets_seen,
                "final_path": str(result.final_path) if result.final_path else str(ODDS_PATH),
            }
        )
    return odds_df


def show_exception(label: str, exc: BaseException) -> None:
    message = str(exc).strip() or repr(exc)
    if "ProcessSingleton" in message or "Lock file can not be created" in message:
        message = (
            f"{message}\n\nThe collector browser profile is already in use. "
            "Close any old collector Chrome window, then try again."
        )
    st.error(f"{label}: {type(exc).__name__}: {message}")
    with st.expander("Error details", expanded=False):
        st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def load_streamlit_collector_settings():
    settings = load_settings()
    STREAMLIT_BROWSER_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    return replace(settings, user_data_dir=STREAMLIT_BROWSER_PROFILE_DIR)


def run_live_results_capture_command(
    season_id: str | None,
    target_matchday: int | None,
    live_gap_matchdays: int,
    timeout_seconds: int | None = 300,
    progress_callback=None,
    timer_callback=None,
    plan_callback=None,
):
    env = os.environ.copy()
    env["USER_DATA_DIR"] = str(STREAMLIT_BROWSER_PROFILE_DIR)
    STREAMLIT_BROWSER_PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(PROJECT_ROOT / "collect_live_results.py"),
        "--live-gap-matchdays",
        str(live_gap_matchdays),
    ]
    if season_id:
        command.extend(["--season-id", str(season_id)])
    if target_matchday:
        command.extend(["--target-matchday", str(target_matchday)])
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    output_lines = []
    output_queue = queue.Queue()

    def read_process_output():
        if process.stdout is None:
            return
        for line in process.stdout:
            output_queue.put(line)

    reader = threading.Thread(target=read_process_output, daemon=True)
    reader.start()
    started_at = time.monotonic()
    last_reported_matchday = None
    last_reported_second = -1

    while process.poll() is None or not output_queue.empty():
        elapsed_seconds = int(time.monotonic() - started_at)
        if timer_callback and elapsed_seconds != last_reported_second:
            timer_callback(elapsed_seconds)
            last_reported_second = elapsed_seconds

        if timeout_seconds is not None and elapsed_seconds > timeout_seconds:
            process.kill()
            process.wait()
            raise subprocess.TimeoutExpired(command, timeout_seconds)

        try:
            line = output_queue.get(timeout=0.25)
        except queue.Empty:
            continue

        output_lines.append(line)
        plan_match = re.search(
            r"Live capture plan season=(\d+) target_md=(\d+) cutoff_md=(\d+) missing=(\d+)",
            line,
        )
        if plan_match and plan_callback:
            plan_callback(
                plan_match.group(1),
                int(plan_match.group(2)),
                int(plan_match.group(3)),
                int(plan_match.group(4)),
            )
        match = re.search(r"Live results season=.*? matchday=(\d+) rows=", line)
        if match and progress_callback:
            reported_matchday = int(match.group(1))
            if reported_matchday != last_reported_matchday:
                progress_callback(reported_matchday)
                last_reported_matchday = reported_matchday

    reader.join(timeout=1)
    while not output_queue.empty():
        output_lines.append(output_queue.get_nowait())

    output = "".join(output_lines)
    if process.returncode != 0:
        raise RuntimeError(output.strip() or f"collect_live_results.py exited with code {process.returncode}")

    summary = parse_live_capture_output(output)
    detected_season_id = str(summary.get("season") or season_id or "")
    detected_target_matchday = int(summary.get("target_md") or target_matchday or 0)
    if not detected_season_id or not detected_target_matchday:
        raise RuntimeError("The collector finished without reporting the detected season and target matchday.")
    detected_cutoff_matchday = int(
        summary.get("cutoff_md") or (detected_target_matchday - live_gap_matchdays)
    )
    output_path = LIVE_RESULTS_DIR / (
        f"season_{detected_season_id}_tgt_md{detected_target_matchday}_results.csv"
    )

    return SimpleNamespace(
        season_id=detected_season_id,
        target_matchday=detected_target_matchday,
        feature_cutoff_matchday=detected_cutoff_matchday,
        output_path=output_path,
        previous_path=None,
        reused_matchdays=parse_int_list(summary.get("reused", "")),
        fetched_matchdays=parse_int_list(summary.get("fetched", "")),
        incomplete_matchdays={},
        rows_written=int(summary.get("rows") or 0),
        raw_output=output,
    )


def run_odds_capture_command():
    env = os.environ.copy()
    env["USER_DATA_DIR"] = str(STREAMLIT_BROWSER_PROFILE_DIR)
    STREAMLIT_BROWSER_PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(PROJECT_ROOT / "collect_odds.py"),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=300,
    )
    output = "\n".join(part for part in [completed.stdout, completed.stderr] if part.strip())
    if completed.returncode != 0:
        raise RuntimeError(output.strip() or f"collect_odds.py exited with code {completed.returncode}")

    summary = parse_odds_capture_output(output)
    return SimpleNamespace(
        rows=int(summary.get("rows") or 0),
        markets_seen=parse_text_list(summary.get("markets", "")),
        final_path=ODDS_PATH,
        raw_output=output,
    )


def parse_odds_capture_output(output: str) -> dict[str, str]:
    done_line = ""
    for line in output.splitlines():
        if "Done. rows=" in line:
            done_line = line
    if not done_line:
        return {}
    return dict(re.findall(r"(\w+)=([^\s]+)", done_line))


def parse_text_list(value: str) -> list[str]:
    if not value or value == "-":
        return []
    return [part for part in value.split(",") if part]


def parse_live_capture_output(output: str) -> dict[str, str]:
    done_line = ""
    for line in output.splitlines():
        if "Done. season=" in line:
            done_line = line
    if not done_line:
        return {}

    pairs = re.findall(r"(\w+)=([^\s]+)", done_line)
    return dict(pairs)


def parse_int_list(value: str) -> list[int]:
    if not value or value == "-":
        return []
    return [int(part) for part in value.split(",") if part.strip().isdigit()]


def apply_compact_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
        h1 { font-size: 1.6rem !important; margin-bottom: 0.25rem !important; }
        h2 { font-size: 1.15rem !important; margin-top: 0.7rem !important; }
        h3 { font-size: 1rem !important; margin-top: 0.5rem !important; }
        p, label, .stCaption, [data-testid="stMarkdownContainer"] { font-size: 0.86rem; }
        div.stButton > button { padding: 0.28rem 0.65rem; min-height: 2rem; font-size: 0.86rem; }
        textarea { font-size: 0.86rem !important; }
        .step-guide {
            font-size: 0.84rem;
            line-height: 1.45;
            color: #b7bcc7;
            margin: 0.25rem 0 0.65rem 0;
        }
        .compact-status {
            font-size: 0.82rem;
            color: #aab0bb;
            margin: 0.2rem 0 0.35rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    apply_compact_styles()
    st.markdown("### BTTS Predictor")

    with st.sidebar:
        st.header("Workspace")
        app_page = st.radio("Page", ["Predict BTTS", "Evaluate results"], index=0)

    if app_page == "Evaluate results":
        show_evaluation_page()
        return

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
        st.caption("Season and matchday are detected from the screenshot/standings text.")
        show_details = st.checkbox("Show detailed panels", value=False)
        st.metric("pick_threshold", f"{pick_threshold:.0%}")
        st.caption(f"Saved model features: {len(feature_columns)}")

    st.markdown("#### Manual input")
    st.markdown(
        """
        <div class="step-guide">
        <b>1.</b> Update odds first, then paste fixture screenshots and standings.<br>
        <b>2.</b> Update live results, check compact status, then predict.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if "manual_screenshot_bytes" not in st.session_state:
        st.session_state["manual_screenshot_bytes"] = []

    odds_col, paste_col, clear_col = st.columns([1, 1.3, 1])
    with odds_col:
        if st.button("1. Update odds"):
            try:
                with st.spinner("Updating current odds..."):
                    odds = refresh_odds_after_live_update(show_details)
            except Exception as exc:
                show_exception("Odds update failed", exc)
    with paste_col:
        if st.button("Paste screenshot"):
            image_bytes, error = read_clipboard_image_png()
            if error:
                st.error(error)
            elif image_bytes:
                st.session_state["manual_screenshot_bytes"].append(image_bytes)
                st.success(f"Screenshots: {len(st.session_state['manual_screenshot_bytes'])}")
    with clear_col:
        if st.button("Clear screenshots"):
            st.session_state["manual_screenshot_bytes"] = []

    screenshot_texts = []
    screenshots = st.session_state["manual_screenshot_bytes"]
    if screenshots:
        if show_details:
            cols = st.columns(min(len(screenshots), 3))
        for idx, image_bytes in enumerate(screenshots):
            if show_details:
                cols[idx % len(cols)].image(image_bytes, caption=f"Screenshot {idx + 1}", width=260)
            text = extract_text_from_image_bytes(image_bytes)
            if text.strip():
                screenshot_texts.append(text)

        if screenshot_texts:
            st.caption(f"Screenshots pasted: {len(screenshots)}")
            if show_details:
                with st.expander("OCR text extracted from screenshots", expanded=False):
                    st.text_area(
                        "Extracted screenshot text",
                        value="\n\n".join(screenshot_texts),
                        height=220,
                        disabled=True,
                    )
        else:
            st.warning(
                "No readable fixture text was extracted from the screenshots. "
                "Copy a clearer screenshot or use automated odds/fixture capture."
            )
    else:
        st.caption("Screenshots pasted: 0")

    standings_text = st.text_area(
        "Standings table text",
        height=130,
        placeholder="Paste the Season # line and the full Club Pts W D L Goals table here.",
    )

    fixture_input_text = "\n".join(screenshot_texts)
    all_input_text = "\n".join([standings_text, fixture_input_text])

    detected_season = parse_first_int(SEASON_RE, all_input_text, "season_id")
    detected_fixture_matchday = parse_first_int(MATCHDAY_RE, fixture_input_text, "matchday")
    detected_standings_matchday = parse_first_int(MATCHDAY_RE, standings_text, "matchday")
    detected_matchday = detected_fixture_matchday or detected_standings_matchday
    active_season_id = str(
        detected_season or st.session_state.get("playwright_season_id") or ""
    )
    active_target_matchday = int(
        detected_matchday or st.session_state.get("playwright_target_matchday") or 0
    ) or None
    active_cutoff = active_target_matchday - live_gap_matchdays if active_target_matchday else None

    if (
        detected_fixture_matchday
        and detected_standings_matchday
        and detected_fixture_matchday != detected_standings_matchday
    ):
        st.warning(
            "Fixture screenshot matchday and standings text matchday do not match. "
            f"Using fixture screenshot matchday {detected_fixture_matchday} for prediction."
        )

    standings = parse_standings_text(standings_text)
    fixture_rows = parse_fixtures_and_odds(fixture_input_text, active_target_matchday or 0)
    btts_rows = parse_btts_odds_text("")
    fixture_odds = merge_fixture_odds(fixture_rows, odds, btts_rows)

    st.caption(
        " | ".join(
            [
                f"season_id: {active_season_id or '-'}",
                f"target_matchday: {active_target_matchday if active_target_matchday else '-'}",
                f"feature_cutoff_matchday: {active_cutoff if active_cutoff is not None else '-'}",
            ]
        )
    )
    st.title(f"MD {active_target_matchday:02d}" if active_target_matchday else "MD —")

    manual_live_path = None
    if active_season_id and active_target_matchday:
        manual_live_path = session_live_results_file(active_season_id, active_target_matchday) or find_exact_live_results_file(
            active_season_id,
            active_target_matchday,
        )

    if st.button("2. Update live results"):
        if not load_settings or not collect_live_results:
            st.error(f"Live results collector is not importable. {COLLECTOR_IMPORT_ERROR}")
        else:
            capture_status = st.empty()
            update_progress_state = {"total": 0, "completed": 0}
            update_progress_bar = st.progress(
                0,
                text="Waiting for Playwright to detect missing matchdays...",
            )

            def show_update_plan(season_id: str, target_md: int, cutoff_md: int, missing: int) -> None:
                update_progress_state["total"] = missing
                capture_status.info(
                    f"Playwright detected season {season_id}, target matchday {target_md}. "
                    f"Capturing missing matchdays through {cutoff_md} ({missing} missing)."
                )
                if missing == 0:
                    update_progress_bar.progress(100, text="No missing matchdays")
                else:
                    update_progress_bar.progress(
                        0,
                        text=f"Preparing to capture {missing} missing matchday(s)...",
                    )

            def update_live_results_progress(matchday: int) -> None:
                update_progress_state["completed"] += 1
                total = max(update_progress_state["total"], 1)
                completed = min(update_progress_state["completed"], total)
                percent = min(int((completed / total) * 100), 99)
                update_progress_bar.progress(
                    percent,
                    text=f"Processed missing matchday {matchday} ({completed}/{total})",
                )

            with st.spinner("Capturing raw completed results up to the cutoff matchday..."):
                try:
                    result = run_live_results_capture_command(
                        None,
                        None,
                        live_gap_matchdays,
                        progress_callback=update_live_results_progress,
                        plan_callback=show_update_plan,
                    )
                    load_results_csv.clear()
                    show_capture_result(result, show_details)
                    manual_live_path = Path(result.output_path)
                    active_season_id = result.season_id
                    active_target_matchday = result.target_matchday
                    active_cutoff = result.feature_cutoff_matchday
                    st.session_state["playwright_season_id"] = result.season_id
                    st.session_state["playwright_target_matchday"] = result.target_matchday
                    update_progress_bar.progress(100, text="Live results update complete")
                except Exception as exc:
                    show_exception("Live results capture failed", exc)

    if st.button("First-time season setup (full history)"):
        if not load_settings or not collect_live_results:
            st.error(f"Live results collector is not importable. {COLLECTOR_IMPORT_ERROR}")
        else:
            cutoff_holder = {"value": None}
            capture_status = st.empty()
            progress_bar = st.progress(
                0,
                text="Waiting for Playwright to detect the current season and matchday...",
            )
            timer_placeholder = st.empty()
            timer_placeholder.metric("Elapsed time", "00:00:00")

            def show_first_time_plan(season_id: str, target_md: int, cutoff_md: int, missing: int) -> None:
                cutoff_holder["value"] = max(cutoff_md, 1)
                capture_status.info(
                    f"Playwright detected season {season_id}, target matchday {target_md}. "
                    f"Backfilling matchdays 1-{cutoff_md} ({missing} missing)."
                )
                progress_bar.progress(
                    0,
                    text=f"Preparing historical matchdays 1-{cutoff_md}...",
                )

            def update_first_time_progress(matchday: int) -> None:
                cutoff_matchday = cutoff_holder["value"] or max(matchday, 1)
                percent = min(max(int((matchday / cutoff_matchday) * 100), 1), 99)
                progress_bar.progress(
                    percent,
                    text=f"Processed matchday {matchday} of {cutoff_matchday}",
                )

            def update_first_time_timer(elapsed_seconds: int) -> None:
                hours, remainder = divmod(elapsed_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                timer_placeholder.metric(
                    "Elapsed time",
                    f"{hours:02d}:{minutes:02d}:{seconds:02d}",
                )

            with st.spinner("Capturing the season history. This first run can take several minutes..."):
                try:
                    result = run_live_results_capture_command(
                        None,
                        None,
                        live_gap_matchdays,
                        timeout_seconds=None,
                        progress_callback=update_first_time_progress,
                        timer_callback=update_first_time_timer,
                        plan_callback=show_first_time_plan,
                    )
                    load_results_csv.clear()
                    show_capture_result(result, show_details)
                    manual_live_path = Path(result.output_path)
                    active_season_id = result.season_id
                    active_target_matchday = result.target_matchday
                    active_cutoff = result.feature_cutoff_matchday
                    st.session_state["playwright_season_id"] = result.season_id
                    st.session_state["playwright_target_matchday"] = result.target_matchday
                    progress_bar.progress(100, text="Season history capture complete")
                except Exception as exc:
                    show_exception("First-time season setup failed", exc)

    if active_season_id and active_target_matchday:
        if manual_live_path:
            st.caption(f"Live results: {manual_live_path.name}")
        else:
            st.caption("Live results: not updated for this matchday yet.")

    manual_live_df = None
    if active_season_id and active_target_matchday and not fixture_odds.empty and not standings.empty:
        if manual_live_path and manual_live_path.exists():
            results = load_results_csv(str(manual_live_path))
            bundle = make_stats_bundle(results, active_season_id, active_cutoff)
            feature_source = f"raw live results CSV: {manual_live_path}"
            manual_live_df = build_prediction_features(
                fixture_odds,
                active_season_id,
                active_cutoff,
                live_gap_matchdays,
                bundle,
            )
            show_manual_input_verification(
                manual_live_df,
                feature_columns,
                standings,
                fixture_odds,
                feature_source,
                show_details,
            )
        else:
            st.caption(
                f"Captured: {len(fixture_odds)} fixtures and {len(standings)} standings teams. "
                "Run Update live results."
            )
            if show_details:
                tabs = st.tabs(["Fixtures", "Standings", "Missing until live results"])
                with tabs[0]:
                    st.dataframe(
                        fixture_odds[["matchday", "home_team", "away_team"]],
                        use_container_width=True,
                        hide_index=True,
                    )
                with tabs[1]:
                    st.dataframe(standings, use_container_width=True, hide_index=True)
                with tabs[2]:
                    st.warning(
                        "The 21 performance/form features below require raw match-by-match results. "
                        "Click Update live results to create the current-season CSV, then this table will become the full model input."
                    )
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "feature": LIVE_RESULTS_REQUIRED_FEATURES,
                                "source_needed": "data/live_seasons raw results CSV",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
    else:
        st.caption("Paste the fixture screenshots and standings text to build the model input table.")

    if st.button("Predict BTTS", type="primary"):
        if not active_season_id:
            st.error("No season_id was detected. Paste standings text containing a line like Season #138605.")
        elif not active_target_matchday:
            st.error("No target matchday was detected. Paste the fixture screenshots first.")
        elif fixture_odds.empty:
            st.error("No fixtures were found from the pasted screenshots.")
        elif standings.empty:
            st.error("No standings rows were found. Paste the full Club Pts W D L Goals table before predicting.")
        elif manual_live_df is None:
            st.error("Run Update live results first, then verify the Model input table before predicting.")
        else:
            show_prediction_output(model, manual_live_df, feature_columns, pick_threshold, active_target_matchday)


def show_prediction_output(
    model: object,
    live_df: pd.DataFrame,
    feature_columns: list[str],
    pick_threshold: float,
    target_matchday: int,
) -> None:
    try:
        predictions = make_predictions(model, live_df, feature_columns, pick_threshold)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    output_path = save_predictions_live_file(predictions)

    st.subheader("BTTS Predictions")
    if len(predictions) < 10:
        st.warning(
            f"Predicted all {len(predictions)} detected fixtures. "
            f"{10 - len(predictions)} fixture(s) were not detected and are not included."
        )
    try:
        prediction_table = format_predictions(predictions)
    except Exception as exc:
        st.warning(f"Table styling was skipped, but predictions were completed and saved: {exc}")
        prediction_table = predictions
    st.dataframe(prediction_table, use_container_width=True, hide_index=True)
    play_prediction_alert(predictions)
    st.caption(f"Updated live predictions file: {output_path}")
    st.download_button(
        "Download predictions CSV",
        data=output_path.read_bytes(),
        file_name=output_path.name,
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
