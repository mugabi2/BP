from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]


def _bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    url: str
    league_id: str
    league_name: str
    headless: bool
    browser_channel: str | None
    chrome_path: str | None
    max_seasons: int
    skip_latest_season: bool
    matchdays_per_season: int
    action_delay_ms: int
    page_ready_timeout_ms: int
    user_data_dir: Path
    database_path: Path
    export_dir: Path
    odds_dir: Path
    log_path: Path


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")

    return Settings(
        url=os.getenv(
            "BETPAWA_URL",
            "https://www.betpawa.ug/virtual-sports?virtualTab=results&leagueId=7794",
        ),
        league_id=os.getenv("LEAGUE_ID", "7794"),
        league_name=os.getenv("LEAGUE_NAME", "English League"),
        headless=_bool(os.getenv("HEADLESS"), default=False),
        browser_channel=os.getenv("BROWSER_CHANNEL", "chrome") or None,
        chrome_path=os.getenv("CHROME_PATH") or None,
        max_seasons=int(os.getenv("MAX_SEASONS", "10")),
        skip_latest_season=_bool(os.getenv("SKIP_LATEST_SEASON"), default=True),
        matchdays_per_season=int(os.getenv("MATCHDAYS_PER_SEASON", "34")),
        action_delay_ms=int(os.getenv("ACTION_DELAY_MS", "500")),
        page_ready_timeout_ms=int(os.getenv("PAGE_READY_TIMEOUT_MS", "30000")),
        user_data_dir=Path(os.getenv("USER_DATA_DIR", ROOT / "auth" / "browser-profile")),
        database_path=ROOT / "data" / "betpawa_virtuals_master.sqlite",
        export_dir=ROOT / "data" / "exports",
        odds_dir=ROOT / "data" / "odds",
        log_path=ROOT / "logs" / "collector.log",
    )
