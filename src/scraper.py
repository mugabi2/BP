from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError, sync_playwright

from .config import Settings


SEASON_RE = re.compile(r"Season\s*#?(\d+)", re.IGNORECASE)
RESULT_RE = re.compile(
    r"(?P<home>[A-Z]{2,4})\s*(?:-|\u2013)\s*(?P<away>[A-Z]{2,4}).*?"
    r"(?P<home_goals>\d+)\s*[:\-]\s*(?P<away_goals>\d+)"
)
FIXTURE_RE = re.compile(r"^(?P<home>[A-Z]{2,4})\s*(?:-|\u2013)\s*(?P<away>[A-Z]{2,4})(?:\s|$)")
SCORE_RE = re.compile(r"(?P<home_goals>\d+)\s*-\s*(?P<away_goals>\d+)")
MATCH_ROW_RE = re.compile(
    r"(?P<home>[A-Z]{2,4})\s*(?:-|\u2013)\s*(?P<away>[A-Z]{2,4})"
    r"\s*\((?P<half_home>\d+)\s*-\s*(?P<half_away>\d+)\)"
    r"\s*(?P<full_home>\d+)\s*-\s*(?P<full_away>\d+)"
)
MATCHDAY_RE = re.compile(r"Matchday\s*(?P<matchday>\d+)", re.IGNORECASE)
ENGLISH_TEAMS = {
    "ARS",
    "AST",
    "BHA",
    "BOU",
    "BRE",
    "BUR",
    "CHE",
    "COV",
    "CRY",
    "EVE",
    "FUL",
    "HUL",
    "IPS",
    "LEE",
    "LIV",
    "MCI",
    "MUN",
    "NEW",
    "NOT",
    "SUN",
    "TOT",
    "WHU",
    "WOL",
}


@dataclass
class CollectionResult:
    seasons_seen: int = 0
    seasons_skipped: int = 0
    match_rows: int = 0


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def open_context(settings: Settings):
    playwright = sync_playwright().start()
    launch_options = {
        "user_data_dir": str(settings.user_data_dir),
        "headless": settings.headless,
        "viewport": {"width": 1366, "height": 900},
        "args": ["--disable-quic"],
    }
    if settings.chrome_path:
        launch_options["executable_path"] = settings.chrome_path
    elif settings.browser_channel:
        launch_options["channel"] = settings.browser_channel
    context = playwright.chromium.launch_persistent_context(**launch_options)
    page = context.pages[0] if context.pages else context.new_page()
    return playwright, context, page


def login_once(settings: Settings) -> None:
    settings.user_data_dir.mkdir(parents=True, exist_ok=True)
    playwright, context, page = open_context(settings)
    try:
        page.goto(settings.url, wait_until="domcontentloaded", timeout=60_000)
        print("Browser opened. Log in manually, then return here and press Enter.")
        input()
    finally:
        context.close()
        playwright.stop()


def collect(settings: Settings, conn, keep_open: bool = False) -> CollectionResult:
    settings.user_data_dir.mkdir(parents=True, exist_ok=True)
    result = CollectionResult()
    playwright, context, page = open_context(settings)
    try:
        open_results_page(page, settings)

        season_ids = discover_seasons(page, settings.max_seasons)
        if not season_ids:
            current = current_season_id(page)
            season_ids = [current] if current else []
        if settings.skip_latest_season and len(season_ids) > 1:
            skipped = season_ids[0]
            season_ids = season_ids[1:]
            logging.info("Skipping latest season %s because it may be incomplete.", skipped)
        season_ids = sorted(season_ids, key=int)
        from .db import season_is_fully_captured

        season_ids_to_collect = []
        for season_id in season_ids:
            if season_is_fully_captured(conn, season_id, settings.matchdays_per_season):
                logging.info(
                    "Skipping season %s because it is already captured.",
                    season_id,
                )
                result.seasons_skipped += 1
                continue
            season_ids_to_collect.append(season_id)
        season_ids = season_ids_to_collect
        logging.info("Discovered seasons: %s", season_ids)
        if not season_ids:
            logging.warning("No seasons found.")

        for season_id in season_ids:
            season_started_at = now_iso()
            timer_started_at = perf_counter()
            match_results = []
            try:
                from .db import export_csvs, finish_season_capture, start_season_capture

                start_season_capture(
                    conn,
                    {
                        "season_id": season_id,
                        "league_id": settings.league_id,
                        "league_name": settings.league_name,
                        "captured_at": season_started_at,
                        "capture_started_at": season_started_at,
                    },
                )
                conn.commit()

                match_results = extract_all_matchday_results(
                    page,
                    conn,
                    settings,
                    season_id,
                    season_started_at,
                    on_matchday=lambda rows: save_matchday_rows(conn, settings, rows),
                )
                result.match_rows += len(match_results)
                result.seasons_seen += 1
                matchdays_captured = len({row["matchday"] for row in match_results})
                expected_matches = settings.matchdays_per_season * 10
                errors = []
                if matchdays_captured != settings.matchdays_per_season:
                    errors.append(f"captured {matchdays_captured}/{settings.matchdays_per_season} matchdays")
                if len(match_results) < expected_matches:
                    errors.append(f"captured {len(match_results)}/{expected_matches} matches")
                successful = "no" if errors else "yes"
                finish_season_capture(
                    conn,
                    {
                        "season_id": season_id,
                        "capture_finished_at": now_iso(),
                        "duration_seconds": int(perf_counter() - timer_started_at),
                        "matchdays_captured": matchdays_captured,
                        "matches_captured": len(match_results),
                        "successful": successful,
                        "error": None if successful == "yes" else "; ".join(errors),
                    },
                )
                conn.commit()
                export_csvs(conn, settings.export_dir)
                logging.info(
                    "Season %s finished: matchdays=%s matches=%s successful=%s",
                    season_id,
                    matchdays_captured,
                    len(match_results),
                    successful,
                )
            except Exception as exc:
                conn.rollback()
                try:
                    from .db import export_csvs, finish_season_capture

                    finish_season_capture(
                        conn,
                        {
                            "season_id": season_id,
                            "capture_finished_at": now_iso(),
                            "duration_seconds": int(perf_counter() - timer_started_at),
                            "matchdays_captured": len({row["matchday"] for row in match_results}),
                            "matches_captured": len(match_results),
                            "successful": "no",
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    conn.commit()
                    export_csvs(conn, settings.export_dir)
                except Exception:
                    conn.rollback()
                logging.exception("Failed collecting season %s", season_id)
        return result
    finally:
        if keep_open:
            print("Keeping browser open. Press Enter here when you want to close it.")
            input()
        context.close()
        playwright.stop()


def open_results_page(page: Page, settings: Settings) -> None:
    page.goto(settings.url, wait_until="domcontentloaded", timeout=60_000)
    wait_for_results_page(page, settings.page_ready_timeout_ms)
    _click_text_if_visible(page, settings.league_name)
    _click_text_if_visible(page, "Results")
    page.wait_for_timeout(settings.action_delay_ms)


def save_matchday_rows(conn, settings: Settings, rows: list[dict]) -> int:
    if not rows:
        return 0

    from .db import export_csvs, upsert_many

    changes = upsert_many(conn, "match_results", rows)
    conn.commit()
    try:
        export_csvs(conn, settings.export_dir)
    except PermissionError:
        logging.warning(
            "Could not refresh CSV because it is probably open in Excel. "
            "Close match_results_master.csv and run python collect_virtuals.py export."
        )
    return changes


def wait_for_results_page(page: Page, timeout_ms: int) -> None:
    try:
        page.wait_for_load_state("networkidle", timeout=timeout_ms)
    except PlaywrightTimeoutError:
        logging.info("Network did not become idle; continuing with visible page checks.")

    try:
        page.get_by_text("Results", exact=False).wait_for(state="visible", timeout=timeout_ms)
    except PlaywrightTimeoutError:
        logging.warning("Results text was not visible before timeout.")


def discover_seasons(page: Page, max_seasons: int) -> list[str]:
    option_ids = _season_ids_from_select_options(page)
    if option_ids:
        return option_ids[:max_seasons]

    _open_season_dropdown(page)
    text = _body_text(page)
    season_ids = []
    for match in SEASON_RE.finditer(text):
        season_id = match.group(1)
        if season_id not in season_ids:
            season_ids.append(season_id)
    return season_ids[:max_seasons]


def current_season_id(page: Page) -> str | None:
    match = SEASON_RE.search(_body_text(page))
    return match.group(1) if match else None


def select_season(page: Page, season_id: str, delay_ms: int) -> None:
    selected = _select_native_season(page, season_id)
    if selected:
        page.wait_for_timeout(delay_ms)
        return

    _open_season_dropdown(page)
    option = page.get_by_text(f"Season #{season_id}", exact=False)
    if option.count() > 0:
        option.first.click(timeout=10_000)
        page.wait_for_timeout(delay_ms)
        return
    logging.warning("Season %s was not clickable; keeping current page state.", season_id)


def extract_all_matchday_results(
    page: Page,
    conn,
    settings: Settings,
    season_id: str,
    captured_at: str,
    on_matchday=None,
) -> list[dict]:
    rows = []
    from .db import captured_matchdays

    already_captured = captured_matchdays(conn, season_id)
    for matchday in range(1, settings.matchdays_per_season + 1):
        if matchday in already_captured:
            rows.extend(existing_matchday_rows(conn, season_id, matchday))
            continue
        try:
            page.goto(
                matchday_url(settings, season_id, matchday),
                wait_until="load",
                timeout=60_000,
            )
            page.wait_for_timeout(settings.action_delay_ms)
            wait_for_matchday_content(page, matchday, settings.page_ready_timeout_ms)
            matchday_rows = extract_matchday_results(page, season_id, matchday, captured_at)
            rows.extend(matchday_rows)
            log_captured_matches(season_id, matchday, matchday_rows)
            if on_matchday:
                on_matchday(matchday_rows)
            if not matchday_rows:
                logging.warning("Season %s matchday %s captured 0 matches.", season_id, matchday)
        except Exception:
            logging.exception("Failed reading season %s matchday %s", season_id, matchday)
    return rows


def existing_matchday_rows(conn, season_id: str, matchday: int) -> list[dict]:
    existing_rows = conn.execute(
        """
        SELECT
            season_id,
            matchday,
            home_team,
            away_team,
            halftime_home_goals,
            halftime_away_goals,
            fulltime_home_goals,
            fulltime_away_goals,
            captured_at
        FROM match_results
        WHERE season_id = ? AND matchday = ?
        """,
        (season_id, matchday),
    ).fetchall()
    return [dict(row) for row in existing_rows]


def log_captured_matches(season_id: str, matchday: int, rows: list[dict]) -> None:
    for row in rows:
        logging.info(
            "Captured S%s MD%02d %s-%s FT %s-%s",
            season_id,
            matchday,
            row["home_team"],
            row["away_team"],
            row["fulltime_home_goals"],
            row["fulltime_away_goals"],
        )


def extract_matchday_results(
    page: Page,
    season_id: str,
    matchday: int,
    captured_at: str,
) -> list[dict]:
    text = _main_results_text(page)
    rows = extract_match_rows_from_text(text, season_id, matchday, captured_at)
    if rows:
        return rows

    lines = _compact_lines(text)
    rows = []
    index = 0

    while index < len(lines):
        fixture_match = FIXTURE_RE.match(lines[index])
        if not fixture_match:
            index += 1
            continue

        if not _valid_fixture(fixture_match.group("home"), fixture_match.group("away")):
            index += 1
            continue

        score = _last_score_in_line(lines[index]) or _first_score_after_fixture(lines, index + 1)
        if score:
            rows.append(
                {
                    "season_id": season_id,
                    "matchday": matchday,
                    "home_team": fixture_match.group("home"),
                    "away_team": fixture_match.group("away"),
                    "halftime_home_goals": None,
                    "halftime_away_goals": None,
                    "fulltime_home_goals": int(score.group("home_goals")),
                    "fulltime_away_goals": int(score.group("away_goals")),
                    "captured_at": captured_at,
                }
            )
        index += 1

    return rows


def extract_match_rows_from_text(
    text: str,
    season_id: str,
    matchday: int,
    captured_at: str,
) -> list[dict]:
    rows = []
    seen = set()
    normalized = re.sub(r"\s+", " ", text)
    for match in MATCH_ROW_RE.finditer(normalized):
        home_team = match.group("home")
        away_team = match.group("away")
        if not _valid_fixture(home_team, away_team):
            continue

        key = (home_team, away_team)
        if key in seen:
            continue
        seen.add(key)

        rows.append(
            {
                "season_id": season_id,
                "matchday": matchday,
                "home_team": home_team,
                "away_team": away_team,
                "halftime_home_goals": int(match.group("half_home")),
                "halftime_away_goals": int(match.group("half_away")),
                "fulltime_home_goals": int(match.group("full_home")),
                "fulltime_away_goals": int(match.group("full_away")),
                "captured_at": captured_at,
            }
        )
    return rows


def _main_results_text(page: Page) -> str:
    text = _body_text(page)
    if "English League" in text:
        text = text.split("English League", 1)[1]
    if "View My Bets" in text:
        text = text.split("View My Bets", 1)[0]
    if "Betslip is empty" in text:
        text = text.split("Betslip is empty", 1)[0]
    return text


def _valid_fixture(home_team: str, away_team: str) -> bool:
    return home_team in ENGLISH_TEAMS and away_team in ENGLISH_TEAMS and home_team != away_team


def _old_extract_matchday_results_unused(
    page: Page,
    season_id: str,
    matchday: int,
    captured_at: str,
) -> list[dict]:
    lines = _compact_lines(_main_results_text(page))
    rows = []
    index = 0

    while index < len(lines):
        fixture_match = FIXTURE_RE.match(lines[index])
        if not fixture_match:
            index += 1
            continue

        score = _last_score_in_line(lines[index]) or _first_score_after_fixture(lines, index + 1)
        if score:
            rows.append(
                {
                    "season_id": season_id,
                    "matchday": matchday,
                    "home_team": fixture_match.group("home"),
                    "away_team": fixture_match.group("away"),
                    "halftime_home_goals": None,
                    "halftime_away_goals": None,
                    "fulltime_home_goals": int(score.group("home_goals")),
                    "fulltime_away_goals": int(score.group("away_goals")),
                    "captured_at": captured_at,
                }
            )
        index += 1

    return rows


def matchday_url(settings: Settings, season_id: str, matchday: int) -> str:
    return (
        f"https://www.betpawa.ug/virtual-sports/matchday/{season_id}"
        f"?matchday={matchday}&leagueId={settings.league_id}"
    )


def wait_for_matchday_content(page: Page, matchday: int, timeout_ms: int) -> None:
    expected_labels = {f"Matchday: {matchday:02d}", f"Matchday: {matchday}"}
    per_attempt_timeout = max(timeout_ms / 2, 5000) / 1000

    for attempt in range(2):
        deadline = datetime.now(timezone.utc).timestamp() + per_attempt_timeout
        while datetime.now(timezone.utc).timestamp() < deadline:
            text = _main_results_text(page)
            has_matchday_label = any(label in text for label in expected_labels)
            if has_matchday_label and extract_match_rows_from_text(text, "probe", matchday, now_iso()):
                return
            page.wait_for_timeout(500)

        if attempt == 0:
            page.reload(wait_until="domcontentloaded", timeout=60_000)


def _has_fixture_result_text(text: str) -> bool:
    lines = _compact_lines(text)
    for index, line in enumerate(lines):
        if not FIXTURE_RE.match(line):
            continue
        fixture = FIXTURE_RE.match(line)
        if fixture and not _valid_fixture(fixture.group("home"), fixture.group("away")):
            continue
        if _last_score_in_line(line) or _first_score_after_fixture(lines, index + 1):
            return True
    return False


def _first_score_after_fixture(lines: list[str], start_index: int) -> re.Match | None:
    for line in lines[start_index : start_index + 4]:
        score = _last_score_in_line(line)
        if score:
            return score
    return None


def _last_score_in_line(line: str) -> re.Match | None:
    clean_line = re.sub(r"\([^)]*\)", " ", line)
    scores = list(SCORE_RE.finditer(clean_line))
    return scores[-1] if scores else None


def _open_season_dropdown(page: Page) -> None:
    control = page.get_by_text("Season #", exact=False)
    try:
        if control.count() > 0:
            control.first.click(timeout=10_000)
            page.wait_for_timeout(500)
    except PlaywrightTimeoutError:
        logging.warning("Could not open season dropdown")


def _season_ids_from_select_options(page: Page) -> list[str]:
    try:
        ids = page.evaluate(
            """
            () => Array.from(document.querySelectorAll('option'))
              .map((option) => ({
                value: option.value || '',
                text: option.textContent || ''
              }))
              .filter((option) => /Season\\s*#?\\d+/i.test(option.text))
              .map((option) => {
                const valueMatch = option.value.match(/\\d+/);
                const textMatch = option.text.match(/Season\\s*#?(\\d+)/i);
                return valueMatch ? valueMatch[0] : (textMatch ? textMatch[1] : null);
              })
              .filter(Boolean)
            """
        )
    except Exception:
        logging.exception("Could not read season options.")
        return []

    season_ids = []
    for season_id in ids:
        if season_id not in season_ids:
            season_ids.append(season_id)
    return season_ids


def _select_native_season(page: Page, season_id: str) -> bool:
    try:
        return bool(
            page.evaluate(
                """
                (seasonId) => {
                  const selects = Array.from(document.querySelectorAll('select'));
                  const select = selects.find((candidate) =>
                    Array.from(candidate.options).some((option) =>
                      option.value === seasonId ||
                      (option.textContent || '').includes(`Season #${seasonId}`)
                    )
                  );

                  if (!select) return false;

                  select.value = seasonId;
                  select.dispatchEvent(new Event('input', { bubbles: true }));
                  select.dispatchEvent(new Event('change', { bubbles: true }));
                  return true;
                }
                """,
                season_id,
            )
        )
    except Exception:
        logging.exception("Could not select native season %s.", season_id)
        return False


def _click_text_if_visible(page: Page, text: str) -> bool:
    locator = page.get_by_text(text, exact=False)
    try:
        if locator.count() > 0 and locator.first.is_visible():
            locator.first.click(timeout=10_000)
            page.wait_for_timeout(500)
            return True
    except PlaywrightTimeoutError:
        return False
    return False


def _body_text(page: Page) -> str:
    try:
        return page.locator("body").inner_text(timeout=15_000)
    except PlaywrightTimeoutError:
        return ""


def write_page_diagnostics(settings: Settings, page: Page) -> None:
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)
    text_path = settings.log_path.parent / "last_page_text.txt"
    try:
        text = _body_text(page)
        text_path.write_text(
            f"URL: {page.url}\nTITLE: {page.title()}\n\n{text}",
            encoding="utf-8",
        )
    except Exception:
        logging.exception("Could not write page diagnostics.")


def write_matchday_diagnostics(settings: Settings, page: Page, season_id: str, matchday: int) -> None:
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)
    text_path = settings.log_path.parent / f"matchday_{season_id}_{matchday}_text.txt"
    try:
        text = _body_text(page)
        text_path.write_text(
            f"URL: {page.url}\nTITLE: {page.title()}\n\n{text}",
            encoding="utf-8",
        )
    except Exception:
        logging.exception("Could not write matchday diagnostics.")


def _compact_lines(text: str) -> list[str]:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return [line for line in lines if line]


def _current_matchday(text: str) -> int:
    match = MATCHDAY_RE.search(text)
    return int(match.group("matchday")) if match else 0
