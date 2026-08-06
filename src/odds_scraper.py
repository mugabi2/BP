from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError

from .config import Settings
from .scraper import _click_text_if_visible, _valid_fixture, now_iso, open_context, write_page_diagnostics


MARKETS = ("1X2", "O/U", "BTTS", "DC", "HT/FT")
BASE_ODDS_FIELDS = [
    "captured_at",
    "source_url",
    "league_id",
    "league_name",
    "matchday",
    "start_label",
    "home_team",
    "away_team",
]
ODDS_COLUMNS = [
    "odd_1",
    "odd_x",
    "odd_2",
    "ou_over_0_5",
    "ou_under_0_5",
    "ou_over_1_5",
    "ou_under_1_5",
    "ou_over_2_5",
    "ou_under_2_5",
    "ou_over_3_5",
    "ou_under_3_5",
    "ou_over_4_5",
    "ou_under_4_5",
    "ou_over_5_5",
    "ou_under_5_5",
    "ou_over_6_5",
    "ou_under_6_5",
    "btts_yes",
    "btts_no",
    "dc_1x",
    "dc_x2",
    "dc_12",
    "htft_1_1",
    "htft_1_x",
    "htft_1_2",
    "htft_x_1",
    "htft_x_x",
    "htft_x_2",
    "htft_2_1",
    "htft_2_x",
    "htft_2_2",
]
ODDS_FIELDS = BASE_ODDS_FIELDS + ODDS_COLUMNS
FIXTURE_RE = re.compile(r"^(?P<home>[A-Z]{2,4})\s*(?:-|\u2013)\s*(?P<away>[A-Z]{2,4})$")
DECIMAL_RE = re.compile(r"\b\d+(?:\.\d{1,3})\b")
MATCHDAY_RE = re.compile(r"Matchday\s*(?P<matchday>\d+)", re.IGNORECASE)


@dataclass
class OddsCollectionResult:
    rows: int
    markets_seen: list[str]
    snapshot_path: Path | None
    daily_path: Path | None
    history_path: Path | None
    final_path: Path | None


def collect_odds(settings: Settings, keep_open: bool = False) -> OddsCollectionResult:
    settings.user_data_dir.mkdir(parents=True, exist_ok=True)
    all_odds_dir(settings).mkdir(parents=True, exist_ok=True)
    final_odds_dir(settings).mkdir(parents=True, exist_ok=True)

    captured_at = now_iso()
    playwright, context, page = open_context(settings)
    try:
        open_upcoming_page(page, settings)
        metadata = read_page_metadata(page, settings)
        raw_rows = []
        markets_seen = []

        for market in MARKETS:
            if select_market(page, market, settings.action_delay_ms):
                market_rows = extract_visible_odds(page, settings, captured_at, metadata, market)
                if market_rows:
                    markets_seen.append(market)
                    raw_rows.extend(market_rows)
                    logging.info("Captured %s odds rows for market %s.", len(market_rows), market)
                else:
                    logging.info("Market %s opened, but no odds rows were parsed.", market)

        if not raw_rows:
            write_page_diagnostics(settings, page)
            logging.warning("Wrote page diagnostics to %s.", settings.log_path.parent / "last_page_text.txt")
            if page_requires_login(page):
                logging.error(
                    "The odds page opened without the saved logged-in session. "
                    "Run python collect_virtuals.py login, log in, then run collect_odds.py again."
                )

        rows = pivot_odds_rows(raw_rows)
        snapshot_path = daily_path = history_path = final_path = None
        if rows:
            captures_dir = all_odds_dir(settings)
            snapshot_path = write_snapshot(captures_dir, rows, captured_at)
            daily_path = append_csv(captures_dir / f"odds_captured_on_{local_date_stamp()}.csv", rows)
            history_path = append_csv(captures_dir / "odds_history.csv", rows)
            final_path = upsert_final_odds(final_odds_dir(settings) / "final_odds.csv", rows)
        else:
            logging.warning("No odds were captured from the currently visible upcoming page.")

        return OddsCollectionResult(
            rows=len(rows),
            markets_seen=markets_seen,
            snapshot_path=snapshot_path,
            daily_path=daily_path,
            history_path=history_path,
            final_path=final_path,
        )
    finally:
        if keep_open:
            print("Keeping browser open. Press Enter here when you want to close it.")
            input()
        context.close()
        playwright.stop()


def open_upcoming_page(page: Page, settings: Settings) -> None:
    page.goto(upcoming_url(settings), wait_until="domcontentloaded", timeout=60_000)
    try:
        page.wait_for_load_state("networkidle", timeout=settings.page_ready_timeout_ms)
    except PlaywrightTimeoutError:
        logging.info("Network did not become idle; continuing with visible page checks.")

    _click_text_if_visible(page, settings.league_name)
    page.wait_for_timeout(settings.action_delay_ms)
    wait_for_any_fixture(page, settings.page_ready_timeout_ms)


def upcoming_url(settings: Settings) -> str:
    query = urlencode({"virtualTab": "upcoming", "leagueId": settings.league_id})
    return f"https://www.betpawa.ug/virtual-sports?{query}"


def wait_for_any_fixture(page: Page, timeout_ms: int) -> None:
    try:
        page.wait_for_function(
            """
            () => /[A-Z]{2,4}\\s*[-–]\\s*[A-Z]{2,4}/.test(document.body.innerText || '')
            """,
            timeout=timeout_ms,
        )
    except PlaywrightTimeoutError:
        logging.warning("No visible fixture text appeared before timeout.")


def select_market(page: Page, market: str, delay_ms: int) -> bool:
    locator = page.get_by_text(market, exact=True)
    try:
        if locator.count() == 0:
            return False
        locator.first.click(timeout=10_000)
        page.wait_for_timeout(delay_ms)
        return True
    except PlaywrightTimeoutError:
        logging.warning("Could not select odds market %s.", market)
        return False


def read_page_metadata(page: Page, settings: Settings) -> dict:
    text = page.locator("body").inner_text(timeout=15_000)
    matchday_match = MATCHDAY_RE.search(text)
    return {
        "source_url": page.url,
        "league_id": settings.league_id,
        "league_name": settings.league_name,
        "matchday": int(matchday_match.group("matchday")) if matchday_match else None,
        "start_label": read_start_label(text),
    }


def read_start_label(text: str) -> str:
    compact = " ".join(line.strip() for line in text.splitlines() if line.strip())
    match = re.search(r"Starts in:\s*([0-9:]+)", compact, re.IGNORECASE)
    if match:
        return f"Starts in: {match.group(1)}"
    match = re.search(r"\b\d{1,2}:\d{2}\s*(?:AM|PM)\b", compact, re.IGNORECASE)
    return match.group(0) if match else ""


def page_requires_login(page: Page) -> bool:
    try:
        text = page.locator("body").inner_text(timeout=5_000)
    except PlaywrightTimeoutError:
        return False
    compact = " ".join(line.strip().upper() for line in text.splitlines() if line.strip())
    return "LOGIN" in compact and "JOIN NOW" in compact and "MATCHDAY" not in compact


def extract_visible_odds(
    page: Page,
    settings: Settings,
    captured_at: str,
    metadata: dict,
    market: str,
) -> list[dict]:
    cards = page.evaluate(
        """
        () => {
          const fixturePattern = /^[A-Z]{2,4}\\s*[-–]\\s*[A-Z]{2,4}$/;
          const decimalPattern = /\\b\\d+(?:\\.\\d{1,3})\\b/;

          function visible(element) {
            const style = window.getComputedStyle(element);
            const rect = element.getBoundingClientRect();
            return style.visibility !== 'hidden'
              && style.display !== 'none'
              && rect.width > 0
              && rect.height > 0;
          }

          function fixtureCount(text) {
            return text.split(/\\n+/).filter((line) => fixturePattern.test(line.trim())).length;
          }

          const fixtureNodes = Array.from(document.querySelectorAll('body *')).filter((node) => {
            const text = (node.innerText || '').trim();
            return visible(node) && fixturePattern.test(text);
          });

          const cards = [];
          const seen = new Set();

          for (const fixtureNode of fixtureNodes) {
            let card = fixtureNode;
            let cursor = fixtureNode.parentElement;

            while (cursor && cursor !== document.body) {
              const text = cursor.innerText || '';
              if (fixtureCount(text) !== 1 || !decimalPattern.test(text)) {
                break;
              }
              card = cursor;
              cursor = cursor.parentElement;
            }

            const rect = card.getBoundingClientRect();
            const key = `${Math.round(rect.top)}:${Math.round(rect.left)}:${(card.innerText || '').trim()}`;
            if (!seen.has(key)) {
              seen.add(key);
              cards.push((card.innerText || '').trim());
            }
          }

          return cards;
        }
        """
    )

    rows = []
    for card_text in cards:
        fixture, selections = parse_card_text(card_text, market)
        if not fixture:
            continue
        home_team, away_team = fixture
        if not _valid_fixture(home_team, away_team):
            continue
        for column, odd in selections:
            rows.append(
                {
                    "captured_at": captured_at,
                    "source_url": metadata["source_url"],
                    "league_id": metadata["league_id"],
                    "league_name": metadata["league_name"],
                    "matchday": metadata["matchday"],
                    "start_label": metadata["start_label"],
                    "home_team": home_team,
                    "away_team": away_team,
                    "column": column,
                    "odd": odd,
                }
            )

    return rows


def parse_card_text(card_text: str, market: str = "1X2") -> tuple[tuple[str, str] | None, list[tuple[str, str]]]:
    lines = [re.sub(r"\s+", " ", line).strip() for line in card_text.splitlines()]
    lines = [line for line in lines if line and line not in {">", "\u203a", "\u276f"}]

    fixture = None
    fixture_index = -1
    for index, line in enumerate(lines):
        fixture_match = FIXTURE_RE.match(line)
        if fixture_match:
            fixture = (fixture_match.group("home"), fixture_match.group("away"))
            fixture_index = index
            break

    if not fixture:
        return None, []

    market_lines = lines[fixture_index + 1 :]
    if market == "O/U":
        return fixture, parse_over_under_lines(market_lines)

    selections = []
    pending_label = ""
    for line in market_lines:
        if FIXTURE_RE.match(line):
            break

        odds = list(DECIMAL_RE.finditer(line))
        if not odds:
            pending_label = line
            continue

        for odd_match in odds:
            odd = odd_match.group(0)
            label = line[: odd_match.start()].strip()
            label = clean_selection_label(label or pending_label)
            column = odds_column(market, label)
            if column:
                selections.append((column, odd))
            pending_label = ""

    return fixture, selections


def parse_over_under_lines(lines: list[str]) -> list[tuple[str, str]]:
    selections = []
    index = 0

    while index < len(lines):
        line = lines[index]
        if FIXTURE_RE.match(line):
            break

        combined_match = re.match(
            r"^(Over|Under)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)$",
            line,
            re.IGNORECASE,
        )
        if combined_match:
            column = over_under_column(combined_match.group(1), combined_match.group(2))
            selections.append((column, combined_match.group(3)))
            index += 1
            continue

        label_match = re.match(r"^(Over|Under)(?:\s+(\d+(?:\.\d+)?))?$", line, re.IGNORECASE)
        if label_match:
            side = label_match.group(1)
            line_value = label_match.group(2)
            odd = None
            consumed = 1

            if line_value is None and index + 1 < len(lines) and is_decimal(lines[index + 1]):
                line_value = lines[index + 1]
                consumed += 1

            if line_value is not None and index + consumed < len(lines) and is_decimal(lines[index + consumed]):
                odd = lines[index + consumed]
                consumed += 1

            if line_value is not None and odd is not None:
                selections.append((over_under_column(side, line_value), odd))
            index += consumed
            continue

        index += 1

    return selections


def clean_selection_label(label: str) -> str:
    label = re.sub(r"\s+", " ", label).strip()
    label = label.strip("-:|")
    return label


def odds_column(market: str, selection: str) -> str | None:
    normalized = selection.strip().lower()

    if market == "1X2":
        mapping = {"1": "odd_1", "x": "odd_x", "2": "odd_2"}
        return mapping.get(normalized)

    if market == "BTTS":
        mapping = {"yes": "btts_yes", "no": "btts_no"}
        return mapping.get(normalized)

    if market == "DC":
        mapping = {"1x": "dc_1x", "x2": "dc_x2", "12": "dc_12"}
        return mapping.get(normalized)

    if market == "HT/FT":
        return "htft_" + normalized.replace("/", "_") if re.fullmatch(r"[12x]/[12x]", normalized) else None

    return None


def over_under_column(side: str, line_value: str) -> str:
    safe_line = line_value.replace(".", "_")
    return f"ou_{side.lower()}_{safe_line}"


def is_decimal(value: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", value.strip()))


def pivot_odds_rows(rows: list[dict]) -> list[dict]:
    fixtures = {}

    for row in rows:
        key = (
            row["captured_at"],
            row["source_url"],
            row["league_id"],
            row["league_name"],
            row["matchday"],
            row["start_label"],
            row["home_team"],
            row["away_team"],
        )
        if key not in fixtures:
            fixtures[key] = {
                "captured_at": row["captured_at"],
                "source_url": row["source_url"],
                "league_id": row["league_id"],
                "league_name": row["league_name"],
                "matchday": row["matchday"],
                "start_label": row["start_label"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                **{column: "" for column in ODDS_COLUMNS},
            }

        column = row["column"]
        if column in ODDS_COLUMNS:
            fixtures[key][column] = row["odd"]

    return sorted(
        fixtures.values(),
        key=lambda row: (
            str(row["captured_at"]),
            int(row["matchday"] or 0),
            str(row["home_team"]),
            str(row["away_team"]),
        ),
    )


def all_odds_dir(settings: Settings) -> Path:
    return settings.odds_dir / "all_odds_files"


def final_odds_dir(settings: Settings) -> Path:
    return settings.odds_dir / "final_odds"


def upsert_final_odds(path: Path, new_rows: list[dict]) -> Path:
    existing_rows = read_wide_odds_csv(path)
    by_fixture = {(row["home_team"], row["away_team"]): row for row in existing_rows}
    order = [(row["home_team"], row["away_team"]) for row in existing_rows]

    for row in new_rows:
        key = (row["home_team"], row["away_team"])
        if key not in by_fixture:
            order.append(key)
        by_fixture[key] = {field: row.get(field, "") for field in ODDS_FIELDS}

    merged_rows = [by_fixture[key] for key in order]
    try:
        write_csv(path, merged_rows)
        return path
    except PermissionError:
        fallback_path = path.with_name("final_odds_latest.csv")
        write_csv(fallback_path, merged_rows)
        logging.warning(
            "Could not update %s because it is probably open in Excel. Wrote %s instead.",
            path,
            fallback_path,
        )
        return fallback_path


def read_wide_odds_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ODDS_FIELDS:
            logging.warning("Ignoring incompatible final odds file %s.", path)
            return []
        return [{field: row.get(field, "") for field in ODDS_FIELDS} for row in reader]


def write_snapshot(odds_dir: Path, rows: list[dict], captured_at: str) -> Path:
    stamp = datetime.fromisoformat(captured_at).astimezone().strftime("%Y%m%d_%H%M%S")
    path = odds_dir / f"odds_snapshot_{stamp}.csv"
    write_csv(path, rows)
    return path


def append_csv(path: Path, rows: list[dict]) -> Path:
    path = ensure_compatible_csv(path)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ODDS_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    return path


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ODDS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def ensure_compatible_csv(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        return path

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])

    if header == ODDS_FIELDS:
        return path

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    legacy_path = path.with_name(f"{path.stem}_long_legacy_{stamp}{path.suffix}")
    try:
        path.replace(legacy_path)
        logging.warning("Archived old long-format odds file as %s.", legacy_path)
        return path
    except PermissionError:
        wide_path = path.with_name(f"{path.stem}_wide_{stamp}{path.suffix}")
        logging.warning(
            "Could not archive locked long-format odds file %s. Writing wide odds to %s instead.",
            path,
            wide_path,
        )
        return wide_path


def local_date_stamp() -> str:
    return datetime.now().strftime("%d_%m_%Y")
