# betPawa Virtuals Dataset Collector

This project collects betPawa Virtual English League table data for study purposes.
It uses Playwright so you can log in once manually and reuse the same browser session.

It saves table data only. It does not save screenshots.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
```

The default `.env` points to your installed Chrome with `CHROME_PATH`.
If Chrome is installed somewhere else, update that path.
If you prefer Playwright's bundled Chromium, remove `CHROME_PATH` and run `playwright install chromium`.

## Run the Streamlit app

From PowerShell:

```powershell
cd "D:\Codex things\BP"
C:\Users\SAMUEL\anaconda3\python.exe -m streamlit run app.py
```

Open the local URL Streamlit prints, usually:

```text
http://localhost:8501
```

In the app, use the manual prediction flow:

1. Click `1. Update odds`.
2. Paste fixture screenshots from the clipboard.
3. Paste the standings table text.
4. Click `2. Update live results`.
5. Click `Predict BTTS`.

Each prediction run updates the single live predictions file:

- `data/outputs/btts_predictions_live_master.csv`

## Login once

```powershell
python collect_virtuals.py login
```

A browser will open. Log in manually, then return to the terminal and press Enter.
The browser session is saved in `auth/browser-profile`.

## Collect data

```powershell
python collect_virtuals.py collect
```

If the browser closes too quickly or no seasons are found, run:

```powershell
python collect_virtuals.py collect --keep-open
```

That keeps Chrome open until you press Enter in the terminal.

Outputs:

- `data/betpawa_virtuals_master.sqlite`
- `data/exports/seasons_master.csv`
- `data/match_day_results/match_results_master.csv`
- `data/match_day_results/match_results_captured_on_DD_MM_YYYY.csv`
- `logs/collector.log`

If a season is already fully captured, the collector skips it on future runs.

## Schedule suggestion

A 20-team, 34-matchday virtual season at 5 minutes per matchday takes about 2 hours 50 minutes.
Run the collector about every 3 hours, or every 2.5 hours if you want a safety margin.

Repeated runs are safe because SQLite uses uniqueness keys to update existing rows instead of duplicating them.

## Collect live season results for prediction

```powershell
python collect_live_results.py --target-matchday 25
```

This keeps one raw current-season results file for the target matchday:

- `data/live_seasons/season_138605_tgt_md25_results.csv`

The file contains only match results through `target_matchday - 2`. For example, target matchday 25 writes results through matchday 23. If an earlier run already saved matchdays 1-16 for the same season, the next run only fetches the missing matchdays such as 17-23.

You can also pass the season explicitly:

```powershell
python collect_live_results.py --season-id 138605 --target-matchday 25
```

This file is intended for Streamlit to load and compute model features itself. It does not include odds or engineered features.

## Collect odds

```powershell
python collect_odds.py
```

The odds collector opens the upcoming virtual fixtures page and snapshots the currently displayed odds into the `data/odds/all_odds_files` folder.
Each row is one fixture for one capture time. Odds are stored in columns such as `odd_1`, `odd_x`, `odd_2`, `ou_over_2_5`, `btts_yes`, `dc_1x`, and `htft_1_1`.

Outputs:

- `data/odds/all_odds_files/odds_snapshot_YYYYMMDD_HHMMSS.csv`
- `data/odds/all_odds_files/odds_captured_on_DD_MM_YYYY.csv`
- `data/odds/all_odds_files/odds_history_master.csv`
- `data/odds/final_odds/final_odds_master.csv`

`final_odds_master.csv` is the source-of-truth file. On every run, the collector looks for the same `home_team` and `away_team`; if the fixture exists, it updates that row with the newest odds, and if it is new, it appends it at the bottom.

If you have older long-format odds CSVs with `selection` and `odd` rows, convert them with:

```powershell
python convert_odds_to_wide.py
```

The converter archives the old file with `_long_legacy_...` in its name, then writes the corrected wide CSV at the original filename.

The upcoming page usually exposes the next matchday, not a full historical season of prices. For that reason, run this script frequently while the virtual league is active. A 1-minute schedule is the safest practical interval because matchdays rotate quickly and repeated captures create an odds history without missing the short pre-match window. If you only need one price per matchday, every 3-4 minutes is lighter, but it has a higher risk of missing a matchday when the page rolls over.

## Notes

- Start with `HEADLESS=false` in `.env` so you can watch the browser while testing.
- After the collector is stable, you can try `HEADLESS=true`.
- If the site changes its layout, the Playwright selectors/parsing in `src/scraper.py` may need adjustment.
