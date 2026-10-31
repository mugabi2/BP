# betPawa Virtuals Dataset Collector

This project collects betPawa Virtual English League data for study purposes.
The main results collector uses Playwright to read the public virtual-results pages and stores collected results in PostgreSQL.

It saves table data only. It does not save screenshots.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
```

On Linux/Docker, leave `BROWSER_CHANNEL` and `CHROME_PATH` blank to use Playwright's bundled Chromium.
For local Windows debugging, you may point those settings to an installed Chrome.

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

- `outputs/btts_predictions_live.csv`

## Collect historical virtual results

```powershell
python collect_virtuals.py collect
```

If you want to inspect the browser during a local run:

```powershell
python collect_virtuals.py collect --keep-open
```

The main historical collector writes directly to PostgreSQL. It no longer uses SQLite or CSV exports for collected match results.

If a season is already fully captured, the collector skips it on future runs.

## VPS schedule

The historical-results collector runs once per day at a fixed wall-clock time.

Default:

```text
COLLECT_TIME=13:00
COLLECT_TIMEZONE=Africa/Kampala
```

That is 1:00 PM Kampala time every day. Successful runs then stay idle until the next day's 1:00 PM schedule. Failed runs retry using `COLLECT_RETRY_SECONDS`. A manual harvest can also be started at any time with `docker compose exec collector python collect_virtuals.py collect`. Overlapping runs are prevented with a PostgreSQL advisory lock.

## Collect live season results for prediction

```powershell
python collect_live_results.py --target-matchday 25
```

This keeps one raw current-season results file for the target matchday:

- `data/live_seasons/season_138605_tgt_md25_results.csv`

The file contains only match results through `target_matchday - 2`. For example, target matchday 25 writes results through matchday 23.

## Collect odds

```powershell
python collect_odds.py
```

The odds collector opens the upcoming virtual fixtures page and snapshots the currently displayed odds into the `odds/all_odds_files` folder.

Odds collection is a separate workflow from the PostgreSQL historical-results collector.

## VPS deployment

See `VPS_DEPLOYMENT.md`.

## Notes

- The VPS collector runs headless.
- The public historical-results collection does not require login or a saved browser session.
- If the site changes its layout, the Playwright selectors/parsing in `src/scraper.py` may need adjustment.


## Metabase

The VPS Docker stack includes Metabase for visual exploration of PostgreSQL data.

```text
Metabase: http://YOUR_VPS_IP:3000
Data source host inside Docker: postgres
Data source database: bp
```

Metabase application metadata is stored separately in `metabase_app`.
