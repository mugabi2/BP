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

## Train the BTTS model (`notebooks/01_btts_model_training.ipynb`)

The training notebook supports two historical data sources via `src/training_data.py` and logs experiment runs via `src/mlflow_tracking.py`:

```text
Historical data (CSV or VPS PostgreSQL)
      ↓
Laptop notebook (notebooks/01_btts_model_training.ipynb)
      ↓
Existing BTTS training + candidate model comparison
      ↓
Local model files (models/btts_model.joblib + models/btts_feature_columns.json)
      ↓
MLflow tracking (optional / non-blocking)
```

### Mode B — Local CSV training (works offline / without VPS)

Set in `.env` or your shell before launching Jupyter:

```powershell
$env:TRAINING_DATA_SOURCE="csv"
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
```

- Loads historical match results from `Match day results/match_results.csv`.
- Trains all 6 candidate models using the `feature_cutoff_matchday = target_matchday - 2` rule.
- Always writes `models/btts_model.joblib` and `models/btts_feature_columns.json` for local Streamlit inference.
- If `MLFLOW_TRACKING_URI` cannot be reached, the notebook emits a clear warning and finishes normally.

### Mode A — VPS PostgreSQL + VPS MLflow (via SSH tunnel)

When VPS access is available, open an SSH tunnel from your laptop to the VPS and set:

```powershell
$env:TRAINING_DATA_SOURCE="postgres"
$env:DATABASE_URL="postgresql://bp_user:YOUR_POSTGRES_PASSWORD@127.0.0.1:5433/bp"
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
$env:MLFLOW_EXPERIMENT_NAME="BTTS Prediction"
$env:MLFLOW_REGISTERED_MODEL_NAME="btts-predictor"
```

See `VPS_DEPLOYMENT.md` for the exact SSH tunnel and verification commands.

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

## MLflow

The VPS Docker stack includes an MLflow Tracking Server for centralized experiment and model tracking.

- Tracking metadata database: `mlflow_app`
- Artifact storage: persistent Docker volume `mlflow_artifacts`
- VPS binding: `127.0.0.1:5000` by default
- Recommended access: SSH tunnel from the laptop

MLflow is not in the live prediction path. Streamlit continues to load the local `models/btts_model.joblib` model.

See `VPS_DEPLOYMENT.md` for deployment and access commands.

## Automated VPS training

The optional `trainer` Compose profile runs the same feature engineering,
six-model comparison, winner selection, Pick/Skip evaluation, and MLflow
logging without requiring a laptop or Jupyter session:

```bash
docker compose --profile training build trainer
docker compose --profile training run --rm trainer
```

The trainer reads PostgreSQL over Docker's private network and writes a
candidate bundle to `training-output/latest-candidate/`. It does not replace
the working Streamlit model automatically. After reviewing the run and model
version in MLflow, promote the approved bundle on Windows with:

```powershell
.\scripts\fetch_vps_model.ps1 `
  -VpsHost root@YOUR_VPS_IP `
  -Approve
```

The script backs up the current local model before copying the approved
`btts_model.joblib` and `btts_feature_columns.json` into `models/`. See
`VPS_DEPLOYMENT.md` for the weekly systemd timer setup.
