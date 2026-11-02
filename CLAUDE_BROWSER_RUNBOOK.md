# BP Predictor: Two-Site Operating Instructions

This activity uses two browser tabs:

1. **BetPawa** supplies the fixtures, screenshots, standings, results, and the betting page.
2. **Streamlit** (`http://localhost:8501`) receives the information manually copied from BetPawa and produces BTTS predictions.

Nothing is transferred automatically from BetPawa into Streamlit. The agent must copy it as described below.

## Rules

- Confidence threshold: **67%**.
- Stake: **UGX 2 on each qualifying match**, placed as a single bet.
- Stop immediately after bets have been successfully placed on **50 different matches**.
- Never bet on the same season, matchday, and fixture twice.
- One virtual matchday lasts about **5 minutes** before the next matchday appears.
- If no prediction reaches 67%, place no bet. Wait for the next matchday and repeat.
- Never guess missing fixtures, standings, predictions, scores, or results.

## 1. Start Streamlit

Open PowerShell and run:

```powershell
cd "D:\Codex things\BP"
.\env_bp\Scripts\python.exe -m streamlit run app.py
```

Wait for Streamlit to display its local address, normally `http://localhost:8501`, and open it in Chrome.

## 2. Open the two browser tabs

- Tab 1: BetPawa Virtual Sports, English League.
- Tab 2: Streamlit BTTS Predictor at `http://localhost:8501`.

In Streamlit, select **Predict BTTS** and set **pick_threshold** to **67%**.

## 3. Prepare one matchday

Follow the instructions displayed in Streamlit:

If this is the first use of a new season, click **First-time season setup (full history)** and wait until it finishes.
1. In Streamlit, click **Clear screenshots** if screenshots from a previous matchday remain.
2. In Streamlit, click **1. Update odds** and wait until it finishes. Do not press it twice while it is running.
3. In BetPawa, open the current English League upcoming fixtures.
4. Take screenshots to capture all fixtures for the current matchday.
5. Switch to Streamlit, click **Paste screenshot**, and paste every fixture screenshot. This is a manual transfer from BetPawa to Streamlit.
6. Return to BetPawa and open **Results → English League → Standings**.
7. Copy the standings text containing the current matchday, the `Season #...` line, like you do a control a . to select all text on that page.
8. Switch to Streamlit and paste it into **Standings table text**. This is also a manual transfer.
9. Confirm Streamlit detects the correct `season_id`, `target_matchday`, and `feature_cutoff_matchday`.
10. In Streamlit, click **2. Update live results** and wait for Playwright to finish.


Do not refresh Streamlit or press another collection button while Playwright is working. Continue only when the spinner stops and Streamlit displays the live-results file and a ready/compact status.

The expected status is 20 standings teams and 64 model features. A complete matchday normally has 10 fixtures. If Streamlit captures 9, warn about the missing fixture but continue with predictions for the 9 captured fixtures; do not fail the entire run.

## 4. Generate predictions

1. Click **Predict BTTS**.
2. Wait for the **BTTS Predictions** table.
3. Use only predictions whose confidence is **at least 67%**.
4. Ignore every prediction below 67%.
5. If none qualify, wait for the next matchday, which normally appears within about 5 minutes, and restart at section 3.

## 5. Place each qualifying bet on BetPawa

For every qualifying Streamlit prediction:

1. Switch to BetPawa and locate the exact fixture in the current matchday.
2. Open the fixture's markets and find **Both Teams To Score**.
3. Select the exact outcome predicted by Streamlit: **Yes** or **No**.
4. Ensure the betslip contains only that match so it is a single bet.
5. Enter a stake of **UGX 2**.
6. Verify the fixture, matchday, BTTS selection, stake, and confidence of at least 67%.
7. Place the bet and wait for BetPawa to confirm it was accepted.
8. Immediately add it to the tracking file.
9. if more that 1 matches have above threshold confidence, place the bets seperately

Do not place the bet if the fixture has started, moved to another matchday, the BTTS market is unavailable, confidence is below 67%, or BetPawa does not accept UGX 2. Record the reason and continue with the next valid match or matchday.

## 6. Keep the prediction and outcome file

Maintain:

```text
D:\Codex things\BP\data\claude_prediction_tracking.csv
```

Use these columns:

```text
predicted_at,season_id,matchday,fixture,prediction,confidence_pct,stake_ugx,bet_placed,bet_reference,final_score,actual_btts,prediction_correct,bet_outcome,settled_at,notes
```

Add one row immediately after every qualifying bet attempt. When the result is available, update that row with:

- `final_score`;
- `actual_btts` — `Yes` when both teams scored, otherwise `No`;
- `prediction_correct` — `Yes` when the prediction matches the actual BTTS result, otherwise `No`;
- `bet_outcome` — `Won`, `Lost`, `Void`, or `Rejected`;
- `settled_at`.

Count only rows where `bet_placed` is `Yes` toward the 50-match limit. Stop when that count reaches 50. At UGX 2 per match, the maximum total stake is UGX 100.

## Instruction for Claude Code with browser control

```text
Follow CLAUDE_BROWSER_RUNBOOK.md exactly. Work between the BetPawa tab and the Streamlit tab. Manually copy fixture screenshots and standings from BetPawa into Streamlit. Set pick_threshold to 67%. Wait for every Streamlit/Playwright operation to finish. If this is the first use of the season, click First-time season setup and wait; otherwise click Update live results and wait. Generate BTTS predictions and place only single UGX 2 bets whose confidence is at least 67%, using the exact Yes/No outcome predicted by Streamlit. If no match qualifies, wait for the next matchday; matchdays change about every 5 minutes. Maintain data/prediction_tracking.csv, update outcomes when results appear, prevent duplicate fixture bets, and stop immediately after 50 successfully placed match bets.
```
