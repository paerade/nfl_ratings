# NFL Ratings

A Streamlit app for opponent-adjusted NFL rankings and game forecasts.
Schedules and play-by-play arrive automatically through `nflreadpy`; no CSV,
Excel files, or API keys are needed. The first load requires internet access.

## Explore

- **Rankings:** Team strength through a selected week.
- **Weekly forecasts:** Projected point margins and comparisons with market lines.
- **Team history:** Weekly ratings and opponent-adjusted game performances.
- **Backtesting:** Forecast errors, winner accuracy, and against-spread results.

Choose the season and postseason setting in the sidebar. Expand **Model
settings** to edit weights, normalization ranges, venue adjustments, prior
strength, and forecast requirements, then select **Apply settings**. Prepared
data are cached for up to one hour; **Refresh data** requests a fresh download.

Forecasts estimate point margins, not individual scores or totals. Backtests
rebuild ratings using prior weeks only. Historical data may receive later
corrections, and results do not establish betting profitability.

## Run locally

Use Python 3.12. From the extracted project folder in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local address shown in the terminal. Stop with Ctrl+C.
