# NFL Ratings

A Streamlit app for opponent-adjusted NFL rankings and game forecasts.
Schedules and play-by-play arrive automatically through `nflreadpy`; no CSV,
Excel files, or API keys are needed. The first load requires internet access.

## Explore

- **Rankings:** Team strength through a selected week, with supplementary
  offense and defense columns after Game variability.
- **Weekly forecasts:** Projected point margins and comparisons with market lines.
- **Team history:** Weekly ratings and opponent-adjusted game performances,
  with a prominent aggregate and distinct offense/defense colors and legends.
- **Backtesting:** Forecast errors, winner accuracy, and against-spread results.

Choose the season and postseason setting in the sidebar. Expand **Model
settings** to edit weights, normalization ranges, venue adjustments, prior
strength, and forecast requirements, then select **Apply settings**. Prepared
data are cached for up to one hour; **Refresh data** requests a fresh download.

Offense and defense sum to the existing Rating and do not change forecasts.
Higher is better for both components. The split uses the existing prior strength
and observed league efficiency through each cutoff, with no new tuning controls.
See the [model description](docs/peter-model.md#supplementary-offense-and-defense)
for assumptions and units.

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

## Development and publishing

Install `pytest` and run `python -m pytest -q` from the repository root to check
the app, model, data validation, and component decomposition.

The live app is [nflratings.streamlit.app](https://nflratings.streamlit.app/).
Changes must reach this repository's deployed branch before they appear there;
editing a local folder alone does not publish anything. Follow the
[GitHub workflow](docs/github-workflow.md) to review and publish updates.
