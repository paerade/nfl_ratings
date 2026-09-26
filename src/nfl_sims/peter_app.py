"""Compact Streamlit interface for the Peter Score model."""

from dataclasses import asdict
from datetime import date
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from nfl_sims.peter_data import DataError, SeasonData, load_season_data
from nfl_sims.peter_model import (
    ModelConfig, backtest, fit_ratings, forecast_week, summarize_backtest,
)

VIEWS = ["Rankings", "Weekly forecasts", "Team history", "Backtesting"]
COLORS = {"positive": "#2DD4BF", "negative": "#F0B56B"}


@st.cache_data(ttl=3600, max_entries=3, show_spinner=False)
def season_data(season: int, refresh_token: int = 0) -> SeasonData:
    """Cache only the small aggregate, never raw play-by-play in session state."""
    return load_season_data(season, refresh=refresh_token > 0)


def reset_model() -> None:
    for key in list(st.session_state):
        if key.startswith("constant_"):
            del st.session_state[key]
    st.session_state["model_config"] = ModelConfig()


def model_controls() -> ModelConfig:
    st.session_state.setdefault("model_config", ModelConfig())
    current = st.session_state["model_config"]
    with st.sidebar.expander("Model settings"):
        st.caption("Changes take effect when you select Apply settings. Defaults are a starting point, not fitted to the selected backtest.")
        with st.form("model_settings"):
            values = {}

            def number(label, name, step, minimum=None):
                kwargs = {} if minimum is None else {"min_value": minimum}
                if isinstance(getattr(current, name), float):
                    kwargs["format"] = "%.4f" if "adjustment" in name else "%.3f" if name.endswith(("_min", "_max")) else "%.2f"
                values[name] = st.number_input(
                    label, value=getattr(current, name), step=step,
                    key=f"constant_{name}", **kwargs,
                )

            number("Points per rating unit", "points_per_rating", 1.0, 0.1)
            number("Home advantage (points)", "home_advantage", 0.1, 0.0)
            number("Prior strength (equivalent games)", "prior_games", 0.25, 0.0)
            st.caption("One average-performance prior game steadies early ratings. Set to 0 for the original unregularized equations.")
            number("Games required per team to forecast", "min_games", 1, 1)
            st.markdown("**Efficiency weights**")
            for metric, label in [("ypp", "Yards per play"), ("fpp", "First downs per play"), ("tdc", "Third-down conversion")]:
                number(f"{label} weight", f"{metric}_weight", 0.1, 0.0)
            st.markdown("**Normalization ranges**")
            st.caption("FPP and third-down rates are fractions: 0.40 means 40%.")
            for metric, label in [("ypp", "YPP"), ("fpp", "FPP"), ("tdc", "Third downs")]:
                left, right = st.columns(2)
                with left:
                    number(f"{label} minimum", f"{metric}_min", 0.01)
                with right:
                    number(f"{label} maximum", f"{metric}_max", 0.01)
            st.markdown("**Venue adjustments**")
            st.caption("Added to away efficiency and subtracted from home efficiency before fitting. Neutral sites receive no adjustment.")
            number("YPP venue adjustment", "ypp_venue_adjustment", 0.001, 0.0)
            number("FPP venue adjustment", "fpp_venue_adjustment", 0.0001, 0.0)
            number("Third-down venue adjustment", "tdc_venue_adjustment", 0.001, 0.0)
            if st.form_submit_button("Apply settings", use_container_width=True):
                try:
                    st.session_state["model_config"] = ModelConfig(**values)
                    st.success("Settings applied.")
                except ValueError as exc:
                    st.error(str(exc))
        st.button("Restore defaults", on_click=reset_model, use_container_width=True)
    return st.session_state["model_config"]


def choose_week(label: str, key: str, weeks: list[int], default: int) -> int:
    if key in st.session_state and st.session_state[key] not in weeks:
        del st.session_state[key]
    return st.selectbox(label, weeks, index=weeks.index(default), key=key,
                        format_func=lambda value: f"Week {value}" if value else "Before Week 1")


def formatted(value, digits=2, suffix="") -> str:
    return "—" if value is None or pd.isna(value) else f"{value:,.{digits}f}{suffix}"


def download_results(frame: pd.DataFrame, filename: str, config: ModelConfig) -> None:
    """Optional outputs; no file is needed to run the app."""
    with st.expander("Export results and settings"):
        st.download_button("Download results", frame.to_csv(index=False), filename,
                           "text/csv", key=f"download_{filename}")
        st.download_button("Download model settings", json.dumps(asdict(config), indent=2),
                           "model_settings.json", "application/json", key=f"settings_{filename}")


def diagnostics(result) -> None:
    for warning in result.diagnostics.get("warnings", []):
        st.warning(warning)
    pending = result.diagnostics.get("pending_games", 0)
    if pending:
        st.caption(f"{pending} scheduled game(s) through this week are awaiting final results.")


def show_rankings(games, season, weeks, default_week, config) -> None:
    cutoff = choose_week("Rankings through", "rank_week", [0] + weeks, default_week)
    result = fit_ratings(games, season, cutoff, config)
    ranked = result.ratings.dropna(subset=["rating"]).sort_values("rating", ascending=False).copy()
    columns = st.columns(3)
    columns[0].metric("Rated teams", len(ranked))
    columns[1].metric("Games used", int(result.ratings["games"].sum() // 2))
    columns[2].metric("Through", f"Week {cutoff}" if cutoff else "Preseason")
    diagnostics(result)
    if ranked.empty:
        st.info("No completed games with usable statistics are available through this week.")
        return
    ranked.insert(0, "rank", ranked["rating"].rank(method="min", ascending=False).astype(int))
    ranked["points_vs_average"] = ranked["rating"] * config.points_per_rating
    table, plot = st.columns([1.15, 1])
    with table:
        st.dataframe(ranked[["rank", "team", "rating", "points_vs_average", "games", "performance_std"]],
                     hide_index=True, use_container_width=True, height=650,
                     column_config={"rank": "Rank", "team": "Team", "games": "Games",
                                    "rating": st.column_config.NumberColumn("Rating", format="%.4f"),
                                    "points_vs_average": st.column_config.NumberColumn("Points vs average", format="%+.1f"),
                                    "performance_std": st.column_config.NumberColumn("Game variability", format="%.4f")})
    with plot:
        st.altair_chart(alt.Chart(ranked).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("team:N", sort=ranked["team"].tolist(), title=None),
            x=alt.X("points_vs_average:Q", title="Points above / below average"),
            color=alt.condition(alt.datum.points_vs_average >= 0, alt.value(COLORS["positive"]), alt.value(COLORS["negative"])),
            tooltip=["team", alt.Tooltip("rating:Q", format=".4f"), "games", alt.Tooltip("points_vs_average:Q", format="+.1f")],
        ).properties(height=600), use_container_width=True)
    st.caption("Ratings use completed games through the selected week, including completed games in a partially played week. Game variability is the sample standard deviation of actual opponent-adjusted game performances; bye weeks are omitted.")
    unplayed = result.ratings.loc[result.ratings["games"].eq(0), "team"].tolist()
    if unplayed:
        st.caption("No usable games yet: " + ", ".join(unplayed))
    download_results(ranked, f"rankings_{season}_week_{cutoff}.csv", config)


def forecast_display(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["matchup"] = result["away_team"] + " @ " + result["home_team"]
    result["forecast"] = result.apply(
        lambda row: ("Unavailable" if pd.isna(row.predicted_margin) else "Even" if abs(row.predicted_margin) < 1e-9
                     else f"{row.away_team if row.predicted_margin > 0 else row.home_team} by {abs(row.predicted_margin):.1f}"), axis=1)
    result["status"] = result["status"].str.replace("_", " ").str.capitalize()
    names = {"matchup": "Matchup", "forecast": "Projected winner", "predicted_margin": "Model margin",
             "market_margin": "Market margin", "actual_margin": "Final margin", "edge": "Model − market",
             "pick": "Selected side", "ats_result": "Against spread", "status": "Status", "week": "Week"}
    columns = [c for c in ["week", "matchup", "forecast", "predicted_margin", "market_margin", "actual_margin", "edge", "pick", "ats_result", "status"] if c in result]
    return result[columns].rename(columns=names)


def show_forecasts(games, season, weeks, config) -> None:
    pending = games.loc[~games["completed"].fillna(False), "week"]
    default = int(pending.min()) if not pending.empty else max(weeks)
    week = choose_week("Forecast week", "forecast_week", weeks, default)
    forecasts = forecast_week(games, season, week, config)
    for warning in forecasts.attrs.get("training_diagnostics", {}).get("warnings", []):
        st.warning(warning)
    columns = st.columns(3)
    columns[0].metric("Matchups", len(forecasts))
    columns[1].metric("Forecasts available", int(forecasts["eligible"].sum()))
    columns[2].metric("Training cutoff", f"Week {week - 1}" if week > 1 else "No games")
    st.caption("Every forecast uses only prior weeks and excludes games dated on or after the forecast week's first game. All margins are away minus home; positive favors away. Home advantage is zero at neutral sites.")
    if not forecasts["eligible"].any():
        st.info(f"Both teams need at least {config.min_games} usable prior games. Choose a later week or change the minimum in Model settings.")
    st.dataframe(forecast_display(forecasts), hide_index=True, use_container_width=True,
                 column_config={name: st.column_config.NumberColumn(name, format="%+.1f") for name in ["Model margin", "Market margin", "Final margin", "Model − market"]})
    st.caption("Market comparisons use the line supplied by nflverse. A zero model–market difference creates no pick. Future or unfinished scores remain unavailable.")
    download_results(forecasts, f"forecasts_{season}_week_{week}.csv", config)


def show_history(games, season, weeks, default_week, config) -> None:
    controls = st.columns(2)
    teams = sorted(set(games["away_team"]) | set(games["home_team"]))
    with controls[0]:
        if st.session_state.get("history_team") not in teams:
            st.session_state.pop("history_team", None)
        team = st.selectbox("Team", teams, key="history_team")
    with controls[1]:
        cutoff = choose_week("History through", "history_week", weeks, max(default_week, min(weeks)))
    result = fit_ratings(games, season, cutoff, config)
    history = result.performances.loc[result.performances["team"].eq(team)].sort_values("week").copy()
    diagnostics(result)
    if history.empty:
        st.info("This team has no usable completed games through the selected week.")
        return
    row = result.ratings.set_index("team").loc[team]
    columns = st.columns(3)
    columns[0].metric("Current rating", formatted(row.rating, 4))
    columns[1].metric("Games", int(row.games))
    columns[2].metric("Game variability", formatted(row.performance_std, 4))
    snapshots = []
    for week in weeks:
        if week > cutoff:
            break
        ratings = fit_ratings(games, season, week, config).ratings.set_index("team")
        snapshots.append({"week": week, "rating": ratings.loc[team, "rating"]})
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("**Rating after each week**")
        st.altair_chart(alt.Chart(pd.DataFrame(snapshots)).mark_line(point=True, color=COLORS["positive"]).encode(
            x=alt.X("week:O", title="Week"), y=alt.Y("rating:Q", title="Rating", scale=alt.Scale(zero=False)),
            tooltip=["week", alt.Tooltip("rating:Q", format=".4f")],
        ).properties(height=300), use_container_width=True)
        st.caption("Each point uses games through that week. Ratings can change during a bye as opponents play.")
    with chart_right:
        st.markdown("**Opponent-adjusted game performances**")
        st.altair_chart(alt.Chart(history).mark_bar(color=COLORS["positive"]).encode(
            x=alt.X("week:O", title="Week played"), y=alt.Y("performance:Q", title="Game performance"),
            tooltip=["week", "opponent", alt.Tooltip("performance:Q", format=".4f")],
        ).properties(height=300), use_container_width=True)
        st.caption("Opponents are rated at the selected cutoff. These retrospective game performances are distinct from the week-by-week ratings.")
    st.dataframe(history[["week", "opponent", "venue", "ps", "ps_diff", "opponent_rating", "performance"]],
                 hide_index=True, use_container_width=True,
                 column_config={"week": "Week", "opponent": "Opponent", "venue": "Venue",
                                **{name: st.column_config.NumberColumn(label, format="%.4f") for name, label in
                                   [("ps", "Game Peter Score"), ("ps_diff", "Peter Score differential"), ("opponent_rating", "Opponent rating"), ("performance", "Adjusted performance")]}})
    download_results(history, f"history_{season}_{team}_week_{cutoff}.csv", config)


def show_backtest(games, season, weeks, config) -> None:
    max_week = max(weeks)
    key = f"backtest_range_{season}_{max_week}"
    if len(weeks) > 1:
        first, last = st.slider("Forecast weeks", min(weeks), max_week,
                                (min(config.min_games + 1, max_week), max_week), key=key)
    else:
        first = last = weeks[0]
    st.caption("For each week, ratings are rebuilt from earlier weeks only. The same settings apply to every forecast. Changing settings after viewing results is exploratory tuning, not independent validation.")
    results = backtest(games, season, first, last, config)
    summary = summarize_backtest(results)
    columns = st.columns(4)
    columns[0].metric("Scored", summary["evaluated_games"], help="Completed games with an eligible forecast.")
    columns[1].metric("MAE", formatted(summary["mae"], 2, " pts"), help="Mean absolute error: the average size of the margin error.")
    columns[2].metric("RMSE", formatted(summary["rmse"], 2, " pts"), help="Root mean square error gives extra weight to larger misses.")
    columns[3].metric("Winners", formatted(summary["winner_accuracy"] * 100, 1, "%") if pd.notna(summary["winner_accuracy"]) else "—", help="Correct winner predictions among games without a tied result or tied prediction.")
    st.caption(f"{summary['eligible_games']} eligible forecasts · {summary['pending_games']} awaiting final results · {summary['excluded_games']} excluded. Winner accuracy excludes tied results and tied model predictions.")
    if not summary["evaluated_games"]:
        st.info("No completed, eligible forecasts in this range. Choose later weeks, another season, or a smaller minimum history.")
    else:
        scored = results.loc[results["eligible"] & results["completed"] & results["actual_margin"].notna()].copy()
        weekly = scored.groupby("week", as_index=False)["absolute_error"].mean().rename(columns={"absolute_error": "Mean absolute error"})
        st.altair_chart(alt.Chart(weekly).mark_bar(color=COLORS["positive"], cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("week:O", title="Forecast week"), y=alt.Y("Mean absolute error:Q", title="Mean absolute error (points)"),
            tooltip=["week", alt.Tooltip("Mean absolute error:Q", format=".2f")],
        ).properties(height=240), use_container_width=True)
    with st.expander("Compare with the market", expanded=True):
        columns = st.columns(3)
        columns[0].metric("Model MAE", formatted(summary["model_mae_vs_market"], 2, " pts"))
        columns[1].metric("Market MAE", formatted(summary["market_mae"], 2, " pts"))
        columns[2].metric("ATS wins", formatted(summary["ats_win_rate"] * 100, 1, "%") if pd.notna(summary["ats_win_rate"]) else "—", help="Against-spread wins divided by wins plus losses.")
        st.caption(f"Error comparison: {summary['market_games']} games with both a forecast and a market line. Picks: {summary['ats_wins']} wins, {summary['ats_losses']} losses, {summary['ats_pushes']} pushes; {summary['ats_no_edge']} with no edge and {summary['ats_no_market']} without a market line. Pushes and no-picks are excluded from the win-rate denominator. No odds-based return is assumed.")
    st.dataframe(forecast_display(results), hide_index=True, use_container_width=True,
                 column_config={name: st.column_config.NumberColumn(name, format="%+.1f") for name in ["Model margin", "Market margin", "Final margin", "Model − market"]})
    download_results(results, f"backtest_{season}_weeks_{first}_{last}.csv", config)


def main() -> None:
    st.set_page_config(page_title="Peter Score · NFL", page_icon="🏈", layout="wide")
    st.markdown("""<style>
        .block-container {max-width: 1380px; padding-top: 1.8rem;}
        h1 {letter-spacing: -0.045em;}
        [data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
        [data-testid="stMetricValue"] {font-size: clamp(1.3rem, 2.1vw, 2rem);}
        [data-testid="stSidebar"] {border-right: 1px solid color-mix(in srgb, currentColor 12%, transparent);}
        div[data-testid="stMetric"] {background: color-mix(in srgb, currentColor 5%, transparent); padding: 1rem; border-radius: .6rem;}
    </style>""", unsafe_allow_html=True)
    today = date.today()
    latest = today.year if today.month >= 9 else today.year - 1
    with st.sidebar:
        st.title("Peter Score")
        st.caption("NFL efficiency, adjusted for the opposition.")
        season = st.selectbox("Season", list(range(latest, 1998, -1)), key="season")
        postseason = st.checkbox("Include postseason", value=False, key="postseason")
        refresh = st.button("Refresh data", use_container_width=True)
        token_key = f"refresh_{season}"
        if refresh:
            st.session_state[token_key] = st.session_state.get(token_key, 0) + 1
            season_data.clear()
    config = model_controls()
    st.caption("PETER SCORE / NFL")
    st.title("NFL team strength")
    st.write("Opponent-adjusted rankings, weekly forecasts, and historical results.")
    try:
        with st.spinner(f"Loading {season} from nflverse through nflreadpy. The first load can take a minute…"):
            loaded = season_data(season, st.session_state.get(token_key, 0))
    except (DataError, OSError, ValueError) as exc:
        st.error(f"Unable to load this season: {exc}")
        st.info("Use Refresh data to retry, or select an earlier season. Internet access is needed for the first download; no files or API keys are required.")
        return
    games = loaded.games.copy()
    if not postseason:
        games = games.loc[games["game_type"].eq("REG")].copy()
    if games.empty:
        st.info("No schedule is available for this selection yet. Select an earlier season or refresh later.")
        return
    weeks = sorted(games["week"].dropna().astype(int).unique().tolist())
    completed = games["completed"].fillna(False)
    usable = completed & games.get("stats_available", pd.Series(True, index=games.index)).fillna(False)
    default_week = int(games.loc[usable, "week"].max()) if usable.any() else 0
    with st.sidebar:
        st.caption(f"Fetched {loaded.fetched_at.strftime('%Y-%m-%d %H:%M UTC')}. Cached for up to one hour.")
        st.caption(f"{int(completed.sum())} completed games · {int(usable.sum())} with usable statistics")
    with st.expander("Data coverage and model method"):
        st.write(f"Season {season}: {len(games)} scheduled games in this selection. Data are fetched automatically using nflreadpy; no spreadsheet inputs are used.")
        for note in loaded.notes:
            st.write(note)
        st.write("Peter Score combines normalized yards per play, first downs per play, and third-down conversion. Ratings solve the opponent-adjusted averaging equations directly, with an optional average-team prior. This avoids the old oscillation and iteration limit.")
        st.write(f"Forecast away margin = {config.points_per_rating:g} × (away rating − home rating) − {config.home_advantage:g} at a home venue. Neutral games have no home advantage. Forecasts are point margins; no total or individual final scores are estimated.")
        st.caption("Offensive plays include sacks, kneels, and spikes; tries and nullified plays are excluded. First downs include penalty awards. Third-down rates use recorded conversions and failures. Historical data can receive later official corrections; these are chronological reconstructions from the currently available data.")
        quality = games.loc[completed & ~usable]
        if not quality.empty:
            cols = [c for c in ["week", "away_team", "home_team", "quality_notes"] if c in quality]
            st.dataframe(quality[cols], hide_index=True, use_container_width=True)
    view = st.radio("View", VIEWS, horizontal=True, label_visibility="collapsed", key="view")
    st.divider()
    try:
        if view == "Rankings":
            show_rankings(games, season, weeks, default_week, config)
        elif view == "Weekly forecasts":
            show_forecasts(games, season, weeks, config)
        elif view == "Team history":
            show_history(games, season, weeks, default_week, config)
        else:
            show_backtest(games, season, weeks, config)
    except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
        st.error(f"This selection could not be calculated: {exc}")
        st.caption("Check data coverage and model settings, then refresh if necessary.")


if __name__ == "__main__":
    main()
