from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from nfl_sims.peter_data import DataError, SeasonData
from nfl_sims.peter_app import performance_history_chart, rating_history_chart
from nfl_sims.peter_model import ModelConfig, fit_ratings

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def season_fixture():
    records = []
    matchups = [("A", "B", "C", "D"), ("A", "C", "B", "D"), ("D", "A", "B", "C")]
    for week in range(1, 7):
        pairings = matchups[(week - 1) % 3]
        for i in (0, 2):
            records.append(dict(
                game_id=f"2025_{week:02}_{i}", season=2025, week=week, game_type="REG",
                gameday=pd.Timestamp(datetime(2025, 9, 7) + timedelta(weeks=week-1)),
                away_team=pairings[i], home_team=pairings[i+1],
                away_score=24.0 if week < 6 else np.nan, home_score=20.0 if week < 6 else np.nan,
                neutral=week == 5, spread_line=3.0, completed=week < 6,
                away_ypp=6.0, away_fpp=.34, away_tdc=.45,
                home_ypp=5.0, home_fpp=.30, home_tdc=.35,
                stats_available=week < 6, quality_notes="" if week < 6 else "Pending",
            ))
    return SeasonData(games=pd.DataFrame(records), fetched_at=datetime.now(timezone.utc), notes=[])


def open_app():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    return app.selectbox(key="season").set_value(2025).run()


def test_four_views_render_and_forecasts_keep_future_scores_missing(season_fixture):
    with patch("nfl_sims.peter_app.season_data", return_value=season_fixture):
        app = open_app()
        assert not app.exception
        assert any(m.label == "Rated teams" and m.value == "4" for m in app.metric)
        for view in ["Weekly forecasts", "Team history", "Backtesting"]:
            app.radio(key="view").set_value(view).run()
            assert not app.exception, (view, app.exception)
            assert not app.error
            if view == "Weekly forecasts":
                table = app.dataframe[-1].value
                assert table["Final margin"].isna().all()
                assert table["Projected winner"].ne("Unavailable").all()
            elif view == "Backtesting":
                assert any(m.label == "Scored" and m.value == "4" for m in app.metric)


def test_settings_apply_validate_and_restore(season_fixture):
    with patch("nfl_sims.peter_app.season_data", return_value=season_fixture):
        app = open_app()
        app.number_input(key="constant_points_per_rating").set_value(40.0)
        next(b for b in app.button if b.label == "Apply settings").click().run()
        assert app.session_state["model_config"].points_per_rating == 40.0
        for key in ["ypp_weight", "fpp_weight", "tdc_weight"]:
            app.number_input(key=f"constant_{key}").set_value(0.0)
        next(b for b in app.button if b.label == "Apply settings").click().run()
        assert any("positive" in e.value for e in app.error)
        assert app.session_state["model_config"].ypp_weight == 1.2
        next(b for b in app.button if b.label == "Restore defaults").click().run()
        assert not app.exception
        assert app.session_state["model_config"].points_per_rating == 55.0
        assert app.number_input(key="constant_ypp_weight").value == 1.2


def test_empty_history_and_preseason_are_explicit(season_fixture):
    with patch("nfl_sims.peter_app.season_data", return_value=season_fixture):
        app = open_app()
        app.selectbox(key="rank_week").set_value(0).run()
        assert not app.exception
        assert any("No completed games" in item.value for item in app.info)
        app.radio(key="view").set_value("Weekly forecasts").run()
        app.selectbox(key="forecast_week").set_value(1).run()
        assert not app.exception
        assert any("at least 3" in item.value for item in app.info)
        assert app.dataframe[-1].value["Model margin"].isna().all()


def test_download_failure_is_actionable():
    with patch("nfl_sims.peter_app.season_data", side_effect=DataError("Network unavailable")):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        assert not app.exception
        assert any("Network unavailable" in e.value for e in app.error)
        assert any("Refresh data" in i.value for i in app.info)


def test_refresh_requests_a_new_load(season_fixture):
    with patch("nfl_sims.peter_app.season_data", return_value=season_fixture) as loader:
        app = open_app()
        loader.reset_mock()
        next(b for b in app.button if b.label == "Refresh data").click().run()
        assert not app.exception
        loader.clear.assert_called_once()
        loader.assert_called_once_with(2025, 1)


def test_rankings_preserve_rating_and_append_components_after_variability(season_fixture):
    with patch("nfl_sims.peter_app.season_data", return_value=season_fixture):
        app = open_app()
        assert not app.exception
        table = app.dataframe[0].value
        assert table.columns.tolist() == [
            "rank", "team", "rating", "points_vs_average", "games", "performance_std", "offense", "defense",
        ]
        expected = fit_ratings(season_fixture.games, 2025, 5, ModelConfig()).ratings.set_index("team")
        shown = table.set_index("team")
        for name in ["rating", "offense", "defense"]:
            np.testing.assert_allclose(shown[name], expected.loc[shown.index, name])
        np.testing.assert_allclose(shown.offense + shown.defense, shown.rating, atol=1e-12)
        assert shown.rating.is_monotonic_decreasing
        assert any("Ratings and forecasts are unchanged" in caption.value for caption in app.caption)


def test_history_uses_weekly_component_snapshots_and_selected_cutoff_performances(season_fixture):
    with (
        patch("nfl_sims.peter_app.season_data", return_value=season_fixture),
        patch("nfl_sims.peter_app.rating_history_chart", wraps=rating_history_chart) as rating_chart,
        patch("nfl_sims.peter_app.performance_history_chart", wraps=performance_history_chart) as performance_chart,
    ):
        app = open_app()
        app.radio(key="view").set_value("Team history").run()
        app.selectbox(key="history_week").set_value(3).run()
        assert not app.exception
        assert [metric.label for metric in app.metric] == ["Offense", "Current rating", "Defense"]
        weekly = rating_chart.call_args.args[0]
        assert weekly.week.tolist() == [1, 2, 3]
        for row in weekly.itertuples():
            expected = fit_ratings(season_fixture.games, 2025, row.week, ModelConfig()).ratings.set_index("team").loc["A"]
            np.testing.assert_allclose([row.rating, row.offense, row.defense], expected[["rating", "offense", "defense"]].to_numpy(dtype=float))
        performances = performance_chart.call_args.args[0]
        expected = fit_ratings(season_fixture.games, 2025, 3, ModelConfig()).performances
        expected = expected.loc[expected.team.eq("A")].sort_values("week")
        pd.testing.assert_frame_equal(performances, expected)
        np.testing.assert_allclose(performances.offense_performance + performances.defense_performance, performances.performance, atol=1e-12)
        assert app.dataframe[0].value.columns[-3:].tolist() == ["performance", "offense_performance", "defense_performance"]
        app.selectbox(key="history_team").set_value("B").run()
        assert not app.exception
        assert performance_chart.call_args.args[0].team.eq("B").all()
        expected_b = fit_ratings(season_fixture.games, 2025, 3, ModelConfig()).ratings.set_index("team").loc["B"]
        assert [metric.value for metric in app.metric] == [
            f"{expected_b.offense:.4f}", f"{expected_b.rating:.4f}", f"{expected_b.defense:.4f}",
        ]


def test_history_charts_keep_aggregate_distinct_and_performance_bar_centered():
    snapshots = pd.DataFrame({"week": [1, 2], "rating": [.3, .4], "offense": [.2, .1], "defense": [.1, .3]})
    line = rating_history_chart(snapshots)
    spec = line.to_dict()
    encoding = spec["encoding"]
    assert encoding["color"]["scale"]["domain"] == ["Rating", "Offense", "Defense"]
    assert encoding["strokeWidth"]["scale"]["range"] == [4, 2, 2]
    assert len({tuple(dash) for dash in encoding["strokeDash"]["scale"]["range"]}) == 3
    assert set(line.data.series) == {"Rating", "Offense", "Defense"}
    assert line.data.loc[line.data.series.eq("Rating"), "value"].tolist() == snapshots.rating.tolist()

    performances = pd.DataFrame({"week": [1, 2], "opponent": ["B", "C"], "performance": [.7, -.2], "offense_performance": [.2, -.3], "defense_performance": [.5, .1]})
    bars = performance_history_chart(performances)
    bar_encoding = bars.to_dict()["encoding"]
    assert bar_encoding["xOffset"]["sort"] == ["Offense", "Aggregate performance", "Defense"]
    assert bar_encoding["color"]["scale"]["range"] == encoding["color"]["scale"]["range"]
    assert bars.data.loc[bars.data.series.eq("Aggregate performance"), "value"].tolist() == performances.performance.tolist()
