"""Independent checks for supplementary offense/defense decomposition."""

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from nfl_sims import peter_model
from nfl_sims.peter_model import ModelConfig, backtest, fit_ratings, forecast_week


def game(week, away, home, away_ps=.8, home_ps=.4, **changes):
    row = {
        "game_id": f"2023_{week:02d}_{away}_{home}", "season": 2023,
        "week": week, "gameday": date(2023, 9, 1) + timedelta(days=7 * (week - 1)),
        "away_team": away, "home_team": home, "away_score": 24., "home_score": 20.,
        "neutral": True, "spread_line": 3., "completed": True,
        "away_ypp": away_ps * 10, "home_ypp": home_ps * 10,
        "away_fpp": .3, "home_fpp": .25, "away_tdc": .45, "home_tdc": .35,
    }
    row.update(changes)
    return row


@pytest.fixture
def config():
    return ModelConfig(
        ypp_weight=1, fpp_weight=0, tdc_weight=0,
        ypp_min=0, ypp_max=10, ypp_venue_adjustment=0,
        fpp_venue_adjustment=0, tdc_venue_adjustment=0,
        points_per_rating=10, home_advantage=0, prior_games=0, min_games=1,
    )


@pytest.mark.parametrize("prior", [0., 1., 1e-30])
@pytest.mark.parametrize("edges", [
    [("A", "B")],
    [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")],
    [("A", "B"), ("B", "C"), ("C", "A")],
    [("A", "B"), ("C", "D"), ("D", "E"), ("E", "C")],
], ids=["one_game", "bipartite_cycle", "odd_cycle", "disconnected"])
def test_components_match_independent_paired_fit_and_preserve_aggregate(config, prior, edges):
    games = pd.DataFrame([
        game(index + 1, away, home, .75 - .08 * index, .25 + .11 * index)
        for index, (away, home) in enumerate(edges)
    ])
    settings = replace(config, prior_games=prior)
    result = fit_ratings(games, 2023, len(edges), settings)
    aggregate_only = fit_ratings(games, 2023, len(edges), settings, include_components=False)
    # Existing outputs stay bit-for-bit unchanged, including order and variability.
    pd.testing.assert_frame_equal(
        result.ratings.drop(columns=["offense", "defense"]),
        aggregate_only.ratings.drop(columns=["offense", "defense"]), check_exact=True,
    )
    pd.testing.assert_frame_equal(
        result.performances.drop(columns=["offense_performance", "defense_performance"]),
        aggregate_only.performances.drop(columns=["offense_performance", "defense_performance"]),
        check_exact=True,
    )

    # Fit individual efficiencies directly with 2N unknowns. This oracle does not
    # use the implementation's aggregate/signless decomposition.
    ratings = result.ratings.set_index("team").sort_index()
    teams = list(ratings.index)
    positions = {team: index for index, team in enumerate(teams)}
    size = len(teams)
    scores = games[["away_ypp", "home_ypp"]].to_numpy() / 10
    baseline = scores.mean()
    design = np.zeros((2 * len(games), 2 * size))
    for index, (away, home) in enumerate(edges):
        design[2 * index, positions[away]] = 1
        design[2 * index, size + positions[home]] = -1
        design[2 * index + 1, positions[home]] = 1
        design[2 * index + 1, size + positions[away]] = -1
    target = (scores - baseline).ravel()
    if prior:
        design = np.vstack([design, np.sqrt(prior) * np.eye(2 * size)])
        target = np.concatenate([target, np.zeros(2 * size)])
    expected = np.linalg.lstsq(design, target, rcond=None)[0]
    np.testing.assert_allclose(ratings.offense, expected[:size], atol=1e-12)
    np.testing.assert_allclose(ratings.defense, expected[size:], atol=1e-12)
    assert result.diagnostics["efficiency_baseline"] == pytest.approx(baseline)
    np.testing.assert_allclose(ratings.offense + ratings.defense, ratings.rating, atol=1e-12)

    performances = result.performances
    np.testing.assert_allclose(
        performances.offense_performance + performances.defense_performance,
        performances.performance, atol=1e-12,
    )
    by_game_team = performances.set_index(["game_id", "team"])
    for row in performances.itertuples():
        opponent = ratings.loc[row.opponent]
        opponent_ps = by_game_team.loc[(row.game_id, row.opponent), "ps"]
        assert row.offense_performance == pytest.approx(row.ps - baseline + opponent.defense)
        assert row.defense_performance == pytest.approx(baseline - opponent_ps + opponent.offense)
    for component in ("offense", "defense"):
        sums = performances.groupby("team")[f"{component}_performance"].sum()
        np.testing.assert_allclose(ratings[component], sums / (ratings.games + prior), atol=1e-12)


def test_distinguishes_offense_from_defense_in_known_paired_data(config):
    offense = {"A": .2, "B": -.1, "C": -.1}
    defense = {"A": -.1, "B": .2, "C": -.1}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    games = pd.DataFrame([
        game(index + 1, away, home,
             .5 + offense[away] - defense[home], .5 + offense[home] - defense[away])
        for index, (away, home) in enumerate(edges)
    ])
    ratings = fit_ratings(games, 2023, 3, config).ratings.set_index("team")
    for team in offense:
        assert ratings.loc[team, "offense"] == pytest.approx(offense[team])
        assert ratings.loc[team, "defense"] == pytest.approx(defense[team])
    assert ratings.loc["A", "rating"] == pytest.approx(ratings.loc["B", "rating"])
    assert ratings.loc["A", "offense"] > ratings.loc["B", "offense"]
    assert ratings.loc["A", "defense"] < ratings.loc["B", "defense"]


@pytest.mark.parametrize("prior", [0., 1.])
def test_common_efficiency_shift_changes_reference_but_not_components(config, prior):
    games = pd.DataFrame([game(1, "A", "B"), game(2, "B", "C", .3, .7)])
    settings = replace(config, prior_games=prior)
    before = fit_ratings(games, 2023, 2, settings)
    shifted = games.copy()
    shifted[["away_ypp", "home_ypp"]] += 3
    after = fit_ratings(shifted, 2023, 2, settings)
    assert after.diagnostics["efficiency_baseline"] == pytest.approx(
        before.diagnostics["efficiency_baseline"] + .3
    )
    pd.testing.assert_frame_equal(before.ratings, after.ratings, atol=1e-12, rtol=1e-12)
    for component in ("offense_performance", "defense_performance"):
        np.testing.assert_allclose(before.performances[component], after.performances[component], atol=1e-12)


def test_only_valid_completed_current_season_cutoff_games_determine_components(config):
    valid = pd.DataFrame([game(1, "A", "B"), game(2, "B", "C", .3, .7)])
    extra = pd.DataFrame([
        game(2, "C", "D", away_ypp=np.nan),
        game(2, "D", "E", completed=False),
        game(2, "E", "F", neutral=None),
        game(3, "G", "A", 100, 90),
        game(1, "X", "Y", 80, 90, season=2022, game_id="other_season"),
    ])
    baseline = fit_ratings(valid, 2023, 2, config)
    result = fit_ratings(pd.concat([valid, extra], ignore_index=True), 2023, 2, config)
    assert result.diagnostics["efficiency_baseline"] == baseline.diagnostics["efficiency_baseline"]
    assert result.diagnostics["included_games"] == 2
    ratings = result.ratings.set_index("team")
    pd.testing.assert_frame_equal(
        ratings.loc[["A", "B", "C"]],
        baseline.ratings.set_index("team").loc[["A", "B", "C"]], check_exact=True,
    )
    assert ratings.loc[["D", "E", "F", "G"], ["rating", "offense", "defense"]].isna().all().all()
    assert ratings.loc[["D", "E", "F", "G"], "games"].eq(0).all()
    pd.testing.assert_frame_equal(result.performances, baseline.performances, check_exact=True)


def test_before_first_game_components_are_missing_and_empty_season_is_safe(config):
    games = pd.DataFrame([game(1, "A", "B")])
    preseason = fit_ratings(games, 2023, 0, config)
    assert preseason.ratings[["rating", "offense", "defense"]].isna().all().all()
    assert preseason.performances.empty
    assert "efficiency_baseline" not in preseason.diagnostics
    assert fit_ratings(games, 2022, 1, config).ratings.empty
    assert fit_ratings(games.iloc[:0], 2023, 1, config).ratings.empty


def test_forecasts_and_backtests_skip_component_solver(config, monkeypatch):
    games = pd.DataFrame([game(week, "A", "B") for week in (1, 2, 3)])
    expected_forecast = forecast_week(games, 2023, 2, config)
    expected_backtest = backtest(games, 2023, 1, 3, config)

    def components_must_not_run(*args, **kwargs):
        raise AssertionError("Supplementary components must not affect the forecast path")

    monkeypatch.setattr(peter_model, "_strength_components", components_must_not_run)
    pd.testing.assert_frame_equal(forecast_week(games, 2023, 2, config), expected_forecast, check_exact=True)
    pd.testing.assert_frame_equal(backtest(games, 2023, 1, 3, config), expected_backtest, check_exact=True)
