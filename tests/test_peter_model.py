"""Behavioral checks for the Peter model's historical failure cases."""

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from nfl_sims.peter_model import (
    ModelConfig,
    backtest,
    fit_ratings,
    forecast_week,
    summarize_backtest,
)


def game(week=1, away="A", home="B", season=2023, **changes):
    row = {
        "game_id": f"{season}_{week:02d}_{away}_{home}", "season": season,
        "week": week, "game_type": "REG",
        "gameday": date(int(season), 9, 1) + timedelta(days=7 * (int(week) - 1)),
        "away_team": away, "home_team": home, "away_score": 24.0, "home_score": 20.0,
        "neutral": False, "spread_line": 3.0,
        "away_ypp": 6.0, "home_ypp": 5.0,
        "away_fpp": 0.30, "home_fpp": 0.25,
        "away_tdc": 0.45, "home_tdc": 0.35,
        "completed": True,
    }
    row.update(changes)
    return row


@pytest.fixture
def pure_ypp():
    return ModelConfig(
        ypp_weight=1, fpp_weight=0, tdc_weight=0,
        ypp_min=0, ypp_max=10, ypp_venue_adjustment=0,
        fpp_venue_adjustment=0, tdc_venue_adjustment=0,
        points_per_rating=10, home_advantage=0, prior_games=0, min_games=1,
    )


def test_first_week_fixed_point_does_not_oscillate_and_does_not_invent_unplayed_rating(pure_ypp):
    games = pd.DataFrame([
        game(1, "A", "B", away_ypp=8, home_ypp=4),
        game(2, "A", "C", completed=False, away_score=np.nan, home_score=np.nan),
    ])
    result = fit_ratings(games, 2023, 1, pure_ypp)
    ratings = result.ratings.set_index("team")
    assert ratings.loc["A", "rating"] == pytest.approx(0.2)
    assert ratings.loc["B", "rating"] == pytest.approx(-0.2)
    assert np.isnan(ratings.loc["C", "rating"])
    assert ratings.loc["C", "games"] == 0
    assert result.diagnostics["converged"]
    assert result.diagnostics["residual"] < 1e-12
    assert result.diagnostics["iterations"] == 1


def test_zero_prior_satisfies_opponent_average_even_when_games_are_inconsistent(pure_ypp):
    games = pd.DataFrame([
        game(1, "A", "B", away_ypp=8, home_ypp=4),
        game(2, "B", "C", away_ypp=6, home_ypp=3),
        game(3, "C", "A", away_ypp=6, home_ypp=5),
    ])
    result = fit_ratings(games, 2023, 3, pure_ypp)
    means = result.performances.groupby("team")["performance"].mean().sort_index()
    ratings = result.ratings.set_index("team")["rating"].sort_index()
    np.testing.assert_allclose(means, ratings, atol=1e-12)
    assert ratings.sum() == pytest.approx(0, abs=1e-12)


def test_prior_is_a_zero_rating_pseudo_game(pure_ypp):
    games = pd.DataFrame([game(away_ypp=8, home_ypp=4)])
    result = fit_ratings(games, 2023, 1, replace(pure_ypp, prior_games=2))
    ratings = result.ratings.set_index("team")
    assert ratings.loc["A", "rating"] == pytest.approx(0.1)
    assert ratings.loc["B", "rating"] == pytest.approx(-0.1)
    performance_sums = result.performances.groupby("team")["performance"].sum()
    for team, row in ratings.iterrows():
        assert row.rating == pytest.approx(performance_sums[team] / (row.games + 2))


def test_disconnected_components_center_independently(pure_ypp):
    games = pd.DataFrame([
        game(1, "A", "B", away_ypp=8, home_ypp=4),
        game(1, "C", "D", away_ypp=6, home_ypp=5),
    ])
    result = fit_ratings(games, 2023, 1, pure_ypp)
    assert result.diagnostics["components"] == 2
    assert any("provisional" in warning for warning in result.diagnostics["warnings"])
    for _, group in result.ratings.groupby("component"):
        assert group.rating.sum() == pytest.approx(0, abs=1e-12)


def test_neutral_game_skips_both_venue_adjustments():
    config = ModelConfig(min_games=1)
    neutral_games = pd.DataFrame([
        game(1, away_ypp=5, home_ypp=5, away_fpp=.25, home_fpp=.25, away_tdc=.4, home_tdc=.4, neutral=True),
        game(2, neutral=True),
    ])
    neutral_ratings = fit_ratings(neutral_games, 2023, 1, config)
    np.testing.assert_allclose(neutral_ratings.ratings.rating, 0, atol=1e-12)
    assert forecast_week(neutral_games, 2023, 2, config).iloc[0].predicted_margin == pytest.approx(0)
    home_games = neutral_games.copy()
    home_games.loc[0, "neutral"] = False
    assert fit_ratings(home_games, 2023, 1, config).ratings.set_index("team").loc["A", "rating"] > 0
    home_games = neutral_games.copy()
    home_games.loc[1, "neutral"] = False
    assert forecast_week(home_games, 2023, 2, config).iloc[0].predicted_margin == pytest.approx(-2.2)


def test_default_weights_match_legacy_and_scale_does_not_change_them():
    games = pd.DataFrame([game(neutral=True)])
    config = ModelConfig(prior_games=0)
    score = fit_ratings(games, 2023, 1, config).performances.set_index("team").loc["A", "ps"]
    expected = .5 * (6 - 2) / 7 + (1 / 3) * (.30 - .16) / .32 + (1 / 6) * (.45 - .05) / .65
    assert score == pytest.approx(expected)
    scaled = replace(config, ypp_weight=12, fpp_weight=8, tdc_weight=4)
    pd.testing.assert_frame_equal(fit_ratings(games, 2023, 1, config).ratings, fit_ratings(games, 2023, 1, scaled).ratings)


def test_statistics_outside_reference_range_are_not_clipped(pure_ypp):
    result = fit_ratings(pd.DataFrame([game(away_ypp=12)]), 2023, 1, pure_ypp)
    assert result.performances.set_index("team").loc["A", "ps"] == pytest.approx(1.2)


def test_missing_stats_unknown_venue_and_live_scores_never_enter_fit(pure_ypp):
    games = pd.DataFrame([
        game(1), game(2, away_ypp=np.nan), game(3, neutral=None),
        game(4, completed=False, away_score=42, home_score=0),
        game(5, home_score=np.nan),
    ])
    result = fit_ratings(games, 2023, 5, pure_ypp)
    assert result.diagnostics["included_games"] == 1
    assert result.diagnostics["excluded_games"] == 4
    assert result.ratings.games.tolist() == [1, 1]
    forecast = forecast_week(games, 2023, 3, pure_ypp).iloc[0]
    assert forecast.status == "unknown_venue"
    assert not forecast.eligible


def test_live_or_missing_scores_stay_missing_in_forecast_and_summary(pure_ypp):
    games = pd.DataFrame([game(1), game(2, completed=False, away_score=42, home_score=0)])
    row = forecast_week(games, 2023, 2, pure_ypp).iloc[0]
    assert row.eligible
    assert np.isnan(row.actual_margin)
    assert np.isnan(row.away_score)
    assert row.ats_result == "ungraded"
    summary = summarize_backtest(forecast_week(games, 2023, 2, pure_ypp))
    assert summary["pending_games"] == 1
    assert summary["evaluated_games"] == 0
    assert np.isnan(summary["mae"])


def test_forecast_is_isolated_from_other_seasons_and_future_or_same_week_outcomes(pure_ypp):
    games = pd.DataFrame([
        game(1), game(2), game(3), game(1, season=2022, away_ypp=100),
    ])
    original = forecast_week(games, 2023, 2, pure_ypp).iloc[0]
    changed = games.copy()
    mask = changed.week.ge(2) | changed.season.eq(2022)
    changed.loc[mask, ["away_ypp", "away_fpp", "away_tdc"]] = [50, .95, .99]
    changed.loc[mask, ["away_score", "home_score"]] = [0, 55]
    after = forecast_week(changed, 2023, 2, pure_ypp).iloc[0]
    for column in ("predicted_margin", "away_rating", "home_rating", "away_games", "home_games"):
        assert original[column] == after[column]
    alone = fit_ratings(games.loc[games.season.eq(2023)], 2023, 1, pure_ypp)
    mixed = fit_ratings(games, 2023, 1, pure_ypp)
    pd.testing.assert_frame_equal(alone.ratings, mixed.ratings)


def test_postponed_earlier_week_result_does_not_leak(pure_ypp):
    games = pd.DataFrame([
        game(1), game(2, gameday=pd.Timestamp("2023-09-22"), away_ypp=100),
        game(3, gameday=pd.Timestamp("2023-09-15")),
    ])
    row = forecast_week(games, 2023, 3, pure_ypp).iloc[0]
    assert row.away_games == 1
    assert row.predicted_margin == pytest.approx(1)


def test_unknown_dates_cannot_be_used_to_claim_chronological_forecasts(pure_ypp):
    games = pd.DataFrame([game(1, gameday=pd.NaT), game(2), game(3, gameday=pd.NaT)])
    assert forecast_week(games, 2023, 2, pure_ypp).iloc[0].status == "no_history"
    assert forecast_week(games, 2023, 3, pure_ypp).iloc[0].status == "unknown_date"


def test_tiny_prior_does_not_make_the_solver_singular(pure_ypp):
    result = fit_ratings(pd.DataFrame([game(away_ypp=8, home_ypp=4)]), 2023, 1, replace(pure_ypp, prior_games=1e-30))
    assert result.diagnostics["converged"]
    assert result.ratings.set_index("team").loc["A", "rating"] == pytest.approx(.2)


def test_minimum_history_and_preseason_cutoff(pure_ypp):
    games = pd.DataFrame([game(1), game(2), game(3)])
    pre = fit_ratings(games, 2023, 0, pure_ypp)
    assert pre.ratings.rating.isna().all()
    assert pre.ratings.games.eq(0).all()
    assert forecast_week(games, 2023, 1, pure_ypp).iloc[0].status == "no_history"
    assert forecast_week(games, 2023, 2, replace(pure_ypp, min_games=3)).iloc[0].status == "insufficient_history"
    assert forecast_week(games, 2023, 2, pure_ypp).iloc[0].eligible


def test_duplicate_games_are_not_double_counted_and_conflicts_are_rejected(pure_ypp):
    same = pd.DataFrame([game(), game()])
    result = fit_ratings(same, 2023, 1, pure_ypp)
    assert result.diagnostics["duplicates_removed"] == 1
    assert result.ratings.games.tolist() == [1, 1]
    assert len(forecast_week(same, 2023, 1, pure_ypp)) == 1
    conflict = pd.DataFrame([game(), game(away_ypp=7)])
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        fit_ratings(conflict, 2023, 1, pure_ypp)


@pytest.mark.parametrize("update", [
    {"ypp_weight": -1}, {"ypp_weight": 0, "fpp_weight": 0, "tdc_weight": 0},
    {"fpp_min": .48}, {"tdc_max": .04}, {"home_advantage": np.nan},
    {"ypp_venue_adjustment": np.inf}, {"points_per_rating": 0},
    {"prior_games": -1}, {"min_games": 1.5}, {"min_games": True},
    {"ypp_weight": 1e308, "fpp_weight": 1e308}, {"ypp_min": -1e308, "ypp_max": 1e308},
])
def test_invalid_config_rejected_on_creation(update):
    with pytest.raises(ValueError):
        ModelConfig(**update)


def test_backtest_keeps_all_fixtures_and_last_game_and_uses_real_error_metrics(pure_ypp):
    # The model predicts one point in week 2 from week 1's efficiency margin.
    # In week 3 the same prediction remains one point; future outcome scores do
    # not alter the efficiency ratings used to make it.
    games = pd.DataFrame([
        game(1), game(2, away_score=23, home_score=20),
        game(3, away_score=29, home_score=20),
    ])
    result = backtest(games, 2023, 1, 3, pure_ypp)
    assert len(result) == 3
    assert result.status.tolist() == ["no_history", "ready", "ready"]
    summary = summarize_backtest(result)
    assert summary["evaluated_games"] == 2
    assert summary["mae"] == pytest.approx((2 + 8) / 2)
    assert summary["rmse"] == pytest.approx(np.sqrt((4 + 64) / 2))


def test_ats_signs_pushes_no_edge_no_market_and_ties_have_honest_denominators(pure_ypp):
    # Repeated identical efficiency margins imply an away-one-point forecast.
    games = pd.DataFrame([
        game(1),
        game(2, away_score=24, home_score=20, spread_line=3),  # market away -3; away wins ATS
        game(3, away_score=17, home_score=20, spread_line=3),  # exact push
        game(4, away_score=14, home_score=20, spread_line=3),  # away loses ATS
        game(5, away_score=21, home_score=20, spread_line=-1),  # no edge; no pick
        game(6, away_score=20, home_score=20, spread_line=np.nan),  # tie; no market
        game(7, away_score=23, home_score=20, spread_line=-5),  # away market +5; home pick wins
    ])
    result = backtest(games, 2023, 2, 7, pure_ypp)
    assert result.ats_result.tolist() == ["win", "push", "loss", "no_pick", "no_pick", "win"]
    assert result.pick_status.tolist() == ["away", "away", "away", "no_edge", "no_market", "home"]
    assert result.iloc[0].market_margin == -3
    assert result.iloc[5].pick == "B"
    summary = summarize_backtest(result)
    assert summary["ats_wins"] == 2
    assert summary["ats_losses"] == 1
    assert summary["ats_pushes"] == 1
    assert summary["ats_no_pick"] == 2
    assert summary["ats_no_edge"] == 1
    assert summary["ats_no_market"] == 1
    assert summary["ats_graded"] == 3
    assert summary["ats_win_rate"] == pytest.approx(2 / 3)
    assert summary["actual_ties"] == 1
    assert summary["winner_games"] == 5
    assert summary["winner_accuracy"] == pytest.approx(3 / 5)
    assert summary["market_games"] == 5
    paired = result[result.market_margin.notna()]
    assert summary["model_mae_vs_market"] == pytest.approx((paired.predicted_margin - paired.actual_margin).abs().mean())
    assert summary["market_mae"] == pytest.approx((paired.market_margin - paired.actual_margin).abs().mean())


def test_predicted_tie_is_not_a_winner_pick(pure_ypp):
    games = pd.DataFrame([game(1, away_ypp=5, home_ypp=5), game(2)])
    result = forecast_week(games, 2023, 2, pure_ypp)
    assert result.iloc[0].winner_pick is None
    summary = summarize_backtest(result)
    assert summary["predicted_ties"] == 1
    assert summary["winner_games"] == 0
    assert np.isnan(summary["winner_accuracy"])


def test_empty_season_empty_week_and_empty_backtest(pure_ypp):
    games = pd.DataFrame([game()])
    assert fit_ratings(games, 2022, 1, pure_ypp).ratings.empty
    assert forecast_week(games, 2023, 2, pure_ypp).empty
    empty = backtest(games, 2023, 2, 3, pure_ypp)
    summary = summarize_backtest(empty)
    assert summary["total_games"] == 0
    assert summary["evaluated_games"] == 0
    assert np.isnan(summary["ats_win_rate"])
    truly_empty = games.iloc[:0]
    assert fit_ratings(truly_empty, 2023, 1, pure_ypp).ratings.empty


def test_reversed_backtest_range_rejected(pure_ypp):
    with pytest.raises(ValueError, match="end_week"):
        backtest(pd.DataFrame([game()]), 2023, 5, 2, pure_ypp)
