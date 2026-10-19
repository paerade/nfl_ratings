from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import polars as pl
import pytest

from nfl_sims import peter_data
from nfl_sims.peter_data import DataError, build_season_data


NOW = datetime(2024, 9, 10, 18, tzinfo=timezone.utc)


def schedule(**changes):
    row = dict(game_id="2024_01_A_B", season=2024, week=1, game_type="REG",
               gameday="2024-09-08", away_team="A", home_team="B", away_score=21,
               home_score=24, location="Home", spread_line=3.0)
    row.update(changes)
    return pl.DataFrame([row])


def play(number, team="A", **changes):
    row = dict(game_id="2024_01_A_B", season=2024, play_id=number, posteam=team,
               away_team="A", home_team="B", play_type="run", yards_gained=5.0,
               first_down_pass=0, first_down_rush=0, first_down_penalty=0,
               third_down_converted=0, third_down_failed=0, two_point_attempt=0,
               extra_point_attempt=0, play_deleted=0, play_type_nfl="RUSH", desc="Run",
               home_score=24, away_score=21)
    row.update(changes)
    return row


def minimal_pbp(**away_changes):
    return pl.DataFrame([
        play(1, third_down_converted=1, first_down_rush=1, **away_changes),
        play(2, "B", third_down_failed=1),
        play(3, None, play_type=None, play_type_nfl="END_GAME", desc="END GAME"),
    ])


def test_offensive_denominator_penalty_downs_sacks_and_tries():
    pbp = pl.DataFrame([
        play(1, play_type="pass", yards_gained=12, first_down_pass=1, third_down_converted=1),
        play(2, play_type="pass", yards_gained=-7, third_down_failed=1, desc="Sacked"),
        play(3, play_type="qb_kneel", yards_gained=-1),
        play(4, play_type="qb_spike", yards_gained=0),
        play(5, play_type="no_play", yards_gained=99, first_down_pass=1, first_down_penalty=1, third_down_converted=1),
        play(6, play_type="run", yards_gained=90, first_down_rush=1, play_deleted=1),
        play(7, play_type="run", yards_gained=2, first_down_rush=1, two_point_attempt=1),
        play(8, play_type="extra_point", yards_gained=1, extra_point_attempt=1),
        play(9, "B", third_down_failed=1),
        play(10, None, play_type=None, play_type_nfl="END_GAME", desc="END GAME"),
    ])
    result = build_season_data(schedule(), pbp, 2024, now=NOW)
    game = result.games.iloc[0]
    assert game.away_plays == 4  # the sack is not counted twice
    assert game.away_ypp == 1  # (12 - 7 - 1 + 0) / 4
    assert game.away_fpp == .5  # pass first down plus accepted penalty first down
    assert game.away_tdc == .5  # no-play row is not a third-down attempt
    assert bool(game.stats_available)
    assert game.spread_line == 3  # positive means HOME favored
    assert result.fetched_at == NOW


def test_no_third_downs_is_missing_not_zero():
    pbp = minimal_pbp().with_columns(pl.lit(0).alias("third_down_converted"), pl.lit(0).alias("third_down_failed"))
    result = build_season_data(schedule(), pbp, 2024, now=NOW)
    assert pd.isna(result.games.iloc[0].away_tdc)
    assert not result.games.iloc[0].stats_available
    assert "1 completed" in " ".join(result.notes)


@pytest.mark.parametrize("field", ["yards_gained", "first_down_pass", "third_down_failed"])
def test_missing_required_measurement_invalidates_game(field):
    pbp = minimal_pbp().with_columns(pl.when(pl.col("play_id") == 1).then(None).otherwise(pl.col(field)).alias(field))
    game = build_season_data(schedule(), pbp, 2024, now=NOW).games.iloc[0]
    assert not game.stats_available
    assert pd.isna(game.away_ypp)


def test_live_score_without_end_marker_does_not_enter_model():
    pbp = minimal_pbp().filter(pl.col("play_type_nfl") != "END_GAME")
    game = build_season_data(schedule(gameday="2024-09-10"), pbp, 2024, now=NOW).games.iloc[0]
    assert not game.completed
    assert not game.stats_available
    assert pd.isna(game.away_ypp)


def test_end_marker_allows_same_day_final_with_matching_scores():
    game = build_season_data(schedule(gameday="2024-09-10"), minimal_pbp(), 2024, now=NOW).games.iloc[0]
    assert game.completed and game.stats_available


def test_old_feed_without_end_marker_uses_conservative_schedule_fallback():
    pbp = minimal_pbp().drop("play_type_nfl", "desc")
    game = build_season_data(schedule(), pbp, 2024, now=NOW).games.iloc[0]
    assert game.completed and game.stats_available


def test_score_mismatch_excludes_partial_or_stale_play_by_play():
    pbp = minimal_pbp().with_columns(pl.lit(17).alias("away_score"))
    game = build_season_data(schedule(), pbp, 2024, now=NOW).games.iloc[0]
    assert game.completed
    assert not game.stats_available
    assert "synchronized" in game.quality_notes


def test_future_game_is_retained_without_fake_scores_or_metrics():
    game = build_season_data(schedule(gameday="2024-09-15", away_score=None, home_score=None), None, 2024, now=NOW).games.iloc[0]
    assert not game.completed and not game.stats_available
    assert pd.isna(game.away_score) and pd.isna(game.away_ypp)


def test_neutral_and_unknown_venue_are_distinct():
    neutral = build_season_data(schedule(location="Neutral"), minimal_pbp(), 2024, now=NOW).games.iloc[0]
    unknown = build_season_data(schedule(location=None), minimal_pbp(), 2024, now=NOW).games.iloc[0]
    assert neutral.neutral and neutral.stats_available
    assert pd.isna(unknown.neutral) and not unknown.stats_available


def test_aliases_normalize_franchise_without_rewriting_game_id():
    sched = schedule(away_team="OAK", home_team="STL")
    pbp = minimal_pbp().with_columns(
        pl.col("posteam").replace({"A": "LV", "B": "LA"}),
        pl.lit("LV").alias("away_team"), pl.lit("LA").alias("home_team"))
    game = build_season_data(sched, pbp, 2024, now=NOW).games.iloc[0]
    assert game.away_team == "LV" and game.home_team == "LA"
    assert game.game_id == "2024_01_A_B"
    assert game.stats_available


def test_duplicate_schedule_and_play_keys_are_rejected():
    with pytest.raises(DataError, match="duplicate game"):
        build_season_data(pl.concat([schedule(), schedule()]), minimal_pbp(), 2024, now=NOW)
    with pytest.raises(DataError, match="duplicate game/play"):
        build_season_data(schedule(), pl.concat([minimal_pbp(), minimal_pbp().head(1)]), 2024, now=NOW)


def test_deleted_duplicate_does_not_double_count_or_fail():
    pbp = pl.concat([minimal_pbp(), minimal_pbp().head(1).with_columns(pl.lit(1, dtype=pl.Int64).alias("play_deleted"))])
    game = build_season_data(schedule(), pbp, 2024, now=NOW).games.iloc[0]
    assert game.away_plays == 1


def test_cross_season_and_wrong_team_data_are_rejected():
    with pytest.raises(DataError, match="outside season"):
        build_season_data(schedule(), minimal_pbp().with_columns(pl.lit(2023).alias("season")), 2024, now=NOW)
    with pytest.raises(DataError, match="possession team"):
        build_season_data(schedule(), minimal_pbp().with_columns(pl.lit("C").alias("posteam")), 2024, now=NOW)
    with pytest.raises(DataError, match="assignments disagree"):
        build_season_data(schedule(away_team="C"), minimal_pbp(), 2024, now=NOW)


def test_playoff_type_comes_from_feed_not_hardcoded_week_cutoff():
    game = build_season_data(schedule(week=18, game_type="WC"), minimal_pbp(), 2024, now=NOW).games.iloc[0]
    assert game.game_type == "WC"


def test_load_uses_nflreadpy_one_season_and_can_refresh(monkeypatch):
    calls = []
    fake = SimpleNamespace(
        clear_cache=lambda: calls.append("clear"),
        load_schedules=lambda **kw: calls.append(("schedules", kw)) or schedule(),
        load_pbp=lambda **kw: calls.append(("pbp", kw)) or minimal_pbp(),
    )
    monkeypatch.setattr(peter_data, "_nfl_module", lambda: fake)
    result = peter_data.load_season_data(2024, refresh=True)
    assert result.games.iloc[0].stats_available
    assert calls == ["clear", ("schedules", {"seasons": 2024}), ("pbp", {"seasons": 2024})]


def test_unavailable_pbp_keeps_schedule_and_explicit_notes(monkeypatch):
    def fail(**kwargs):
        raise ConnectionError("Offline")
    monkeypatch.setattr(peter_data, "_nfl_module", lambda: SimpleNamespace(load_schedules=lambda **kw: schedule(), load_pbp=fail))
    result = peter_data.load_season_data(2024)
    assert len(result.games) == 1 and not result.games.iloc[0].stats_available
    assert "currently unavailable" in " ".join(result.notes)


def test_schedule_download_failure_is_actionable(monkeypatch):
    def fail(**kwargs):
        raise ConnectionError("Offline")
    monkeypatch.setattr(peter_data, "_nfl_module", lambda: SimpleNamespace(load_schedules=fail))
    with pytest.raises(DataError, match="Refresh data"):
        peter_data.load_season_data(2024)


def test_missing_schema_field_and_invalid_season_fail_clearly():
    with pytest.raises(DataError, match="first_down_penalty"):
        build_season_data(schedule(), minimal_pbp().drop("first_down_penalty"), 2024, now=NOW)
    for invalid in (True, 1998, "2024", 9999):
        with pytest.raises(DataError, match="1999"):
            peter_data.load_season_data(invalid)


def test_invalid_game_date_cannot_bypass_chronological_cutoff():
    with pytest.raises(DataError, match="chronological cutoffs"):
        build_season_data(schedule(gameday=None), minimal_pbp(), 2024, now=NOW)
