"""Automatic nflreadpy inputs for the Peter efficiency model.

Only a small game table is returned; the large play-by-play table stays in
Polars and is released after aggregation. No user-maintained data files exist.
Definitions follow https://nflreadr.nflverse.com/articles/dictionary_pbp.html.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os
from typing import Any

import numpy as np
import pandas as pd
import polars as pl


class DataError(RuntimeError):
    """An actionable download or data-integrity problem."""


@dataclass
class SeasonData:
    games: pd.DataFrame
    fetched_at: datetime
    notes: list[str]


# Franchise abbreviations stay stable when schedules and PBP use older names.
TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "JAC": "JAX", "WSH": "WAS"}
GAME_TYPES = {"REG", "WC", "DIV", "CON", "SB"}
METRICS = ("ypp", "fpp", "tdc")
PBP_REQUIRED = {
    "game_id", "play_id", "posteam", "home_team", "away_team", "play_type",
    "yards_gained", "first_down_pass", "first_down_rush", "first_down_penalty",
    "third_down_converted", "third_down_failed", "two_point_attempt", "extra_point_attempt",
}
PBP_OPTIONAL = {
    "season", "play_deleted", "play_type_nfl", "desc", "home_score", "away_score",
    "total_home_score", "total_away_score",
}
SCHEDULE_REQUIRED = {
    "game_id", "season", "week", "game_type", "gameday", "away_team", "home_team",
    "away_score", "home_score",
}


def _nfl_module() -> Any:
    # Streamlit caches the compact SeasonData result. Retaining every 300+ column
    # season in nflreadpy's default memory cache would waste cloud RAM.
    os.environ.setdefault("NFLREADPY_CACHE", "off")
    os.environ.setdefault("NFLREADPY_TIMEOUT", "90")
    try:
        import nflreadpy as nfl
    except ImportError as exc:
        raise DataError("Install the app requirements, including nflreadpy, before loading NFL data.") from exc
    return nfl


def _require_columns(frame: pl.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise DataError(f"The {label} feed is missing required fields: {', '.join(sorted(missing))}.")


def _normalize_teams(frame: pl.DataFrame, names: tuple[str, ...]) -> pl.DataFrame:
    return frame.with_columns(
        pl.col(name).cast(pl.String).str.to_uppercase().replace(TEAM_ALIASES) for name in names
    )


def _flag(name: str) -> pl.Expr:
    return pl.col(name).cast(pl.Float64, strict=False).fill_nan(None).fill_null(0).eq(1)


def _aggregate_pbp(pbp: pl.DataFrame, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one row per game/team plus one game-completion evidence row.

    YPP: yards_gained / counted offensive snaps. Sacks are pass plays and count
    once; kneels/spikes count; tries and deleted/no-play rows do not count.
    FPP: first downs by run/pass or accepted penalty / the same snap count.
    Penalty-awarded first downs can occur on no-play rows; those contribute to
    the numerator without inventing an offensive snap. A first down counts once.
    TDC: recorded converted / (converted + failed), on counted offensive snaps.
    No attempts means unavailable, never a manufactured zero conversion rate.
    """
    _require_columns(pbp, PBP_REQUIRED, "play-by-play")
    pbp = pbp.select(sorted(PBP_REQUIRED | (PBP_OPTIONAL & set(pbp.columns))))
    if "season" in pbp.columns:
        if pbp.filter(pl.col("season").is_not_null() & (pl.col("season") != season)).height:
            raise DataError(f"The play-by-play feed contains games outside season {season}.")
    pbp = _normalize_teams(pbp, ("posteam", "home_team", "away_team"))
    if "play_deleted" in pbp.columns:
        pbp = pbp.filter(~_flag("play_deleted"))
    if pbp.select(pl.struct("game_id", "play_id").is_duplicated().any()).item():
        raise DataError("The play-by-play feed contains duplicate game/play identifiers; refresh the data.")
    if pbp.filter(pl.col("game_id").is_null() | pl.col("play_id").is_null()).height:
        raise DataError("The play-by-play feed contains missing game/play identifiers.")
    for name in ("home_team", "away_team"):
        if pbp.group_by("game_id").agg(pl.col(name).drop_nulls().n_unique().alias("n")).filter(pl.col("n") != 1).height:
            raise DataError("A play-by-play game has inconsistent home/away team identifiers.")
    wrong_team = pl.col("posteam").is_not_null() & ~(
        (pl.col("posteam") == pl.col("home_team")) | (pl.col("posteam") == pl.col("away_team"))
    )
    if pbp.filter(wrong_team).height:
        raise DataError("A play-by-play possession team does not match its game's teams.")

    ended = pl.lit(False)
    if "play_type_nfl" in pbp.columns:
        ended = ended | pl.col("play_type_nfl").eq("END_GAME").fill_null(False)
    if "desc" in pbp.columns:
        ended = ended | pl.col("desc").str.to_uppercase().str.contains(r"^\s*END (OF )?GAME\s*$").fill_null(False)
    evidence = [ended.any().alias("pbp_ended"), pl.len().alias("pbp_rows")]
    evidence.extend(pl.col(side + "_team").drop_nulls().first().alias("pbp_" + side + "_team") for side in ("home", "away"))
    for side in ("home", "away"):
        final_name = side + "_score"
        # The feed's final-score field is repeated on every play. Fall back to
        # end-game scoreboard only, never the maximum of a live scoreboard.
        if final_name in pbp.columns:
            expr = pl.col(final_name).drop_nulls().last()
        elif "total_" + final_name in pbp.columns:
            expr = pl.col("total_" + final_name).filter(ended).drop_nulls().last()
        else:
            expr = pl.lit(None, dtype=pl.Float64)
        evidence.append(expr.alias("pbp_" + final_name))
    game_evidence = pbp.group_by("game_id").agg(evidence).to_pandas()

    nontry = ~(_flag("two_point_attempt") | _flag("extra_point_attempt"))
    normal = pl.col("play_type").is_in(["pass", "run", "qb_kneel", "qb_spike"]) & nontry
    penalty_down = _flag("first_down_penalty") & nontry
    first_down = (normal & (_flag("first_down_pass") | _flag("first_down_rush"))) | penalty_down
    relevant = pbp.filter(pl.col("posteam").is_not_null()).with_columns(
        normal.fill_null(False).alias("counted_play"),
        first_down.fill_null(False).alias("counted_first_down"),
        (normal & _flag("third_down_converted")).fill_null(False).alias("counted_conversion"),
        (normal & _flag("third_down_failed")).fill_null(False).alias("counted_failure"),
    )
    aggregated = relevant.group_by("game_id", "posteam").agg(
        pl.col("counted_play").sum().alias("plays"),
        pl.col("yards_gained").filter(pl.col("counted_play")).sum().alias("yards"),
        pl.col("yards_gained").filter(pl.col("counted_play")).is_null().sum().alias("missing_yards"),
        pl.col("counted_first_down").sum().alias("first_downs"),
        pl.col("counted_conversion").sum().alias("third_converted"),
        pl.col("counted_failure").sum().alias("third_failed"),
        (pl.col("counted_play") & pl.any_horizontal(pl.col(name).cast(pl.Float64, strict=False).fill_nan(None).is_null() for name in ("first_down_pass", "first_down_rush", "first_down_penalty"))).sum().alias("missing_first_down_flags"),
        (pl.col("counted_play") & pl.any_horizontal(pl.col(name).cast(pl.Float64, strict=False).fill_nan(None).is_null() for name in ("third_down_converted", "third_down_failed"))).sum().alias("missing_third_down_flags"),
    ).with_columns((pl.col("third_converted") + pl.col("third_failed")).alias("third_attempts"))
    aggregated = aggregated.with_columns(
        pl.when((pl.col("plays") > 0) & (pl.col("missing_yards") == 0)).then(pl.col("yards") / pl.col("plays")).otherwise(None).alias("ypp"),
        pl.when((pl.col("plays") > 0) & (pl.col("missing_first_down_flags") == 0)).then(pl.col("first_downs") / pl.col("plays")).otherwise(None).alias("fpp"),
        pl.when((pl.col("third_attempts") > 0) & (pl.col("missing_third_down_flags") == 0)).then(pl.col("third_converted") / pl.col("third_attempts")).otherwise(None).alias("tdc"),
    )
    return aggregated.to_pandas(), game_evidence


def build_season_data(
    schedules: pl.DataFrame,
    pbp: pl.DataFrame | None,
    season: int,
    *,
    now: datetime | None = None,
    notes: list[str] | None = None,
) -> SeasonData:
    """Validate/assemble supplied feeds (also the deterministic test seam)."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    notes = list(notes or [])
    _require_columns(schedules, SCHEDULE_REQUIRED, "schedule")
    schedules = schedules.filter(pl.col("season") == season)
    schedules = _normalize_teams(schedules, ("home_team", "away_team"))
    schedules = schedules.filter(pl.col("game_type").is_in(sorted(GAME_TYPES)))
    if schedules.is_empty():
        raise DataError(f"nflverse has not published an NFL schedule for {season} yet. Choose another season or refresh later.")
    if schedules.get_column("game_id").is_null().any() or schedules.get_column("game_id").is_duplicated().any():
        raise DataError("The schedule contains missing or duplicate game identifiers.")
    if schedules.filter(pl.col("home_team").is_null() | pl.col("away_team").is_null() | (pl.col("home_team") == pl.col("away_team"))).height:
        raise DataError("The schedule has missing or inconsistent home/away teams.")
    selected = SCHEDULE_REQUIRED | ({"location", "spread_line", "gametime"} & set(schedules.columns))
    games = schedules.select(sorted(selected)).to_pandas()
    games["game_id"] = games["game_id"].astype(str)
    games["season"] = games["season"].astype(int)
    numeric_week = pd.to_numeric(games["week"], errors="coerce")
    if numeric_week.isna().any() or not numeric_week.between(1, 30).all() or not numeric_week.eq(np.floor(numeric_week)).all():
        raise DataError("The schedule has invalid season-week numbers.")
    games["week"] = numeric_week.astype(int)
    games["gameday"] = pd.to_datetime(games["gameday"], errors="coerce")
    if games["gameday"].isna().any():
        raise DataError("The schedule has missing or invalid game dates, so chronological cutoffs cannot be verified. Refresh the data.")
    for name in ("home_score", "away_score", "spread_line"):
        games[name] = pd.to_numeric(games[name], errors="coerce") if name in games else np.nan
    games["neutral"] = games.get("location", pd.Series(index=games.index, dtype="object")).map({"Home": False, "Neutral": True}).astype("boolean")
    games["quality_notes"] = ""
    if games["neutral"].isna().any():
        notes.append("Some venue locations are unknown. Those games are excluded from venue-adjusted rankings/forecasts.")
    games["pbp_ended"] = False
    games["pbp_rows"] = 0
    for side in ("home", "away"):
        games["pbp_" + side + "_score"] = np.nan
        for metric in METRICS:
            games[side + "_" + metric] = np.nan

    if pbp is not None and not pbp.is_empty():
        stats, evidence = _aggregate_pbp(pbp, season)
        if not set(evidence["game_id"]).issubset(set(games["game_id"])):
            # PBP may include a canceled game omitted from the schedule. Never
            # join it to some other game just because team/week happen to match.
            notes.append("Play-by-play games absent from the schedule were ignored.")
        games = games.drop(columns=[name for name in evidence.columns if name != "game_id" and name in games]).merge(evidence, on="game_id", how="left", validate="one_to_one")
        for side in ("home", "away"):
            known = games["pbp_" + side + "_team"].notna()
            if not games.loc[known, side + "_team"].eq(games.loc[known, "pbp_" + side + "_team"]).all():
                raise DataError("Schedule and play-by-play team assignments disagree for the same game.")
            renamed = stats.rename(columns={"posteam": side + "_team", **{name: side + "_" + name for name in stats.columns if name not in ("game_id", "posteam")}})
            games = games.drop(columns=[side + "_" + metric for metric in METRICS]).merge(renamed, on=["game_id", side + "_team"], how="left", validate="one_to_one")

    games["pbp_ended"] = games["pbp_ended"].eq(True)
    games["pbp_rows"] = games["pbp_rows"].fillna(0).astype(int)
    scored = games[["away_score", "home_score"]].notna().all(axis=1)
    score_valid = games[["away_score", "home_score"]].ge(0).all(axis=1)
    # A conservative 36h fallback accommodates older PBP without END_GAME rows.
    # Recent games need affirmative terminal evidence, not just nonempty scores.
    cutoff = np.datetime64((now - timedelta(hours=36)).replace(tzinfo=None), "ns")
    aged = pd.Series(games["gameday"].to_numpy(dtype="datetime64[ns]") <= cutoff, index=games.index)
    games["completed"] = (scored & score_valid & (games["pbp_ended"] | aged)).astype(bool)
    corroborated = games["away_score"].eq(games["pbp_away_score"]) & games["home_score"].eq(games["pbp_home_score"])
    metric_columns = [side + "_" + name for side in ("home", "away") for name in METRICS]
    metrics_finite = np.isfinite(games[metric_columns].to_numpy(dtype=float)).all(axis=1)
    stats_ready = games["completed"] & corroborated & metrics_finite & games["neutral"].notna()
    games["stats_available"] = stats_ready.astype(bool)
    # Partial live PBP must never leak into rankings, even if a caller forgets
    # to inspect stats_available. Keep raw counts for quality inspection only.
    games.loc[~stats_ready, metric_columns] = np.nan
    games.loc[~games["completed"], "quality_notes"] = "Not yet confirmed final"
    games.loc[games["completed"] & ~corroborated, "quality_notes"] = "Missing play-by-play or scores not synchronized"
    games.loc[games["completed"] & corroborated & ~metrics_finite, "quality_notes"] = "Missing or invalid efficiency inputs"
    games.loc[games["neutral"].isna(), "quality_notes"] = "Unknown home/neutral venue"
    excluded = int((games["completed"] & ~games["stats_available"]).sum())
    if excluded:
        notes.append(f"{excluded} completed game(s) lack verified efficiency/venue inputs and are excluded from rankings.")
    unconfirmed = int((scored & ~games["completed"]).sum())
    if unconfirmed:
        notes.append(f"{unconfirmed} scored game(s) are awaiting confirmation that play has ended.")
    notes.append("Inputs are derived from nflverse plays; rare lateral/fumble scoring can differ from the old TeamRankings spreadsheets.")
    return SeasonData(games.sort_values(["week", "gameday", "game_id"]).reset_index(drop=True), now, notes)


def load_season_data(season: int, refresh: bool = False) -> SeasonData:
    """Fetch one season automatically; caller may cache the compact result.

    Missing PBP is a partial-data state (fixtures remain useful); invalid feed
    schemas and contradictory identifiers are errors rather than guessed inputs.
    ``fetched_at`` is retrieval time, not an upstream publication timestamp.
    """
    if isinstance(season, bool) or not isinstance(season, int) or not 1999 <= season <= datetime.now(timezone.utc).year:
        raise DataError("Choose an NFL season from 1999 through the current calendar year.")
    nfl = _nfl_module()
    if refresh:
        # 0.1.5 hashes cache keys, so its documented substring clearing does not
        # invalidate URL patterns reliably. Full raw-cache clearing does.
        nfl.clear_cache()
    try:
        schedules = nfl.load_schedules(seasons=season)
    except Exception as exc:
        raise DataError(f"Could not retrieve the {season} schedule from nflverse. Check the internet connection and try Refresh data. ({type(exc).__name__})") from exc
    notes: list[str] = []
    try:
        pbp = nfl.load_pbp(seasons=season)
    except Exception as exc:
        pbp = None
        notes.append(f"Play-by-play for {season} is currently unavailable ({type(exc).__name__}). The schedule is shown; try Refresh data later.")
    return build_season_data(schedules, pbp, season, notes=notes)
