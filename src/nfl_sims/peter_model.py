"""Peter efficiency ratings and strictly prior-week NFL forecasts.

The original opponent-averaging fixed point is a graph least-squares problem.
Solving that system directly avoids the oscillation of the old iteration on a
bipartite (especially first-week) schedule. This module has no network or UI
dependencies and never estimates model constants from a backtest's outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd


_METRICS = ("ypp", "fpp", "tdc")
_SCORE_COLUMNS = ("away_score", "home_score")
_RATE_COLUMNS = tuple(f"{side}_{metric}" for side in ("away", "home") for metric in _METRICS)
_SCHEDULE_COLUMNS = ("game_id", "season", "week", "away_team", "home_team", "neutral")
_RATING_COLUMNS = ("team", "rating", "games", "performance_std", "component")
_PERFORMANCE_COLUMNS = (
    "game_id", "season", "week", "gameday", "team", "opponent", "venue",
    "ps", "ps_diff", "opponent_rating", "performance",
)
_FORECAST_COLUMNS = (
    "game_id", "season", "week", "gameday", "away_team", "home_team", "neutral",
    "away_score", "home_score", "completed", "away_rating", "home_rating",
    "away_games", "home_games", "predicted_margin", "market_margin", "actual_margin",
    "edge", "pick", "pick_status", "eligible", "status", "winner_pick",
    "winner_correct", "ats_result", "absolute_error", "squared_error",
)
_EPSILON = 1e-10


@dataclass(frozen=True)
class ModelConfig:
    """Manually chosen constants; weights are normalized to sum to one.

    ``prior_games`` adds zero-rating pseudo games to each active team. Zero
    reproduces the original unregularized fixed point. ``min_games`` controls
    forecast eligibility only, so early rankings remain available to inspect.
    Normalized metrics are deliberately not clipped to their reference ranges.
    """

    ypp_weight: float = 1.2
    fpp_weight: float = 0.8
    tdc_weight: float = 0.4
    ypp_min: float = 2.0
    ypp_max: float = 9.0
    fpp_min: float = 0.16
    fpp_max: float = 0.48
    tdc_min: float = 0.05
    tdc_max: float = 0.70
    ypp_venue_adjustment: float = 0.108
    fpp_venue_adjustment: float = 0.0058
    tdc_venue_adjustment: float = 0.014
    points_per_rating: float = 55.0
    home_advantage: float = 2.2
    prior_games: float = 1.0
    min_games: int = 3

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{field.name} must be a finite number.")
        for metric in _METRICS:
            span = float(getattr(self, f"{metric}_max")) - float(getattr(self, f"{metric}_min"))
            if not np.isfinite(span) or span <= 0:
                raise ValueError(f"{metric}_max must exceed {metric}_min by a finite amount.")
            if getattr(self, f"{metric}_weight") < 0:
                raise ValueError("Metric weights must be nonnegative.")
        total_weight = sum(float(getattr(self, f"{m}_weight")) for m in _METRICS)
        if not np.isfinite(total_weight) or total_weight <= 0:
            raise ValueError("Metric weights must have a finite positive sum.")
        if self.points_per_rating <= 0:
            raise ValueError("points_per_rating must be positive.")
        if self.prior_games < 0:
            raise ValueError("prior_games must be nonnegative.")
        if not isinstance(self.min_games, Integral) or self.min_games < 0:
            raise ValueError("min_games must be a nonnegative integer.")


@dataclass
class RatingResult:
    ratings: pd.DataFrame
    performances: pd.DataFrame
    diagnostics: dict[str, Any]


def _known_bool(value: Any) -> bool | None:
    """Only explicit booleans/0/1 qualify; an unknown venue is not home field."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Real) and np.isfinite(value) and value in (0, 1):
        return bool(value)
    return None


def _validate_week(value: int, *, allow_zero: bool = False) -> None:
    minimum = 0 if allow_zero else 1
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"Week must be an integer of at least {minimum}.")


def _prepare_games(games: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    missing = set(_SCHEDULE_COLUMNS + _SCORE_COLUMNS + _RATE_COLUMNS).difference(games.columns)
    if missing:
        raise ValueError(f"Game data is missing columns: {', '.join(sorted(missing))}.")
    frame = games.copy()
    for column in ("game_id", "away_team", "home_team"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} must be present on every game.")
        frame[column] = frame[column].astype(str)
    if frame["away_team"].eq(frame["home_team"]).any():
        raise ValueError("A team cannot play itself.")
    for column in ("season", "week"):
        numbers = pd.to_numeric(frame[column], errors="coerce")
        if (~np.isfinite(numbers) | numbers.mod(1).ne(0) | numbers.lt(1)).any():
            raise ValueError(f"Every game needs a positive integer {column}.")
        frame[column] = numbers.astype(int)
    for column in _SCORE_COLUMNS + _RATE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    if "spread_line" not in frame:
        frame["spread_line"] = np.nan
    frame["spread_line"] = pd.to_numeric(frame["spread_line"], errors="coerce").astype(float)
    frame["neutral"] = frame["neutral"].map(_known_bool)
    if "gameday" not in frame:
        frame["gameday"] = pd.NaT
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="coerce", utc=True).dt.tz_localize(None)
    final_scores = np.isfinite(frame[list(_SCORE_COLUMNS)]).all(axis=1) & frame[list(_SCORE_COLUMNS)].ge(0).all(axis=1)
    if "completed" in frame:
        frame["completed"] = frame["completed"].map(_known_bool).eq(True) & final_scores
    else:
        frame["completed"] = final_scores
    # Ignore irrelevant provider metadata when identifying redundant copies, but
    # reject any conflict in fields that could affect a fit or an evaluation.
    comparison = list(_SCHEDULE_COLUMNS + _SCORE_COLUMNS + _RATE_COLUMNS) + ["gameday", "spread_line", "completed"]
    if "game_type" in frame:
        comparison.append("game_type")
    original_count = len(frame)
    frame = frame.drop_duplicates(subset=comparison)
    duplicate_ids = frame.loc[frame["game_id"].duplicated(keep=False), "game_id"].unique()
    if len(duplicate_ids):
        raise ValueError(f"Conflicting duplicate game IDs: {', '.join(duplicate_ids[:5])}.")
    return frame.reset_index(drop=True), original_count - len(frame)


def _components(teams: list[str], games: pd.DataFrame) -> dict[str, int]:
    neighbours: dict[str, set[str]] = {team: set() for team in teams}
    for row in games.itertuples():
        neighbours[row.away_team].add(row.home_team)
        neighbours[row.home_team].add(row.away_team)
    result: dict[str, int] = {}
    component = 0
    for first in teams:
        if first in result:
            continue
        component += 1
        todo = [first]
        while todo:
            team = todo.pop()
            if team in result:
                continue
            result[team] = component
            todo.extend(sorted(neighbours[team] - result.keys()))
    return result


def _peter_scores(games: pd.DataFrame, config: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
    weights = np.array([getattr(config, f"{metric}_weight") for metric in _METRICS], dtype=float)
    weights /= weights.sum()
    low = np.array([getattr(config, f"{metric}_min") for metric in _METRICS])
    span = np.array([getattr(config, f"{metric}_max") - getattr(config, f"{metric}_min") for metric in _METRICS])
    adjustment = np.array([getattr(config, f"{metric}_venue_adjustment") for metric in _METRICS])
    venue_effect = (~games["neutral"].astype(bool)).to_numpy(dtype=float)[:, None] * adjustment
    away = games[[f"away_{metric}" for metric in _METRICS]].to_numpy() + venue_effect
    home = games[[f"home_{metric}" for metric in _METRICS]].to_numpy() - venue_effect
    return ((away - low) / span) @ weights, ((home - low) / span) @ weights


def fit_ratings(
    games: pd.DataFrame,
    season: int,
    through_week: int,
    config: ModelConfig = ModelConfig(),
) -> RatingResult:
    """Fit one season through an inclusive week, from completed valid games.

    Teams with no valid games retain a missing rating. With a zero prior each
    connected schedule component is centered separately, because their relative
    strength cannot be identified from results. ``residual`` is the maximum
    absolute residual of the solved normal equations.
    """
    _validate_week(through_week, allow_zero=True)
    frame, duplicates_removed = _prepare_games(games)
    season_games = frame.loc[frame["season"].eq(season)]
    teams = sorted(set(season_games["away_team"]) | set(season_games["home_team"]))
    candidate = season_games.loc[season_games["week"].le(through_week)]
    metric_valid = pd.Series(np.isfinite(candidate[list(_RATE_COLUMNS)]).all(axis=1), index=candidate.index)
    venue_valid = candidate["neutral"].notna()
    valid = candidate.loc[candidate["completed"] & metric_valid & venue_valid].copy()
    active = sorted(set(valid["away_team"]) | set(valid["home_team"]))
    component_by_team = _components(active, valid)
    component_count = len(set(component_by_team.values()))
    warnings: list[str] = []
    if component_count > 1:
        warnings.append("Some groups of teams have not faced one another yet. Comparisons between those groups are provisional.")
    if not len(valid):
        warnings.append("No completed games with valid efficiency statistics and venue information are available at this cutoff.")
    excluded = len(candidate) - len(valid)
    invalid_completed = int((candidate["completed"] & ~(metric_valid & venue_valid)).sum())
    if invalid_completed:
        warnings.append(f"Excluded {invalid_completed} completed game(s) with missing efficiency statistics or unknown venue.")
    diagnostics: dict[str, Any] = {
        "converged": True,
        "iterations": 0 if not len(valid) else 1,
        "residual": 0.0,
        "components": component_count,
        "excluded_games": excluded,
        "pending_games": int((~candidate["completed"]).sum()),
        "invalid_completed_games": invalid_completed,
        "included_games": len(valid),
        "duplicates_removed": duplicates_removed,
        "method": "regularized least squares" if config.prior_games > 0 else "component-centered least squares",
        "warnings": warnings,
    }
    rating_values: dict[str, float] = {}
    counts: dict[str, int] = {}
    performance_rows: list[dict[str, Any]] = []
    if len(valid):
        positions = {team: index for index, team in enumerate(active)}
        away_indices = valid["away_team"].map(positions).to_numpy(dtype=int)
        home_indices = valid["home_team"].map(positions).to_numpy(dtype=int)
        design = np.zeros((len(valid), len(active)), dtype=float)
        design[np.arange(len(valid)), away_indices] = 1.0
        design[np.arange(len(valid)), home_indices] = -1.0
        away_ps, home_ps = _peter_scores(valid, config)
        differential = away_ps - home_ps
        if not np.isfinite(differential).all():
            raise ValueError("The model constants produced nonfinite Peter Scores; use less extreme ranges or weights.")
        lhs = design.T @ design + config.prior_games * np.eye(len(active))
        rhs = design.T @ differential
        if config.prior_games > 0:
            # The augmented least-squares form is stable even if a manually
            # chosen prior is too small to affect the rounded normal matrix.
            augmented = np.vstack([design, np.sqrt(config.prior_games) * np.eye(len(active))])
            target = np.concatenate([differential, np.zeros(len(active))])
            values = np.linalg.lstsq(augmented, target, rcond=None)[0]
        else:
            # Solve the original design instead of squaring its condition number.
            # The minimum-norm solution sets every component's mean to zero.
            values = np.linalg.lstsq(design, differential, rcond=None)[0]
        if not np.isfinite(values).all():
            raise ValueError("The ranking solve did not produce finite ratings.")
        residual = float(np.max(np.abs(lhs @ values - rhs)))
        diagnostics["residual"] = residual
        diagnostics["converged"] = residual <= 1e-8 * max(1.0, float(np.max(np.abs(rhs))))
        rating_values = dict(zip(active, values, strict=True))
        counts = pd.concat([valid["away_team"], valid["home_team"]]).value_counts().to_dict()
        for index, row in enumerate(valid.itertuples()):
            for side, opponent, score, difference in (
                ("away", row.home_team, away_ps[index], differential[index]),
                ("home", row.away_team, home_ps[index], -differential[index]),
            ):
                team = getattr(row, f"{side}_team")
                opponent_rating = rating_values[opponent]
                performance_rows.append({
                    "game_id": row.game_id, "season": row.season, "week": row.week,
                    "gameday": row.gameday, "team": team, "opponent": opponent,
                    "venue": "neutral" if row.neutral else side,
                    "ps": float(score), "ps_diff": float(difference),
                    "opponent_rating": opponent_rating,
                    "performance": opponent_rating + float(difference),
                })
    performances = pd.DataFrame(performance_rows, columns=_PERFORMANCE_COLUMNS)
    deviations = performances.groupby("team")["performance"].std().to_dict() if len(performances) else {}
    ratings = pd.DataFrame([
        {"team": team, "rating": rating_values.get(team, np.nan), "games": counts.get(team, 0),
         "performance_std": deviations.get(team, np.nan), "component": component_by_team.get(team, pd.NA)}
        for team in teams
    ], columns=_RATING_COLUMNS)
    ratings["component"] = ratings["component"].astype("Int64")
    ratings = ratings.sort_values(["rating", "team"], ascending=[False, True], na_position="last").reset_index(drop=True)
    if len(performances):
        performances = performances.sort_values(["week", "gameday", "game_id", "team"]).reset_index(drop=True)
    return RatingResult(ratings, performances, diagnostics)


def forecast_week(
    games: pd.DataFrame,
    season: int,
    week: int,
    config: ModelConfig = ModelConfig(),
) -> pd.DataFrame:
    """Forecast a whole week using results from strictly earlier weeks only.

    ``spread_line`` is the provider's expected HOME margin (positive means home
    favored). All exposed margins are AWAY minus HOME. Outcomes are graded only
    when ``completed`` is true, and missing outcomes remain missing.
    """
    _validate_week(week)
    frame, _ = _prepare_games(games)
    fixtures = frame.loc[frame["season"].eq(season) & frame["week"].eq(week)].sort_values(["gameday", "game_id"])
    if fixtures.empty:
        return pd.DataFrame(columns=_FORECAST_COLUMNS)
    training = frame.copy()
    earliest_game = fixtures["gameday"].min()
    if pd.notna(earliest_game):
        # Conservatively exclude postponed earlier-week games played on/after
        # the first date of the forecast week. Same-week scores are never used.
        after_cutoff = training["gameday"].ge(earliest_game) | training["gameday"].isna()
        training.loc[after_cutoff, "completed"] = False
    result = fit_ratings(training, season, week - 1, config)
    lookup = result.ratings.set_index("team").to_dict(orient="index")
    output: list[dict[str, Any]] = []
    for game in fixtures.itertuples():
        away = lookup[game.away_team]
        home = lookup[game.home_team]
        status = "ready"
        if pd.isna(game.gameday):
            status = "unknown_date"
        elif _known_bool(game.neutral) is None:
            status = "unknown_venue"
        elif not np.isfinite(away["rating"]) or not np.isfinite(home["rating"]):
            status = "no_history"
        elif min(away["games"], home["games"]) < config.min_games:
            status = "insufficient_history"
        eligible = status == "ready"
        prediction = (
            config.points_per_rating * (away["rating"] - home["rating"])
            - (0.0 if game.neutral else config.home_advantage)
        ) if eligible else np.nan
        market = -float(game.spread_line) if np.isfinite(game.spread_line) else np.nan
        actual = float(game.away_score - game.home_score) if game.completed else np.nan
        edge = prediction - market if eligible and np.isfinite(market) else np.nan
        pick: str | None = None
        if not eligible:
            pick_status = "unavailable"
        elif not np.isfinite(market):
            pick_status = "no_market"
        elif abs(edge) <= _EPSILON:
            pick_status = "no_edge"
        elif edge > 0:
            pick_status, pick = "away", game.away_team
        else:
            pick_status, pick = "home", game.home_team
        ats_result = "ungraded"
        if eligible and game.completed:
            if pick is None:
                ats_result = "no_pick"
            else:
                cover_margin = (actual - market) * (1 if pick_status == "away" else -1)
                ats_result = "push" if abs(cover_margin) <= _EPSILON else ("win" if cover_margin > 0 else "loss")
        winner_pick = None
        winner_correct = np.nan
        if eligible and abs(prediction) > _EPSILON:
            winner_pick = game.away_team if prediction > 0 else game.home_team
            if game.completed and abs(actual) > _EPSILON:
                winner_correct = float(np.sign(prediction) == np.sign(actual))
        error = prediction - actual if eligible and game.completed else np.nan
        output.append({
            "game_id": game.game_id, "season": game.season, "week": game.week,
            "gameday": game.gameday, "away_team": game.away_team, "home_team": game.home_team,
            "neutral": game.neutral, "away_score": game.away_score if game.completed else np.nan,
            "home_score": game.home_score if game.completed else np.nan,
            "completed": bool(game.completed), "away_rating": away["rating"], "home_rating": home["rating"],
            "away_games": away["games"], "home_games": home["games"],
            "predicted_margin": prediction, "market_margin": market, "actual_margin": actual,
            "edge": edge, "pick": pick, "pick_status": pick_status,
            "eligible": eligible, "status": status, "winner_pick": winner_pick,
            "winner_correct": winner_correct, "ats_result": ats_result,
            "absolute_error": abs(error), "squared_error": error * error,
        })
    forecast = pd.DataFrame(output, columns=_FORECAST_COLUMNS)
    forecast.attrs["training_diagnostics"] = result.diagnostics
    return forecast


def backtest(
    games: pd.DataFrame,
    season: int,
    start_week: int,
    end_week: int,
    config: ModelConfig = ModelConfig(),
) -> pd.DataFrame:
    """Return every fixture in the requested range with its rolling forecast.

    Missing/ineligible fixtures remain in the output for auditability; use
    ``summarize_backtest`` to evaluate only completed, eligible predictions.
    """
    _validate_week(start_week)
    _validate_week(end_week)
    if end_week < start_week:
        raise ValueError("end_week must not precede start_week.")
    frame, _ = _prepare_games(games)
    weeks = sorted(frame.loc[frame["season"].eq(season) & frame["week"].between(start_week, end_week), "week"].unique())
    if not len(weeks):
        return pd.DataFrame(columns=_FORECAST_COLUMNS)
    pieces = [forecast_week(frame, season, int(week), config) for week in weeks]
    # DataFrame attrs contain week-specific diagnostics and should not survive
    # concatenation as if one week's diagnostics described the entire backtest.
    for piece in pieces:
        piece.attrs.clear()
    return pd.concat(pieces, ignore_index=True)


def summarize_backtest(forecasts: pd.DataFrame) -> dict[str, float | int]:
    """Honest denominators, true MAE/RMSE, and paired market comparison.

    A push is neither a win nor a loss. No-edge and missing-market games have no
    ATS pick. Actual ties and predicted ties are excluded from winner accuracy.
    Market errors and the corresponding model errors use exactly the same games.
    All metrics without an eligible sample return NaN instead of a false zero.
    """
    if forecasts.empty:
        evaluated = forecasts.copy()
        paired = forecasts.copy()
        winner_values = pd.Series(dtype=float)
        eligible_count = 0
        pending_count = 0
        ats_counts: dict[str, int] = {}
    else:
        eligible_mask = forecasts["eligible"].fillna(False).astype(bool)
        finite_predictions = np.isfinite(pd.to_numeric(forecasts["predicted_margin"], errors="coerce"))
        finite_actual = np.isfinite(pd.to_numeric(forecasts["actual_margin"], errors="coerce"))
        complete = forecasts["completed"].fillna(False).astype(bool)
        evaluated = forecasts.loc[eligible_mask & complete & finite_predictions & finite_actual]
        paired = evaluated.loc[np.isfinite(pd.to_numeric(evaluated["market_margin"], errors="coerce"))]
        winner_values = pd.to_numeric(evaluated["winner_correct"], errors="coerce").dropna()
        eligible_count = int(eligible_mask.sum())
        pending_count = int((eligible_mask & ~complete).sum())
        ats_counts = evaluated["ats_result"].value_counts().to_dict()
    count = len(evaluated)
    paired_count = len(paired)
    error = (evaluated["predicted_margin"] - evaluated["actual_margin"]).astype(float) if count else pd.Series(dtype=float)
    paired_error = (paired["predicted_margin"] - paired["actual_margin"]).astype(float) if paired_count else pd.Series(dtype=float)
    market_error = (paired["market_margin"] - paired["actual_margin"]).astype(float) if paired_count else pd.Series(dtype=float)
    wins, losses, pushes = (int(ats_counts.get(key, 0)) for key in ("win", "loss", "push"))
    return {
        "total_games": len(forecasts), "eligible_games": eligible_count,
        "evaluated_games": count, "excluded_games": len(forecasts) - eligible_count,
        "pending_games": pending_count,
        "mae": float(error.abs().mean()) if count else np.nan,
        "rmse": float(np.sqrt((error * error).mean())) if count else np.nan,
        "winner_games": len(winner_values),
        "winner_accuracy": float(winner_values.mean()) if len(winner_values) else np.nan,
        "actual_ties": int(evaluated["actual_margin"].abs().le(_EPSILON).sum()) if count else 0,
        "predicted_ties": int(evaluated["predicted_margin"].abs().le(_EPSILON).sum()) if count else 0,
        "market_games": paired_count,
        "model_mae_vs_market": float(paired_error.abs().mean()) if paired_count else np.nan,
        "market_mae": float(market_error.abs().mean()) if paired_count else np.nan,
        "model_rmse_vs_market": float(np.sqrt((paired_error * paired_error).mean())) if paired_count else np.nan,
        "market_rmse": float(np.sqrt((market_error * market_error).mean())) if paired_count else np.nan,
        "ats_wins": wins, "ats_losses": losses, "ats_pushes": pushes,
        "ats_no_pick": int(ats_counts.get("no_pick", 0)),
        "ats_no_edge": int(evaluated["pick_status"].eq("no_edge").sum()) if count else 0,
        "ats_no_market": int(evaluated["pick_status"].eq("no_market").sum()) if count else 0,
        "ats_graded": wins + losses,
        "ats_win_rate": wins / (wins + losses) if wins + losses else np.nan,
    }
