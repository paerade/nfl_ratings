# Peter Score: inputs, ratings, forecasts, and evaluation

The app preserves the original model's three efficiency measures and
opponent-adjusted rating equations. It retrieves its inputs automatically and
solves the rating equations directly. The default adds a small average-team
prior to steady early-season estimates. Set **Prior strength** to **0** to use
the original unregularized equations.

Forecasts are point margins. The app does not estimate individual final scores,
game totals, win probabilities, or betting returns.

## Data and counting rules

`nflreadpy` loads one season's schedule and play-by-play. The app supports seasons
from 1999 onward, subject to upstream availability. Games join by `game_id` and
team, not by row order or a name-only lookup. No Excel or CSV input files are
required. Optional result downloads are outputs only. Package downloads and
supported seasons are described in the
[official nflreadpy API](https://nflreadpy.nflverse.com/api/load_functions/).

Regular-season games are the default. **Include postseason** includes the
provider's `WC`, `DIV`, `CON`, and `SB` game types. A fixed 17-week season is not
assumed. Team aliases normalize OAK→LV, SD→LAC, STL/LAR→LA, JAC→JAX, and WSH→WAS;
the original game identifier remains intact.

Each team-game receives three measures:

| Measure | Numerator | Denominator |
| --- | --- | --- |
| Yards per play, YPP | Sum of `yards_gained` on counted offensive plays | Counted offensive plays |
| First downs per play, FPP | Passing/rushing first downs on counted plays, plus penalty-awarded first downs | The same offensive-play count |
| Third-down conversion, TDC | Recorded converted third downs on counted plays | Recorded converted plus failed third downs on counted plays |

Counted offensive plays have `play_type` of `pass`, `run`, `qb_kneel`, or
`qb_spike`. Sacks count once as pass plays. Deleted plays, no-play rows, extra
points, and two-point tries do not count as offensive plays. A first down
awarded by penalty can contribute to FPP even on a no-play row; it does not
create an extra offensive play. Multiple first-down flags on a row count once.
Zero third-down attempts produces an unavailable rate, not a 0% rate.

These are the app's explicit aggregation choices using the
[nflverse play-by-play fields](https://nflreadr.nflverse.com/articles/dictionary_pbp.html).
They are not a claim of exact box-score reconciliation for every play. In
particular, lateral and fumble yardage, historical scorer revisions, rounding,
and differences from the old TeamRankings source can change the inputs. Exact
numerical agreement with the archived spreadsheet ratings has not been
established for the package-derived data.

The loader checks duplicate identifiers, missing required fields, valid game
dates, season consistency, and agreement between schedule and play-by-play team assignments.
It retains missing statistics as missing. Known neutral sites receive no home
adjustment; an unknown venue is excluded from adjusted calculations.

Final-game confirmation requires nonnegative schedule scores and either an
end-game marker in play-by-play or a fallback for a game date at least 36 hours
old. That age is measured from the schedule date, not kickoff time. To use
efficiency statistics, schedule scores must also match the play-by-play final
scores, and all six team metrics must be finite. Recent games without terminal
evidence remain unconfirmed; missing or unsynchronized statistics remain
unavailable. This favors omitting uncertain inputs over treating live or
partial statistics as final.

The app caches the small prepared game table for up to one hour. **Refresh
data** clears the app cache and requests fresh package data. Its displayed
timestamp is retrieval time, not the provider's publication time. See
[publishing and operating instructions](github-workflow.md) for setup and network
behavior.

## From efficiency to a game Peter Score

At a normal home venue, first remove the assumed venue benefit from each
observed efficiency measure: add the adjustment to the away team and subtract
it from the home team. At a neutral site, make no adjustment.

| Metric | Lower reference | Upper reference | Venue adjustment per team | Default raw weight | Normalized weight |
| --- | ---: | ---: | ---: | ---: | ---: |
| YPP | 2.00 | 9.00 | 0.108 | 1.2 | 1/2 |
| FPP | 0.16 | 0.48 | 0.0058 | 0.8 | 1/3 |
| TDC | 0.05 | 0.70 | 0.014 | 0.4 | 1/6 |

For metric `k`, adjusted observation `x`, reference bounds `L` and `U`, and
raw weight `w`:

```text
normalized_metric_k = (adjusted_x_k − L_k) / (U_k − L_k)
normalized_weight_k = w_k / sum(all three raw weights)
PeterScore = sum(normalized_weight_k × normalized_metric_k)
```

The reference bounds are scaling anchors, not caps: observations outside them
are not clipped. Weights must be nonnegative with at least one positive weight;
every upper bound must exceed its lower bound. All model constants must be
finite. Defaults are carried-forward modeling assumptions, not fitted against
the selected backtest.

For game `g`, orient its efficiency differential as away minus home:

```text
d_g = PeterScore_away,g − PeterScore_home,g
```

YPP, FPP, and TDC describe offensive efficiency. Comparing both teams in a game,
then adjusting for opponent strength, makes the final rating reflect overall
team performance. This aggregate remains the sole input to forecasts. The
supplementary offense/defense decomposition below adds insight without changing
that aggregate, its ranking order, or any forecast.

## Opponent adjustment and the direct solve

Let `r_i` be team `i`'s rating, `n_i` its usable completed games, `j(g)` its
opponent, and `d_i,g` the game differential oriented in team `i`'s favor. The
original averaging fixed point is:

```text
r_i = sum over i's games [r_j(g) + d_i,g] / n_i
```

The old implementation repeatedly recomputed these averages. The app solves
the same equations as a least-squares problem, avoiding an arbitrary iteration
cap and oscillation on disconnected or bipartite early-season schedules.

Build matrix `X` with one row per usable game, `+1` for its away team, `−1` for
its home team, and zeros elsewhere. With `d` the vector of game differentials,
the app minimizes:

```text
sum_g (r_away,g − r_home,g − d_g)^2 + lambda × sum_i r_i^2

(X'X + lambda I) r = X'd
```

Here `lambda` is **Prior strength**, default **1.0**. Its equivalent team equation
is:

```text
r_i = sum over i's games [r_j(g) + d_i,g] / (n_i + lambda)
```

This shrinks ratings toward zero, the average-team baseline. It is equivalent
to adding `lambda` zero-performance observations to each active team's
averaging equation. These prior observations do not inflate displayed game
counts or enter game histories and variability. A team with no usable games
still has no rating, rather than a fabricated zero rating.

With `lambda = 0`, the app uses a minimum-norm least-squares solution and centers
each connected schedule component separately. With positive `lambda`, it solves
the equivalent augmented least-squares system, stacking `sqrt(lambda) I` under
`X` and zeros under `d`. This avoids numerical problems from directly solving
the normal matrix with an extremely small prior. A component is a group of teams joined
by the games already played. If groups have not faced each other, the results
cannot identify their relative level; the zero baseline is an assumption.
The fitting diagnostics report disconnected components and the maximum normal-
equation residual.

The zero-prior setting restores the original mathematical fixed point for the
same valid input metrics. It does not promise identical finite-iteration
outputs, spreadsheet rounding, old counting conventions, or corrected source
errors. The default prior intentionally changes the ratings.

## Supplementary offense and defense

The split is an extension of the aggregate model, not uniquely implied by the
original differential equations. It assumes that venue-adjusted offensive
efficiency satisfies `PeterScore_i - mu = offense_i - defense_opponent`, with
positive defense indicating stronger suppression. `mu` is the arithmetic mean
of the two teams' Peter Scores in every usable game in the selected season
through the cutoff. Each team-game contributes once. Future games, invalid games,
other seasons, and prior pseudo-games do not enter this reference.

The extension uses the existing prior strength on both components, with no new
tuning controls. If `r = offense + defense` and `q = offense - defense`, the
paired-efficiency least-squares objective separates into the original fit for
`r` and an independent fit for `q`. Let `H = abs(X)` and
`t_g = PeterScore_away,g + PeterScore_home,g - 2 mu`. Then:

```text
q = argmin [sum_g (q_away,g + q_home,g - t_g)^2 + lambda sum_i q_i^2]
offense = (existing_rating + q) / 2
defense = existing_rating - offense
```

Only `q` is newly calculated; the original aggregate solve is unchanged.
Offense plus defense therefore equals Rating (up to floating-point rounding).
Forecasts explicitly skip this calculation and continue to use the original
ratings alone. These components are in dimensionless Peter Score units, not
points scored or allowed, and do not add forecast uncertainty estimates.

Zero denotes the empirical league-efficiency reference after opponent adjustment
and shrinkage. Positive offense means stronger production; positive defense
means stronger suppression. Neither component is independently recentered to
its team median or mean; unequal schedules can produce a nonzero component mean.
The reference is recalculated at each historical cutoff, so movement measures
relative strength within that season rather than an absolute cross-season scale.

With a zero prior, the additional solve uses the minimum-norm solution. Sparse,
disconnected, or bipartite schedules can leave the split unidentified by games
alone; minimum norm supplies a convention. A positive prior regularizes these
directions. Early component comparisons should therefore be treated as provisional.

For an individual game, the supplementary performances are:

```text
offense_performance_i = PeterScore_i - mu + defense_opponent
defense_performance_i = mu - PeterScore_opponent + offense_opponent
```

Their sum is the existing `opponent_rating + game_differential`. Component
ratings equal the sum of these game components divided by `games + prior_games`
(within numerical tolerance), rather than an unshrunk average when a prior is used.

## Rankings and team history

Rankings use one season through the selected week, inclusive, and only usable
completed games. In a partly played week, completed games can appear while
pending games do not. Results from other seasons are not mixed in.
Offense and Defense follow Game variability in the rankings table; Rating remains
the third column. Exports include the supplementary values.

For each real game, the displayed adjusted performance is
`opponent_rating + game_differential`. **Game variability** is the sample
standard deviation of those performances, using only actual usable games.
It is unavailable after one game. Byes do not add zeros or observations.

The rating-history chart refits at each weekly cutoff. A team's rating can
change during its bye as opponents play. The separate game-performance chart
uses opponent ratings at the currently selected cutoff, so those bars are
retrospective and can change when the cutoff changes. Variability is a
descriptive measure; the app does not turn it into a forecast confidence band.
The history line chart emphasizes the existing aggregate Rating and distinguishes
Offense and Defense by color, line style, and legend. Grouped game-performance
bars keep the aggregate in the center with its component bars alongside it.
The bars represent individual game performances, not weekly fitted ratings.

## Weekly forecasts and chronology

To forecast Week `w`, the app fits on the same season's games with week less
than `w`. It also excludes earlier-week games dated on or after the first
scheduled game date of the forecast week. This protects against postponed
games entering an earlier forecast. The whole week uses one prior-week fit;
Thursday's result does not update Sunday's forecast within that week.

Both teams need a finite rating, a known home/neutral venue, and at least
**3 usable prior games** by default. Changing the minimum affects forecast
eligibility, not the fitting of early rankings. Missing history produces an
explicit unavailable forecast. A fixture without a valid date cannot be
forecast; earlier training games with unknown dates are excluded when applying
the chronological cutoff. The standard loader rejects an invalid schedule date
before reaching this stage.

```text
predicted_away_margin = 55 × (away_rating − home_rating) − home_advantage
home_advantage = 2.2 points at a home venue, 0 at a neutral venue
```

Positive margins favor away; negative margins favor home. The venue adjustment
in the input metrics removes the assumed benefit from observed performance;
the forecast advantage applies that benefit at the upcoming game's venue.
The two sets of constants remain separately configurable.

Historical backtesting rebuilds ratings independently for every forecast
week using these rules and the same settings throughout. It uses the
currently available historical data, which may contain later statistical
corrections; it is not a replay of archived weekly data releases. Adjusting
settings after inspecting performance is exploratory tuning. Independent
validation requires separate held-out seasons or periods. The inherited
55-point scale is consistent with the archived 2010–2018 calibration analysis,
so a backtest over those calibration years is not independent validation of
that scale, even with correctly ordered rating updates.

## Market comparison and evaluation denominators

The schedule feed's `spread_line` is positive when home is favored. The app
negates it to align with its away-minus-home margins. See the
[nflverse schedule dictionary](https://nflreadr.nflverse.com/articles/dictionary_schedules.html).

```text
market_away_margin = −spread_line
edge = predicted_away_margin − market_away_margin
```

A positive edge selects away; a negative edge selects home. An effectively zero
edge produces no pick. Missing market lines also produce no pick. The app uses
the currently supplied line; it does not reconstruct which line was available
at the historical forecast cutoff or assume a particular sportsbook price.

Only eligible forecasts with confirmed final scores and finite model/actual
margins enter error metrics. All scheduled games remain in result tables, so
pending and excluded rows are visible.

| Output | Calculation and sample |
| --- | --- |
| MAE | Mean of `abs(predicted_margin − actual_margin)` over evaluated games |
| RMSE | Square root of the mean squared margin error over those same games |
| Winner accuracy | Correct predicted winners divided by games with a non-tied prediction and non-tied final result |
| Model versus market errors | Both use exactly the evaluated games with a finite market line |
| Against-spread win rate | Wins divided by wins plus losses; pushes and no-picks excluded |

For an away pick, covering means `actual_away_margin > market_away_margin`;
for a home pick it means the reverse. Equality is a push. Tied final games
still enter margin error metrics and can be graded against a nonzero spread.
Numerical comparisons use a `1e-10` tolerance for equal model margins, no edge,
and pushes. A metric with no qualifying sample is unavailable, not zero.

These definitions fix the archived worksheets' blank-scores-as-zero results,
omitted final rows, loss treatment of pushes, and ambiguous “Correct?” column.
The former `sqrt(abs(error))` total is replaced by actual MAE and RMSE. No
odds-based return or betting profitability is inferred.
