"""Market probability helpers derived from a soccer score matrix."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, inf
from typing import Iterator, Mapping


@dataclass(frozen=True)
class ScoreProbabilityTable:
    """Probability table where rows are home goals and columns are away goals."""

    home_team: str
    away_team: str
    max_goals: int
    probabilities: tuple[tuple[float, ...], ...]

    @property
    def total_probability(self) -> float:
        return sum(sum(row) for row in self.probabilities)

    def probability(self, home_goals: int, away_goals: int) -> float:
        return self.probabilities[home_goals][away_goals]

    def scores(self) -> Iterator[tuple[int, int, float]]:
        for home_goals, row in enumerate(self.probabilities):
            for away_goals, probability in enumerate(row):
                yield home_goals, away_goals, probability


@dataclass(frozen=True)
class DrawNoBetProbabilities:
    home_win: float
    draw_push: float
    away_win: float

    @property
    def home_fair_odds(self) -> float:
        if self.home_win == 0:
            return inf
        return (1.0 - self.draw_push) / self.home_win

    @property
    def away_fair_odds(self) -> float:
        if self.away_win == 0:
            return inf
        return (1.0 - self.draw_push) / self.away_win


def score_matrix(
    home_expected_goals: float,
    away_expected_goals: float,
    *,
    max_goals: int = 10,
    home_team: str = "Home",
    away_team: str = "Away",
    normalize: bool = True,
    dixon_coles_rho: float | None = None,
) -> ScoreProbabilityTable:
    """Build a Poisson score matrix from expected goals for both teams."""

    if home_expected_goals < 0 or away_expected_goals < 0:
        raise ValueError("Expected goals must be non-negative.")
    if max_goals < 1:
        raise ValueError("max_goals must be at least 1.")

    home_probs = _poisson_probabilities(home_expected_goals, max_goals)
    away_probs = _poisson_probabilities(away_expected_goals, max_goals)
    matrix = [[home_prob * away_prob for away_prob in away_probs] for home_prob in home_probs]
    if dixon_coles_rho is not None:
        matrix = [
            [
                probability
                * _dixon_coles_adjustment(
                    home_goals,
                    away_goals,
                    home_expected_goals,
                    away_expected_goals,
                    dixon_coles_rho,
                )
                for away_goals, probability in enumerate(row)
            ]
            for home_goals, row in enumerate(matrix)
        ]

    if normalize:
        total = sum(sum(row) for row in matrix)
        matrix = [[probability / total for probability in row] for row in matrix]

    return ScoreProbabilityTable(
        home_team=home_team,
        away_team=away_team,
        max_goals=max_goals,
        probabilities=tuple(tuple(row) for row in matrix),
    )


def three_way(table: ScoreProbabilityTable) -> dict[str, float]:
    """Return full-time 1/X/2 probabilities."""

    home = 0.0
    draw = 0.0
    away = 0.0

    for home_goals, away_goals, probability in table.scores():
        if home_goals > away_goals:
            home += probability
        elif home_goals == away_goals:
            draw += probability
        else:
            away += probability

    return {"1": home, "X": draw, "2": away}


def double_chance(table: ScoreProbabilityTable) -> dict[str, float]:
    result = three_way(table)
    return {
        "1X": result["1"] + result["X"],
        "12": result["1"] + result["2"],
        "X2": result["X"] + result["2"],
    }


def over_under(table: ScoreProbabilityTable, line: float) -> dict[str, float]:
    """Return over/under probabilities for a total-goals line."""

    over = 0.0
    under = 0.0
    push = 0.0

    for home_goals, away_goals, probability in table.scores():
        total_goals = home_goals + away_goals
        if total_goals > line:
            over += probability
        elif total_goals < line:
            under += probability
        else:
            push += probability

    return {"+": over, "-": under, "push": push}


def both_teams_to_score(table: ScoreProbabilityTable) -> dict[str, float]:
    yes = sum(
        probability
        for home_goals, away_goals, probability in table.scores()
        if home_goals > 0 and away_goals > 0
    )
    return {"Y": yes, "N": 1.0 - yes}


def correct_score(table: ScoreProbabilityTable) -> dict[str, float]:
    return {
        f"{home_goals}:{away_goals}": probability
        for home_goals, away_goals, probability in table.scores()
    }


def draw_no_bet(table: ScoreProbabilityTable) -> DrawNoBetProbabilities:
    result = three_way(table)
    return DrawNoBetProbabilities(
        home_win=result["1"],
        draw_push=result["X"],
        away_win=result["2"],
    )


def european_handicap_3way(
    table: ScoreProbabilityTable,
    *,
    home_goals_adjustment: int = 0,
    away_goals_adjustment: int = 0,
) -> dict[str, float]:
    """Return 1/X/2 probabilities after applying a European handicap."""

    home = 0.0
    draw = 0.0
    away = 0.0

    for home_goals, away_goals, probability in table.scores():
        adjusted_home = home_goals + home_goals_adjustment
        adjusted_away = away_goals + away_goals_adjustment
        if adjusted_home > adjusted_away:
            home += probability
        elif adjusted_home == adjusted_away:
            draw += probability
        else:
            away += probability

    return {"1": home, "X": draw, "2": away}


def match_result_and_over_under(table: ScoreProbabilityTable, line: float) -> dict[str, float]:
    """Return combined result and total-goals probabilities, such as 1+, X-, 2+."""

    probabilities = {"1+": 0.0, "1-": 0.0, "X+": 0.0, "X-": 0.0, "2+": 0.0, "2-": 0.0}

    for home_goals, away_goals, probability in table.scores():
        if home_goals > away_goals:
            result = "1"
        elif home_goals == away_goals:
            result = "X"
        else:
            result = "2"

        total_goals = home_goals + away_goals
        if total_goals > line:
            side = "+"
        elif total_goals < line:
            side = "-"
        else:
            continue

        probabilities[f"{result}{side}"] += probability

    return probabilities


def fair_decimal_odds(probability: float) -> float:
    """Convert a fair probability into fair decimal odds."""

    if probability < 0 or probability > 1:
        raise ValueError("Probability must be between 0 and 1.")
    if probability == 0:
        return inf
    return 1.0 / probability


def expected_value(probability: float, decimal_odds: float, *, stake: float = 1.0) -> float:
    """Return expected profit for a win/loss bet with decimal odds."""

    if probability < 0 or probability > 1:
        raise ValueError("Probability must be between 0 and 1.")
    if decimal_odds <= 1:
        raise ValueError("Decimal odds must be greater than 1.")
    if stake <= 0:
        raise ValueError("Stake must be positive.")

    return stake * ((probability * decimal_odds) - 1.0)


def remove_vig(decimal_odds: Mapping[str, float]) -> dict[str, float]:
    """Normalize bookmaker implied probabilities so they sum to 1."""

    implied = {}
    for outcome, odds in decimal_odds.items():
        if odds <= 1:
            raise ValueError("Decimal odds must be greater than 1.")
        implied[outcome] = 1.0 / odds

    total = sum(implied.values())
    return {outcome: probability / total for outcome, probability in implied.items()}


def _poisson_probabilities(rate: float, max_goals: int) -> list[float]:
    probabilities = [exp(-rate)]
    for goals in range(1, max_goals + 1):
        probabilities.append(probabilities[-1] * rate / goals)
    return probabilities


def _dixon_coles_adjustment(
    home_goals: int,
    away_goals: int,
    home_expected_goals: float,
    away_expected_goals: float,
    rho: float,
) -> float:
    if home_goals == 0 and away_goals == 0:
        adjustment = 1.0 - (home_expected_goals * away_expected_goals * rho)
    elif home_goals == 0 and away_goals == 1:
        adjustment = 1.0 + (home_expected_goals * rho)
    elif home_goals == 1 and away_goals == 0:
        adjustment = 1.0 + (away_expected_goals * rho)
    elif home_goals == 1 and away_goals == 1:
        adjustment = 1.0 - rho
    else:
        adjustment = 1.0

    return max(adjustment, 1e-12)
