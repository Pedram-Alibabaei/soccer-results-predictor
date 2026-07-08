"""Chronological Elo feature engineering for soccer matches."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EloConfig:
    initial_rating: float = 1500.0
    k_factor: float = 30.0
    home_advantage: float = 75.0
    neutral_home_advantage: float = 0.0
    goal_rating_weight: float = 0.08
    goal_decay: float = 0.90
    initial_goals_for: float = 1.25
    initial_goals_against: float = 1.25


@dataclass(frozen=True)
class EloState:
    elo: dict[str, float]
    goals_for: dict[str, float]
    goals_against: dict[str, float]
    matches_played: dict[str, int]

    @classmethod
    def empty(cls) -> "EloState":
        return cls(elo={}, goals_for={}, goals_against={}, matches_played={})

    def copy(self) -> "EloState":
        return EloState(
            elo=dict(self.elo),
            goals_for=dict(self.goals_for),
            goals_against=dict(self.goals_against),
            matches_played=dict(self.matches_played),
        )


def add_elo_features(
    matches: pd.DataFrame,
    config: EloConfig | None = None,
    *,
    initial_state: EloState | None = None,
    update_ratings: bool = True,
) -> pd.DataFrame:
    """Attach pre-match Elo and rolling goal-strength features to matches.

    Ratings are read before each match and updated after the result, which prevents
    future results from leaking into a row's features.
    """

    cfg = config or EloConfig()
    data = matches.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)

    state = initial_state.copy() if initial_state else EloState.empty()

    rows = []
    for row in data.itertuples(index=False):
        home_team = row.home_team
        away_team = row.away_team

        home_elo = state.elo.get(home_team, cfg.initial_rating)
        away_elo = state.elo.get(away_team, cfg.initial_rating)
        home_attack = state.goals_for.get(home_team, cfg.initial_goals_for)
        away_attack = state.goals_for.get(away_team, cfg.initial_goals_for)
        home_defense = state.goals_against.get(home_team, cfg.initial_goals_against)
        away_defense = state.goals_against.get(away_team, cfg.initial_goals_against)
        home_matches = state.matches_played.get(home_team, 0)
        away_matches = state.matches_played.get(away_team, 0)

        home_advantage = cfg.neutral_home_advantage if row.neutral else cfg.home_advantage
        expected_home_result = expected_result(home_elo + home_advantage, away_elo)
        actual_home_result = actual_result(row.home_goals, row.away_goals)
        multiplier = goal_difference_multiplier(abs(row.home_goals - row.away_goals))
        rating_change = cfg.k_factor * multiplier * (actual_home_result - expected_home_result)

        enriched = row._asdict()
        enriched.update(
            {
                "home_elo_pre": round(home_elo, 3),
                "away_elo_pre": round(away_elo, 3),
                "elo_diff_pre": round(home_elo + home_advantage - away_elo, 3),
                "home_attack_pre": round(home_attack, 3),
                "away_attack_pre": round(away_attack, 3),
                "home_defense_pre": round(home_defense, 3),
                "away_defense_pre": round(away_defense, 3),
                "home_matches_played_pre": home_matches,
                "away_matches_played_pre": away_matches,
                "home_expected_result_elo": round(expected_home_result, 6),
            }
        )
        rows.append(enriched)

        if update_ratings:
            _update_state(
                state,
                home_team=home_team,
                away_team=away_team,
                home_elo=home_elo,
                away_elo=away_elo,
                home_attack=home_attack,
                away_attack=away_attack,
                home_defense=home_defense,
                away_defense=away_defense,
                home_matches=home_matches,
                away_matches=away_matches,
                home_goals=row.home_goals,
                away_goals=row.away_goals,
                rating_change=rating_change,
                cfg=cfg,
            )

    return pd.DataFrame(rows)


def fit_elo_state(matches: pd.DataFrame, config: EloConfig | None = None) -> EloState:
    """Return the Elo state after processing matches chronologically."""

    cfg = config or EloConfig()
    data = matches.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)
    state = EloState.empty()

    for row in data.itertuples(index=False):
        home_team = row.home_team
        away_team = row.away_team
        home_elo = state.elo.get(home_team, cfg.initial_rating)
        away_elo = state.elo.get(away_team, cfg.initial_rating)
        home_attack = state.goals_for.get(home_team, cfg.initial_goals_for)
        away_attack = state.goals_for.get(away_team, cfg.initial_goals_for)
        home_defense = state.goals_against.get(home_team, cfg.initial_goals_against)
        away_defense = state.goals_against.get(away_team, cfg.initial_goals_against)
        home_matches = state.matches_played.get(home_team, 0)
        away_matches = state.matches_played.get(away_team, 0)
        home_advantage = cfg.neutral_home_advantage if row.neutral else cfg.home_advantage
        expected_home_result = expected_result(home_elo + home_advantage, away_elo)
        actual_home_result = actual_result(row.home_goals, row.away_goals)
        multiplier = goal_difference_multiplier(abs(row.home_goals - row.away_goals))
        rating_change = cfg.k_factor * multiplier * (actual_home_result - expected_home_result)

        _update_state(
            state,
            home_team=home_team,
            away_team=away_team,
            home_elo=home_elo,
            away_elo=away_elo,
            home_attack=home_attack,
            away_attack=away_attack,
            home_defense=home_defense,
            away_defense=away_defense,
            home_matches=home_matches,
            away_matches=away_matches,
            home_goals=row.home_goals,
            away_goals=row.away_goals,
            rating_change=rating_change,
            cfg=cfg,
        )

    return state


def expected_result(team_rating: float, opponent_rating: float) -> float:
    """Return Elo expected result for the first team."""

    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - team_rating) / 400.0))


def actual_result(team_goals: int, opponent_goals: int) -> float:
    if team_goals > opponent_goals:
        return 1.0
    if team_goals < opponent_goals:
        return 0.0
    return 0.5


def goal_difference_multiplier(goal_difference: int) -> float:
    """Return a modest Elo multiplier for wider wins."""

    if goal_difference <= 1:
        return 1.0
    if goal_difference == 2:
        return 1.5
    return (11.0 + goal_difference) / 8.0


def _update_exponential_average(previous: float, observed: float, decay: float) -> float:
    return (decay * previous) + ((1.0 - decay) * observed)


def _update_state(
    state: EloState,
    *,
    home_team: str,
    away_team: str,
    home_elo: float,
    away_elo: float,
    home_attack: float,
    away_attack: float,
    home_defense: float,
    away_defense: float,
    home_matches: int,
    away_matches: int,
    home_goals: int,
    away_goals: int,
    rating_change: float,
    cfg: EloConfig,
) -> None:
    state.elo[home_team] = home_elo + rating_change
    state.elo[away_team] = away_elo - rating_change
    state.goals_for[home_team] = _update_exponential_average(home_attack, home_goals, cfg.goal_decay)
    state.goals_for[away_team] = _update_exponential_average(away_attack, away_goals, cfg.goal_decay)
    state.goals_against[home_team] = _update_exponential_average(
        home_defense,
        away_goals,
        cfg.goal_decay,
    )
    state.goals_against[away_team] = _update_exponential_average(
        away_defense,
        home_goals,
        cfg.goal_decay,
    )
    state.matches_played[home_team] = home_matches + 1
    state.matches_played[away_team] = away_matches + 1
