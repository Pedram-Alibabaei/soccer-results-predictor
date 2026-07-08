"""Odds conversion, value-bet selection, and flat-stake backtesting helpers."""

from __future__ import annotations

from math import inf
from typing import Mapping

import pandas as pd

from .markets import expected_value, fair_decimal_odds, remove_vig


THREE_WAY_ODDS_COLUMNS = {"1": "odds_1", "X": "odds_X", "2": "odds_2"}
OVER_UNDER_25_ODDS_COLUMNS = {"+": "odds_over_2_5", "-": "odds_under_2_5"}


def bookmaker_probabilities(
    frame: pd.DataFrame,
    odds_columns: Mapping[str, str],
) -> pd.DataFrame:
    """Return no-vig bookmaker probabilities for each row and outcome."""

    rows = []
    for row in frame.itertuples():
        odds = {}
        for outcome, column in odds_columns.items():
            value = getattr(row, column)
            odds[outcome] = float(value) if pd.notna(value) else float("nan")

        if any(not is_valid_decimal_odds(value) for value in odds.values()):
            rows.append({outcome: float("nan") for outcome in odds_columns})
            continue

        rows.append(remove_vig(odds))

    return pd.DataFrame(rows, columns=list(odds_columns), index=frame.index)


def odds_columns_for_market(market: str, timing: str) -> dict[str, str]:
    """Return project odds columns for a market and timing choice."""

    if market == "1x2":
        if timing == "opening":
            return {"1": "odds_open_1", "X": "odds_open_X", "2": "odds_open_2"}
        if timing == "closing":
            return {"1": "odds_close_1", "X": "odds_close_X", "2": "odds_close_2"}
        if timing == "best_available":
            return dict(THREE_WAY_ODDS_COLUMNS)
    if market == "over_under_2_5":
        if timing == "opening":
            return {"+": "odds_open_over_2_5", "-": "odds_open_under_2_5"}
        if timing == "closing":
            return {"+": "odds_close_over_2_5", "-": "odds_close_under_2_5"}
        if timing == "best_available":
            return dict(OVER_UNDER_25_ODDS_COLUMNS)
    raise ValueError(f"Unsupported market/timing: {market}/{timing}")


def complete_odds_mask(frame: pd.DataFrame, odds_columns: Mapping[str, str]) -> pd.Series:
    """Return rows with complete valid decimal odds for the requested outcomes."""

    return frame[list(odds_columns.values())].apply(pd.to_numeric, errors="coerce").gt(1.0).all(axis=1)


def find_value_bets(
    matches: pd.DataFrame,
    probabilities: pd.DataFrame,
    *,
    market: str,
    odds_columns: Mapping[str, str],
    actual_results: pd.Series,
    min_edge: float = 0.03,
    stake: float = 1.0,
    max_bets_per_match: int = 1,
) -> pd.DataFrame:
    """Find positive expected-value bets and settle them with flat stakes."""

    bookmaker = bookmaker_probabilities(matches, odds_columns)
    candidates = []
    for match_index in matches.index:
        match = matches.loc[match_index]
        for outcome, odds_column in odds_columns.items():
            decimal_odds = match[odds_column]
            model_probability = probabilities.at[match_index, outcome]
            if pd.isna(decimal_odds) or pd.isna(model_probability):
                continue
            if not is_valid_decimal_odds(float(decimal_odds)):
                continue

            edge = expected_value(float(model_probability), float(decimal_odds), stake=1.0)
            if edge < min_edge:
                continue

            actual_result = actual_results.at[match_index]
            won = actual_result == outcome
            profit = stake * (float(decimal_odds) - 1.0) if won else -stake
            bookmaker_probability = bookmaker.at[match_index, outcome]
            candidates.append(
                {
                    "date": match["date"],
                    "league": match.get("league_name", match.get("tournament", "")),
                    "season": match.get("season", ""),
                    "home_team": match["home_team"],
                    "away_team": match["away_team"],
                    "market": market,
                    "outcome": outcome,
                    "actual_result": actual_result,
                    "model_probability": float(model_probability),
                    "bookmaker_probability": float(bookmaker_probability),
                    "probability_gap": float(model_probability) - float(bookmaker_probability),
                    "decimal_odds": float(decimal_odds),
                    "fair_odds": fair_decimal_odds(float(model_probability)),
                    "edge": edge,
                    "stake": stake,
                    "won": won,
                    "profit": profit,
                    "expected_profit": stake * edge,
                }
            )

    bets = pd.DataFrame(candidates)
    if bets.empty:
        return empty_bets_frame()

    bets = bets.sort_values(["date", "home_team", "away_team", "edge"], ascending=[True, True, True, False])
    if max_bets_per_match > 0:
        group_columns = ["date", "home_team", "away_team", "market"]
        bets = bets.groupby(group_columns, group_keys=False).head(max_bets_per_match)
    return bets.reset_index(drop=True)


def find_value_bets_1x2(
    matches: pd.DataFrame,
    probabilities: pd.DataFrame,
    *,
    odds_columns: Mapping[str, str] | None = None,
    min_edge: float = 0.03,
    stake: float = 1.0,
    max_bets_per_match: int = 1,
) -> pd.DataFrame:
    return find_value_bets(
        matches,
        probabilities,
        market="1x2",
        odds_columns=odds_columns or THREE_WAY_ODDS_COLUMNS,
        actual_results=matches["result"],
        min_edge=min_edge,
        stake=stake,
        max_bets_per_match=max_bets_per_match,
    )


def find_value_bets_over_under_2_5(
    matches: pd.DataFrame,
    probabilities: pd.DataFrame,
    *,
    odds_columns: Mapping[str, str] | None = None,
    min_edge: float = 0.03,
    stake: float = 1.0,
    max_bets_per_match: int = 1,
) -> pd.DataFrame:
    actual_results = actual_over_under_2_5(matches)
    return find_value_bets(
        matches,
        probabilities,
        market="over_under_2_5",
        odds_columns=odds_columns or OVER_UNDER_25_ODDS_COLUMNS,
        actual_results=actual_results,
        min_edge=min_edge,
        stake=stake,
        max_bets_per_match=max_bets_per_match,
    )


def actual_over_under_2_5(matches: pd.DataFrame) -> pd.Series:
    return pd.Series(
        ["+" if total_goals > 2.5 else "-" for total_goals in matches["total_goals"]],
        index=matches.index,
    )


def summarize_bets(bets: pd.DataFrame) -> dict[str, float | int]:
    if bets.empty:
        return {
            "bets": 0,
            "total_staked": 0.0,
            "profit": 0.0,
            "roi": 0.0,
            "hit_rate": 0.0,
            "avg_odds": 0.0,
            "avg_edge": 0.0,
            "expected_profit": 0.0,
        }

    total_staked = float(bets["stake"].sum())
    profit = float(bets["profit"].sum())
    return {
        "bets": int(len(bets)),
        "total_staked": total_staked,
        "profit": profit,
        "roi": profit / total_staked if total_staked else 0.0,
        "hit_rate": float(bets["won"].mean()),
        "avg_odds": float(bets["decimal_odds"].mean()),
        "avg_edge": float(bets["edge"].mean()),
        "expected_profit": float(bets["expected_profit"].sum()),
    }


def kelly_fraction(
    probability: float,
    decimal_odds: float,
    *,
    fraction: float = 0.25,
    cap: float = 0.02,
) -> float:
    """Return a capped fractional-Kelly stake fraction for later bankroll sizing."""

    if probability < 0 or probability > 1:
        raise ValueError("Probability must be between 0 and 1.")
    if decimal_odds <= 1:
        raise ValueError("Decimal odds must be greater than 1.")
    if fraction <= 0 or cap <= 0:
        raise ValueError("fraction and cap must be positive.")

    net_odds = decimal_odds - 1.0
    full_kelly = ((probability * decimal_odds) - 1.0) / net_odds
    return min(max(full_kelly * fraction, 0.0), cap)


def is_valid_decimal_odds(value: float) -> bool:
    return value != inf and value > 1.0 and value == value


def empty_bets_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "league",
            "season",
            "home_team",
            "away_team",
            "market",
            "outcome",
            "actual_result",
            "model_probability",
            "bookmaker_probability",
            "probability_gap",
            "decimal_odds",
            "fair_odds",
            "edge",
            "stake",
            "won",
            "profit",
            "expected_profit",
        ]
    )
