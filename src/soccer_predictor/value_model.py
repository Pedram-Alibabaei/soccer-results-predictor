"""Residual value model for bet-candidate filtering."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Mapping

import numpy as np
import pandas as pd

from .markets import expected_value, fair_decimal_odds, over_under, score_matrix
from .odds import (
    actual_over_under_2_5,
    bookmaker_probabilities,
    complete_odds_mask,
    is_valid_decimal_odds,
    odds_columns_for_market,
)


VALUE_FEATURE_COLUMNS = (
    "model_probability",
    "bookmaker_probability",
    "probability_gap",
    "abs_probability_gap",
    "raw_edge",
    "decimal_odds",
    "log_decimal_odds",
    "market_1x2",
    "market_over_under_2_5",
    "outcome_home",
    "outcome_draw",
    "outcome_away",
    "outcome_over",
    "outcome_under",
    "elo_diff_pre",
    "abs_elo_diff_pre",
    "rest_days_diff_pre",
    "form_diff_pre",
    "venue_form_diff_pre",
    "goal_form_diff_pre",
    "defensive_form_diff_pre",
    "shot_form_diff_pre",
    "shot_concession_diff_pre",
    "sot_form_diff_pre",
    "sot_concession_diff_pre",
    "xg_form_diff_pre",
    "xg_concession_diff_pre",
    "selection_elo_diff_pre",
    "selection_rest_diff_pre",
    "selection_form_diff_pre",
    "selection_venue_form_diff_pre",
    "selection_goal_form_diff_pre",
    "selection_defensive_advantage_pre",
    "selection_shot_diff_pre",
    "selection_sot_diff_pre",
    "selection_total_goal_signal_pre",
    "selection_total_shot_signal_pre",
    "selection_total_sot_signal_pre",
)


@dataclass(frozen=True)
class ResidualValueModel:
    feature_columns: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    weights: np.ndarray

    def predict_raw_probability(self, candidates: pd.DataFrame) -> pd.Series:
        features = transform_value_features(candidates, self.feature_columns, self.means, self.scales)
        probabilities = sigmoid(add_intercept(features) @ self.weights)
        return pd.Series(probabilities, index=candidates.index)

    def score_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        scored = candidates.copy()
        scored["value_probability_raw"] = self.predict_raw_probability(scored)
        scored["value_probability"] = normalize_candidate_probabilities(scored, "value_probability_raw")
        scored["value_edge"] = scored["value_probability"] * scored["decimal_odds"] - 1.0
        scored["value_expected_profit"] = scored["value_edge"]
        return scored

    def score_clv_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        scored = candidates.copy()
        scored["clv_model_probability"] = self.predict_raw_probability(scored)
        return scored


def fit_residual_value_model(
    candidates: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = VALUE_FEATURE_COLUMNS,
    learning_rate: float = 0.05,
    epochs: int = 2500,
    l2: float = 0.01,
) -> ResidualValueModel:
    """Fit a binary model that estimates whether each candidate outcome wins."""

    return fit_candidate_probability_model(
        candidates,
        target_column="won",
        feature_columns=feature_columns,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
    )


def fit_clv_model(
    candidates: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = VALUE_FEATURE_COLUMNS,
    learning_rate: float = 0.05,
    epochs: int = 2500,
    l2: float = 0.01,
) -> ResidualValueModel:
    """Fit a binary model that estimates whether an opening bet beats the close."""

    return fit_candidate_probability_model(
        add_clv_target(candidates),
        target_column="positive_clv",
        feature_columns=feature_columns,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
    )


def fit_candidate_probability_model(
    candidates: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: tuple[str, ...] = VALUE_FEATURE_COLUMNS,
    learning_rate: float = 0.05,
    epochs: int = 2500,
    l2: float = 0.01,
) -> ResidualValueModel:
    """Fit a binary model over one-row-per-candidate training data."""

    if candidates.empty:
        raise ValueError("Need at least one candidate row to fit the candidate probability model.")
    if target_column not in candidates.columns:
        raise ValueError(f"Missing target column for candidate probability model: {target_column}")

    usable_candidates = candidates[pd.notna(candidates[target_column])].copy()
    if usable_candidates.empty:
        raise ValueError(f"No usable candidate rows with target column: {target_column}")

    features_raw = build_value_features(usable_candidates, feature_columns)
    means = features_raw.mean(axis=0).to_numpy()
    scales = features_raw.std(axis=0).replace(0.0, 1.0).fillna(1.0).to_numpy()
    features = (features_raw.to_numpy(dtype=float) - means) / scales
    target = usable_candidates[target_column].astype(float).to_numpy()
    weights = fit_logistic_weights(
        add_intercept(features),
        target,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
    )
    return ResidualValueModel(
        feature_columns=feature_columns,
        means=means,
        scales=scales,
        weights=weights,
    )


def add_clv_target(candidates: pd.DataFrame) -> pd.DataFrame:
    if "clv_decimal" not in candidates.columns:
        raise ValueError("CLV model training needs a clv_decimal column.")

    targeted = candidates[pd.notna(candidates["clv_decimal"])].copy()
    targeted["positive_clv"] = targeted["clv_decimal"] > 0.0
    return targeted


def build_bet_candidates(
    matches: pd.DataFrame,
    probabilities_by_market: Mapping[str, pd.DataFrame],
    odds_columns_by_market: Mapping[str, Mapping[str, str]],
) -> pd.DataFrame:
    """Create one row per possible bet outcome, with actual settlement attached."""

    frames = []
    if "1x2" in probabilities_by_market:
        frames.append(
            build_market_candidates(
                matches,
                probabilities_by_market["1x2"],
                market="1x2",
                odds_columns=odds_columns_by_market["1x2"],
                actual_results=matches["result"],
            )
        )
    if "over_under_2_5" in probabilities_by_market:
        frames.append(
            build_market_candidates(
                matches,
                probabilities_by_market["over_under_2_5"],
                market="over_under_2_5",
                odds_columns=odds_columns_by_market["over_under_2_5"],
                actual_results=actual_over_under_2_5(matches),
            )
        )
    if not frames:
        return empty_candidate_frame()
    return pd.concat(frames, ignore_index=True)


def build_market_candidates(
    matches: pd.DataFrame,
    model_probabilities: pd.DataFrame,
    *,
    market: str,
    odds_columns: Mapping[str, str],
    actual_results: pd.Series,
) -> pd.DataFrame:
    usable_matches = matches[complete_odds_mask(matches, odds_columns)].copy()
    if usable_matches.empty:
        return empty_candidate_frame()

    bookmaker = bookmaker_probabilities(usable_matches, odds_columns)
    closing_columns = odds_columns_for_market(market, "closing")
    has_closing_columns = all(column in usable_matches.columns for column in closing_columns.values())
    closing_bookmaker = (
        bookmaker_probabilities(usable_matches, closing_columns)
        if has_closing_columns
        else pd.DataFrame(
            float("nan"),
            index=usable_matches.index,
            columns=list(closing_columns),
        )
    )
    rows = []
    for match_index in usable_matches.index:
        match = usable_matches.loc[match_index]
        actual_result = actual_results.at[match_index]
        for outcome, odds_column in odds_columns.items():
            decimal_odds = float(match[odds_column])
            closing_decimal_odds = safe_odds(match.get(closing_columns[outcome], float("nan")))
            model_probability = float(model_probabilities.at[match_index, outcome])
            bookmaker_probability = float(bookmaker.at[match_index, outcome])
            closing_bookmaker_probability = float(closing_bookmaker.at[match_index, outcome])
            won = actual_result == outcome
            profit = decimal_odds - 1.0 if won else -1.0
            raw_edge = expected_value(model_probability, decimal_odds)
            clv_decimal = (
                decimal_odds / closing_decimal_odds - 1.0
                if is_valid_decimal_odds(closing_decimal_odds)
                else float("nan")
            )
            clv_probability = (
                closing_bookmaker_probability - bookmaker_probability
                if pd.notna(closing_bookmaker_probability)
                else float("nan")
            )
            rows.append(
                {
                    "match_id": int(match["match_id"]),
                    "date": match["date"],
                    "season_start_year": int(match["season_start_year"]),
                    "season": match.get("season", ""),
                    "league": match.get("league_name", match.get("tournament", "")),
                    "home_team": match["home_team"],
                    "away_team": match["away_team"],
                    "market": market,
                    "outcome": outcome,
                    "actual_result": actual_result,
                    "model_probability": model_probability,
                    "bookmaker_probability": bookmaker_probability,
                    "closing_bookmaker_probability": closing_bookmaker_probability,
                    "probability_gap": model_probability - bookmaker_probability,
                    "abs_probability_gap": abs(model_probability - bookmaker_probability),
                    "decimal_odds": decimal_odds,
                    "closing_decimal_odds": closing_decimal_odds,
                    "clv_decimal": clv_decimal,
                    "clv_probability": clv_probability,
                    "fair_odds": fair_decimal_odds(model_probability),
                    "raw_edge": raw_edge,
                    "won": bool(won),
                    "profit": profit,
                    **candidate_match_features(match, market, outcome, decimal_odds=decimal_odds),
                }
            )
    return pd.DataFrame(rows)


def over_under_2_5_probabilities(expected_goals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in expected_goals.itertuples(index=False):
        probabilities = over_under(score_matrix(row.home_xg, row.away_xg), 2.5)
        rows.append({"+": probabilities["+"], "-": probabilities["-"]})
    return pd.DataFrame(rows, index=expected_goals.index)


def select_value_bets(
    scored_candidates: pd.DataFrame,
    *,
    min_edge: float,
    stake: float = 1.0,
    max_bets_per_match_market: int = 1,
    max_bets_per_match: int = 1,
    max_decimal_odds: float | None = None,
    max_bets_per_weekend: int | None = None,
    edge_column: str = "value_edge",
    probability_column: str = "value_probability",
    selection_mode: str = "conservative",
    min_probability: float | None = None,
    secondary_probability_column: str | None = None,
    min_secondary_probability: float | None = None,
) -> pd.DataFrame:
    if selection_mode not in {"value", "conservative"}:
        raise ValueError("selection_mode must be 'value' or 'conservative'.")
    if scored_candidates.empty:
        return empty_selected_bets_frame()

    selected = scored_candidates[scored_candidates[edge_column] >= min_edge].copy()
    if max_decimal_odds is not None:
        selected = selected[selected["decimal_odds"] <= max_decimal_odds].copy()
    if min_probability is not None:
        selected = selected[
            pd.to_numeric(selected[probability_column], errors="coerce") >= min_probability
        ].copy()
    if secondary_probability_column is not None and min_secondary_probability is not None:
        selected = selected[
            pd.to_numeric(selected[secondary_probability_column], errors="coerce")
            >= min_secondary_probability
        ].copy()
    if selected.empty:
        return empty_selected_bets_frame()

    ranking_columns = (
        [edge_column, probability_column]
        if selection_mode == "value"
        else [probability_column, edge_column]
    )
    selected = selected.sort_values(
        ["date", "match_id", "market", *ranking_columns],
        ascending=[True, True, True, False, False],
    )
    if max_bets_per_match_market > 0:
        selected = selected.groupby(["match_id", "market"], group_keys=False).head(max_bets_per_match_market)
    if max_bets_per_match > 0:
        selected = selected.sort_values(
            ["date", "match_id", *ranking_columns],
            ascending=[True, True, False, False],
        )
        selected = selected.groupby("match_id", group_keys=False).head(max_bets_per_match)
    if max_bets_per_weekend is not None and max_bets_per_weekend > 0:
        selected["weekend"] = weekend_key(selected["date"])
        selected = selected.sort_values(
            ["weekend", *ranking_columns],
            ascending=[True, False, False],
        )
        selected = selected.groupby("weekend", group_keys=False).head(max_bets_per_weekend)
        selected = selected.sort_values(["date", "match_id"], kind="mergesort")

    selected["stake"] = stake
    selected["settled_profit"] = selected["profit"] * stake
    selected["selected_probability"] = selected[probability_column]
    selected["selected_edge"] = selected[edge_column]
    selected["selection_mode"] = selection_mode
    return selected.reset_index(drop=True)


def summarize_selected_bets(bets: pd.DataFrame) -> dict[str, float | int]:
    if bets.empty:
        return {
            "bets": 0,
            "total_staked": 0.0,
            "profit": 0.0,
            "roi": 0.0,
            "hit_rate": 0.0,
            "avg_odds": 0.0,
            "avg_edge": 0.0,
        }
    total_staked = float(bets["stake"].sum())
    profit = float(bets["settled_profit"].sum())
    return {
        "bets": int(len(bets)),
        "total_staked": total_staked,
        "profit": profit,
        "roi": profit / total_staked if total_staked else 0.0,
        "hit_rate": float(bets["won"].mean()),
        "avg_odds": float(bets["decimal_odds"].mean()),
        "avg_edge": float(bets["selected_edge"].mean()),
    }


def summarize_clv(bets: pd.DataFrame) -> dict[str, float | int]:
    if bets.empty or "clv_decimal" not in bets.columns:
        return {
            "bets": 0,
            "with_closing_odds": 0,
            "positive_clv_rate": 0.0,
            "avg_decimal_clv": 0.0,
            "avg_probability_clv": 0.0,
        }
    clv_rows = bets[pd.notna(bets["clv_decimal"])].copy()
    if clv_rows.empty:
        return {
            "bets": int(len(bets)),
            "with_closing_odds": 0,
            "positive_clv_rate": 0.0,
            "avg_decimal_clv": 0.0,
            "avg_probability_clv": 0.0,
        }
    return {
        "bets": int(len(bets)),
        "with_closing_odds": int(len(clv_rows)),
        "positive_clv_rate": float((clv_rows["clv_decimal"] > 0).mean()),
        "avg_decimal_clv": float(clv_rows["clv_decimal"].mean()),
        "avg_probability_clv": float(clv_rows["clv_probability"].mean()),
    }


def summarize_thresholds(
    scored_candidates: pd.DataFrame,
    thresholds: list[float],
    *,
    edge_column: str,
    probability_column: str,
    stake: float,
    max_decimal_odds: float | None = None,
    max_bets_per_weekend: int | None = None,
    selection_mode: str = "conservative",
    min_probability: float | None = None,
    secondary_probability_column: str | None = None,
    min_secondary_probability: float | None = None,
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        bets = select_value_bets(
            scored_candidates,
            min_edge=threshold,
            stake=stake,
            edge_column=edge_column,
            probability_column=probability_column,
            selection_mode=selection_mode,
            min_probability=min_probability,
            max_decimal_odds=max_decimal_odds,
            max_bets_per_weekend=max_bets_per_weekend,
            secondary_probability_column=secondary_probability_column,
            min_secondary_probability=min_secondary_probability,
        )
        rows.append(
            {
                "selection_mode": selection_mode,
                "min_probability": min_probability or 0.0,
                "min_edge": threshold,
                **summarize_selected_bets(bets),
                **summarize_clv(bets),
            }
        )
    return pd.DataFrame(rows)


def summarize_combined_thresholds(
    scored_candidates: pd.DataFrame,
    edge_thresholds: list[float],
    clv_probability_thresholds: list[float],
    *,
    edge_column: str,
    probability_column: str,
    stake: float,
    max_decimal_odds: float | None = None,
    max_bets_per_weekend: int | None = None,
    selection_mode: str = "conservative",
    min_probability: float | None = None,
) -> pd.DataFrame:
    rows = []
    for edge_threshold in edge_thresholds:
        for clv_threshold in clv_probability_thresholds:
            bets = select_value_bets(
                scored_candidates,
                min_edge=edge_threshold,
                stake=stake,
                edge_column=edge_column,
                probability_column=probability_column,
                selection_mode=selection_mode,
                min_probability=min_probability,
                max_decimal_odds=max_decimal_odds,
                max_bets_per_weekend=max_bets_per_weekend,
                secondary_probability_column="clv_model_probability",
                min_secondary_probability=clv_threshold,
            )
            rows.append(
                {
                    "selection_mode": selection_mode,
                    "min_probability": min_probability or 0.0,
                    "min_edge": edge_threshold,
                    "min_clv_model_probability": clv_threshold,
                    **summarize_selected_bets(bets),
                    **summarize_clv(bets),
                }
            )
    return pd.DataFrame(rows)


def choose_threshold(
    validation_summary: pd.DataFrame,
    *,
    min_bets: int,
) -> float:
    eligible = validation_summary[validation_summary["bets"] >= min_bets]
    if eligible.empty:
        return float("inf")
    profitable = eligible.sort_values(["profit", "roi"], ascending=[False, False])
    best = profitable.iloc[0]
    if float(best["profit"]) <= 0:
        return float("inf")
    return float(best["min_edge"])


def choose_combined_threshold(
    validation_summary: pd.DataFrame,
    *,
    min_bets: int,
) -> tuple[float, float]:
    eligible = validation_summary[validation_summary["bets"] >= min_bets]
    if eligible.empty:
        return float("inf"), float("inf")
    profitable = eligible.sort_values(
        ["profit", "roi", "positive_clv_rate", "avg_decimal_clv"],
        ascending=[False, False, False, False],
    )
    best = profitable.iloc[0]
    if float(best["profit"]) <= 0:
        return float("inf"), float("inf")
    return float(best["min_edge"]), float(best["min_clv_model_probability"])


def normalize_candidate_probabilities(candidates: pd.DataFrame, column: str) -> pd.Series:
    group_sums = candidates.groupby(["match_id", "market"])[column].transform("sum")
    group_sizes = candidates.groupby(["match_id", "market"])[column].transform("size")
    safe_sums = group_sums.where(group_sums > 0)
    normalized = candidates[column] / safe_sums
    return normalized.fillna(1.0 / group_sizes)


def candidate_match_features(
    match: pd.Series,
    market: str,
    outcome: str,
    *,
    decimal_odds: float,
) -> dict[str, float]:
    directional = directional_selection_features(match, market, outcome)
    return {
        "log_decimal_odds": log(decimal_odds),
        "market_1x2": 1.0 if market == "1x2" else 0.0,
        "market_over_under_2_5": 1.0 if market == "over_under_2_5" else 0.0,
        "outcome_home": 1.0 if outcome == "1" else 0.0,
        "outcome_draw": 1.0 if outcome == "X" else 0.0,
        "outcome_away": 1.0 if outcome == "2" else 0.0,
        "outcome_over": 1.0 if outcome == "+" else 0.0,
        "outcome_under": 1.0 if outcome == "-" else 0.0,
        "elo_diff_pre": numeric(match, "elo_diff_pre"),
        "abs_elo_diff_pre": abs(numeric(match, "elo_diff_pre")),
        "rest_days_diff_pre": numeric(match, "rest_days_diff_pre"),
        "form_diff_pre": numeric(match, "form_diff_pre"),
        "venue_form_diff_pre": numeric(match, "venue_form_diff_pre"),
        "goal_form_diff_pre": numeric(match, "goal_form_diff_pre"),
        "defensive_form_diff_pre": numeric(match, "defensive_form_diff_pre"),
        "shot_form_diff_pre": numeric(match, "shot_form_diff_pre"),
        "shot_concession_diff_pre": numeric(match, "shot_concession_diff_pre"),
        "sot_form_diff_pre": numeric(match, "sot_form_diff_pre"),
        "sot_concession_diff_pre": numeric(match, "sot_concession_diff_pre"),
        "xg_form_diff_pre": numeric(match, "xg_form_diff_pre"),
        "xg_concession_diff_pre": numeric(match, "xg_concession_diff_pre"),
        **directional,
    }


def directional_selection_features(match: pd.Series, market: str, outcome: str) -> dict[str, float]:
    elo_diff = numeric(match, "elo_diff_pre")
    rest_diff = numeric(match, "rest_days_diff_pre")
    form_diff = numeric(match, "form_diff_pre")
    venue_form_diff = numeric(match, "venue_form_diff_pre")
    goal_form_diff = numeric(match, "goal_form_diff_pre")
    defensive_diff = numeric(match, "defensive_form_diff_pre")
    shot_diff = numeric(match, "shot_form_diff_pre")
    sot_diff = numeric(match, "sot_form_diff_pre")
    total_goal_signal = (
        numeric(match, "home_goals_for_last5_pre", 1.25)
        + numeric(match, "away_goals_for_last5_pre", 1.25)
        + numeric(match, "home_goals_against_last5_pre", 1.25)
        + numeric(match, "away_goals_against_last5_pre", 1.25)
    )
    total_shot_signal = (
        numeric(match, "home_shots_for_last5_pre", 10.0)
        + numeric(match, "away_shots_for_last5_pre", 10.0)
        + numeric(match, "home_shots_against_last5_pre", 10.0)
        + numeric(match, "away_shots_against_last5_pre", 10.0)
    )
    total_sot_signal = (
        numeric(match, "home_sot_for_last5_pre", 4.0)
        + numeric(match, "away_sot_for_last5_pre", 4.0)
        + numeric(match, "home_sot_against_last5_pre", 4.0)
        + numeric(match, "away_sot_against_last5_pre", 4.0)
    )

    if market == "1x2" and outcome == "1":
        return {
            "selection_elo_diff_pre": elo_diff,
            "selection_rest_diff_pre": rest_diff,
            "selection_form_diff_pre": form_diff,
            "selection_venue_form_diff_pre": venue_form_diff,
            "selection_goal_form_diff_pre": goal_form_diff,
            "selection_defensive_advantage_pre": -defensive_diff,
            "selection_shot_diff_pre": shot_diff,
            "selection_sot_diff_pre": sot_diff,
            "selection_total_goal_signal_pre": 0.0,
            "selection_total_shot_signal_pre": 0.0,
            "selection_total_sot_signal_pre": 0.0,
        }
    if market == "1x2" and outcome == "2":
        return {
            "selection_elo_diff_pre": -elo_diff,
            "selection_rest_diff_pre": -rest_diff,
            "selection_form_diff_pre": -form_diff,
            "selection_venue_form_diff_pre": -venue_form_diff,
            "selection_goal_form_diff_pre": -goal_form_diff,
            "selection_defensive_advantage_pre": defensive_diff,
            "selection_shot_diff_pre": -shot_diff,
            "selection_sot_diff_pre": -sot_diff,
            "selection_total_goal_signal_pre": 0.0,
            "selection_total_shot_signal_pre": 0.0,
            "selection_total_sot_signal_pre": 0.0,
        }
    if market == "1x2" and outcome == "X":
        return {
            "selection_elo_diff_pre": -abs(elo_diff),
            "selection_rest_diff_pre": -abs(rest_diff),
            "selection_form_diff_pre": -abs(form_diff),
            "selection_venue_form_diff_pre": -abs(venue_form_diff),
            "selection_goal_form_diff_pre": -abs(goal_form_diff),
            "selection_defensive_advantage_pre": -abs(defensive_diff),
            "selection_shot_diff_pre": -abs(shot_diff),
            "selection_sot_diff_pre": -abs(sot_diff),
            "selection_total_goal_signal_pre": 0.0,
            "selection_total_shot_signal_pre": 0.0,
            "selection_total_sot_signal_pre": 0.0,
        }

    total_multiplier = 1.0 if outcome == "+" else -1.0
    return {
        "selection_elo_diff_pre": 0.0,
        "selection_rest_diff_pre": 0.0,
        "selection_form_diff_pre": 0.0,
        "selection_venue_form_diff_pre": 0.0,
        "selection_goal_form_diff_pre": 0.0,
        "selection_defensive_advantage_pre": 0.0,
        "selection_shot_diff_pre": 0.0,
        "selection_sot_diff_pre": 0.0,
        "selection_total_goal_signal_pre": total_multiplier * total_goal_signal,
        "selection_total_shot_signal_pre": total_multiplier * total_shot_signal,
        "selection_total_sot_signal_pre": total_multiplier * total_sot_signal,
    }


def build_value_features(candidates: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    features = pd.DataFrame(index=candidates.index)
    for column in columns:
        if column in candidates.columns:
            features[column] = pd.to_numeric(candidates[column], errors="coerce").fillna(0.0)
        else:
            features[column] = 0.0
    return features


def transform_value_features(
    candidates: pd.DataFrame,
    columns: tuple[str, ...],
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    values = build_value_features(candidates, columns).to_numpy(dtype=float)
    return (values - means) / scales


def fit_logistic_weights(
    features: np.ndarray,
    target: np.ndarray,
    *,
    learning_rate: float,
    epochs: int,
    l2: float,
) -> np.ndarray:
    weights = np.zeros(features.shape[1], dtype=float)
    positive_rate = min(max(float(target.mean()), 1e-4), 1.0 - 1e-4)
    weights[0] = np.log(positive_rate / (1.0 - positive_rate))

    for _ in range(epochs):
        probabilities = sigmoid(features @ weights)
        error = probabilities - target
        gradient = features.T @ error / len(target)
        gradient[1:] += l2 * weights[1:]
        weights -= learning_rate * gradient

    return weights


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def add_intercept(features: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(features.shape[0]), features])


def numeric(row: pd.Series, column: str, default: float = 0.0) -> float:
    if column not in row:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return float(value)


def safe_odds(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    value = float(value)
    return value if is_valid_decimal_odds(value) else float("nan")


def weekend_key(dates: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(dates)
    # Tuesday-Monday periods keep the Friday-Monday club weekend together.
    return parsed.dt.to_period("W-MON").astype(str)


def empty_candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "match_id",
            "date",
            "season_start_year",
            "season",
            "league",
            "home_team",
            "away_team",
            "market",
            "outcome",
            "actual_result",
            "model_probability",
            "bookmaker_probability",
            "closing_bookmaker_probability",
            "probability_gap",
            "abs_probability_gap",
            "decimal_odds",
            "closing_decimal_odds",
            "clv_decimal",
            "clv_probability",
            "fair_odds",
            "raw_edge",
            "won",
            "profit",
        ]
    )


def empty_selected_bets_frame() -> pd.DataFrame:
    frame = empty_candidate_frame()
    frame["stake"] = []
    frame["settled_profit"] = []
    frame["selected_probability"] = []
    frame["selected_edge"] = []
    return frame
