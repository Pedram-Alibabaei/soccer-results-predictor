"""Context-adjusted and overdispersed goal probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma, log
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .markets import (
    ScoreProbabilityTable,
    negative_binomial_score_matrix,
    over_under,
    three_way,
)
from .modeling import (
    BASELINE_FEATURE_COLUMNS,
    PoissonGoalModel,
    fit_poisson_goal_model,
    recency_weights,
)


CONTEXTUAL_BASE_FEATURE_COLUMNS = tuple(
    column for column in BASELINE_FEATURE_COLUMNS if column != "is_world_cup_third_place"
)
DEFAULT_DISPERSION_GRID = tuple(np.linspace(0.0, 0.75, 61))


@dataclass(frozen=True)
class StageAdjustment:
    multiplier: float
    match_count: int
    observed_goals: float
    predicted_goals: float
    prior_exposure: float


@dataclass(frozen=True)
class ContextualGoalModel:
    base_model: PoissonGoalModel
    third_place_adjustment: StageAdjustment
    dispersion: float
    max_goals: int = 15

    def predict_expected_goals(self, frame: pd.DataFrame) -> pd.DataFrame:
        expected = self.base_model.predict_expected_goals(frame).copy()
        if "is_world_cup_third_place" not in frame.columns:
            return expected

        mask = frame["is_world_cup_third_place"].fillna(False).astype(bool)
        expected.loc[mask, ["home_xg", "away_xg"]] *= self.third_place_adjustment.multiplier
        return expected

    def predict_score_tables(self, frame: pd.DataFrame) -> list[ScoreProbabilityTable]:
        expected = self.predict_expected_goals(frame)
        tables = []
        for index, row in expected.iterrows():
            match = frame.loc[index]
            tables.append(
                negative_binomial_score_matrix(
                    float(row["home_xg"]),
                    float(row["away_xg"]),
                    dispersion=self.dispersion,
                    max_goals=self.max_goals,
                    home_team=str(match.get("home_team", "Home")),
                    away_team=str(match.get("away_team", "Away")),
                )
            )
        return tables

    def predict_proba(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows = [three_way(table) for table in self.predict_score_tables(frame)]
        return pd.DataFrame(rows, index=frame.index).loc[:, ["1", "X", "2"]]

    def predict_risk(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for table in self.predict_score_tables(frame):
            totals = [(home + away, probability) for home, away, probability in table.scores()]
            mean = sum(total * probability for total, probability in totals)
            variance = sum(((total - mean) ** 2) * probability for total, probability in totals)
            dispersion_index = variance / mean if mean > 1e-12 else 1.0
            over_1_5 = over_under(table, 1.5)["+"]
            over_2_5 = over_under(table, 2.5)["+"]
            over_3_5 = over_under(table, 3.5)["+"]
            probability_5_plus = over_under(table, 4.5)["+"]
            probability_6_plus = over_under(table, 5.5)["+"]
            probability_10_plus = over_under(table, 9.5)["+"]
            rows.append(
                {
                    "expected_total_goals": mean,
                    "total_goals_variance": variance,
                    "dispersion_index": dispersion_index,
                    "probability_over_1_5": over_1_5,
                    "probability_over_2_5": over_2_5,
                    "probability_over_3_5": over_3_5,
                    "probability_under_3_5": 1.0 - over_3_5,
                    "probability_5_plus": probability_5_plus,
                    "probability_6_plus": probability_6_plus,
                    "probability_10_plus": probability_10_plus,
                    "high_variance": bool(
                        dispersion_index >= 1.25 or probability_6_plus >= 0.08
                    ),
                }
            )
        return pd.DataFrame(rows, index=frame.index)


def load_third_place_history(path: Path) -> pd.DataFrame:
    history = pd.read_csv(path, parse_dates=["date"])
    required = {
        "date",
        "home_team",
        "away_team",
        "regulation_home_goals",
        "regulation_away_goals",
    }
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Third-place history is missing columns: {sorted(missing)}")
    return history.sort_values("date", kind="mergesort").reset_index(drop=True)


def fit_contextual_goal_model(
    train: pd.DataFrame,
    third_place_history: pd.DataFrame,
    *,
    learning_rate: float = 0.03,
    epochs: int = 2500,
    l2: float = 0.01,
    half_life_days: float | None = 3650.0,
    context_half_life_days: float | None = 7300.0,
    prior_exposure: float = 30.0,
    max_goals: int = 15,
    dispersion_grid: Iterable[float] = DEFAULT_DISPERSION_GRID,
) -> ContextualGoalModel:
    base_model = fit_poisson_goal_model(
        train,
        feature_columns=CONTEXTUAL_BASE_FEATURE_COLUMNS,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
        half_life_days=half_life_days,
        max_goals=max_goals,
    )
    adjustment = fit_third_place_adjustment(
        train,
        third_place_history,
        base_model,
        prior_exposure=prior_exposure,
        half_life_days=context_half_life_days,
    )

    calibration = (
        train[train["is_world_cup"].astype(bool)].copy()
        if "is_world_cup" in train.columns
        else train.copy()
    )
    if calibration.empty:
        calibration = train.copy()
    expected = base_model.predict_expected_goals(calibration)
    expected_total = expected["home_xg"].to_numpy() + expected["away_xg"].to_numpy()
    if "is_world_cup_third_place" in calibration.columns:
        context_mask = calibration["is_world_cup_third_place"].fillna(False).astype(bool).to_numpy()
        expected_total[context_mask] *= adjustment.multiplier
    actual_total = (
        calibration["home_goals"].to_numpy(dtype=float)
        + calibration["away_goals"].to_numpy(dtype=float)
    )
    weights = recency_weights(calibration["date"], half_life_days)
    dispersion = fit_negative_binomial_dispersion(
        actual_total,
        expected_total,
        sample_weights=weights,
        grid=dispersion_grid,
    )
    return ContextualGoalModel(
        base_model=base_model,
        third_place_adjustment=adjustment,
        dispersion=dispersion,
        max_goals=max_goals,
    )


def fit_third_place_adjustment(
    train: pd.DataFrame,
    history: pd.DataFrame,
    base_model: PoissonGoalModel,
    *,
    prior_exposure: float = 30.0,
    half_life_days: float | None = 7300.0,
) -> StageAdjustment:
    if prior_exposure <= 0:
        raise ValueError("prior_exposure must be positive.")

    cutoff = pd.to_datetime(train["date"]).max()
    eligible = history[pd.to_datetime(history["date"]) <= cutoff].copy()
    keyed_train = train.copy()
    if "tournament" in keyed_train.columns:
        keyed_train = keyed_train[keyed_train["tournament"].eq("FIFA World Cup")]
    keyed_train = keyed_train.reset_index().rename(columns={"index": "_train_index"})
    matched = eligible.merge(
        keyed_train,
        on=["date", "home_team", "away_team"],
        how="inner",
        validate="one_to_one",
    )
    if matched.empty:
        return StageAdjustment(1.0, 0, 0.0, 0.0, prior_exposure)

    matched_frame = train.loc[matched["_train_index"].to_numpy()]
    predictions = base_model.predict_expected_goals(matched_frame)
    predicted_total = predictions["home_xg"].to_numpy() + predictions["away_xg"].to_numpy()
    observed_total = (
        matched["regulation_home_goals"].to_numpy(dtype=float)
        + matched["regulation_away_goals"].to_numpy(dtype=float)
    )
    weights = recency_weights(matched["date"], half_life_days)
    weighted_observed = float(np.dot(weights, observed_total))
    weighted_predicted = float(np.dot(weights, predicted_total))
    multiplier = (prior_exposure + weighted_observed) / (
        prior_exposure + weighted_predicted
    )
    multiplier = float(np.clip(multiplier, 0.75, 1.35))
    return StageAdjustment(
        multiplier=multiplier,
        match_count=len(matched),
        observed_goals=weighted_observed,
        predicted_goals=weighted_predicted,
        prior_exposure=prior_exposure,
    )


def fit_negative_binomial_dispersion(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    sample_weights: np.ndarray | None = None,
    grid: Iterable[float] = DEFAULT_DISPERSION_GRID,
) -> float:
    actual = np.asarray(actual, dtype=float)
    expected = np.clip(np.asarray(expected, dtype=float), 1e-9, None)
    if len(actual) != len(expected):
        raise ValueError("actual and expected must have the same length.")
    weights = (
        np.ones(len(actual), dtype=float)
        if sample_weights is None
        else np.asarray(sample_weights, dtype=float)
    )
    if len(weights) != len(actual):
        raise ValueError("sample_weights must match actual.")

    candidates = [float(value) for value in grid]
    if not candidates or any(value < 0 for value in candidates):
        raise ValueError("dispersion grid must contain non-negative values.")
    losses = [
        float(np.average(_negative_binomial_nll(actual, expected, value), weights=weights))
        for value in candidates
    ]
    return candidates[int(np.argmin(losses))]


def _negative_binomial_nll(actual: np.ndarray, expected: np.ndarray, dispersion: float) -> np.ndarray:
    if dispersion <= 1e-12:
        return expected - (actual * np.log(expected)) + np.array(
            [lgamma(value + 1.0) for value in actual]
        )

    shape = 1.0 / dispersion
    success_probability = shape / (shape + expected)
    return -np.array(
        [
            lgamma(value + shape)
            - lgamma(shape)
            - lgamma(value + 1.0)
            + (shape * log(probability))
            + (value * log(1.0 - probability))
            for value, probability in zip(actual, success_probability, strict=True)
        ]
    )
