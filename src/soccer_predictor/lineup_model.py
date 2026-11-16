"""Regularized confirmed-lineup adjustment for a stable goal model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .markets import score_matrix, three_way
from .modeling import (
    CLASS_LABELS,
    FeatureScaler,
    PoissonGoalModel,
    add_intercept,
    fit_feature_scaler,
    recency_weights,
)


ROTATION_ADJUSTMENT_FEATURE_COLUMNS = (
    "home_lineup_changes_pre",
    "away_lineup_changes_pre",
    "home_goalkeeper_changed_pre",
    "away_goalkeeper_changed_pre",
    "home_defensive_continuity_pre",
    "away_defensive_continuity_pre",
    "home_attacking_continuity_pre",
    "away_attacking_continuity_pre",
)

POSITION_STRENGTH_ADJUSTMENT_FEATURE_COLUMNS = (
    "home_goalkeeper_strength_change_pre",
    "away_goalkeeper_strength_change_pre",
    "home_defense_strength_change_pre",
    "away_defense_strength_change_pre",
    "home_midfield_strength_change_pre",
    "away_midfield_strength_change_pre",
    "home_attack_strength_change_pre",
    "away_attack_strength_change_pre",
    "home_lineup_strength_change_pre",
    "away_lineup_strength_change_pre",
    "min_player_strength_coverage_pre",
)

LINEUP_ADJUSTMENT_FEATURE_COLUMNS = (
    ROTATION_ADJUSTMENT_FEATURE_COLUMNS + POSITION_STRENGTH_ADJUSTMENT_FEATURE_COLUMNS
)


@dataclass(frozen=True)
class LineupAdjustedGoalModel:
    baseline_model: PoissonGoalModel
    lineup_scaler: FeatureScaler
    home_adjustment_weights: np.ndarray
    away_adjustment_weights: np.ndarray
    correction_strength: float = 0.25
    max_log_adjustment: float = 0.35

    def predict_expected_goals(self, frame: pd.DataFrame) -> pd.DataFrame:
        baseline = self.baseline_model.predict_expected_goals(frame)
        features = add_intercept(self.lineup_scaler.transform(frame))
        available = pd.to_numeric(
            frame.get("has_confirmed_lineups", pd.Series(0.0, index=frame.index)),
            errors="coerce",
        ).fillna(0.0).to_numpy()
        home_log_adjustment = np.clip(
            self.correction_strength * (features @ self.home_adjustment_weights),
            -self.max_log_adjustment,
            self.max_log_adjustment,
        ) * available
        away_log_adjustment = np.clip(
            self.correction_strength * (features @ self.away_adjustment_weights),
            -self.max_log_adjustment,
            self.max_log_adjustment,
        ) * available
        return pd.DataFrame(
            {
                "home_xg": baseline["home_xg"].to_numpy() * np.exp(home_log_adjustment),
                "away_xg": baseline["away_xg"].to_numpy() * np.exp(away_log_adjustment),
            },
            index=frame.index,
        )

    def predict_proba(self, frame: pd.DataFrame) -> pd.DataFrame:
        expected = self.predict_expected_goals(frame)
        rows = []
        for row in expected.itertuples(index=False):
            rows.append(
                three_way(
                    score_matrix(
                        row.home_xg,
                        row.away_xg,
                        max_goals=self.baseline_model.max_goals,
                        dixon_coles_rho=self.baseline_model.dixon_coles_rho,
                    )
                )
            )
        return pd.DataFrame(rows, index=frame.index).loc[:, list(CLASS_LABELS)]


def fit_lineup_adjusted_goal_model(
    baseline_model: PoissonGoalModel,
    train: pd.DataFrame,
    *,
    baseline_expected_goals: pd.DataFrame | None = None,
    feature_columns: tuple[str, ...] = LINEUP_ADJUSTMENT_FEATURE_COLUMNS,
    correction_strength: float = 0.25,
    learning_rate: float = 0.03,
    epochs: int = 2000,
    l2: float = 1.0,
    half_life_days: float | None = None,
    max_log_adjustment: float = 0.35,
) -> LineupAdjustedGoalModel:
    """Fit a small Poisson offset layer using only confirmed-lineup matches."""

    if len(train) < 30:
        raise ValueError("At least 30 confirmed-lineup matches are required for adjustment fitting.")
    if not pd.to_numeric(train["has_confirmed_lineups"], errors="coerce").eq(1).all():
        raise ValueError("Lineup adjustment training rows must all have confirmed lineups.")

    baseline = (
        baseline_model.predict_expected_goals(train)
        if baseline_expected_goals is None
        else baseline_expected_goals.loc[train.index]
    )
    scaler = fit_feature_scaler(train, feature_columns)
    features = add_intercept(scaler.transform(train))
    sample_weights = recency_weights(train["date"], half_life_days)
    home_weights = fit_poisson_offset_weights(
        features,
        train["home_goals"].to_numpy(dtype=float),
        baseline["home_xg"].to_numpy(dtype=float),
        sample_weights=sample_weights,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
    )
    away_weights = fit_poisson_offset_weights(
        features,
        train["away_goals"].to_numpy(dtype=float),
        baseline["away_xg"].to_numpy(dtype=float),
        sample_weights=sample_weights,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
    )
    return LineupAdjustedGoalModel(
        baseline_model=baseline_model,
        lineup_scaler=scaler,
        home_adjustment_weights=home_weights,
        away_adjustment_weights=away_weights,
        correction_strength=correction_strength,
        max_log_adjustment=max_log_adjustment,
    )


def fit_poisson_offset_weights(
    features: np.ndarray,
    target: np.ndarray,
    baseline_rate: np.ndarray,
    *,
    sample_weights: np.ndarray,
    learning_rate: float,
    epochs: int,
    l2: float,
) -> np.ndarray:
    """Fit multiplicative goal-rate adjustments around fixed baseline rates."""

    weights = np.zeros(features.shape[1], dtype=float)
    normalized_weights = sample_weights / sample_weights.sum()
    safe_baseline = np.clip(baseline_rate, 0.05, 10.0)
    for _ in range(epochs):
        linear = np.clip(np.log(safe_baseline) + (features @ weights), -5.0, 3.5)
        expected = np.exp(linear)
        gradient = features.T @ ((expected - target) * normalized_weights)
        gradient[1:] += l2 * weights[1:]
        weights -= learning_rate * gradient
    return weights
