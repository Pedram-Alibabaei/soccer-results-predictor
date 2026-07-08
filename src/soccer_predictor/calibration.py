"""Probability calibration helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .modeling import evaluate_predictions


def blend_probabilities(
    model_probabilities: pd.DataFrame,
    anchor_probabilities: pd.DataFrame,
    *,
    alpha: float,
    labels: tuple[str, ...],
) -> pd.DataFrame:
    """Blend model probabilities with an anchor such as bookmaker no-vig prices."""

    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be between 0 and 1.")

    blended = (
        alpha * model_probabilities.loc[:, labels].astype(float)
        + (1.0 - alpha) * anchor_probabilities.loc[:, labels].astype(float)
    )
    row_sums = blended.sum(axis=1).replace(0.0, np.nan)
    return blended.div(row_sums, axis=0).fillna(1.0 / len(labels))


def fit_blend_alpha(
    actual: pd.Series,
    model_probabilities: pd.DataFrame,
    anchor_probabilities: pd.DataFrame,
    *,
    labels: tuple[str, ...],
    candidates: tuple[float, ...] | None = None,
) -> tuple[float, pd.DataFrame]:
    """Choose the blend alpha with the lowest validation log loss."""

    alpha_candidates = candidates or tuple(np.linspace(0.0, 1.0, 21))
    rows = []
    best_alpha = 0.0
    best_log_loss = float("inf")
    for alpha in alpha_candidates:
        blended = blend_probabilities(
            model_probabilities,
            anchor_probabilities,
            alpha=float(alpha),
            labels=labels,
        )
        metrics = evaluate_predictions(actual, blended, labels=labels)
        log_loss = float(metrics["log_loss"])
        rows.append(
            {
                "alpha": float(alpha),
                "accuracy": float(metrics["accuracy"]),
                "log_loss": log_loss,
                "brier": float(metrics["brier"]),
            }
        )
        if log_loss < best_log_loss:
            best_alpha = float(alpha)
            best_log_loss = log_loss

    return best_alpha, pd.DataFrame(rows)
