"""Market-anchored probability scoring for Conservative V2."""

from __future__ import annotations

from math import isinf
from typing import Mapping

import numpy as np
import pandas as pd

from .value_model import empty_selected_bets_frame, select_value_bets, weekend_key


DEFAULT_ANCHOR_ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


def score_market_anchored_candidates(
    scored_candidates: pd.DataFrame,
    alpha_by_market: Mapping[str, float],
    *,
    model_probability_column: str = "value_probability",
) -> pd.DataFrame:
    """Shrink residual probabilities toward no-vig opening-market probabilities."""

    scored = scored_candidates.copy()
    scored["anchor_alpha"] = scored["market"].map(alpha_by_market).fillna(0.0).astype(float)
    if ((scored["anchor_alpha"] < 0.0) | (scored["anchor_alpha"] > 1.0)).any():
        raise ValueError("Market anchor alphas must be between 0 and 1.")

    model = pd.to_numeric(scored[model_probability_column], errors="coerce")
    bookmaker = pd.to_numeric(scored["bookmaker_probability"], errors="coerce")
    scored["conservative_probability_raw"] = (
        scored["anchor_alpha"] * model + (1.0 - scored["anchor_alpha"]) * bookmaker
    )
    group_sums = scored.groupby(["match_id", "market"])[
        "conservative_probability_raw"
    ].transform("sum")
    scored["conservative_probability"] = (
        scored["conservative_probability_raw"] / group_sums.where(group_sums > 0.0)
    )
    scored["conservative_probability"] = scored["conservative_probability"].fillna(bookmaker)
    scored["conservative_edge"] = (
        scored["conservative_probability"] * scored["decimal_odds"] - 1.0
    )
    return scored


def fit_market_anchor_alphas(
    validation_candidates: pd.DataFrame,
    *,
    model_probability_column: str = "value_probability",
    alpha_candidates: tuple[float, ...] = DEFAULT_ANCHOR_ALPHAS,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Choose one capped model weight per market using validation log loss."""

    if not alpha_candidates:
        raise ValueError("Need at least one anchor alpha candidate.")
    if any(alpha < 0.0 or alpha > 1.0 for alpha in alpha_candidates):
        raise ValueError("Anchor alpha candidates must be between 0 and 1.")

    chosen: dict[str, float] = {}
    rows: list[dict[str, float | str]] = []
    for market in sorted(validation_candidates["market"].dropna().unique()):
        market_candidates = validation_candidates[
            validation_candidates["market"] == market
        ].copy()
        best_alpha = float(alpha_candidates[0])
        best_log_loss = float("inf")
        for alpha in alpha_candidates:
            anchored = score_market_anchored_candidates(
                market_candidates,
                {str(market): float(alpha)},
                model_probability_column=model_probability_column,
            )
            metrics = market_probability_metrics(anchored, "conservative_probability")
            rows.append({"market": str(market), "alpha": float(alpha), **metrics})
            if metrics["log_loss"] < best_log_loss:
                best_log_loss = metrics["log_loss"]
                best_alpha = float(alpha)
        chosen[str(market)] = best_alpha
    return chosen, pd.DataFrame(rows)


def market_probability_metrics(
    candidates: pd.DataFrame,
    probability_column: str,
) -> dict[str, float]:
    """Return match-level multiclass metrics from one-row-per-outcome candidates."""

    usable = candidates[
        pd.notna(candidates[probability_column]) & pd.notna(candidates["won"])
    ].copy()
    if usable.empty:
        return {"accuracy": 0.0, "log_loss": float("inf"), "brier": float("inf")}

    probabilities = np.clip(
        pd.to_numeric(usable[probability_column], errors="coerce").to_numpy(dtype=float),
        1e-12,
        1.0,
    )
    targets = usable["won"].astype(float).to_numpy()
    usable["_probability"] = probabilities
    usable["_squared_error"] = (probabilities - targets) ** 2

    winning_rows = usable[usable["won"].astype(bool)]
    if winning_rows.empty:
        return {"accuracy": 0.0, "log_loss": float("inf"), "brier": float("inf")}
    log_loss = float(-np.log(winning_rows["_probability"]).mean())

    predicted_indices = usable.groupby(["match_id", "market"])["_probability"].idxmax()
    accuracy = float(usable.loc[predicted_indices, "won"].astype(bool).mean())
    brier = float(
        usable.groupby(["match_id", "market"])["_squared_error"].sum().mean()
    )
    return {"accuracy": accuracy, "log_loss": log_loss, "brier": brier}


def probability_diagnostics(
    candidates: pd.DataFrame,
    probability_columns: Mapping[str, str],
) -> pd.DataFrame:
    rows = []
    for market in sorted(candidates["market"].dropna().unique()):
        market_candidates = candidates[candidates["market"] == market]
        for source, column in probability_columns.items():
            rows.append(
                {
                    "market": market,
                    "probability_source": source,
                    **market_probability_metrics(market_candidates, column),
                }
            )
    return pd.DataFrame(rows)


def calibration_table(
    candidates: pd.DataFrame,
    *,
    probability_column: str,
    bin_width: float = 0.1,
) -> pd.DataFrame:
    if bin_width <= 0.0 or bin_width > 1.0:
        raise ValueError("bin_width must be between 0 and 1.")
    usable = candidates[pd.notna(candidates[probability_column])].copy()
    probabilities = pd.to_numeric(usable[probability_column], errors="coerce").clip(0.0, 1.0)
    bin_index = np.minimum((probabilities / bin_width).astype(int), int(1.0 / bin_width) - 1)
    usable["probability_bin"] = bin_index.map(
        lambda index: f"{index * bin_width:.1f}-{(index + 1) * bin_width:.1f}"
    )
    return (
        usable.groupby(["market", "probability_bin"], observed=True)
        .agg(
            candidates=("won", "size"),
            predicted_probability=(probability_column, "mean"),
            observed_frequency=("won", "mean"),
        )
        .reset_index()
    )


def select_conservative_v2_bets(
    scored_candidates: pd.DataFrame,
    thresholds_by_market: Mapping[str, float],
    *,
    min_probability: float = 0.55,
    max_decimal_odds: float = 2.5,
    stake: float = 1.0,
    max_bets_per_market_weekend: int = 1,
    max_bets_per_weekend: int = 3,
) -> pd.DataFrame:
    """Select independently within each market, then enforce global match/weekend limits."""

    frames = []
    for market, threshold in thresholds_by_market.items():
        if isinf(threshold):
            continue
        market_candidates = scored_candidates[scored_candidates["market"] == market]
        selected = select_value_bets(
            market_candidates,
            min_edge=threshold,
            stake=stake,
            max_decimal_odds=max_decimal_odds,
            max_bets_per_weekend=max_bets_per_market_weekend,
            edge_column="conservative_edge",
            probability_column="conservative_probability",
            selection_mode="conservative",
            min_probability=min_probability,
        )
        if not selected.empty:
            frames.append(selected)
    if not frames:
        return empty_selected_bets_frame()

    selected = pd.concat(frames, ignore_index=True)
    selected["weekend"] = weekend_key(selected["date"])
    selected = selected.sort_values(
        ["date", "match_id", "selected_edge", "selected_probability"],
        ascending=[True, True, False, False],
    )
    selected = selected.groupby("match_id", group_keys=False).head(1)
    if max_bets_per_weekend > 0:
        selected = selected.sort_values(
            ["weekend", "selected_edge", "selected_probability"],
            ascending=[True, False, False],
        )
        selected = selected.groupby("weekend", group_keys=False).head(max_bets_per_weekend)
    selected["selection_mode"] = "conservative_v2"
    return selected.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)
