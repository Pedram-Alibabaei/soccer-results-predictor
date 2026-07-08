"""Small baseline models and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .markets import score_matrix, three_way


CLASS_LABELS = ("1", "X", "2")

BASELINE_FEATURE_COLUMNS = (
    "home_elo_pre",
    "away_elo_pre",
    "elo_diff_pre",
    "home_attack_pre",
    "away_attack_pre",
    "home_defense_pre",
    "away_defense_pre",
    "home_matches_played_pre",
    "away_matches_played_pre",
    "neutral",
    "abs_elo_diff_pre",
    "attack_gap_pre",
    "defense_gap_pre",
    "combined_attack_pre",
    "combined_defense_pre",
    "goal_balance_gap_pre",
    "experience_gap_pre",
    "min_matches_played_pre",
    "is_world_cup",
    "world_cup_match_number_scaled",
    "is_world_cup_group_stage",
    "is_world_cup_knockout_stage",
    "is_world_cup_round_of_16",
    "is_world_cup_quarter_final",
    "is_world_cup_semi_final",
    "is_world_cup_third_place",
    "is_world_cup_final",
    "world_cup_group_matchday",
    "is_world_cup_final_group_match",
    "home_rest_days_pre",
    "away_rest_days_pre",
    "rest_days_diff_pre",
    "home_points_per_match_last5_pre",
    "away_points_per_match_last5_pre",
    "form_diff_pre",
    "home_home_points_per_match_last5_pre",
    "away_away_points_per_match_last5_pre",
    "venue_form_diff_pre",
    "home_goals_for_last5_pre",
    "away_goals_for_last5_pre",
    "home_goals_against_last5_pre",
    "away_goals_against_last5_pre",
    "goal_form_diff_pre",
    "defensive_form_diff_pre",
    "home_shots_for_last5_pre",
    "away_shots_for_last5_pre",
    "home_shots_against_last5_pre",
    "away_shots_against_last5_pre",
    "shot_form_diff_pre",
    "shot_concession_diff_pre",
    "home_sot_for_last5_pre",
    "away_sot_for_last5_pre",
    "home_sot_against_last5_pre",
    "away_sot_against_last5_pre",
    "sot_form_diff_pre",
    "sot_concession_diff_pre",
    "home_xg_for_last5_pre",
    "away_xg_for_last5_pre",
    "home_xg_against_last5_pre",
    "away_xg_against_last5_pre",
    "xg_form_diff_pre",
    "xg_concession_diff_pre",
)

WORLD_CUP_CONTEXT_FEATURE_COLUMNS = {
    "is_world_cup",
    "world_cup_match_number_scaled",
    "is_world_cup_group_stage",
    "is_world_cup_knockout_stage",
    "is_world_cup_round_of_16",
    "is_world_cup_quarter_final",
    "is_world_cup_semi_final",
    "is_world_cup_third_place",
    "is_world_cup_final",
    "world_cup_group_matchday",
    "is_world_cup_final_group_match",
}

OPTIONAL_NUMERIC_FEATURE_DEFAULTS = {
    "home_rest_days_pre": 7.0,
    "away_rest_days_pre": 7.0,
    "rest_days_diff_pre": 0.0,
    "home_points_per_match_last5_pre": 1.0,
    "away_points_per_match_last5_pre": 1.0,
    "form_diff_pre": 0.0,
    "home_home_points_per_match_last5_pre": 1.0,
    "away_away_points_per_match_last5_pre": 1.0,
    "venue_form_diff_pre": 0.0,
    "home_goals_for_last5_pre": 1.25,
    "away_goals_for_last5_pre": 1.25,
    "home_goals_against_last5_pre": 1.25,
    "away_goals_against_last5_pre": 1.25,
    "goal_form_diff_pre": 0.0,
    "defensive_form_diff_pre": 0.0,
    "home_shots_for_last5_pre": 10.0,
    "away_shots_for_last5_pre": 10.0,
    "home_shots_against_last5_pre": 10.0,
    "away_shots_against_last5_pre": 10.0,
    "shot_form_diff_pre": 0.0,
    "shot_concession_diff_pre": 0.0,
    "home_sot_for_last5_pre": 4.0,
    "away_sot_for_last5_pre": 4.0,
    "home_sot_against_last5_pre": 4.0,
    "away_sot_against_last5_pre": 4.0,
    "sot_form_diff_pre": 0.0,
    "sot_concession_diff_pre": 0.0,
    "home_xg_for_last5_pre": 0.0,
    "away_xg_for_last5_pre": 0.0,
    "home_xg_against_last5_pre": 0.0,
    "away_xg_against_last5_pre": 0.0,
    "xg_form_diff_pre": 0.0,
    "xg_concession_diff_pre": 0.0,
}


@dataclass(frozen=True)
class FeatureScaler:
    columns: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = build_model_features(frame, self.columns).astype(float).to_numpy()
        return (values - self.means) / self.scales


@dataclass(frozen=True)
class SoftmaxModel:
    labels: tuple[str, ...]
    feature_scaler: FeatureScaler
    weights: np.ndarray

    def predict_proba(self, frame: pd.DataFrame) -> pd.DataFrame:
        features = add_intercept(self.feature_scaler.transform(frame))
        probabilities = softmax(features @ self.weights)
        return pd.DataFrame(probabilities, columns=list(self.labels), index=frame.index)

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        probabilities = self.predict_proba(frame)
        return probabilities.idxmax(axis=1)


@dataclass(frozen=True)
class PoissonGoalModel:
    feature_scaler: FeatureScaler
    home_weights: np.ndarray
    away_weights: np.ndarray
    max_goals: int = 10
    dixon_coles_rho: float | None = None

    def predict_expected_goals(self, frame: pd.DataFrame) -> pd.DataFrame:
        features = add_intercept(self.feature_scaler.transform(frame))
        home_xg = np.exp(np.clip(features @ self.home_weights, -5.0, 3.5))
        away_xg = np.exp(np.clip(features @ self.away_weights, -5.0, 3.5))
        return pd.DataFrame({"home_xg": home_xg, "away_xg": away_xg}, index=frame.index)

    def predict_proba(self, frame: pd.DataFrame) -> pd.DataFrame:
        expected_goals = self.predict_expected_goals(frame)
        rows = []
        for row in expected_goals.itertuples(index=False):
            table = score_matrix(
                row.home_xg,
                row.away_xg,
                max_goals=self.max_goals,
                dixon_coles_rho=self.dixon_coles_rho,
            )
            rows.append(three_way(table))
        return pd.DataFrame(rows, index=frame.index).loc[:, list(CLASS_LABELS)]

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        probabilities = self.predict_proba(frame)
        return probabilities.idxmax(axis=1)


@dataclass(frozen=True)
class DrawAwarePoissonModel:
    goal_model: PoissonGoalModel
    draw_scaler: FeatureScaler
    draw_weights: np.ndarray
    draw_temperature: float = 1.0

    def predict_expected_goals(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.goal_model.predict_expected_goals(frame)

    def predict_draw_probability(self, frame: pd.DataFrame) -> np.ndarray:
        features = add_intercept(self.draw_scaler.transform(frame))
        scores = (features @ self.draw_weights) / self.draw_temperature
        probabilities = softmax(scores)
        return probabilities[:, 1]

    def predict_proba(self, frame: pd.DataFrame) -> pd.DataFrame:
        base_probabilities = self.goal_model.predict_proba(frame)
        draw_probability = self.predict_draw_probability(frame)
        non_draw_probability = 1.0 - draw_probability
        base_win_mass = (base_probabilities["1"] + base_probabilities["2"]).to_numpy()
        safe_win_mass = np.where(base_win_mass <= 1e-12, 1.0, base_win_mass)
        home_share = base_probabilities["1"].to_numpy() / safe_win_mass
        away_share = base_probabilities["2"].to_numpy() / safe_win_mass

        probabilities = pd.DataFrame(
            {
                "1": non_draw_probability * home_share,
                "X": draw_probability,
                "2": non_draw_probability * away_share,
            },
            index=frame.index,
        )
        return probabilities.loc[:, list(CLASS_LABELS)]

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        probabilities = self.predict_proba(frame)
        return probabilities.idxmax(axis=1)


def fit_softmax_baseline(
    train: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = BASELINE_FEATURE_COLUMNS,
    labels: tuple[str, ...] = CLASS_LABELS,
    learning_rate: float = 0.05,
    epochs: int = 2500,
    l2: float = 0.01,
    half_life_days: float | None = 3650.0,
) -> SoftmaxModel:
    """Fit a compact multinomial logistic baseline with optional recency weights."""

    scaler = fit_feature_scaler(train, feature_columns)
    features = add_intercept(scaler.transform(train))
    target = encode_labels(train["result"], labels)
    sample_weights = recency_weights(train["date"], half_life_days)
    weights = fit_softmax_weights(
        features,
        target,
        len(labels),
        sample_weights=sample_weights,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
    )
    return SoftmaxModel(labels=labels, feature_scaler=scaler, weights=weights)


def fit_poisson_goal_model(
    train: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = BASELINE_FEATURE_COLUMNS,
    learning_rate: float = 0.03,
    epochs: int = 2500,
    l2: float = 0.01,
    half_life_days: float | None = 3650.0,
    max_goals: int = 10,
) -> PoissonGoalModel:
    """Fit separate Poisson goal models for home and away goals."""

    scaler = fit_feature_scaler(train, feature_columns)
    features = add_intercept(scaler.transform(train))
    sample_weights = recency_weights(train["date"], half_life_days)
    home_weights = fit_poisson_weights(
        features,
        train["home_goals"].to_numpy(dtype=float),
        sample_weights=sample_weights,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
    )
    away_weights = fit_poisson_weights(
        features,
        train["away_goals"].to_numpy(dtype=float),
        sample_weights=sample_weights,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
    )
    return PoissonGoalModel(
        feature_scaler=scaler,
        home_weights=home_weights,
        away_weights=away_weights,
        max_goals=max_goals,
    )


def fit_dixon_coles_goal_model(
    train: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = BASELINE_FEATURE_COLUMNS,
    learning_rate: float = 0.03,
    epochs: int = 2500,
    l2: float = 0.01,
    half_life_days: float | None = 3650.0,
    max_goals: int = 10,
) -> PoissonGoalModel:
    """Fit a Poisson goal model plus a Dixon-Coles low-score correction."""

    base_model = fit_poisson_goal_model(
        train,
        feature_columns=feature_columns,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
        half_life_days=half_life_days,
        max_goals=max_goals,
    )
    rho = fit_dixon_coles_rho(train, base_model, half_life_days=half_life_days)
    return PoissonGoalModel(
        feature_scaler=base_model.feature_scaler,
        home_weights=base_model.home_weights,
        away_weights=base_model.away_weights,
        max_goals=base_model.max_goals,
        dixon_coles_rho=rho,
    )


def fit_draw_aware_poisson_model(
    train: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = BASELINE_FEATURE_COLUMNS,
    learning_rate: float = 0.03,
    epochs: int = 2500,
    l2: float = 0.01,
    half_life_days: float | None = 3650.0,
    max_goals: int = 10,
    draw_temperature: float = 1.0,
) -> DrawAwarePoissonModel:
    """Fit a two-part model: draw probability plus non-draw winner split."""

    goal_model = fit_poisson_goal_model(
        train,
        feature_columns=feature_columns,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
        half_life_days=half_life_days,
        max_goals=max_goals,
    )
    draw_scaler = fit_feature_scaler(train, feature_columns)
    features = add_intercept(draw_scaler.transform(train))
    target = (train["result"] == "X").astype(int).to_numpy()
    sample_weights = recency_weights(train["date"], half_life_days)
    draw_weights = fit_softmax_weights(
        features,
        target,
        2,
        sample_weights=sample_weights,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
    )
    return DrawAwarePoissonModel(
        goal_model=goal_model,
        draw_scaler=draw_scaler,
        draw_weights=draw_weights,
        draw_temperature=draw_temperature,
    )


def evaluate_predictions(
    actual: pd.Series,
    probabilities: pd.DataFrame,
    *,
    labels: tuple[str, ...] = CLASS_LABELS,
) -> dict[str, object]:
    """Return headline classification and probability metrics."""

    y_true = encode_labels(actual, labels)
    y_pred = probabilities.loc[:, labels].to_numpy().argmax(axis=1)
    probability_values = probabilities.loc[:, labels].to_numpy()
    clipped = np.clip(probability_values, 1e-15, 1.0)

    accuracy = float((y_true == y_pred).mean())
    log_loss = float(-np.log(clipped[np.arange(len(y_true)), y_true]).mean())
    brier = float(np.mean(np.sum((probability_values - one_hot(y_true, len(labels))) ** 2, axis=1)))

    confusion = pd.DataFrame(0, index=list(labels), columns=list(labels))
    for actual_label, predicted_label in zip(actual, probabilities.idxmax(axis=1), strict=True):
        confusion.loc[actual_label, predicted_label] += 1

    return {
        "accuracy": accuracy,
        "log_loss": log_loss,
        "brier": brier,
        "confusion": confusion,
        "predicted_counts": probabilities.idxmax(axis=1).value_counts().reindex(labels, fill_value=0),
        "mean_probabilities": probabilities.mean().reindex(labels),
    }


def majority_class_probabilities(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    labels: tuple[str, ...] = CLASS_LABELS,
) -> pd.DataFrame:
    """Return constant train-distribution probabilities for all test rows."""

    distribution = train["result"].value_counts(normalize=True).reindex(labels, fill_value=0.0)
    values = np.tile(distribution.to_numpy(), (len(test), 1))
    return pd.DataFrame(values, columns=list(labels), index=test.index)


def write_holdout_report(
    path: Path,
    *,
    title: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_probabilities: pd.DataFrame,
    model_metrics: dict[str, object],
    majority_metrics: dict[str, object],
    model_name: str = "Softmax Elo baseline",
    expected_goals: pd.DataFrame | None = None,
    labels: tuple[str, ...] = CLASS_LABELS,
) -> None:
    """Write a Markdown holdout report."""

    prediction_columns = [
        column
        for column in [
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "result",
            "world_cup_stage",
        ]
        if column in test.columns
    ]
    predictions = test.loc[:, prediction_columns].copy()
    predictions["predicted"] = model_probabilities.idxmax(axis=1).to_numpy()
    for label in labels:
        predictions[f"p_{label}"] = model_probabilities[label].round(4).to_numpy()
    if expected_goals is not None:
        predictions["home_xg"] = expected_goals["home_xg"].round(3).to_numpy()
        predictions["away_xg"] = expected_goals["away_xg"].round(3).to_numpy()
    predictions["correct"] = predictions["result"] == predictions["predicted"]

    high_confidence_errors = predictions.assign(confidence=model_probabilities.max(axis=1)).query(
        "correct == False"
    )
    high_confidence_errors = high_confidence_errors.sort_values("confidence", ascending=False).head(10)

    lines = [
        f"# {title}",
        "",
        "## Data",
        "",
        f"- Train rows: {len(train)}",
        f"- Train date range: {train['date'].min().date()} to {train['date'].max().date()}",
        f"- Test rows: {len(test)}",
        f"- Test date range: {test['date'].min().date()} to {test['date'].max().date()}",
        "",
        "## Headline Metrics",
        "",
        "| Model | Accuracy | Log loss | Brier |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| Majority baseline | {majority_metrics['accuracy']:.3f} | "
            f"{majority_metrics['log_loss']:.3f} | {majority_metrics['brier']:.3f} |"
        ),
        (
            f"| {model_name} | {model_metrics['accuracy']:.3f} | "
            f"{model_metrics['log_loss']:.3f} | {model_metrics['brier']:.3f} |"
        ),
        "",
        "## Confusion Matrix",
        "",
        "Rows are actual results; columns are predicted results.",
        "",
        frame_to_markdown(model_metrics["confusion"].reset_index(names="actual")),
        "",
        "## Predicted Result Distribution",
        "",
        frame_to_markdown(model_metrics["predicted_counts"].rename_axis("result").reset_index(name="count")),
        "",
        "## Average Predicted Probabilities",
        "",
        frame_to_markdown(
            model_metrics["mean_probabilities"].rename_axis("result").reset_index(name="probability"),
            float_digits=3,
        ),
        "",
        "## Highest-Confidence Misses",
        "",
        frame_to_markdown(
            high_confidence_errors[
                [
                    column
                    for column in [
                        "date",
                        "home_team",
                        "away_team",
                        "home_goals",
                        "away_goals",
                        "result",
                        "world_cup_stage",
                        "predicted",
                        "confidence",
                    ]
                    if column in high_confidence_errors.columns
                ]
            ],
            float_digits=3,
        ),
        "",
        "## Match Predictions",
        "",
        frame_to_markdown(predictions),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def fit_feature_scaler(frame: pd.DataFrame, columns: tuple[str, ...]) -> FeatureScaler:
    values = build_model_features(frame, columns).astype(float).to_numpy()
    means = values.mean(axis=0)
    scales = values.std(axis=0)
    scales = np.where(scales == 0, 1.0, scales)
    return FeatureScaler(columns=columns, means=means, scales=scales)


def build_model_features(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Return raw and derived modeling features."""

    features = pd.DataFrame(index=frame.index)
    for column in columns:
        if column in frame.columns:
            features[column] = frame[column]
        elif column == "abs_elo_diff_pre":
            features[column] = frame["elo_diff_pre"].abs()
        elif column == "attack_gap_pre":
            features[column] = (frame["home_attack_pre"] - frame["away_attack_pre"]).abs()
        elif column == "defense_gap_pre":
            features[column] = (frame["home_defense_pre"] - frame["away_defense_pre"]).abs()
        elif column == "combined_attack_pre":
            features[column] = frame["home_attack_pre"] + frame["away_attack_pre"]
        elif column == "combined_defense_pre":
            features[column] = frame["home_defense_pre"] + frame["away_defense_pre"]
        elif column == "goal_balance_gap_pre":
            home_goal_signal = (frame["home_attack_pre"] + frame["away_defense_pre"]) / 2.0
            away_goal_signal = (frame["away_attack_pre"] + frame["home_defense_pre"]) / 2.0
            features[column] = (home_goal_signal - away_goal_signal).abs()
        elif column == "experience_gap_pre":
            features[column] = (
                frame["home_matches_played_pre"] - frame["away_matches_played_pre"]
            ).abs()
        elif column == "min_matches_played_pre":
            features[column] = frame[["home_matches_played_pre", "away_matches_played_pre"]].min(axis=1)
        elif column in WORLD_CUP_CONTEXT_FEATURE_COLUMNS:
            features[column] = 0
        elif column in OPTIONAL_NUMERIC_FEATURE_DEFAULTS:
            features[column] = OPTIONAL_NUMERIC_FEATURE_DEFAULTS[column]
        else:
            raise ValueError(f"Cannot build unknown feature column: {column}")

    return features


def add_intercept(features: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(features.shape[0]), features])


def fit_softmax_weights(
    features: np.ndarray,
    target: np.ndarray,
    class_count: int,
    *,
    sample_weights: np.ndarray,
    learning_rate: float,
    epochs: int,
    l2: float,
) -> np.ndarray:
    weights = np.zeros((features.shape[1], class_count), dtype=float)
    y = one_hot(target, class_count)
    normalized_weights = sample_weights / sample_weights.sum()

    for _ in range(epochs):
        probabilities = softmax(features @ weights)
        error = probabilities - y
        weighted_error = error * normalized_weights[:, None]
        gradient = features.T @ weighted_error
        gradient[1:, :] += l2 * weights[1:, :]
        weights -= learning_rate * gradient

    return weights


def fit_poisson_weights(
    features: np.ndarray,
    target: np.ndarray,
    *,
    sample_weights: np.ndarray,
    learning_rate: float,
    epochs: int,
    l2: float,
) -> np.ndarray:
    clipped_target_mean = max(float(np.average(target, weights=sample_weights)), 0.05)
    weights = np.zeros(features.shape[1], dtype=float)
    weights[0] = np.log(clipped_target_mean)
    normalized_weights = sample_weights / sample_weights.sum()

    for _ in range(epochs):
        linear = np.clip(features @ weights, -5.0, 3.5)
        expected = np.exp(linear)
        error = expected - target
        gradient = features.T @ (error * normalized_weights)
        gradient[1:] += l2 * weights[1:]
        weights -= learning_rate * gradient

    return weights


def fit_dixon_coles_rho(
    train: pd.DataFrame,
    model: PoissonGoalModel,
    *,
    half_life_days: float | None,
    rho_values: np.ndarray | None = None,
) -> float:
    """Fit the Dixon-Coles rho parameter by weighted exact-score likelihood."""

    candidate_values = rho_values if rho_values is not None else np.linspace(-0.25, 0.15, 81)
    expected_goals = model.predict_expected_goals(train)
    sample_weights = recency_weights(train["date"], half_life_days)
    home_goals = train["home_goals"].to_numpy(dtype=int)
    away_goals = train["away_goals"].to_numpy(dtype=int)
    home_xg = expected_goals["home_xg"].to_numpy(dtype=float)
    away_xg = expected_goals["away_xg"].to_numpy(dtype=float)
    base_log_probability = poisson_log_probability(home_goals, home_xg) + poisson_log_probability(
        away_goals,
        away_xg,
    )

    best_rho = 0.0
    best_loss = float("inf")
    for rho in candidate_values:
        adjustment = dixon_coles_adjustment_vector(home_goals, away_goals, home_xg, away_xg, rho)
        log_probability = base_log_probability + np.log(np.clip(adjustment, 1e-15, None))
        loss = float(np.average(-log_probability, weights=sample_weights))
        if loss < best_loss:
            best_loss = loss
            best_rho = float(rho)

    return best_rho


def poisson_log_probability(goals: np.ndarray, rate: np.ndarray) -> np.ndarray:
    safe_rate = np.clip(rate, 1e-12, None)
    return (goals * np.log(safe_rate)) - safe_rate - log_factorial(goals)


def log_factorial(values: np.ndarray) -> np.ndarray:
    max_value = int(values.max()) if len(values) else 0
    lookup = np.zeros(max_value + 1)
    for value in range(1, max_value + 1):
        lookup[value] = lookup[value - 1] + np.log(value)
    return lookup[values]


def dixon_coles_adjustment_vector(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home_xg: np.ndarray,
    away_xg: np.ndarray,
    rho: float,
) -> np.ndarray:
    adjustment = np.ones(len(home_goals), dtype=float)
    mask_00 = (home_goals == 0) & (away_goals == 0)
    mask_01 = (home_goals == 0) & (away_goals == 1)
    mask_10 = (home_goals == 1) & (away_goals == 0)
    mask_11 = (home_goals == 1) & (away_goals == 1)

    adjustment[mask_00] = 1.0 - (home_xg[mask_00] * away_xg[mask_00] * rho)
    adjustment[mask_01] = 1.0 + (home_xg[mask_01] * rho)
    adjustment[mask_10] = 1.0 + (away_xg[mask_10] * rho)
    adjustment[mask_11] = 1.0 - rho
    return np.clip(adjustment, 1e-12, None)


def encode_labels(values: pd.Series, labels: tuple[str, ...]) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(labels)}
    return values.map(label_to_index).to_numpy(dtype=int)


def one_hot(indices: np.ndarray, class_count: int) -> np.ndarray:
    result = np.zeros((len(indices), class_count), dtype=float)
    result[np.arange(len(indices)), indices] = 1.0
    return result


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def recency_weights(dates: pd.Series, half_life_days: float | None) -> np.ndarray:
    if half_life_days is None:
        return np.ones(len(dates), dtype=float)

    parsed_dates = pd.to_datetime(dates)
    most_recent = parsed_dates.max()
    age_days = (most_recent - parsed_dates).dt.days.to_numpy(dtype=float)
    return np.power(0.5, age_days / half_life_days)


def frame_to_markdown(frame: pd.DataFrame, *, float_digits: int = 4) -> str:
    """Render a compact Markdown table without optional pandas dependencies."""

    columns = [str(column) for column in frame.columns]
    rows = []
    for row in frame.itertuples(index=False, name=None):
        rows.append([format_markdown_cell(value, float_digits=float_digits) for value in row])

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def format_markdown_cell(value: object, *, float_digits: int) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, float):
        return f"{value:.{float_digits}f}"
    return str(value)
