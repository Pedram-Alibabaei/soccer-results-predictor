from __future__ import annotations

import argparse
from math import log
from pathlib import Path

import numpy as np
import pandas as pd

from soccer_predictor.contextual_goals import (
    ContextualGoalModel,
    load_third_place_history,
    fit_contextual_goal_model,
)
from soccer_predictor.elo import add_elo_features, fit_elo_state
from soccer_predictor.markets import ScoreProbabilityTable, over_under, score_matrix
from soccer_predictor.modeling import evaluate_predictions, frame_to_markdown


DEFAULT_HOLDOUTS = ("world_cup_2018", "world_cup_2022")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the context-adjusted, overdispersed V3 goal model."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--holdouts", nargs="+", default=list(DEFAULT_HOLDOUTS))
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--half-life-days", type=float, default=3650.0)
    parser.add_argument("--context-half-life-days", type=float, default=7300.0)
    parser.add_argument("--context-prior-exposure", type=float, default=30.0)
    parser.add_argument("--max-goals", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    holdout_dir = args.project_root / "data" / "processed" / "tournament_holdouts"
    history = load_third_place_history(
        args.project_root / "data" / "external" / "world_cup_third_place_regulation.csv"
    )
    half_life_days = None if args.half_life_days <= 0 else args.half_life_days
    context_half_life_days = (
        None if args.context_half_life_days <= 0 else args.context_half_life_days
    )

    metric_rows: list[dict[str, object]] = []
    context_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for holdout in args.holdouts:
        train = pd.read_csv(
            holdout_dir / f"{holdout}_train.csv",
            parse_dates=["date"],
            low_memory=False,
        )
        test = pd.read_csv(
            holdout_dir / f"{holdout}_test.csv",
            parse_dates=["date"],
            low_memory=False,
        )
        test = apply_live_tournament_ratings(train, test)
        model = fit_contextual_goal_model(
            train,
            history,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            l2=args.l2,
            half_life_days=half_life_days,
            context_half_life_days=context_half_life_days,
            prior_exposure=args.context_prior_exposure,
            max_goals=args.max_goals,
        )

        base_expected = model.base_model.predict_expected_goals(test)
        contextual_expected = model.predict_expected_goals(test)
        base_tables = independent_tables(test, base_expected, max_goals=args.max_goals)
        contextual_tables = model.predict_score_tables(test)
        actual_home = test["market_3way_home_goals"].to_numpy(dtype=int)
        actual_away = test["market_3way_away_goals"].to_numpy(dtype=int)

        for name, tables in (
            ("Independent Poisson", base_tables),
            ("Contextual NB V3", contextual_tables),
        ):
            distribution_metrics = evaluate_goal_distributions(
                tables,
                actual_home,
                actual_away,
            )
            probabilities = pd.DataFrame(
                [table_three_way(table) for table in tables],
                index=test.index,
            )
            result_metrics = evaluate_predictions(
                test["market_3way_result"],
                probabilities,
            )
            metric_rows.append(
                {
                    "holdout": holdout,
                    "model": name,
                    **distribution_metrics,
                    "1x2_log_loss": result_metrics["log_loss"],
                    "1x2_brier": result_metrics["brier"],
                }
            )

        risk = model.predict_risk(test)
        predictions = build_predictions(
            holdout,
            test,
            base_expected,
            contextual_expected,
            risk,
        )
        prediction_frames.append(predictions)
        third_place = predictions[predictions["world_cup_stage"] == "third_place"].iloc[0]
        adjustment = model.third_place_adjustment
        context_rows.append(
            {
                "holdout": holdout,
                "history_matches": adjustment.match_count,
                "history_weighted_observed_goals": adjustment.observed_goals,
                "history_weighted_predicted_goals": adjustment.predicted_goals,
                "history_raw_ratio": (
                    adjustment.observed_goals / adjustment.predicted_goals
                    if adjustment.predicted_goals > 0
                    else 1.0
                ),
                "stage_multiplier": adjustment.multiplier,
                "dispersion": model.dispersion,
                "fixture": f"{third_place.home_team} vs {third_place.away_team}",
                "actual_score": (
                    f"{int(third_place.actual_home_goals)}-"
                    f"{int(third_place.actual_away_goals)}"
                ),
                "base_expected_total": third_place.base_expected_total,
                "v3_expected_total": third_place.expected_total_goals,
                "p_over_2_5": third_place.probability_over_2_5,
                "p_under_3_5": third_place.probability_under_3_5,
                "p_6_plus": third_place.probability_6_plus,
                "high_variance": third_place.high_variance,
            }
        )

    metrics = pd.DataFrame(metric_rows)
    contexts = pd.DataFrame(context_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    report_dir = args.project_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "contextual_goal_risk_v3_holdouts.md"
    predictions_path = report_dir / "contextual_goal_risk_v3_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    write_report(report_path, metrics, contexts, args)

    print("Contextual goal-risk V3 evaluation complete")
    print(frame_to_markdown(metrics, float_digits=4))
    print(frame_to_markdown(contexts, float_digits=4))
    print(f"report: {report_path}")
    print(f"predictions: {predictions_path}")


def apply_live_tournament_ratings(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    return add_elo_features(
        test,
        initial_state=fit_elo_state(train),
        update_ratings=True,
    )


def independent_tables(
    test: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    max_goals: int,
) -> list[ScoreProbabilityTable]:
    tables = []
    for index, row in expected.iterrows():
        match = test.loc[index]
        tables.append(
            score_matrix(
                float(row["home_xg"]),
                float(row["away_xg"]),
                max_goals=max_goals,
                home_team=str(match["home_team"]),
                away_team=str(match["away_team"]),
            )
        )
    return tables


def table_three_way(table: ScoreProbabilityTable) -> dict[str, float]:
    home = draw = away = 0.0
    for home_goals, away_goals, probability in table.scores():
        if home_goals > away_goals:
            home += probability
        elif home_goals == away_goals:
            draw += probability
        else:
            away += probability
    return {"1": home, "X": draw, "2": away}


def evaluate_goal_distributions(
    tables: list[ScoreProbabilityTable],
    actual_home: np.ndarray,
    actual_away: np.ndarray,
) -> dict[str, float]:
    exact_losses = []
    total_losses = []
    absolute_errors = []
    over_2_5_errors = []
    under_3_5_errors = []
    for table, home_goals, away_goals in zip(
        tables,
        actual_home,
        actual_away,
        strict=True,
    ):
        exact_probability = table.probability(home_goals, away_goals)
        actual_total = int(home_goals + away_goals)
        total_probability = sum(
            probability
            for predicted_home, predicted_away, probability in table.scores()
            if predicted_home + predicted_away == actual_total
        )
        expected_total = sum(
            (predicted_home + predicted_away) * probability
            for predicted_home, predicted_away, probability in table.scores()
        )
        probability_over_2_5 = over_under(table, 2.5)["+"]
        probability_under_3_5 = over_under(table, 3.5)["-"]
        exact_losses.append(-log(max(exact_probability, 1e-15)))
        total_losses.append(-log(max(total_probability, 1e-15)))
        absolute_errors.append(abs(expected_total - actual_total))
        over_2_5_errors.append((probability_over_2_5 - (actual_total >= 3)) ** 2)
        under_3_5_errors.append((probability_under_3_5 - (actual_total <= 3)) ** 2)

    return {
        "exact_score_log_loss": float(np.mean(exact_losses)),
        "total_goals_log_loss": float(np.mean(total_losses)),
        "total_goals_mae": float(np.mean(absolute_errors)),
        "over_2_5_brier": float(np.mean(over_2_5_errors)),
        "under_3_5_brier": float(np.mean(under_3_5_errors)),
    }


def build_predictions(
    holdout: str,
    test: pd.DataFrame,
    base_expected: pd.DataFrame,
    contextual_expected: pd.DataFrame,
    risk: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "holdout": holdout,
            "date": test["date"].to_numpy(),
            "home_team": test["home_team"].to_numpy(),
            "away_team": test["away_team"].to_numpy(),
            "world_cup_stage": test["world_cup_stage"].to_numpy(),
            "actual_home_goals": test["market_3way_home_goals"].to_numpy(dtype=int),
            "actual_away_goals": test["market_3way_away_goals"].to_numpy(dtype=int),
            "base_home_xg": base_expected["home_xg"].to_numpy(),
            "base_away_xg": base_expected["away_xg"].to_numpy(),
            "base_expected_total": (
                base_expected["home_xg"].to_numpy()
                + base_expected["away_xg"].to_numpy()
            ),
            "v3_home_xg": contextual_expected["home_xg"].to_numpy(),
            "v3_away_xg": contextual_expected["away_xg"].to_numpy(),
            **{column: risk[column].to_numpy() for column in risk.columns},
        }
    )


def write_report(
    path: Path,
    metrics: pd.DataFrame,
    contexts: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    comparison = metrics.copy()
    numeric = comparison.select_dtypes(include="number").columns
    comparison[numeric] = comparison[numeric].round(4)
    context_table = contexts.copy()
    numeric = context_table.select_dtypes(include="number").columns
    context_table[numeric] = context_table[numeric].round(4)

    lines = [
        "# Contextual Goal-Risk V3 Holdout Evaluation",
        "",
        "## Protocol",
        "",
        "- Holdouts: World Cup 2018 and World Cup 2022.",
        "- Every base model is trained only on matches before the held-out tournament.",
        "- Test ratings update match-by-match using only already completed tournament games.",
        "- Third-place history is filtered at each training cutoff; 2022 cannot inform 2018.",
        "- Regulation-time goals are used for goal-market evaluation.",
        "- The base fit excludes the unstable third-place one-hot coefficient.",
        "- V3 applies a Gamma-Poisson shrinkage multiplier and a Negative Binomial total-goals distribution.",
        f"- Context prior exposure: {args.context_prior_exposure:.1f} expected goals.",
        f"- Maximum modeled goals per team: {args.max_goals}.",
        "",
        "## Tournament Metrics",
        "",
        frame_to_markdown(comparison, float_digits=4),
        "",
        "Lower is better for every metric in this table.",
        "",
        "## Third-Place Forecasts",
        "",
        frame_to_markdown(context_table, float_digits=4),
        "",
        "`high_variance` is a risk warning, not a bet recommendation. It is triggered by an inflated total-goals variance or material six-plus-goal probability.",
        "",
        "## Interpretation Rule",
        "",
        "V3 should only replace the independent score distribution if it improves total-goals log loss and market Brier scores across multiple untouched holdouts. A better explanation of one memorable match is not sufficient evidence.",
        "",
        "## Current Verdict",
        "",
        comparison_verdict(metrics),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def comparison_verdict(metrics: pd.DataFrame) -> str:
    averages = metrics.groupby("model").mean(numeric_only=True)
    baseline = averages.loc["Independent Poisson"]
    contextual = averages.loc["Contextual NB V3"]
    improved = [
        column
        for column in averages.columns
        if contextual[column] < baseline[column]
    ]
    worsened = [
        column
        for column in averages.columns
        if contextual[column] > baseline[column]
    ]
    if len(improved) == len(averages.columns):
        return "V3 improved every averaged metric, but it still requires broader holdouts before promotion."
    return (
        "V3 is **not promoted**. Its holdout evidence is mixed: it improved "
        f"{', '.join(improved) or 'no averaged metrics'}, while worsening "
        f"{', '.join(worsened) or 'none'}. Keep it as a tail-risk diagnostic until "
        "broader international-tournament validation shows a consistent gain."
    )


if __name__ == "__main__":
    main()
