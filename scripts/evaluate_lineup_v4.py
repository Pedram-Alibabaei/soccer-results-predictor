from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from evaluate_holdout import apply_live_tournament_ratings, apply_target_filter
from soccer_predictor.lineup_features import attach_lineup_features
from soccer_predictor.lineup_model import (
    LINEUP_ADJUSTMENT_FEATURE_COLUMNS,
    POSITION_STRENGTH_ADJUSTMENT_FEATURE_COLUMNS,
    ROTATION_ADJUSTMENT_FEATURE_COLUMNS,
    fit_lineup_adjusted_goal_model,
)
from soccer_predictor.modeling import evaluate_predictions, fit_poisson_goal_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chronologically validate the regularized confirmed-lineup V4 model."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--baseline-epochs", type=int, default=2500)
    parser.add_argument("--adjustment-epochs", type=int, default=2000)
    parser.add_argument(
        "--feature-set",
        choices=("combined", "rotation", "position_strength"),
        default="combined",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_columns = {
        "combined": LINEUP_ADJUSTMENT_FEATURE_COLUMNS,
        "rotation": ROTATION_ADJUSTMENT_FEATURE_COLUMNS,
        "position_strength": POSITION_STRENGTH_ADJUSTMENT_FEATURE_COLUMNS,
    }[args.feature_set]
    holdout_dir = args.project_root / "data" / "processed" / "tournament_holdouts"
    lineup_path = (
        args.project_root
        / "data"
        / "processed"
        / "lineups"
        / "international_lineup_features.csv"
    )
    train = pd.read_csv(
        holdout_dir / "world_cup_2022_train.csv",
        parse_dates=["date"],
        low_memory=False,
    )
    test = pd.read_csv(
        holdout_dir / "world_cup_2022_test.csv",
        parse_dates=["date"],
        low_memory=False,
    )
    lineup_features = pd.read_csv(lineup_path, parse_dates=["date"])
    test = apply_live_tournament_ratings(train, test)
    train = attach_lineup_features(train, lineup_features)
    test = attach_lineup_features(test, lineup_features)
    train, test = apply_target_filter(train, test, "clean_3way")

    confirmed = train[train["has_confirmed_lineups"].eq(1)].copy()
    adjustment_train = confirmed[confirmed["date"].dt.year == 2018].copy()
    adjustment_validation = confirmed[confirmed["date"].dt.year == 2021].copy()
    if len(adjustment_train) != 64 or len(adjustment_validation) != 51:
        raise ValueError(
            "Expected 64 World Cup 2018 and 51 Euro 2020 confirmed-lineup matches; "
            f"found {len(adjustment_train)} and {len(adjustment_validation)}."
        )

    baseline_before_2018 = fit_baseline(
        train[train["date"] < adjustment_train["date"].min()], args.baseline_epochs
    )
    baseline_before_euro = fit_baseline(
        train[train["date"] < adjustment_validation["date"].min()], args.baseline_epochs
    )
    final_baseline = fit_baseline(train, args.baseline_epochs)

    train_baseline_xg = baseline_before_2018.predict_expected_goals(adjustment_train)
    validation_baseline_xg = baseline_before_euro.predict_expected_goals(adjustment_validation)
    validation_rows = []
    fitted_by_l2 = {}
    for l2 in (0.1, 0.5, 1.0, 2.0, 5.0):
        adjustment = fit_lineup_adjusted_goal_model(
            baseline_before_2018,
            adjustment_train,
            baseline_expected_goals=train_baseline_xg,
            correction_strength=1.0,
            epochs=args.adjustment_epochs,
            l2=l2,
            feature_columns=feature_columns,
        )
        fitted_by_l2[l2] = adjustment
        for strength in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0):
            candidate = replace(
                adjustment,
                baseline_model=baseline_before_euro,
                correction_strength=strength,
            )
            metrics = evaluate_predictions(
                adjustment_validation["result"], candidate.predict_proba(adjustment_validation)
            )
            validation_rows.append(
                {
                    "l2": l2,
                    "correction_strength": strength,
                    "accuracy": metrics["accuracy"],
                    "log_loss": metrics["log_loss"],
                    "brier": metrics["brier"],
                }
            )

    validation = pd.DataFrame(validation_rows).sort_values(
        ["log_loss", "brier", "correction_strength"], kind="mergesort"
    )
    best = validation.iloc[0]

    correction_train = pd.concat([adjustment_train, adjustment_validation]).sort_values("date")
    out_of_sample_baseline_xg = pd.concat(
        [train_baseline_xg, validation_baseline_xg]
    ).loc[correction_train.index]
    final_model = fit_lineup_adjusted_goal_model(
        final_baseline,
        correction_train,
        baseline_expected_goals=out_of_sample_baseline_xg,
        correction_strength=float(best["correction_strength"]),
        epochs=args.adjustment_epochs,
        l2=float(best["l2"]),
        feature_columns=feature_columns,
    )

    baseline_probabilities = final_baseline.predict_proba(test)
    adjusted_probabilities = final_model.predict_proba(test)
    baseline_metrics = evaluate_predictions(test["result"], baseline_probabilities)
    adjusted_metrics = evaluate_predictions(test["result"], adjusted_probabilities)
    baseline_xg = final_baseline.predict_expected_goals(test)
    adjusted_xg = final_model.predict_expected_goals(test)

    report_path = (
        args.project_root
        / "reports"
        / f"lineup_v4_{args.feature_set}_world_cup_2022_summary.md"
    )
    predictions_path = (
        args.project_root
        / "reports"
        / f"lineup_v4_{args.feature_set}_world_cup_2022_predictions.csv"
    )
    write_report(
        report_path,
        validation,
        best,
        baseline_metrics,
        adjusted_metrics,
        correction_train,
        test,
        feature_set=args.feature_set,
    )
    write_predictions(
        predictions_path,
        test,
        baseline_probabilities,
        adjusted_probabilities,
        baseline_xg,
        adjusted_xg,
    )

    print(f"Evaluated confirmed-lineup V4 ({args.feature_set}) on World Cup 2022")
    print(
        f"  selected on Euro 2020: l2={best['l2']:.1f}, "
        f"correction_strength={best['correction_strength']:.2f}"
    )
    print(
        f"  baseline: accuracy={baseline_metrics['accuracy']:.3f}, "
        f"log_loss={baseline_metrics['log_loss']:.3f}, brier={baseline_metrics['brier']:.3f}"
    )
    print(
        f"  lineup V4: accuracy={adjusted_metrics['accuracy']:.3f}, "
        f"log_loss={adjusted_metrics['log_loss']:.3f}, brier={adjusted_metrics['brier']:.3f}"
    )
    print(f"  report: {report_path}")


def fit_baseline(train: pd.DataFrame, epochs: int):
    return fit_poisson_goal_model(
        train,
        epochs=epochs,
        learning_rate=0.03,
        l2=0.01,
        half_life_days=3650.0,
        max_goals=10,
    )


def write_report(
    path: Path,
    validation: pd.DataFrame,
    best: pd.Series,
    baseline_metrics: dict[str, object],
    adjusted_metrics: dict[str, object],
    correction_train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_set: str,
) -> None:
    top_validation = validation.head(8)
    lines = [
        f"# Confirmed-Lineup V4 ({feature_set.replace('_', ' ').title()}): World Cup 2022",
        "",
        "## Design",
        "",
        "- Stable Poisson baseline: all historical international matches before each forecast date.",
        "- Lineup correction training: World Cup 2018 and Euro 2020 only, using confirmed starting XIs.",
        "- Player strength: latest Transfermarkt valuation on or before kickoff, log-scaled by position.",
        "- Rotation: changes, goalkeeper replacement, defensive continuity, and attacking continuity.",
        "- Hyperparameters selected on Euro 2020 after fitting the first correction on World Cup 2018.",
        "- Final untouched test: all 64 World Cup 2022 matches, scored at 90 minutes.",
        "",
        "## Selected Correction",
        "",
        f"- L2 regularization: {best['l2']:.1f}",
        f"- Feature set: {feature_set}",
        f"- Correction strength: {best['correction_strength']:.2f}",
        f"- Confirmed-lineup training rows: {len(correction_train)}",
        f"- Confirmed-lineup test rows: {int(test['has_confirmed_lineups'].sum())}/{len(test)}",
        "- Maximum per-team goal-rate adjustment: +/-35% in log space.",
        "",
        "## World Cup 2022 Results",
        "",
        "| Model | Accuracy | Log loss | Brier |",
        "| --- | ---: | ---: | ---: |",
        _metric_row("Poisson baseline", baseline_metrics),
        _metric_row("Confirmed-lineup V4", adjusted_metrics),
        "",
        "Lower log loss and Brier are better; accuracy is secondary because betting needs calibrated probabilities.",
        "",
        "## Euro 2020 Selection Grid",
        "",
        "| L2 | Strength | Accuracy | Log loss | Brier |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top_validation.itertuples(index=False):
        lines.append(
            f"| {row.l2:.1f} | {row.correction_strength:.2f} | {row.accuracy:.3f} | "
            f"{row.log_loss:.3f} | {row.brier:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "Promote V4 only if it improves probability quality on the untouched 2022 holdout. "
            "A correction strength of zero means validation rejected lineup adjustments and the stable baseline remains active.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _metric_row(name: str, metrics: dict[str, object]) -> str:
    return (
        f"| {name} | {metrics['accuracy']:.3f} | "
        f"{metrics['log_loss']:.3f} | {metrics['brier']:.3f} |"
    )


def write_predictions(
    path: Path,
    test: pd.DataFrame,
    baseline_probabilities: pd.DataFrame,
    adjusted_probabilities: pd.DataFrame,
    baseline_xg: pd.DataFrame,
    adjusted_xg: pd.DataFrame,
) -> None:
    output = test.loc[
        :, ["date", "home_team", "away_team", "home_goals", "away_goals", "result"]
    ].copy()
    for label in ("1", "X", "2"):
        output[f"baseline_p_{label}"] = baseline_probabilities[label].to_numpy()
        output[f"lineup_v4_p_{label}"] = adjusted_probabilities[label].to_numpy()
    output["baseline_home_xg"] = baseline_xg["home_xg"].to_numpy()
    output["baseline_away_xg"] = baseline_xg["away_xg"].to_numpy()
    output["lineup_v4_home_xg"] = adjusted_xg["home_xg"].to_numpy()
    output["lineup_v4_away_xg"] = adjusted_xg["away_xg"].to_numpy()
    output.to_csv(path, index=False)


if __name__ == "__main__":
    main()
