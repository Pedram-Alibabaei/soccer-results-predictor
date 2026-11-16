from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from soccer_predictor.modeling import (
    evaluate_predictions,
    fit_dixon_coles_goal_model,
    fit_draw_aware_poisson_model,
    fit_poisson_goal_model,
    fit_softmax_baseline,
    majority_class_probabilities,
    write_holdout_report,
)
from soccer_predictor.elo import add_elo_features, fit_elo_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a tournament holdout.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root folder.",
    )
    parser.add_argument("--holdout", default="world_cup_2018")
    parser.add_argument(
        "--model",
        choices=("softmax", "poisson", "dc_poisson", "draw_poisson"),
        default="softmax",
    )
    parser.add_argument(
        "--half-life-days",
        type=float,
        default=3650.0,
        help="Recency weighting half-life. Use 0 for unweighted training.",
    )
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--max-goals", type=int, default=10)
    parser.add_argument("--draw-temperature", type=float, default=1.0)
    parser.add_argument(
        "--target",
        choices=("source_result", "clean_3way", "qualification"),
        default="source_result",
        help="source_result uses source scores; clean_3way uses corrected 90-minute scores; qualification uses knockout advancement labels.",
    )
    parser.add_argument(
        "--live-tournament-ratings",
        action="store_true",
        help="Update test Elo/rolling features after each completed holdout match, matching in-tournament forecasting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    holdout_dir = args.project_root / "data" / "processed" / "tournament_holdouts"
    report_dir = args.project_root / "reports"
    train_path = holdout_dir / f"{args.holdout}_train.csv"
    test_path = holdout_dir / f"{args.holdout}_test.csv"

    train = pd.read_csv(train_path, parse_dates=["date"], low_memory=False)
    test = pd.read_csv(test_path, parse_dates=["date"], low_memory=False)
    if args.live_tournament_ratings:
        test = apply_live_tournament_ratings(train, test)
    train, test = apply_target_filter(train, test, args.target)
    labels = ("1", "2") if args.target == "qualification" else ("1", "X", "2")
    if args.target == "qualification" and args.model != "softmax":
        raise ValueError("qualification target currently supports only --model softmax.")

    half_life_days = None if args.half_life_days <= 0 else args.half_life_days
    if args.model == "draw_poisson":
        model = fit_draw_aware_poisson_model(
            train,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            l2=args.l2,
            half_life_days=half_life_days,
            max_goals=args.max_goals,
            draw_temperature=args.draw_temperature,
        )
        model_name = "Draw-aware Poisson goal baseline"
        expected_goals = model.predict_expected_goals(test)
    elif args.model == "dc_poisson":
        model = fit_dixon_coles_goal_model(
            train,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            l2=args.l2,
            half_life_days=half_life_days,
            max_goals=args.max_goals,
        )
        model_name = f"Dixon-Coles Poisson goal baseline (rho={model.dixon_coles_rho:.3f})"
        expected_goals = model.predict_expected_goals(test)
    elif args.model == "poisson":
        model = fit_poisson_goal_model(
            train,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            l2=args.l2,
            half_life_days=half_life_days,
            max_goals=args.max_goals,
        )
        model_name = "Poisson goal baseline"
        expected_goals = model.predict_expected_goals(test)
    else:
        model = fit_softmax_baseline(
            train,
            labels=labels,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            l2=args.l2,
            half_life_days=half_life_days,
        )
        model_name = "Softmax Elo baseline"
        expected_goals = None

    model_probabilities = model.predict_proba(test)
    majority_probabilities = majority_class_probabilities(train, test, labels=labels)

    model_metrics = evaluate_predictions(test["result"], model_probabilities, labels=labels)
    majority_metrics = evaluate_predictions(test["result"], majority_probabilities, labels=labels)

    target_suffix = "" if args.target == "source_result" else f"_{args.target}"
    live_suffix = "_live" if args.live_tournament_ratings else ""
    report_path = report_dir / f"{args.holdout}_{args.model}{target_suffix}{live_suffix}_report.md"
    predictions_path = report_dir / f"{args.holdout}_{args.model}{target_suffix}{live_suffix}_predictions.csv"
    write_holdout_report(
        report_path,
        title=(
            f"{args.holdout.replace('_', ' ').title()} {model_name} Report "
            f"({args.target}{', live tournament ratings' if args.live_tournament_ratings else ''})"
        ),
        train=train,
        test=test,
        model_probabilities=model_probabilities,
        model_metrics=model_metrics,
        majority_metrics=majority_metrics,
        model_name=model_name,
        expected_goals=expected_goals,
        labels=labels,
    )
    write_predictions_csv(
        test,
        model_probabilities,
        predictions_path,
        expected_goals=expected_goals,
        labels=labels,
    )

    print(f"Evaluated {args.holdout} with {args.model} for {args.target}")
    if args.live_tournament_ratings:
        print("  mode: live tournament ratings")
    print(f"  train rows: {len(train)}")
    print(f"  test rows: {len(test)}")
    print("  majority baseline:")
    print(f"    accuracy: {majority_metrics['accuracy']:.3f}")
    print(f"    log_loss: {majority_metrics['log_loss']:.3f}")
    print(f"    brier: {majority_metrics['brier']:.3f}")
    print(f"  {model_name}:")
    print(f"    accuracy: {model_metrics['accuracy']:.3f}")
    print(f"    log_loss: {model_metrics['log_loss']:.3f}")
    print(f"    brier: {model_metrics['brier']:.3f}")
    print("  confusion matrix:")
    print(model_metrics["confusion"].to_string())
    print(f"  report: {report_path}")
    print(f"  predictions: {predictions_path}")

def write_predictions_csv(
    test: pd.DataFrame,
    probabilities: pd.DataFrame,
    path: Path,
    *,
    expected_goals: pd.DataFrame | None = None,
    labels: tuple[str, ...] = ("1", "X", "2"),
) -> None:
    base_columns = ["date", "home_team", "away_team", "home_goals", "away_goals", "result"]
    if "world_cup_stage" in test.columns:
        base_columns.append("world_cup_stage")
    predictions = test.loc[:, base_columns].copy()
    predictions["predicted"] = probabilities.idxmax(axis=1).to_numpy()
    for label in labels:
        predictions[f"p_{label}"] = probabilities[label].to_numpy()
    if expected_goals is not None:
        predictions["home_xg"] = expected_goals["home_xg"].to_numpy()
        predictions["away_xg"] = expected_goals["away_xg"].to_numpy()
    predictions["correct"] = predictions["result"] == predictions["predicted"]
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)


def apply_live_tournament_ratings(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Recompute holdout test features as if forecasting match-by-match live."""

    pre_tournament_state = fit_elo_state(train)
    return add_elo_features(
        test,
        initial_state=pre_tournament_state,
        update_ratings=True,
    )


def apply_target_filter(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if target == "source_result":
        return train, test
    if target == "clean_3way":
        required_columns = {
            "is_clean_3way_result",
            "market_3way_home_goals",
            "market_3way_away_goals",
            "market_3way_result",
        }
        missing = required_columns.difference(train.columns).union(required_columns.difference(test.columns))
        if missing:
            raise ValueError(f"clean_3way target requires rebuilt datasets. Missing: {sorted(missing)}")

        train_clean = train[train["is_clean_3way_result"].astype(bool)].copy()
        test_clean = test[test["is_clean_3way_result"].astype(bool)].copy()
        return (
            apply_3way_market_result(train_clean).reset_index(drop=True),
            apply_3way_market_result(test_clean).reset_index(drop=True),
        )

    if target == "qualification":
        required_columns = {"is_world_cup_knockout_stage", "qualification_result"}
        missing = required_columns.difference(train.columns).union(required_columns.difference(test.columns))
        if missing:
            raise ValueError(f"qualification target requires rebuilt datasets. Missing: {sorted(missing)}")

        train = train.copy()
        test = test.copy()
        train["qualification_result"] = normalize_result_labels(train["qualification_result"])
        test["qualification_result"] = normalize_result_labels(test["qualification_result"])
        train_qualification = train[
            train["is_world_cup_knockout_stage"].astype(bool)
            & train["qualification_result"].isin(["1", "2"])
        ].copy()
        test_qualification = test[
            test["is_world_cup_knockout_stage"].astype(bool)
            & test["qualification_result"].isin(["1", "2"])
        ].copy()
        train_qualification["result"] = train_qualification["qualification_result"]
        test_qualification["result"] = test_qualification["qualification_result"]
        return train_qualification.reset_index(drop=True), test_qualification.reset_index(drop=True)

    raise ValueError(f"Unsupported target: {target}")


def apply_3way_market_result(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["home_goals"] = result["market_3way_home_goals"].astype(int)
    result["away_goals"] = result["market_3way_away_goals"].astype(int)
    result["result"] = result["market_3way_result"].astype(str)
    result["total_goals"] = result["home_goals"] + result["away_goals"]
    result["goal_difference"] = result["home_goals"] - result["away_goals"]
    result["both_teams_scored"] = (result["home_goals"] > 0) & (result["away_goals"] > 0)
    return result


def normalize_result_labels(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace({"nan": "", "None": ""})
    )


if __name__ == "__main__":
    main()
