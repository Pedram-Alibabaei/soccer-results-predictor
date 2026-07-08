from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from soccer_predictor.calibration import blend_probabilities, fit_blend_alpha
from soccer_predictor.markets import over_under, score_matrix
from soccer_predictor.modeling import (
    evaluate_predictions,
    fit_dixon_coles_goal_model,
    fit_draw_aware_poisson_model,
    fit_poisson_goal_model,
    fit_softmax_baseline,
    frame_to_markdown,
)
from soccer_predictor.odds import (
    OVER_UNDER_25_ODDS_COLUMNS,
    THREE_WAY_ODDS_COLUMNS,
    actual_over_under_2_5,
    bookmaker_probabilities,
    complete_odds_mask,
    find_value_bets_1x2,
    find_value_bets_over_under_2_5,
    odds_columns_for_market,
    summarize_bets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train on club matches and backtest value bets.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root folder.",
    )
    parser.add_argument("--test-season", type=int, default=2024, help="Season start year to hold out.")
    parser.add_argument(
        "--model",
        choices=("softmax", "poisson", "dc_poisson", "draw_poisson"),
        default="poisson",
    )
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--half-life-days", type=float, default=730.0)
    parser.add_argument("--max-goals", type=int, default=10)
    parser.add_argument("--draw-temperature", type=float, default=1.0)
    parser.add_argument("--min-edge", type=float, default=0.03)
    parser.add_argument("--stake", type=float, default=1.0)
    parser.add_argument(
        "--odds-timing",
        choices=("opening", "closing", "best_available"),
        default="opening",
        help="Which odds columns to use for value decisions and bookmaker baseline.",
    )
    parser.add_argument(
        "--blend-with-bookmaker",
        action="store_true",
        help="Learn a validation-season blend between model probabilities and no-vig bookmaker probabilities.",
    )
    parser.add_argument(
        "--validation-season",
        type=int,
        default=None,
        help="Season start year used only to fit bookmaker-blend alpha. Defaults to test-season - 1.",
    )
    parser.add_argument("--blend-step", type=float, default=0.05)
    parser.add_argument(
        "--thresholds",
        default="0,0.02,0.03,0.05,0.08,0.10",
        help="Comma-separated EV thresholds for the report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = (
        args.project_root
        / "data"
        / "processed"
        / "club_odds"
        / "club_matches_with_elo_features.csv"
    )
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Club odds dataset not found: {dataset_path}. Run scripts/build_club_odds_dataset.py first."
        )

    data = pd.read_csv(dataset_path, parse_dates=["date"], low_memory=False)
    train = data[data["season_start_year"] < args.test_season].copy()
    test = data[data["season_start_year"] == args.test_season].copy()
    if train.empty or test.empty:
        raise ValueError(f"Need non-empty train and test splits for test season {args.test_season}.")

    half_life_days = None if args.half_life_days <= 0 else args.half_life_days
    blend = None
    if args.blend_with_bookmaker:
        validation_season = args.validation_season or args.test_season - 1
        blend = fit_bookmaker_blend(
            data,
            model_type=args.model,
            validation_season=validation_season,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            l2=args.l2,
            half_life_days=half_life_days,
            max_goals=args.max_goals,
            draw_temperature=args.draw_temperature,
            blend_step=args.blend_step,
            odds_timing=args.odds_timing,
        )

    model, model_name, expected_goals = fit_model(
        train,
        test,
        model_type=args.model,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        l2=args.l2,
        half_life_days=half_life_days,
        max_goals=args.max_goals,
        draw_temperature=args.draw_temperature,
    )
    decision_name = model_name

    probabilities_1x2 = model.predict_proba(test)
    odds_columns_1x2 = odds_columns_for_market("1x2", args.odds_timing)
    odds_columns_ou = odds_columns_for_market("over_under_2_5", args.odds_timing)
    test_1x2 = test[complete_odds_mask(test, odds_columns_1x2)].copy()
    probabilities_1x2_odds = probabilities_1x2.loc[test_1x2.index]
    bookmaker_1x2 = bookmaker_probabilities(test_1x2, odds_columns_1x2)
    decision_probabilities_1x2 = probabilities_1x2_odds
    if blend is not None:
        decision_probabilities_1x2 = blend_probabilities(
            probabilities_1x2_odds,
            bookmaker_1x2,
            alpha=blend["1x2_alpha"],
            labels=("1", "X", "2"),
        )
        decision_name = f"{model_name} + bookmaker blend"

    model_1x2_metrics = evaluate_predictions(test_1x2["result"], probabilities_1x2_odds)
    decision_1x2_metrics = evaluate_predictions(test_1x2["result"], decision_probabilities_1x2)
    bookmaker_1x2_metrics = evaluate_predictions(test_1x2["result"], bookmaker_1x2)

    thresholds = parse_thresholds(args.thresholds)
    threshold_rows = []
    selected_bets = []
    for threshold in thresholds:
        bets = find_value_bets_1x2(
            test_1x2,
            decision_probabilities_1x2,
            odds_columns=odds_columns_1x2,
            min_edge=threshold,
            stake=args.stake,
        )
        threshold_rows.append({"market": "1x2", "min_edge": threshold, **summarize_bets(bets)})
        if abs(threshold - args.min_edge) < 1e-12:
            selected_bets.append(bets)

    ou_metrics = None
    if expected_goals is not None:
        test_ou = test[complete_odds_mask(test, odds_columns_ou)].copy()
        probabilities_ou = over_under_probabilities(expected_goals.loc[test_ou.index])
        bookmaker_ou = bookmaker_probabilities(test_ou, odds_columns_ou)
        decision_probabilities_ou = probabilities_ou
        if blend is not None and blend["ou_2_5_alpha"] is not None:
            decision_probabilities_ou = blend_probabilities(
                probabilities_ou,
                bookmaker_ou,
                alpha=blend["ou_2_5_alpha"],
                labels=("+", "-"),
            )
        actual_ou = actual_over_under_2_5(test_ou)
        ou_metrics = {
            "model": evaluate_predictions(actual_ou, probabilities_ou, labels=("+", "-")),
            "decision": evaluate_predictions(actual_ou, decision_probabilities_ou, labels=("+", "-")),
            "bookmaker": evaluate_predictions(actual_ou, bookmaker_ou, labels=("+", "-")),
            "rows": len(test_ou),
        }
        for threshold in thresholds:
            bets = find_value_bets_over_under_2_5(
                test_ou,
                decision_probabilities_ou,
                odds_columns=odds_columns_ou,
                min_edge=threshold,
                stake=args.stake,
            )
            threshold_rows.append(
                {"market": "over_under_2_5", "min_edge": threshold, **summarize_bets(bets)}
            )
            if abs(threshold - args.min_edge) < 1e-12:
                selected_bets.append(bets)

    selected = pd.concat(selected_bets, ignore_index=True) if selected_bets else pd.DataFrame()
    report_dir = args.project_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_suffix = (
        f"{args.model}_{args.odds_timing}_bookmaker_blend"
        if blend is not None
        else f"{args.model}_{args.odds_timing}"
    )
    report_path = report_dir / f"club_odds_{args.test_season}_{report_suffix}_value_backtest_report.md"
    bets_path = report_dir / f"club_odds_{args.test_season}_{report_suffix}_value_bets.csv"
    selected.to_csv(bets_path, index=False)
    write_report(
        report_path,
        model_name=decision_name,
        odds_timing=args.odds_timing,
        train=train,
        test=test,
        test_1x2=test_1x2,
        model_1x2_metrics=model_1x2_metrics,
        decision_1x2_metrics=decision_1x2_metrics,
        bookmaker_1x2_metrics=bookmaker_1x2_metrics,
        ou_metrics=ou_metrics,
        threshold_summary=pd.DataFrame(threshold_rows),
        selected_bets=selected,
        min_edge=args.min_edge,
        bets_path=bets_path,
        blend=blend,
    )

    print(f"Backtested club odds with {decision_name}")
    print(f"  train rows: {len(train)}")
    print(f"  test rows: {len(test)}")
    print(f"  odds timing: {args.odds_timing}")
    print(f"  1X2 odds rows: {len(test_1x2)}")
    print("  1X2 probability metrics:")
    print(f"    bookmaker log_loss: {bookmaker_1x2_metrics['log_loss']:.3f}")
    print(f"    model log_loss:     {model_1x2_metrics['log_loss']:.3f}")
    print(f"    decision log_loss:  {decision_1x2_metrics['log_loss']:.3f}")
    print(f"    bookmaker brier:    {bookmaker_1x2_metrics['brier']:.3f}")
    print(f"    model brier:        {model_1x2_metrics['brier']:.3f}")
    print(f"    decision brier:     {decision_1x2_metrics['brier']:.3f}")
    if blend is not None:
        print(f"  validation blend alpha 1X2: {blend['1x2_alpha']:.2f}")
        if blend["ou_2_5_alpha"] is not None:
            print(f"  validation blend alpha O/U 2.5: {blend['ou_2_5_alpha']:.2f}")
    selected_summary = summarize_bets(selected)
    print(f"  selected bets at edge >= {args.min_edge:.2%}: {selected_summary['bets']}")
    print(f"  profit: {selected_summary['profit']:.2f}")
    print(f"  ROI: {selected_summary['roi']:.2%}")
    print(f"  report: {report_path}")
    print(f"  bets: {bets_path}")


def fit_bookmaker_blend(
    data: pd.DataFrame,
    *,
    model_type: str,
    validation_season: int,
    learning_rate: float,
    epochs: int,
    l2: float,
    half_life_days: float | None,
    max_goals: int,
    draw_temperature: float,
    blend_step: float,
    odds_timing: str = "opening",
) -> dict[str, object]:
    calibration_train = data[data["season_start_year"] < validation_season].copy()
    validation = data[data["season_start_year"] == validation_season].copy()
    if calibration_train.empty or validation.empty:
        raise ValueError(f"Need data before and during validation season {validation_season}.")

    model, _, expected_goals = fit_model(
        calibration_train,
        validation,
        model_type=model_type,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
        half_life_days=half_life_days,
        max_goals=max_goals,
        draw_temperature=draw_temperature,
    )
    candidates = blend_candidates(blend_step)

    odds_columns_1x2 = odds_columns_for_market("1x2", odds_timing)
    odds_columns_ou = odds_columns_for_market("over_under_2_5", odds_timing)
    validation_1x2 = validation[complete_odds_mask(validation, odds_columns_1x2)].copy()
    model_1x2 = model.predict_proba(validation_1x2)
    bookmaker_1x2 = bookmaker_probabilities(validation_1x2, odds_columns_1x2)
    alpha_1x2, alpha_1x2_frame = fit_blend_alpha(
        validation_1x2["result"],
        model_1x2,
        bookmaker_1x2,
        labels=("1", "X", "2"),
        candidates=candidates,
    )

    alpha_ou = None
    alpha_ou_frame = pd.DataFrame()
    if expected_goals is not None:
        validation_ou = validation[complete_odds_mask(validation, odds_columns_ou)].copy()
        model_ou = over_under_probabilities(expected_goals.loc[validation_ou.index])
        bookmaker_ou = bookmaker_probabilities(validation_ou, odds_columns_ou)
        alpha_ou, alpha_ou_frame = fit_blend_alpha(
            actual_over_under_2_5(validation_ou),
            model_ou,
            bookmaker_ou,
            labels=("+", "-"),
            candidates=candidates,
        )

    return {
        "validation_season": validation_season,
        "1x2_alpha": alpha_1x2,
        "1x2_alpha_metrics": alpha_1x2_frame,
        "ou_2_5_alpha": alpha_ou,
        "ou_2_5_alpha_metrics": alpha_ou_frame,
    }


def blend_candidates(step: float) -> tuple[float, ...]:
    if step <= 0 or step > 1:
        raise ValueError("blend step must be in (0, 1].")
    values = []
    current = 0.0
    while current < 1.0:
        values.append(round(current, 10))
        current += step
    values.append(1.0)
    return tuple(values)


def fit_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    model_type: str,
    learning_rate: float,
    epochs: int,
    l2: float,
    half_life_days: float | None,
    max_goals: int,
    draw_temperature: float,
):
    if model_type == "draw_poisson":
        model = fit_draw_aware_poisson_model(
            train,
            learning_rate=learning_rate,
            epochs=epochs,
            l2=l2,
            half_life_days=half_life_days,
            max_goals=max_goals,
            draw_temperature=draw_temperature,
        )
        return model, "Draw-aware Poisson club model", model.predict_expected_goals(test)
    if model_type == "dc_poisson":
        model = fit_dixon_coles_goal_model(
            train,
            learning_rate=learning_rate,
            epochs=epochs,
            l2=l2,
            half_life_days=half_life_days,
            max_goals=max_goals,
        )
        return model, f"Dixon-Coles Poisson club model (rho={model.dixon_coles_rho:.3f})", model.predict_expected_goals(test)
    if model_type == "poisson":
        model = fit_poisson_goal_model(
            train,
            learning_rate=learning_rate,
            epochs=epochs,
            l2=l2,
            half_life_days=half_life_days,
            max_goals=max_goals,
        )
        return model, "Poisson club model", model.predict_expected_goals(test)

    model = fit_softmax_baseline(
        train,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
        half_life_days=half_life_days,
    )
    return model, "Softmax club model", None


def over_under_probabilities(expected_goals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in expected_goals.itertuples(index=False):
        probabilities = over_under(score_matrix(row.home_xg, row.away_xg), 2.5)
        rows.append({"+": probabilities["+"], "-": probabilities["-"]})
    return pd.DataFrame(rows, index=expected_goals.index)


def parse_thresholds(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def write_report(
    path: Path,
    *,
    model_name: str,
    odds_timing: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    test_1x2: pd.DataFrame,
    model_1x2_metrics: dict[str, object],
    decision_1x2_metrics: dict[str, object],
    bookmaker_1x2_metrics: dict[str, object],
    ou_metrics: dict[str, object] | None,
    threshold_summary: pd.DataFrame,
    selected_bets: pd.DataFrame,
    min_edge: float,
    bets_path: Path,
    blend: dict[str, object] | None,
) -> None:
    threshold_display = threshold_summary.copy()
    for column in ["roi", "hit_rate", "avg_edge"]:
        threshold_display[column] = threshold_display[column].astype(float)

    lines = [
        f"# Club Odds Value Backtest - {model_name}",
        "",
        "## Data",
        "",
        f"- Train rows: {len(train)}",
        f"- Train date range: {train['date'].min().date()} to {train['date'].max().date()}",
        f"- Test rows: {len(test)}",
        f"- Test date range: {test['date'].min().date()} to {test['date'].max().date()}",
        f"- Odds timing: {odds_timing}",
        f"- Test rows with 1X2 odds: {len(test_1x2)}",
        "",
        "## 1X2 Probability Metrics",
        "",
        "| Source | Accuracy | Log loss | Brier |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| Bookmaker no-vig | {bookmaker_1x2_metrics['accuracy']:.3f} | "
            f"{bookmaker_1x2_metrics['log_loss']:.3f} | {bookmaker_1x2_metrics['brier']:.3f} |"
        ),
        (
            f"| Pure model | {model_1x2_metrics['accuracy']:.3f} | "
            f"{model_1x2_metrics['log_loss']:.3f} | {model_1x2_metrics['brier']:.3f} |"
        ),
        (
            f"| Decision probabilities | {decision_1x2_metrics['accuracy']:.3f} | "
            f"{decision_1x2_metrics['log_loss']:.3f} | {decision_1x2_metrics['brier']:.3f} |"
        ),
        "",
    ]
    if blend is not None:
        lines.extend(
            [
                "## Bookmaker Blend Calibration",
                "",
                f"- Validation season: {blend['validation_season']}",
                f"- 1X2 alpha: {blend['1x2_alpha']:.2f} (`0` = bookmaker only, `1` = model only)",
                "",
                "### 1X2 Alpha Search",
                "",
                frame_to_markdown(blend["1x2_alpha_metrics"], float_digits=3),
                "",
            ]
        )
        if blend["ou_2_5_alpha"] is not None:
            lines.extend(
                [
                    f"- Over/Under 2.5 alpha: {blend['ou_2_5_alpha']:.2f}",
                    "",
                    "### Over/Under 2.5 Alpha Search",
                    "",
                    frame_to_markdown(blend["ou_2_5_alpha_metrics"], float_digits=3),
                    "",
                ]
            )

    if ou_metrics is not None:
        lines.extend(
            [
                "## Over/Under 2.5 Probability Metrics",
                "",
                f"- Test rows with O/U 2.5 odds: {ou_metrics['rows']}",
                "",
                "| Source | Accuracy | Log loss | Brier |",
                "| --- | ---: | ---: | ---: |",
                (
                    f"| Bookmaker no-vig | {ou_metrics['bookmaker']['accuracy']:.3f} | "
                    f"{ou_metrics['bookmaker']['log_loss']:.3f} | {ou_metrics['bookmaker']['brier']:.3f} |"
                ),
                (
                    f"| Pure model | {ou_metrics['model']['accuracy']:.3f} | "
                    f"{ou_metrics['model']['log_loss']:.3f} | {ou_metrics['model']['brier']:.3f} |"
                ),
                (
                    f"| Decision probabilities | {ou_metrics['decision']['accuracy']:.3f} | "
                    f"{ou_metrics['decision']['log_loss']:.3f} | {ou_metrics['decision']['brier']:.3f} |"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Flat-Stake Value Backtest",
            "",
            "Each row bets one unit on outcomes where model probability x decimal odds - 1 is at least the threshold.",
            "",
            frame_to_markdown(
                threshold_display[
                    [
                        "market",
                        "min_edge",
                        "bets",
                        "total_staked",
                        "profit",
                        "roi",
                        "hit_rate",
                        "avg_odds",
                        "avg_edge",
                        "expected_profit",
                    ]
                ],
                float_digits=3,
            ),
            "",
            f"## Selected Bets At Edge >= {min_edge:.1%}",
            "",
            f"- CSV: `{bets_path}`",
            f"- Bets: {len(selected_bets)}",
            f"- Profit: {summarize_bets(selected_bets)['profit']:.2f}",
            f"- ROI: {summarize_bets(selected_bets)['roi']:.2%}",
            "",
            "## Largest Selected Edges",
            "",
            frame_to_markdown(
                selected_bets.sort_values("edge", ascending=False).head(20)
                if not selected_bets.empty
                else selected_bets,
                float_digits=3,
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
