from __future__ import annotations

import argparse
from math import isinf
from pathlib import Path

import pandas as pd

from soccer_predictor.modeling import (
    fit_dixon_coles_goal_model,
    fit_draw_aware_poisson_model,
    fit_poisson_goal_model,
    fit_softmax_baseline,
    frame_to_markdown,
)
from soccer_predictor.odds import odds_columns_for_market
from soccer_predictor.value_model import (
    add_clv_target,
    build_bet_candidates,
    choose_combined_threshold,
    choose_threshold,
    fit_clv_model,
    fit_residual_value_model,
    over_under_2_5_probabilities,
    select_value_bets,
    summarize_combined_thresholds,
    summarize_clv,
    summarize_selected_bets,
    summarize_thresholds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest residual value-bet filtering on club odds.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root folder.",
    )
    parser.add_argument("--test-season", type=int, default=2024)
    parser.add_argument("--validation-season", type=int, default=None)
    parser.add_argument(
        "--base-model",
        choices=("softmax", "poisson", "dc_poisson", "draw_poisson"),
        default="dc_poisson",
    )
    parser.add_argument(
        "--odds-timing",
        choices=("opening", "closing", "best_available"),
        default="opening",
    )
    parser.add_argument("--min-base-train-seasons", type=int, default=2)
    parser.add_argument("--base-epochs", type=int, default=1500)
    parser.add_argument("--base-learning-rate", type=float, default=0.03)
    parser.add_argument("--base-l2", type=float, default=0.01)
    parser.add_argument("--half-life-days", type=float, default=730.0)
    parser.add_argument("--max-goals", type=int, default=10)
    parser.add_argument("--draw-temperature", type=float, default=1.0)
    parser.add_argument("--value-epochs", type=int, default=2500)
    parser.add_argument("--value-learning-rate", type=float, default=0.05)
    parser.add_argument("--value-l2", type=float, default=0.01)
    parser.add_argument(
        "--use-clv-model",
        action="store_true",
        help="Add a second filter that predicts whether an opening bet will beat closing odds.",
    )
    parser.add_argument("--clv-epochs", type=int, default=2500)
    parser.add_argument("--clv-learning-rate", type=float, default=0.05)
    parser.add_argument("--clv-l2", type=float, default=0.01)
    parser.add_argument(
        "--clv-thresholds",
        default="0.45,0.50,0.52,0.55,0.58,0.60,0.65",
        help="Comma-separated minimum CLV-model probability thresholds.",
    )
    parser.add_argument("--stake", type=float, default=1.0)
    parser.add_argument("--max-odds", type=float, default=4.0)
    parser.add_argument("--max-bets-per-weekend", type=int, default=3)
    parser.add_argument("--min-validation-bets", type=int, default=50)
    parser.add_argument(
        "--thresholds",
        default="0,0.01,0.02,0.03,0.05,0.08,0.10,0.15,0.20",
        help="Comma-separated residual EV thresholds.",
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
    validation_season = args.validation_season or args.test_season - 1
    thresholds = parse_thresholds(args.thresholds)
    clv_thresholds = parse_thresholds(args.clv_thresholds)
    half_life_days = None if args.half_life_days <= 0 else args.half_life_days
    odds_columns_by_market = {
        "1x2": odds_columns_for_market("1x2", args.odds_timing),
        "over_under_2_5": odds_columns_for_market("over_under_2_5", args.odds_timing),
    }

    value_train_candidates, training_detail = build_walk_forward_candidates(
        data,
        candidate_seasons=[
            season
            for season in sorted(data["season_start_year"].unique())
            if season < validation_season
        ],
        min_base_train_seasons=args.min_base_train_seasons,
        odds_columns_by_market=odds_columns_by_market,
        args=args,
        half_life_days=half_life_days,
    )
    if value_train_candidates.empty:
        raise ValueError("No residual training candidates were produced. Lower min-base-train-seasons.")

    residual_model = fit_residual_value_model(
        value_train_candidates,
        learning_rate=args.value_learning_rate,
        epochs=args.value_epochs,
        l2=args.value_l2,
    )
    clv_train_candidates = pd.DataFrame()
    clv_model = None
    if args.use_clv_model:
        clv_train_candidates = add_clv_target(value_train_candidates)
        clv_model = fit_clv_model(
            value_train_candidates,
            learning_rate=args.clv_learning_rate,
            epochs=args.clv_epochs,
            l2=args.clv_l2,
        )

    validation_candidates, validation_model_name = candidates_for_season(
        data,
        season=validation_season,
        odds_columns_by_market=odds_columns_by_market,
        args=args,
        half_life_days=half_life_days,
    )
    validation_scored = residual_model.score_candidates(validation_candidates)
    if clv_model is not None:
        validation_scored = clv_model.score_clv_candidates(validation_scored)
    validation_value_summary = summarize_thresholds(
        validation_scored,
        thresholds,
        edge_column="value_edge",
        probability_column="value_probability",
        stake=args.stake,
        max_decimal_odds=args.max_odds,
        max_bets_per_weekend=args.max_bets_per_weekend,
    )
    validation_combined_summary = pd.DataFrame()
    if clv_model is not None:
        validation_combined_summary = summarize_combined_thresholds(
            validation_scored,
            thresholds,
            clv_thresholds,
            edge_column="value_edge",
            probability_column="value_probability",
            stake=args.stake,
            max_decimal_odds=args.max_odds,
            max_bets_per_weekend=args.max_bets_per_weekend,
        )
        chosen_threshold, chosen_clv_threshold = choose_combined_threshold(
            validation_combined_summary,
            min_bets=args.min_validation_bets,
        )
    else:
        chosen_threshold = choose_threshold(
            validation_value_summary,
            min_bets=args.min_validation_bets,
        )
        chosen_clv_threshold = float("inf")

    test_candidates, test_model_name = candidates_for_season(
        data,
        season=args.test_season,
        odds_columns_by_market=odds_columns_by_market,
        args=args,
        half_life_days=half_life_days,
    )
    test_scored = residual_model.score_candidates(test_candidates)
    if clv_model is not None:
        test_scored = clv_model.score_clv_candidates(test_scored)
    test_value_summary = summarize_thresholds(
        test_scored,
        thresholds,
        edge_column="value_edge",
        probability_column="value_probability",
        stake=args.stake,
        max_decimal_odds=args.max_odds,
        max_bets_per_weekend=args.max_bets_per_weekend,
    )
    test_combined_summary = (
        summarize_combined_thresholds(
            test_scored,
            thresholds,
            clv_thresholds,
            edge_column="value_edge",
            probability_column="value_probability",
            stake=args.stake,
            max_decimal_odds=args.max_odds,
            max_bets_per_weekend=args.max_bets_per_weekend,
        )
        if clv_model is not None
        else pd.DataFrame()
    )
    test_raw_summary = summarize_thresholds(
        test_scored,
        thresholds,
        edge_column="raw_edge",
        probability_column="model_probability",
        stake=args.stake,
        max_decimal_odds=args.max_odds,
        max_bets_per_weekend=args.max_bets_per_weekend,
    )
    selected = (
        select_value_bets(
            test_scored,
            min_edge=chosen_threshold,
            stake=args.stake,
            edge_column="value_edge",
            probability_column="value_probability",
            max_decimal_odds=args.max_odds,
            max_bets_per_weekend=args.max_bets_per_weekend,
            secondary_probability_column="clv_model_probability" if clv_model is not None else None,
            min_secondary_probability=chosen_clv_threshold if clv_model is not None else None,
        )
        if not isinf(chosen_threshold) and (clv_model is None or not isinf(chosen_clv_threshold))
        else select_value_bets(
            test_scored.iloc[0:0],
            min_edge=0.0,
            stake=args.stake,
            edge_column="value_edge",
            probability_column="value_probability",
        )
    )

    report_dir = args.project_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    filter_slug = "residual_value_clvmodel" if args.use_clv_model else "residual_value"
    report_stem = (
        f"club_odds_{args.test_season}_{args.base_model}_{args.odds_timing}"
        f"_maxodds_{format_slug(args.max_odds)}_top{args.max_bets_per_weekend}_{filter_slug}"
    )
    report_path = report_dir / f"{report_stem}_report.md"
    bets_path = report_dir / f"{report_stem}_bets.csv"
    selected.to_csv(bets_path, index=False)
    write_report(
        report_path,
        args=args,
        base_model_name=test_model_name,
        validation_model_name=validation_model_name,
        value_train_candidates=value_train_candidates,
        training_detail=training_detail,
        validation_scored=validation_scored,
        test_scored=test_scored,
        validation_value_summary=validation_value_summary,
        validation_combined_summary=validation_combined_summary,
        test_value_summary=test_value_summary,
        test_combined_summary=test_combined_summary,
        test_raw_summary=test_raw_summary,
        chosen_threshold=chosen_threshold,
        chosen_clv_threshold=chosen_clv_threshold,
        clv_train_candidates=clv_train_candidates,
        selected=selected,
        bets_path=bets_path,
    )

    summary = summarize_selected_bets(selected)
    threshold_text = "no-bet" if isinf(chosen_threshold) else f"{chosen_threshold:.2%}"
    clv_threshold_text = (
        "disabled"
        if clv_model is None
        else ("no-bet" if isinf(chosen_clv_threshold) else f"{chosen_clv_threshold:.2%}")
    )
    print("Backtested residual value model")
    print(f"  base model: {test_model_name}")
    print(f"  odds timing: {args.odds_timing}")
    print(f"  max odds: {args.max_odds:.2f}")
    print(f"  max bets per weekend: {args.max_bets_per_weekend}")
    print(f"  value train candidates: {len(value_train_candidates)}")
    if clv_model is not None:
        print(f"  CLV train candidates: {len(clv_train_candidates)}")
    print(f"  validation candidates: {len(validation_scored)}")
    print(f"  test candidates: {len(test_scored)}")
    print(f"  chosen residual edge threshold: {threshold_text}")
    print(f"  chosen CLV-model threshold: {clv_threshold_text}")
    print(f"  selected bets: {summary['bets']}")
    print(f"  profit: {summary['profit']:.2f}")
    print(f"  ROI: {summary['roi']:.2%}")
    clv = summarize_clv(selected)
    print(f"  positive CLV rate: {clv['positive_clv_rate']:.2%}")
    print(f"  avg decimal CLV: {clv['avg_decimal_clv']:.2%}")
    print(f"  report: {report_path}")
    print(f"  bets: {bets_path}")


def build_walk_forward_candidates(
    data: pd.DataFrame,
    *,
    candidate_seasons: list[int],
    min_base_train_seasons: int,
    odds_columns_by_market: dict[str, dict[str, str]],
    args: argparse.Namespace,
    half_life_days: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    details = []
    for season in candidate_seasons:
        base_train = data[data["season_start_year"] < season].copy()
        if base_train["season_start_year"].nunique() < min_base_train_seasons:
            continue
        season_candidates, base_model_name = candidates_for_season(
            data,
            season=season,
            odds_columns_by_market=odds_columns_by_market,
            args=args,
            half_life_days=half_life_days,
        )
        season_candidates["candidate_role"] = "value_train"
        frames.append(season_candidates)
        details.append(
            {
                "season": season,
                "base_train_rows": int(len(base_train)),
                "base_model": base_model_name,
                "candidates": int(len(season_candidates)),
                "winning_candidates": int(season_candidates["won"].sum()),
            }
        )

    if not frames:
        return pd.DataFrame(), pd.DataFrame(details)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(details)


def candidates_for_season(
    data: pd.DataFrame,
    *,
    season: int,
    odds_columns_by_market: dict[str, dict[str, str]],
    args: argparse.Namespace,
    half_life_days: float | None,
) -> tuple[pd.DataFrame, str]:
    train = data[data["season_start_year"] < season].copy()
    test = data[data["season_start_year"] == season].copy()
    if train.empty or test.empty:
        raise ValueError(f"Need train and test rows for season {season}.")

    model, model_name, expected_goals = fit_base_model(
        train,
        test,
        model_type=args.base_model,
        learning_rate=args.base_learning_rate,
        epochs=args.base_epochs,
        l2=args.base_l2,
        half_life_days=half_life_days,
        max_goals=args.max_goals,
        draw_temperature=args.draw_temperature,
    )
    probabilities_by_market = {"1x2": model.predict_proba(test)}
    if expected_goals is not None:
        probabilities_by_market["over_under_2_5"] = over_under_2_5_probabilities(expected_goals)

    candidates = build_bet_candidates(test, probabilities_by_market, odds_columns_by_market)
    candidates["base_model"] = model_name
    return candidates, model_name


def fit_base_model(
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


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    base_model_name: str,
    validation_model_name: str,
    value_train_candidates: pd.DataFrame,
    training_detail: pd.DataFrame,
    validation_scored: pd.DataFrame,
    test_scored: pd.DataFrame,
    validation_value_summary: pd.DataFrame,
    validation_combined_summary: pd.DataFrame,
    test_value_summary: pd.DataFrame,
    test_combined_summary: pd.DataFrame,
    test_raw_summary: pd.DataFrame,
    chosen_threshold: float,
    chosen_clv_threshold: float,
    clv_train_candidates: pd.DataFrame,
    selected: pd.DataFrame,
    bets_path: Path,
) -> None:
    chosen_text = "no-bet" if isinf(chosen_threshold) else f"{chosen_threshold:.3f}"
    chosen_clv_text = (
        "disabled"
        if not args.use_clv_model
        else ("no-bet" if isinf(chosen_clv_threshold) else f"{chosen_clv_threshold:.3f}")
    )
    selected_summary = summarize_selected_bets(selected)
    clv_summary = summarize_clv(selected)
    clv_train_positive_rate = (
        float(clv_train_candidates["positive_clv"].mean()) if not clv_train_candidates.empty else 0.0
    )
    market_summary = (
        selected.groupby("market")
        .agg(
            bets=("market", "size"),
            profit=("settled_profit", "sum"),
            hit_rate=("won", "mean"),
            avg_odds=("decimal_odds", "mean"),
            avg_edge=("selected_edge", "mean"),
            positive_clv_rate=("clv_decimal", lambda values: float((values > 0).mean())),
            avg_decimal_clv=("clv_decimal", "mean"),
            avg_probability_clv=("clv_probability", "mean"),
        )
        .reset_index()
        if not selected.empty
        else pd.DataFrame(
            columns=[
                "market",
                "bets",
                "profit",
                "hit_rate",
                "avg_odds",
                "avg_edge",
                "positive_clv_rate",
                "avg_decimal_clv",
                "avg_probability_clv",
            ]
        )
    )

    lines = [
        f"# Residual Value Backtest - {base_model_name}",
        "",
        "## Setup",
        "",
        f"- Test season: {args.test_season}",
        f"- Validation season: {args.validation_season or args.test_season - 1}",
        f"- Odds timing: {args.odds_timing}",
        f"- Max odds: {args.max_odds:.2f}",
        f"- Max bets per weekend: {args.max_bets_per_weekend}",
        f"- Base model: {base_model_name}",
        f"- Validation base model: {validation_model_name}",
        f"- CLV model filter: {'enabled' if args.use_clv_model else 'disabled'}",
        f"- Value train candidates: {len(value_train_candidates)}",
        f"- CLV train candidates: {len(clv_train_candidates)}",
        f"- CLV train positive rate: {clv_train_positive_rate:.2%}",
        f"- Validation candidates: {len(validation_scored)}",
        f"- Test candidates: {len(test_scored)}",
        f"- Chosen residual edge threshold: {chosen_text}",
        f"- Chosen CLV-model probability threshold: {chosen_clv_text}",
        "",
        "## Walk-Forward Value Training Seasons",
        "",
        frame_to_markdown(training_detail if not training_detail.empty else pd.DataFrame()),
        "",
        "## Validation Threshold Search",
        "",
        frame_to_markdown(validation_value_summary, float_digits=3),
        "",
        "## Validation Combined Residual + CLV Search",
        "",
        frame_to_markdown(validation_combined_summary, float_digits=3),
        "",
        "## Test: Raw Base-Model Edge",
        "",
        frame_to_markdown(test_raw_summary, float_digits=3),
        "",
        "## Test: Residual Value Edge",
        "",
        frame_to_markdown(test_value_summary, float_digits=3),
        "",
        "## Test: Residual + CLV Model Edge",
        "",
        frame_to_markdown(test_combined_summary, float_digits=3),
        "",
        "## Selected Bets",
        "",
        f"- CSV: `{bets_path}`",
        f"- Bets: {selected_summary['bets']}",
        f"- Profit: {selected_summary['profit']:.2f}",
        f"- ROI: {selected_summary['roi']:.2%}",
        f"- Hit rate: {selected_summary['hit_rate']:.2%}",
        f"- Positive CLV rate: {clv_summary['positive_clv_rate']:.2%}",
        f"- Average decimal CLV: {clv_summary['avg_decimal_clv']:.2%}",
        f"- Average no-vig probability CLV: {clv_summary['avg_probability_clv']:.2%}",
        "",
        "### By Market",
        "",
        frame_to_markdown(market_summary, float_digits=3),
        "",
        "### Largest Selected Edges",
        "",
        frame_to_markdown(
            selected.sort_values("selected_edge", ascending=False).head(25)
            if not selected.empty
            else selected,
            float_digits=3,
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_thresholds(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def format_slug(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


if __name__ == "__main__":
    main()
