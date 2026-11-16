from __future__ import annotations

import argparse
import json
from math import isinf
from pathlib import Path

import pandas as pd

from backtest_residual_value import (
    build_walk_forward_candidates,
    candidates_for_season,
    parse_thresholds,
)
from soccer_predictor.conservative import (
    DEFAULT_ANCHOR_ALPHAS,
    calibration_table,
    fit_market_anchor_alphas,
    probability_diagnostics,
    score_market_anchored_candidates,
    select_conservative_v2_bets,
)
from soccer_predictor.modeling import frame_to_markdown
from soccer_predictor.odds import odds_columns_for_market
from soccer_predictor.value_model import (
    choose_threshold,
    fit_residual_value_model,
    summarize_clv,
    summarize_selected_bets,
    summarize_thresholds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest bookmaker-anchored Conservative V2.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--test-season", type=int, default=2025)
    parser.add_argument("--validation-season", type=int, default=None)
    parser.add_argument("--base-model", choices=("poisson", "dc_poisson"), default="dc_poisson")
    parser.add_argument("--odds-timing", choices=("opening",), default="opening")
    parser.add_argument("--min-base-train-seasons", type=int, default=1)
    parser.add_argument("--base-epochs", type=int, default=1500)
    parser.add_argument("--base-learning-rate", type=float, default=0.03)
    parser.add_argument("--base-l2", type=float, default=0.01)
    parser.add_argument("--half-life-days", type=float, default=730.0)
    parser.add_argument("--max-goals", type=int, default=10)
    parser.add_argument("--draw-temperature", type=float, default=1.0)
    parser.add_argument("--value-epochs", type=int, default=2500)
    parser.add_argument("--value-learning-rate", type=float, default=0.05)
    parser.add_argument("--value-l2", type=float, default=0.01)
    parser.add_argument("--stake", type=float, default=1.0)
    parser.add_argument("--max-odds", type=float, default=2.5)
    parser.add_argument("--min-probability", type=float, default=0.55)
    parser.add_argument("--max-bets-per-market-weekend", type=int, default=1)
    parser.add_argument("--max-bets-per-weekend", type=int, default=3)
    parser.add_argument("--min-validation-bets-per-market", type=int, default=20)
    parser.add_argument(
        "--thresholds",
        default="0,0.01,0.02,0.03,0.05,0.08,0.10",
        help="Comma-separated anchored-EV thresholds.",
    )
    parser.add_argument(
        "--anchor-alphas",
        default=",".join(str(alpha) for alpha in DEFAULT_ANCHOR_ALPHAS),
        help="Candidate residual-model weights; bookmaker weight is one minus alpha.",
    )
    parser.add_argument("--holdout-league", default=None)
    parser.add_argument("--report-subdir", default="validation_runs_v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
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
    available_leagues = set(data["league_code"].dropna().astype(str))
    if args.holdout_league and args.holdout_league not in available_leagues:
        raise ValueError(
            f"Unknown holdout league {args.holdout_league!r}; available: {sorted(available_leagues)}"
        )
    development_data = (
        data[data["league_code"] != args.holdout_league].copy()
        if args.holdout_league
        else data
    )
    validation_season = args.validation_season or args.test_season - 1
    thresholds = parse_thresholds(args.thresholds)
    anchor_alphas = tuple(parse_thresholds(args.anchor_alphas))
    half_life_days = None if args.half_life_days <= 0 else args.half_life_days
    odds_columns_by_market = {
        "1x2": odds_columns_for_market("1x2", args.odds_timing),
        "over_under_2_5": odds_columns_for_market("over_under_2_5", args.odds_timing),
    }

    value_train_candidates, training_detail = build_walk_forward_candidates(
        development_data,
        candidate_seasons=[
            season
            for season in sorted(development_data["season_start_year"].unique())
            if season < validation_season
        ],
        min_base_train_seasons=args.min_base_train_seasons,
        odds_columns_by_market=odds_columns_by_market,
        args=args,
        half_life_days=half_life_days,
    )
    if value_train_candidates.empty:
        raise ValueError("No residual training candidates were produced.")
    residual_model = fit_residual_value_model(
        value_train_candidates,
        learning_rate=args.value_learning_rate,
        epochs=args.value_epochs,
        l2=args.value_l2,
    )

    validation_candidates, validation_model_name = candidates_for_season(
        development_data,
        season=validation_season,
        odds_columns_by_market=odds_columns_by_market,
        args=args,
        half_life_days=half_life_days,
    )
    validation_residual = residual_model.score_candidates(validation_candidates)
    alpha_by_market, alpha_search = fit_market_anchor_alphas(
        validation_residual,
        alpha_candidates=anchor_alphas,
    )
    validation_scored = score_market_anchored_candidates(
        validation_residual,
        alpha_by_market,
    )
    validation_thresholds, chosen_thresholds = summarize_market_thresholds(
        validation_scored,
        thresholds=thresholds,
        args=args,
    )

    test_candidates, test_model_name = candidates_for_season(
        data,
        season=args.test_season,
        odds_columns_by_market=odds_columns_by_market,
        args=args,
        half_life_days=half_life_days,
        train_league_codes=(
            set(development_data["league_code"].astype(str).unique())
            if args.holdout_league
            else None
        ),
        test_league_codes={args.holdout_league} if args.holdout_league else None,
    )
    test_residual = residual_model.score_candidates(test_candidates)
    test_scored = score_market_anchored_candidates(test_residual, alpha_by_market)
    test_thresholds, _ = summarize_market_thresholds(
        test_scored,
        thresholds=thresholds,
        args=args,
        choose=False,
    )
    selected = select_conservative_v2_bets(
        test_scored,
        chosen_thresholds,
        min_probability=args.min_probability,
        max_decimal_odds=args.max_odds,
        stake=args.stake,
        max_bets_per_market_weekend=args.max_bets_per_market_weekend,
        max_bets_per_weekend=args.max_bets_per_weekend,
    )

    validation_diagnostics = probability_diagnostics(
        validation_scored,
        {
            "bookmaker_no_vig": "bookmaker_probability",
            "base_model": "model_probability",
            "residual_v1": "value_probability",
            "anchored_v2": "conservative_probability",
        },
    )
    test_diagnostics = probability_diagnostics(
        test_scored,
        {
            "bookmaker_no_vig": "bookmaker_probability",
            "base_model": "model_probability",
            "residual_v1": "value_probability",
            "anchored_v2": "conservative_probability",
        },
    )
    reliability = calibration_table(
        test_scored,
        probability_column="conservative_probability",
    )
    outcome_summary = summarize_outcomes(selected)

    report_dir = args.project_root / "reports" / args.report_subdir
    report_dir.mkdir(parents=True, exist_ok=True)
    scope_slug = (
        f"holdout_{args.holdout_league}"
        if args.holdout_league
        else f"{data['league_code'].nunique()}leagues"
    )
    stem = f"club_odds_{args.test_season}_{args.base_model}_opening_{scope_slug}_conservative_v2"
    report_path = report_dir / f"{stem}_report.md"
    bets_path = report_dir / f"{stem}_bets.csv"
    summary_path = report_dir / f"{stem}_summary.json"
    selected.to_csv(bets_path, index=False)

    summary = summarize_selected_bets(selected)
    clv = summarize_clv(selected)
    write_report(
        report_path,
        args=args,
        validation_season=validation_season,
        validation_model_name=validation_model_name,
        test_model_name=test_model_name,
        value_train_candidates=value_train_candidates,
        training_detail=training_detail,
        alpha_by_market=alpha_by_market,
        alpha_search=alpha_search,
        chosen_thresholds=chosen_thresholds,
        validation_thresholds=validation_thresholds,
        test_thresholds=test_thresholds,
        validation_diagnostics=validation_diagnostics,
        test_diagnostics=test_diagnostics,
        reliability=reliability,
        outcome_summary=outcome_summary,
        selected=selected,
        bets_path=bets_path,
    )
    summary_path.write_text(
        json.dumps(
            {
                "strategy": "conservative_v2",
                "test_season": args.test_season,
                "validation_season": validation_season,
                "holdout_league": args.holdout_league,
                "alpha_by_market": alpha_by_market,
                "thresholds_by_market": {
                    market: None if isinf(value) else value
                    for market, value in chosen_thresholds.items()
                },
                "wins": int(selected["won"].sum()) if not selected.empty else 0,
                **summary,
                "with_closing_odds": clv["with_closing_odds"],
                "positive_clv_bets": (
                    int((selected["clv_decimal"] > 0).sum()) if not selected.empty else 0
                ),
                "positive_clv_rate": clv["positive_clv_rate"],
                "avg_decimal_clv": clv["avg_decimal_clv"],
                "avg_probability_clv": clv["avg_probability_clv"],
                "report_path": str(report_path),
                "bets_path": str(bets_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Backtested Conservative V2")
    print(f"  test season: {args.test_season}")
    print(f"  anchor alphas: {alpha_by_market}")
    print(f"  market thresholds: {format_thresholds(chosen_thresholds)}")
    print(f"  bets: {summary['bets']}")
    print(f"  profit: {summary['profit']:.2f}")
    print(f"  ROI: {summary['roi']:.2%}")
    print(f"  hit rate: {summary['hit_rate']:.2%}")
    print(f"  positive CLV rate: {clv['positive_clv_rate']:.2%}")
    print(f"  report: {report_path}")
    print(f"  summary: {summary_path}")


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.min_probability <= 1.0:
        raise ValueError("min-probability must be between 0 and 1.")
    if args.max_odds <= 1.0:
        raise ValueError("max-odds must be greater than 1.")
    if args.min_validation_bets_per_market < 1:
        raise ValueError("min-validation-bets-per-market must be positive.")


def summarize_market_thresholds(
    scored: pd.DataFrame,
    *,
    thresholds: list[float],
    args: argparse.Namespace,
    choose: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    frames = []
    chosen: dict[str, float] = {}
    for market in sorted(scored["market"].dropna().unique()):
        summary = summarize_thresholds(
            scored[scored["market"] == market],
            thresholds,
            edge_column="conservative_edge",
            probability_column="conservative_probability",
            stake=args.stake,
            max_decimal_odds=args.max_odds,
            max_bets_per_weekend=args.max_bets_per_market_weekend,
            selection_mode="conservative",
            min_probability=args.min_probability,
        )
        summary.insert(0, "market", market)
        frames.append(summary)
        chosen[market] = (
            choose_threshold(summary, min_bets=args.min_validation_bets_per_market)
            if choose
            else float("inf")
        )
    return pd.concat(frames, ignore_index=True), chosen


def summarize_outcomes(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame(
            columns=["market", "outcome", "bets", "profit", "roi", "hit_rate", "avg_odds"]
        )
    summary = (
        selected.groupby(["market", "outcome"])
        .agg(
            bets=("outcome", "size"),
            profit=("settled_profit", "sum"),
            hit_rate=("won", "mean"),
            avg_odds=("decimal_odds", "mean"),
        )
        .reset_index()
    )
    summary["roi"] = summary["profit"] / summary["bets"]
    return summary.loc[:, ["market", "outcome", "bets", "profit", "roi", "hit_rate", "avg_odds"]]


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    validation_season: int,
    validation_model_name: str,
    test_model_name: str,
    value_train_candidates: pd.DataFrame,
    training_detail: pd.DataFrame,
    alpha_by_market: dict[str, float],
    alpha_search: pd.DataFrame,
    chosen_thresholds: dict[str, float],
    validation_thresholds: pd.DataFrame,
    test_thresholds: pd.DataFrame,
    validation_diagnostics: pd.DataFrame,
    test_diagnostics: pd.DataFrame,
    reliability: pd.DataFrame,
    outcome_summary: pd.DataFrame,
    selected: pd.DataFrame,
    bets_path: Path,
) -> None:
    summary = summarize_selected_bets(selected)
    clv = summarize_clv(selected)
    lines = [
        "# Conservative V2 Backtest",
        "",
        "## Setup",
        "",
        f"- Test season: {args.test_season}",
        f"- Validation season: {validation_season}",
        f"- League scope: {'hold out ' + args.holdout_league if args.holdout_league else 'all leagues'}",
        "- Odds used for decisions and settlement: opening",
        "- Closing odds used only for CLV diagnostics",
        f"- Base model: {test_model_name}",
        f"- Validation base model: {validation_model_name}",
        f"- Residual-model weights by market: {alpha_by_market}",
        f"- Chosen EV thresholds by market: {format_thresholds(chosen_thresholds)}",
        f"- Minimum probability: {args.min_probability:.2%}",
        f"- Maximum odds: {args.max_odds:.2f}",
        f"- Maximum bets per market/weekend: {args.max_bets_per_market_weekend}",
        f"- Global maximum bets per weekend: {args.max_bets_per_weekend}",
        f"- Value-training candidates: {len(value_train_candidates)}",
        "",
        "## Walk-Forward Training",
        "",
        frame_to_markdown(training_detail, float_digits=3),
        "",
        "## Anchor Alpha Search",
        "",
        frame_to_markdown(alpha_search, float_digits=3),
        "",
        "## Validation Probability Diagnostics",
        "",
        frame_to_markdown(validation_diagnostics, float_digits=3),
        "",
        "## Test Probability Diagnostics",
        "",
        frame_to_markdown(test_diagnostics, float_digits=3),
        "",
        "## Test Reliability",
        "",
        frame_to_markdown(reliability, float_digits=3),
        "",
        "## Validation Threshold Search by Market",
        "",
        frame_to_markdown(validation_thresholds, float_digits=3),
        "",
        "## Test Threshold Sensitivity by Market",
        "",
        frame_to_markdown(test_thresholds, float_digits=3),
        "",
        "## Selected Bets",
        "",
        f"- CSV: `{bets_path}`",
        f"- Bets: {summary['bets']}",
        f"- Profit: {summary['profit']:.2f}",
        f"- ROI: {summary['roi']:.2%}",
        f"- Hit rate: {summary['hit_rate']:.2%}",
        f"- Positive CLV rate: {clv['positive_clv_rate']:.2%}",
        f"- Average decimal CLV: {clv['avg_decimal_clv']:.2%}",
        "",
        "### Outcome Bias Check",
        "",
        frame_to_markdown(outcome_summary, float_digits=3),
        "",
        "### Selected Rows",
        "",
        frame_to_markdown(selected, float_digits=3),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def format_thresholds(thresholds: dict[str, float]) -> dict[str, str]:
    return {
        market: ("no-bet" if isinf(value) else f"{value:.2%}")
        for market, value in thresholds.items()
    }


if __name__ == "__main__":
    main()
