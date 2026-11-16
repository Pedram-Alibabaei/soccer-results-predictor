from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    from scripts.summarize_betting_validation import aggregate_results
except ModuleNotFoundError:  # Direct execution adds scripts/, not the project root, to sys.path.
    from summarize_betting_validation import aggregate_results

from soccer_predictor.modeling import frame_to_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare bookmaker-anchored Conservative V2 with Conservative V1."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--v1-run-subdir", default="validation_runs")
    parser.add_argument("--v2-run-subdir", default="validation_runs_v2")
    parser.add_argument("--output-name", default="conservative_v2_validation_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = args.project_root / "reports"
    v1_dir = reports_dir / args.v1_run_subdir
    v2_dir = reports_dir / args.v2_run_subdir
    v1_paths = sorted(
        v1_dir.glob("*10leagues_conservative_maxodds_2p50_top3_residual_value_summary.json")
    )
    v2_paths = sorted(v2_dir.glob("*10leagues_conservative_v2_summary.json"))
    if not v1_paths:
        raise FileNotFoundError(f"No pooled Conservative V1 summaries found in {v1_dir}")
    if not v2_paths:
        raise FileNotFoundError(f"No pooled Conservative V2 summaries found in {v2_dir}")

    results = build_comparison_frame(
        read_json_records(v1_paths),
        read_json_records(v2_paths),
    )
    aggregate = aggregate_results(results)
    csv_path = v2_dir / "conservative_v2_comparison.csv"
    results.to_csv(csv_path, index=False)

    report_path = reports_dir / args.output_name
    report_path.write_text(build_report(results, aggregate), encoding="utf-8")

    print("Consolidated Conservative V2 validation")
    print(f"  runs: {len(results)}")
    print(f"  report: {report_path}")
    print(f"  csv: {csv_path}")


def read_json_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def build_comparison_frame(
    v1_records: Iterable[dict[str, Any]],
    v2_records: Iterable[dict[str, Any]],
) -> pd.DataFrame:
    rows = [normalize_v1_record(record) for record in v1_records]
    rows.extend(normalize_v2_record(record) for record in v2_records)
    frame = pd.DataFrame(rows)
    return frame.sort_values(["test_season", "selection_mode"]).reset_index(drop=True)


def normalize_v1_record(record: dict[str, Any]) -> dict[str, Any]:
    threshold = record.get("chosen_edge_threshold")
    row = normalize_common_record(record)
    row.update(
        {
            "selection_mode": "conservative_v1",
            "status": "active" if threshold is not None else "abstained",
            "model_weights": "residual only",
            "edge_thresholds": (
                f"all={float(threshold):.1%}" if threshold is not None else "no-bet"
            ),
        }
    )
    return row


def normalize_v2_record(record: dict[str, Any]) -> dict[str, Any]:
    thresholds = record.get("thresholds_by_market") or {}
    alphas = record.get("alpha_by_market") or {}
    active = any(value is not None for value in thresholds.values())
    row = normalize_common_record(record)
    row.update(
        {
            "selection_mode": "conservative_v2",
            "status": "active" if active else "abstained",
            "model_weights": format_mapping(alphas, percent=True),
            "edge_thresholds": format_mapping(thresholds, percent=True, none="no-bet"),
        }
    )
    return row


def normalize_common_record(record: dict[str, Any]) -> dict[str, Any]:
    bets = int(record.get("bets", 0))
    with_closing_odds = int(record.get("with_closing_odds", 0))
    return {
        "test_season": int(record["test_season"]),
        "bets": bets,
        "total_staked": float(record.get("total_staked", bets)),
        "profit": float(record.get("profit", 0.0)),
        "roi": float(record.get("roi", 0.0)),
        "wins": int(record.get("wins", round(float(record.get("hit_rate", 0.0)) * bets))),
        "hit_rate": float(record.get("hit_rate", 0.0)),
        "avg_odds": float(record.get("avg_odds", 0.0)),
        "with_closing_odds": with_closing_odds,
        "positive_clv_bets": int(
            record.get(
                "positive_clv_bets",
                round(float(record.get("positive_clv_rate", 0.0)) * with_closing_odds),
            )
        ),
        "positive_clv_rate": float(record.get("positive_clv_rate", 0.0)),
        "avg_decimal_clv": float(record.get("avg_decimal_clv", 0.0)),
    }


def format_mapping(
    values: dict[str, Any],
    *,
    percent: bool = False,
    none: str = "none",
) -> str:
    parts = []
    for key in sorted(values):
        value = values[key]
        if value is None:
            rendered = none
        elif percent:
            rendered = f"{float(value):.0%}"
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
    return ", ".join(parts) if parts else none


def build_report(results: pd.DataFrame, aggregate: pd.DataFrame) -> str:
    aggregate_by_mode = aggregate.set_index("selection_mode")
    v1 = aggregate_by_mode.loc["conservative_v1"]
    v2 = aggregate_by_mode.loc["conservative_v2"]
    active_v2_runs = int(v2["active_runs"])
    columns = [
        "test_season",
        "selection_mode",
        "status",
        "model_weights",
        "edge_thresholds",
        "bets",
        "profit",
        "roi",
        "hit_rate",
        "avg_odds",
        "positive_clv_rate",
        "avg_decimal_clv",
    ]
    lines = [
        "# Conservative V2 Validation",
        "",
        "## Frozen Policy",
        "",
        "- Use opening odds for decisions and settlement; use closing odds only for CLV.",
        "- Anchor residual probabilities to bookmaker no-vig probabilities separately for `1X2` and `Over/Under 2.5`.",
        "- Select the residual weight from `0%` through `50%` on validation log loss.",
        "- Select a separate edge threshold for each market on the immediately preceding validation season.",
        "- Require at least 20 validation bets and positive validation profit for an active market.",
        "- Require at least `55%` estimated probability and odds no higher than `2.50`.",
        "- Use flat one-unit stakes, at most one bet per market per weekend, one bet per match, and three bets per weekend overall.",
        "",
        "## Rolling Holdouts",
        "",
        frame_to_markdown(results.loc[:, columns], float_digits=3),
        "",
        "## Aggregate Comparison",
        "",
        frame_to_markdown(aggregate, float_digits=3),
        "",
        "## Verdict",
        "",
        (
            f"Conservative V1 placed {int(v1['bets'])} bets for {v1['profit']:+.2f} units "
            f"and {v1['roi']:+.2%} ROI. Conservative V2 placed {int(v2['bets'])} bets for "
            f"{v2['profit']:+.2f} units and {v2['roi']:+.2%} ROI."
        ),
        "",
        (
            f"V2 was active in only {active_v2_runs} of {int(v2['runs'])} test seasons. Its positive "
            "aggregate result therefore comes entirely from the 2022-2023 holdout and is not enough "
            "evidence of a repeatable betting edge. The later abstentions are useful capital protection, "
            "but the current policy is too sparse for practical deployment."
        ),
        "",
        "V2 remains a research candidate, not a real-money strategy. The next improvement should add genuinely independent information, especially historical xG and lineup availability, then repeat this exact rolling validation without tuning the test seasons.",
        "",
        "## Reproducibility",
        "",
        "Run each holdout with `scripts/backtest_conservative_v2.py`, then run `scripts/summarize_conservative_v2.py`.",
        "",
        "Detailed reports, bet CSVs, JSON summaries, and the comparison CSV are stored under `reports/validation_runs_v2/` and intentionally ignored by Git. This file is the compact tracked result.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
