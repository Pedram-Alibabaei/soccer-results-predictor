from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from soccer_predictor.modeling import frame_to_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate betting-mode validation run summaries.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--run-subdir", default="validation_runs")
    parser.add_argument("--output-name", default="expanded_betting_validation_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.project_root / "reports" / args.run_subdir
    paths = sorted(run_dir.glob("*_summary.json"))
    if not paths:
        raise FileNotFoundError(f"No validation summary JSON files found in {run_dir}")

    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    results = pd.DataFrame(records)
    results["status"] = results["chosen_edge_threshold"].apply(
        lambda value: "abstained" if pd.isna(value) else "active"
    )
    results["wins"] = results.apply(
        lambda row: resolved_count(row, "wins", round(row["hit_rate"] * row["bets"])),
        axis=1,
    )
    results["positive_clv_bets"] = results.apply(
        lambda row: resolved_count(
            row,
            "positive_clv_bets",
            round(row["positive_clv_rate"] * row["with_closing_odds"]),
        ),
        axis=1,
    )

    pooled = results[results["holdout_league"].isna()].copy()
    holdouts = results[results["holdout_league"].notna()].copy()
    aggregate = aggregate_results(pooled)
    holdout_aggregate = aggregate_results(holdouts)

    csv_path = run_dir / "expanded_betting_validation_results.csv"
    results.to_csv(csv_path, index=False)
    report_path = args.project_root / "reports" / args.output_name
    report_path.write_text(
        build_report(
            pooled,
            aggregate,
            holdouts,
            holdout_aggregate,
            csv_path=csv_path,
        ),
        encoding="utf-8",
    )

    print("Consolidated betting validation")
    print(f"  run summaries: {len(results)}")
    print(f"  pooled runs: {len(pooled)}")
    print(f"  league-holdout runs: {len(holdouts)}")
    print(f"  report: {report_path}")
    print(f"  csv: {csv_path}")


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, group in results.groupby("selection_mode", sort=True):
        bets = int(group["bets"].sum())
        staked = float(group["total_staked"].sum())
        closing_rows = int(group["with_closing_odds"].sum())
        rows.append(
            {
                "selection_mode": mode,
                "runs": int(len(group)),
                "active_runs": int((group["status"] == "active").sum()),
                "profitable_runs": int(((group["status"] == "active") & (group["profit"] > 0)).sum()),
                "bets": bets,
                "profit": float(group["profit"].sum()),
                "roi": float(group["profit"].sum()) / staked if staked else float("nan"),
                "hit_rate": float(group["wins"].sum()) / bets if bets else float("nan"),
                "avg_odds": (
                    float((group["avg_odds"] * group["bets"]).sum()) / bets
                    if bets
                    else float("nan")
                ),
                "positive_clv_rate": (
                    float(group["positive_clv_bets"].sum()) / closing_rows
                    if closing_rows
                    else float("nan")
                ),
                "avg_decimal_clv": (
                    float((group["avg_decimal_clv"] * group["with_closing_odds"]).sum())
                    / closing_rows
                    if closing_rows
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def resolved_count(row: pd.Series, column: str, fallback: int) -> int:
    value = row.get(column)
    return int(fallback if pd.isna(value) else value)


def build_report(
    pooled: pd.DataFrame,
    aggregate: pd.DataFrame,
    holdouts: pd.DataFrame,
    holdout_aggregate: pd.DataFrame,
    *,
    csv_path: Path,
) -> str:
    aggregate_by_mode = aggregate.set_index("selection_mode")
    conservative = aggregate_by_mode.loc["conservative"]
    value = aggregate_by_mode.loc["value"]
    latest_season = int(pooled["test_season"].max())
    latest = pooled[pooled["test_season"] == latest_season]
    latest_abstained = bool((latest["status"] == "abstained").all())
    pooled_columns = [
        "test_season",
        "selection_mode",
        "status",
        "chosen_edge_threshold",
        "bets",
        "profit",
        "roi",
        "hit_rate",
        "avg_odds",
        "positive_clv_rate",
        "avg_decimal_clv",
    ]
    holdout_columns = [
        "test_season",
        "holdout_league",
        "selection_mode",
        "status",
        "chosen_edge_threshold",
        "bets",
        "profit",
        "roi",
        "hit_rate",
        "positive_clv_rate",
        "avg_decimal_clv",
    ]
    lines = [
        "# Expanded Betting Validation",
        "",
        "## Fixed Policies",
        "",
        "- Value: opening odds, positive residual EV, odds at most `4.00`, rank by edge.",
        "- Conservative: opening odds, positive residual EV, probability at least `55%`, odds at most `2.50`, rank by probability.",
        "- Both: flat one-unit stakes, one bet per match, at most three bets per club weekend.",
        "- Edge threshold: selected on the immediately preceding validation season with at least 50 validation bets.",
        "",
        "## Pooled Rolling Holdouts",
        "",
        "These runs pool all ten leagues and are the closest current approximation to deploying one mode across the full club schedule.",
        "",
        frame_to_markdown(
            pooled.sort_values(["test_season", "selection_mode"]).loc[:, pooled_columns],
            float_digits=3,
        ),
        "",
        "### Pooled Aggregate",
        "",
        frame_to_markdown(aggregate, float_digits=3),
        "",
        "## Leave-One-League-Out Diagnostic",
        "",
        "For each row, model development and threshold selection exclude the named league. The final test contains only that league. These rows diagnose cross-league generalization; their profits must not be combined as a production portfolio because each run independently applies the three-bets-per-weekend cap.",
        "",
        frame_to_markdown(
            holdouts.sort_values(["holdout_league", "selection_mode"]).loc[:, holdout_columns],
            float_digits=3,
        ),
        "",
        "### Holdout Aggregate Diagnostic",
        "",
        frame_to_markdown(holdout_aggregate, float_digits=3),
        "",
        "## Current Verdict",
        "",
        (
            f"The pooled conservative mode placed {int(conservative['bets'])} bets for "
            f"{conservative['profit']:+.2f} units and {conservative['roi']:+.2%} ROI. "
            f"The pooled value mode placed {int(value['bets'])} bets for "
            f"{value['profit']:+.2f} units and {value['roi']:+.2%} ROI."
        ),
        "",
        (
            f"Both modes abstained in the fresh {latest_season}-{latest_season + 1} holdout because "
            "the preceding validation season did not produce a sufficiently supported profitable threshold."
            if latest_abstained
            else f"At least one mode was active in the fresh {latest_season}-{latest_season + 1} holdout."
        ),
        "",
        "The broader pooled evidence does not support deploying either mode for real-money betting. Conservative selection generalizes better than value selection in the league-holdout diagnostic, but its pooled ROI is still negative and the holdout samples are often small. Positive CLV alongside negative ROI is useful research evidence, not proof of profitability.",
        "",
        "## Reproducibility",
        "",
        f"Machine-readable results: `{csv_path}`",
        "",
        "Detailed run reports, selected-bet CSVs, and JSON summaries are stored under `reports/validation_runs/` and intentionally ignored by Git. This report is the compact tracked artifact.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
