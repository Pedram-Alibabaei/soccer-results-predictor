from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from soccer_predictor.club_data import summarize_club_data_coverage
from soccer_predictor.modeling import frame_to_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit club odds and feature coverage by league-season.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--minimum-rate",
        type=float,
        default=0.95,
        help="Coverage rate below which a league-season is flagged.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = (
        args.project_root
        / "data"
        / "processed"
        / "club_odds"
        / "club_matches.csv"
    )
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Club dataset not found: {dataset_path}. Run scripts/build_club_odds_dataset.py first."
        )

    data = pd.read_csv(dataset_path, parse_dates=["date"], low_memory=False)
    coverage = summarize_club_data_coverage(data)
    output_csv = dataset_path.parent / "club_data_coverage.csv"
    coverage.to_csv(output_csv, index=False)

    rate_columns = [
        "open_1x2_rate",
        "close_1x2_rate",
        "open_ou_2_5_rate",
        "close_ou_2_5_rate",
        "shot_stat_rate",
    ]
    issues = coverage[(coverage[rate_columns] < args.minimum_rate).any(axis=1)].copy()
    report_path = args.project_root / "reports" / "club_data_coverage_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_report(data, coverage, issues, minimum_rate=args.minimum_rate, csv_path=output_csv),
        encoding="utf-8",
    )

    print("Audited club data coverage")
    print(f"  matches: {len(data)}")
    print(f"  leagues: {data['league_code'].nunique()}")
    print(f"  seasons: {data['season_start_year'].nunique()}")
    print(f"  league-seasons: {len(coverage)}")
    print(f"  flagged league-seasons: {len(issues)}")
    print(f"  report: {report_path}")
    print(f"  csv: {output_csv}")


def build_report(
    data: pd.DataFrame,
    coverage: pd.DataFrame,
    issues: pd.DataFrame,
    *,
    minimum_rate: float,
    csv_path: Path,
) -> str:
    display_columns = [
        "season_start_year",
        "league_code",
        "league_name",
        "matches",
        "open_1x2_rate",
        "close_1x2_rate",
        "open_ou_2_5_rate",
        "close_ou_2_5_rate",
        "shot_stat_rate",
        "fully_comparable_rate",
    ]
    overall = pd.DataFrame(
        [
            {
                "matches": len(data),
                "leagues": data["league_code"].nunique(),
                "seasons": data["season_start_year"].nunique(),
                "league_seasons": len(coverage),
                "first_match": data["date"].min().date().isoformat(),
                "last_match": data["date"].max().date().isoformat(),
                "flagged_league_seasons": len(issues),
            }
        ]
    )
    lines = [
        "# Club Data Coverage Audit",
        "",
        "## Scope",
        "",
        frame_to_markdown(overall, float_digits=3),
        "",
        f"Coverage CSV: `{csv_path}`",
        "",
        f"A league-season is flagged when any required rate is below {minimum_rate:.0%}.",
        "",
        "## Flagged League-Seasons",
        "",
        frame_to_markdown(issues.loc[:, display_columns], float_digits=3),
        "",
        "## Complete Coverage Table",
        "",
        frame_to_markdown(coverage.loc[:, display_columns], float_digits=3),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
