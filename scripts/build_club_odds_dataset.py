from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from soccer_predictor.club_data import (
    DEFAULT_LEAGUES,
    DEFAULT_SEASON_START_YEARS,
    LeagueConfig,
    build_club_odds_dataset,
    summarize_club_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and process free club odds data.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root folder.",
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=min(DEFAULT_SEASON_START_YEARS),
        help="First season start year, e.g. 2019 for 2019-2020.",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=max(DEFAULT_SEASON_START_YEARS),
        help="Last season start year, e.g. 2024 for 2024-2025.",
    )
    parser.add_argument(
        "--leagues",
        default=",".join(league.code for league in DEFAULT_LEAGUES),
        help="Comma-separated Football-Data league codes. Defaults to the big five leagues.",
    )
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    league_lookup = {league.code: league for league in DEFAULT_LEAGUES}
    league_codes = [code.strip() for code in args.leagues.split(",") if code.strip()]
    unknown = sorted(set(league_codes).difference(league_lookup))
    if unknown:
        raise ValueError(f"Unknown default league code(s): {unknown}. Add them to DEFAULT_LEAGUES first.")

    leagues = tuple(league_lookup[code] for code in league_codes)
    season_start_years = tuple(range(args.start_season, args.end_season + 1))
    output_paths = build_club_odds_dataset(
        args.project_root,
        leagues=leagues,
        season_start_years=season_start_years,
        force_download=args.force_download,
    )

    features = pd.read_csv(output_paths["features"], parse_dates=["date"], low_memory=False)
    summary = summarize_club_dataset(features)
    print("Built free club odds dataset")
    print(f"  matches: {summary['rows']}")
    print(f"  leagues: {summary['leagues']}")
    print(f"  seasons: {summary['seasons']}")
    print(f"  teams: {summary['teams']}")
    print(f"  date range: {summary['first_match']} to {summary['last_match']}")
    print(f"  rows with 1X2 odds: {summary['rows_with_1x2_odds']}")
    print(f"  rows with O/U 2.5 odds: {summary['rows_with_ou_2_5_odds']}")
    for name, path in output_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
