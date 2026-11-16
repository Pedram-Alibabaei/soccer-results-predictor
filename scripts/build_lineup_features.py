from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import json
from pathlib import Path

import pandas as pd
import requests

from soccer_predictor.lineup_features import (
    add_transfermarkt_player_strength,
    build_match_lineup_features,
    parse_statsbomb_starters,
)


STATSBOMB_RAW_ROOT = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
TRANSFERMARKT_DATABASE_URL = (
    "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/"
    "data/transfermarkt-datasets.duckdb"
)
COMPETITIONS = (
    (43, 3, "FIFA World Cup", "2018"),
    (55, 43, "UEFA Euro", "2020"),
    (43, 106, "FIFA World Cup", "2022"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build confirmed-lineup and dated player-strength features."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--download-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_statsbomb = args.project_root / "data" / "raw" / "statsbomb"
    raw_transfermarkt = args.project_root / "data" / "raw" / "transfermarkt"
    processed_dir = args.project_root / "data" / "processed" / "lineups"
    report_path = args.project_root / "reports" / "lineup_features_coverage_report.md"
    raw_statsbomb.mkdir(parents=True, exist_ok=True)
    raw_transfermarkt.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    transfermarkt_path = raw_transfermarkt / "transfermarkt-datasets.duckdb"
    download_file(
        TRANSFERMARKT_DATABASE_URL,
        transfermarkt_path,
        force=args.force_download,
    )

    all_starters = []
    for competition_id, season_id, competition_name, season_name in COMPETITIONS:
        matches_path = raw_statsbomb / "matches" / f"{competition_id}_{season_id}.json"
        download_file(
            f"{STATSBOMB_RAW_ROOT}/matches/{competition_id}/{season_id}.json",
            matches_path,
            force=args.force_download,
        )
        matches = json.loads(matches_path.read_text(encoding="utf-8"))
        lineup_jobs = [
            (
                f"{STATSBOMB_RAW_ROOT}/lineups/{int(match['match_id'])}.json",
                raw_statsbomb / "lineups" / f"{int(match['match_id'])}.json",
                args.force_download,
            )
            for match in matches
        ]
        with ThreadPoolExecutor(max_workers=max(args.download_workers, 1)) as executor:
            list(executor.map(lambda job: download_file(job[0], job[1], force=job[2]), lineup_jobs))

        for match in matches:
            lineup_path = raw_statsbomb / "lineups" / f"{int(match['match_id'])}.json"
            lineup_payload = json.loads(lineup_path.read_text(encoding="utf-8"))
            all_starters.append(
                parse_statsbomb_starters(
                    match,
                    lineup_payload,
                    competition_name=competition_name,
                    season_name=season_name,
                )
            )

    starters = pd.concat(all_starters, ignore_index=True)
    starters = add_transfermarkt_player_strength(starters, transfermarkt_path)
    features = build_match_lineup_features(starters)

    starters_path = processed_dir / "international_starting_xi.csv"
    features_path = processed_dir / "international_lineup_features.csv"
    audit_path = processed_dir / "player_matching_audit.csv"
    metadata_path = processed_dir / "lineup_features_metadata.json"
    starters.to_csv(starters_path, index=False)
    features.to_csv(features_path, index=False)
    starters.loc[
        :,
        [
            "statsbomb_player_id",
            "player_name",
            "player_nickname",
            "team",
            "transfermarkt_player_id",
            "transfermarkt_player_name",
            "player_match_method",
            "player_match_score",
        ],
    ].drop_duplicates().sort_values(["team", "player_name"]).to_csv(audit_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "statsbomb_source": "https://github.com/statsbomb/open-data",
                "transfermarkt_dataset_source": "https://github.com/dcaribou/transfermarkt-datasets",
                "transfermarkt_license": "CC0-1.0",
                "competitions": [
                    {
                        "competition_id": competition_id,
                        "season_id": season_id,
                        "competition_name": competition_name,
                        "season_name": season_name,
                    }
                    for competition_id, season_id, competition_name, season_name in COMPETITIONS
                ],
                "starter_rows": int(len(starters)),
                "match_rows": int(len(features)),
                "player_match_rate": float(starters["transfermarkt_player_id"].notna().mean()),
                "valuation_coverage": float(starters["market_value_in_eur"].notna().mean()),
                "leakage_rule": "Each value is the latest Transfermarkt valuation dated on or before kickoff.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_coverage_report(report_path, starters, features)

    print(f"Built {len(features)} match-level lineup rows from {len(starters)} starters")
    print(f"  player match rate: {starters['transfermarkt_player_id'].notna().mean():.1%}")
    print(f"  valuation coverage: {starters['market_value_in_eur'].notna().mean():.1%}")
    print(f"  features: {features_path}")
    print(f"  audit: {audit_path}")
    print(f"  report: {report_path}")


def download_file(url: str, path: Path, *, force: bool) -> Path:
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".part")
    with requests.get(url, timeout=120, stream=True) as response:
        response.raise_for_status()
        with temporary_path.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
    temporary_path.replace(path)
    return path


def write_coverage_report(path: Path, starters: pd.DataFrame, features: pd.DataFrame) -> None:
    grouped = starters.groupby(["competition_name", "season_name"], sort=False)
    lines = [
        "# Confirmed Lineup Feature Coverage",
        "",
        "## Sources",
        "",
        "- Starting XIs and positions: StatsBomb Open Data.",
        "- Historical player strength proxy: Transfermarkt market values from the CC0 transfermarkt-datasets project.",
        "- Leakage control: every valuation is the latest record dated on or before match day.",
        "",
        "## Coverage",
        "",
        "| Competition | Season | Matches | Starter rows | Player matches | Dated values |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (competition, season), group in grouped:
        lines.append(
            f"| {competition} | {season} | {group['statsbomb_match_id'].nunique()} | "
            f"{len(group)} | {group['transfermarkt_player_id'].notna().mean():.1%} | "
            f"{group['market_value_in_eur'].notna().mean():.1%} |"
        )

    unmatched = (
        starters[starters["transfermarkt_player_id"].isna()]
        .loc[:, ["team", "player_name", "player_match_method", "player_match_score"]]
        .drop_duplicates()
        .sort_values(["team", "player_name"])
    )
    lines.extend(
        [
            "",
            "## Readiness",
            "",
            f"- Match rows with both complete starting XIs: {int(features['has_confirmed_lineups'].sum())}/{len(features)}.",
            f"- Unique unmatched player records requiring review: {len(unmatched)}.",
            "- Rotation is calculated against the team's previous match within the same tournament.",
            "- Player strength uses log market value by goalkeeper, defense, midfield, and attack.",
            "- These are confirmed-lineup features and must not be used before official XIs are available.",
            "",
            "## Unmatched Players",
            "",
            "| Team | StatsBomb player | Status | Best score |",
            "| --- | --- | --- | ---: |",
        ]
    )
    if unmatched.empty:
        lines.append("| - | - | None | - |")
    else:
        for row in unmatched.itertuples(index=False):
            lines.append(
                f"| {row.team} | {row.player_name} | {row.player_match_method} | "
                f"{float(row.player_match_score):.3f} |"
            )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
