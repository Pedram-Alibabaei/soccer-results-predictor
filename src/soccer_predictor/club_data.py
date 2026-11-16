"""Club match and odds ingestion from Football-Data.co.uk."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import pandas as pd

from .club_features import add_club_pre_match_features
from .elo import add_elo_features


FOOTBALL_DATA_BASE_URL = "https://www.football-data.co.uk/mmz4281"
CLUB_RAW_DIR = "club_odds"
CLUB_PROCESSED_DIR = "club_odds"
CLUB_MATCHES_FILE = "club_matches.csv"
CLUB_FEATURES_FILE = "club_matches_with_elo_features.csv"
CLUB_COVERAGE_FILE = "club_data_coverage.csv"
CLUB_METADATA_FILE = "club_dataset_metadata.json"


@dataclass(frozen=True)
class LeagueConfig:
    code: str
    name: str
    country: str


DEFAULT_LEAGUES = (
    LeagueConfig("E0", "Premier League", "England"),
    LeagueConfig("E1", "Championship", "England"),
    LeagueConfig("SP1", "La Liga", "Spain"),
    LeagueConfig("I1", "Serie A", "Italy"),
    LeagueConfig("D1", "Bundesliga", "Germany"),
    LeagueConfig("F1", "Ligue 1", "France"),
    LeagueConfig("N1", "Eredivisie", "Netherlands"),
    LeagueConfig("P1", "Primeira Liga", "Portugal"),
    LeagueConfig("B1", "Belgian Pro League", "Belgium"),
    LeagueConfig("SC0", "Scottish Premiership", "Scotland"),
)
DEFAULT_SEASON_START_YEARS = tuple(range(2019, 2026))

RESULT_MAP = {"H": "1", "D": "X", "A": "2"}

OPENING_1X2_TRIPLETS = (
    ("Avg", ("AvgH", "AvgD", "AvgA")),
    ("B365", ("B365H", "B365D", "B365A")),
    ("Pinnacle", ("PSH", "PSD", "PSA")),
    ("Betfair", ("BFH", "BFD", "BFA")),
    ("WilliamHill", ("WHH", "WHD", "WHA")),
)
CLOSING_1X2_TRIPLETS = (
    ("AvgC", ("AvgCH", "AvgCD", "AvgCA")),
    ("B365C", ("B365CH", "B365CD", "B365CA")),
    ("PinnacleC", ("PSCH", "PSCD", "PSCA")),
    ("BetfairC", ("BFCH", "BFCD", "BFCA")),
    ("WilliamHillC", ("WHCH", "WHCD", "WHCA")),
)
OPENING_OU_25_PAIRS = (
    ("Avg", ("Avg>2.5", "Avg<2.5")),
    ("B365", ("B365>2.5", "B365<2.5")),
    ("Pinnacle", ("P>2.5", "P<2.5")),
)
CLOSING_OU_25_PAIRS = (
    ("AvgC", ("AvgC>2.5", "AvgC<2.5")),
    ("B365C", ("B365C>2.5", "B365C<2.5")),
    ("PinnacleC", ("PC>2.5", "PC<2.5")),
)


def season_code(season_start_year: int) -> str:
    """Return Football-Data's compact season code, e.g. 2024 -> 2425."""

    start = season_start_year % 100
    end = (season_start_year + 1) % 100
    return f"{start:02d}{end:02d}"


def football_data_csv_url(league_code: str, season_start_year: int) -> str:
    return f"{FOOTBALL_DATA_BASE_URL}/{season_code(season_start_year)}/{league_code}.csv"


def download_football_data_csv(
    raw_dir: Path,
    league: LeagueConfig,
    season_start_year: int,
    *,
    force: bool = False,
) -> Path:
    """Download one Football-Data CSV and return the local path."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / f"{season_code(season_start_year)}_{league.code}.csv"
    if destination.exists() and not force:
        return destination

    url = football_data_csv_url(league.code, season_start_year)
    try:
        urlretrieve(url, destination)
    except (HTTPError, URLError) as exc:
        if destination.exists():
            destination.unlink()
        raise RuntimeError(f"Could not download {url}") from exc

    first_line = destination.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    if "Date" not in first_line or "HomeTeam" not in first_line:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file did not look like a Football-Data CSV: {url}")

    return destination


def build_club_odds_dataset(
    project_root: Path,
    *,
    leagues: tuple[LeagueConfig, ...] = DEFAULT_LEAGUES,
    season_start_years: tuple[int, ...] = DEFAULT_SEASON_START_YEARS,
    force_download: bool = False,
) -> dict[str, Path]:
    """Download free club odds files and write model-ready CSV outputs."""

    raw_dir = project_root / "data" / "raw" / CLUB_RAW_DIR
    processed_dir = project_root / "data" / "processed" / CLUB_PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_paths: list[Path] = []
    processed_frames: list[pd.DataFrame] = []
    for season_start_year in season_start_years:
        for league in leagues:
            raw_path = download_football_data_csv(
                raw_dir,
                league,
                season_start_year,
                force=force_download,
            )
            raw_paths.append(raw_path)
            source = pd.read_csv(raw_path, low_memory=False)
            processed_frames.append(
                process_football_data_frame(
                    source,
                    league=league,
                    season_start_year=season_start_year,
                    source_url=football_data_csv_url(league.code, season_start_year),
                )
            )

    matches = pd.concat(processed_frames, ignore_index=True)
    matches = matches.sort_values(
        ["date", "league_code", "home_team", "away_team"],
        kind="mergesort",
    ).reset_index(drop=True)
    matches.insert(0, "match_id", matches.index + 1)

    enriched_matches = add_club_pre_match_features(matches)
    features = add_elo_features(enriched_matches)

    output_paths = {
        "matches": processed_dir / CLUB_MATCHES_FILE,
        "features": processed_dir / CLUB_FEATURES_FILE,
        "coverage": processed_dir / CLUB_COVERAGE_FILE,
        "metadata": processed_dir / CLUB_METADATA_FILE,
    }
    matches.to_csv(output_paths["matches"], index=False)
    features.to_csv(output_paths["features"], index=False)
    summarize_club_data_coverage(matches).to_csv(output_paths["coverage"], index=False)
    write_club_metadata(
        output_paths["metadata"],
        matches=matches,
        features=features,
        raw_paths=raw_paths,
        leagues=leagues,
        season_start_years=season_start_years,
    )
    return output_paths


def process_football_data_frame(
    source: pd.DataFrame,
    *,
    league: LeagueConfig,
    season_start_year: int,
    source_url: str = "",
) -> pd.DataFrame:
    """Normalize one Football-Data league-season CSV."""

    required_columns = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    missing = required_columns.difference(source.columns)
    if missing:
        raise ValueError(f"Missing required Football-Data columns: {sorted(missing)}")

    matches = source.copy()
    matches = matches.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]).copy()
    matches["date"] = pd.to_datetime(matches["Date"], dayfirst=True, errors="coerce")
    matches["home_goals"] = pd.to_numeric(matches["FTHG"], errors="coerce")
    matches["away_goals"] = pd.to_numeric(matches["FTAG"], errors="coerce")
    matches["result"] = matches["FTR"].map(RESULT_MAP)
    matches = matches.dropna(subset=["date", "home_goals", "away_goals", "result"]).copy()

    matches["home_team"] = matches["HomeTeam"].astype("string").str.strip()
    matches["away_team"] = matches["AwayTeam"].astype("string").str.strip()
    matches["home_goals"] = matches["home_goals"].astype(int)
    matches["away_goals"] = matches["away_goals"].astype(int)
    matches["home_shots"] = numeric_column(matches, "HS")
    matches["away_shots"] = numeric_column(matches, "AS")
    matches["home_shots_on_target"] = numeric_column(matches, "HST")
    matches["away_shots_on_target"] = numeric_column(matches, "AST")
    matches["total_goals"] = matches["home_goals"] + matches["away_goals"]
    matches["goal_difference"] = matches["home_goals"] - matches["away_goals"]
    matches["both_teams_scored"] = (matches["home_goals"] > 0) & (matches["away_goals"] > 0)
    matches["neutral"] = False
    matches["tournament"] = league.name
    matches["league_code"] = league.code
    matches["league_name"] = league.name
    matches["country"] = league.country
    matches["venue_country"] = league.country
    matches["city"] = ""
    matches["season_start_year"] = int(season_start_year)
    matches["season"] = f"{season_start_year}-{season_start_year + 1}"
    matches["source_url"] = source_url
    matches["kickoff_time"] = matches["Time"].astype("string").str.strip() if "Time" in matches else ""

    opening_1x2 = coalesce_triplet(matches, OPENING_1X2_TRIPLETS)
    closing_1x2 = coalesce_triplet(matches, CLOSING_1X2_TRIPLETS)
    opening_ou = coalesce_pair(matches, OPENING_OU_25_PAIRS)
    closing_ou = coalesce_pair(matches, CLOSING_OU_25_PAIRS)

    matches["odds_open_1"] = opening_1x2["first"]
    matches["odds_open_X"] = opening_1x2["second"]
    matches["odds_open_2"] = opening_1x2["third"]
    matches["odds_open_1x2_source"] = opening_1x2["source"]
    matches["odds_close_1"] = closing_1x2["first"]
    matches["odds_close_X"] = closing_1x2["second"]
    matches["odds_close_2"] = closing_1x2["third"]
    matches["odds_close_1x2_source"] = closing_1x2["source"]
    matches["has_open_1x2_odds"] = matches[["odds_open_1", "odds_open_X", "odds_open_2"]].notna().all(axis=1)
    matches["has_close_1x2_odds"] = matches[["odds_close_1", "odds_close_X", "odds_close_2"]].notna().all(axis=1)
    matches["odds_1"] = matches["odds_close_1"].fillna(matches["odds_open_1"])
    matches["odds_X"] = matches["odds_close_X"].fillna(matches["odds_open_X"])
    matches["odds_2"] = matches["odds_close_2"].fillna(matches["odds_open_2"])
    matches["odds_1x2_source"] = matches["odds_close_1x2_source"].where(
        matches["odds_close_1"].notna(),
        matches["odds_open_1x2_source"],
    )
    matches["has_1x2_odds"] = matches[["odds_1", "odds_X", "odds_2"]].notna().all(axis=1)

    matches["odds_open_over_2_5"] = opening_ou["first"]
    matches["odds_open_under_2_5"] = opening_ou["second"]
    matches["odds_open_ou_2_5_source"] = opening_ou["source"]
    matches["odds_close_over_2_5"] = closing_ou["first"]
    matches["odds_close_under_2_5"] = closing_ou["second"]
    matches["odds_close_ou_2_5_source"] = closing_ou["source"]
    matches["has_open_ou_2_5_odds"] = matches[
        ["odds_open_over_2_5", "odds_open_under_2_5"]
    ].notna().all(axis=1)
    matches["has_close_ou_2_5_odds"] = matches[
        ["odds_close_over_2_5", "odds_close_under_2_5"]
    ].notna().all(axis=1)
    matches["odds_over_2_5"] = matches["odds_close_over_2_5"].fillna(matches["odds_open_over_2_5"])
    matches["odds_under_2_5"] = matches["odds_close_under_2_5"].fillna(matches["odds_open_under_2_5"])
    matches["odds_ou_2_5_source"] = matches["odds_close_ou_2_5_source"].where(
        matches["odds_close_over_2_5"].notna(),
        matches["odds_open_ou_2_5_source"],
    )
    matches["has_ou_2_5_odds"] = matches[["odds_over_2_5", "odds_under_2_5"]].notna().all(axis=1)

    return matches.loc[
        :,
        [
            "date",
            "kickoff_time",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "home_shots",
            "away_shots",
            "home_shots_on_target",
            "away_shots_on_target",
            "result",
            "total_goals",
            "goal_difference",
            "both_teams_scored",
            "neutral",
            "tournament",
            "league_code",
            "league_name",
            "country",
            "city",
            "venue_country",
            "season_start_year",
            "season",
            "source_url",
            "odds_open_1",
            "odds_open_X",
            "odds_open_2",
            "odds_open_1x2_source",
            "odds_close_1",
            "odds_close_X",
            "odds_close_2",
            "odds_close_1x2_source",
            "has_open_1x2_odds",
            "has_close_1x2_odds",
            "odds_1",
            "odds_X",
            "odds_2",
            "odds_1x2_source",
            "has_1x2_odds",
            "odds_open_over_2_5",
            "odds_open_under_2_5",
            "odds_open_ou_2_5_source",
            "odds_close_over_2_5",
            "odds_close_under_2_5",
            "odds_close_ou_2_5_source",
            "has_open_ou_2_5_odds",
            "has_close_ou_2_5_odds",
            "odds_over_2_5",
            "odds_under_2_5",
            "odds_ou_2_5_source",
            "has_ou_2_5_odds",
        ],
    ]


def numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float("nan"), index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def coalesce_triplet(
    frame: pd.DataFrame,
    candidates: tuple[tuple[str, tuple[str, str, str]], ...],
) -> pd.DataFrame:
    """Pick the first available valid decimal-odds triplet for each row."""

    result = pd.DataFrame(
        {
            "first": float("nan"),
            "second": float("nan"),
            "third": float("nan"),
            "source": "",
        },
        index=frame.index,
    )

    for source_name, columns in candidates:
        if any(column not in frame.columns for column in columns):
            continue
        values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
        valid = values.gt(1.0).all(axis=1)
        missing = result["first"].isna()
        mask = missing & valid
        if not mask.any():
            continue
        result.loc[mask, "first"] = values.loc[mask, columns[0]]
        result.loc[mask, "second"] = values.loc[mask, columns[1]]
        result.loc[mask, "third"] = values.loc[mask, columns[2]]
        result.loc[mask, "source"] = source_name

    return result


def coalesce_pair(
    frame: pd.DataFrame,
    candidates: tuple[tuple[str, tuple[str, str]], ...],
) -> pd.DataFrame:
    """Pick the first available valid decimal-odds pair for each row."""

    result = pd.DataFrame(
        {
            "first": float("nan"),
            "second": float("nan"),
            "source": "",
        },
        index=frame.index,
    )

    for source_name, columns in candidates:
        if any(column not in frame.columns for column in columns):
            continue
        values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
        valid = values.gt(1.0).all(axis=1)
        missing = result["first"].isna()
        mask = missing & valid
        if not mask.any():
            continue
        result.loc[mask, "first"] = values.loc[mask, columns[0]]
        result.loc[mask, "second"] = values.loc[mask, columns[1]]
        result.loc[mask, "source"] = source_name

    return result


def summarize_club_dataset(dataset: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(dataset)),
        "leagues": int(dataset["league_code"].nunique()),
        "seasons": int(dataset["season_start_year"].nunique()),
        "teams": int(pd.unique(pd.concat([dataset["home_team"], dataset["away_team"]])).size),
        "first_match": dataset["date"].min().date().isoformat(),
        "last_match": dataset["date"].max().date().isoformat(),
        "rows_with_1x2_odds": int(dataset["has_1x2_odds"].sum()),
        "rows_with_ou_2_5_odds": int(dataset["has_ou_2_5_odds"].sum()),
    }


def summarize_club_data_coverage(dataset: pd.DataFrame) -> pd.DataFrame:
    """Summarize model-feature and odds coverage for every league-season."""

    rows: list[dict[str, object]] = []
    group_columns = ["season_start_year", "league_code", "league_name", "country"]
    for keys, group in dataset.groupby(group_columns, sort=True, dropna=False):
        season, league_code, league_name, country = keys
        matches = len(group)
        shots_complete = group[
            ["home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target"]
        ].notna().all(axis=1)
        open_1x2 = group["has_open_1x2_odds"].fillna(False).astype(bool)
        close_1x2 = group["has_close_1x2_odds"].fillna(False).astype(bool)
        open_ou = group["has_open_ou_2_5_odds"].fillna(False).astype(bool)
        close_ou = group["has_close_ou_2_5_odds"].fillna(False).astype(bool)
        comparable = open_1x2 & close_1x2 & open_ou & close_ou
        rows.append(
            {
                "season_start_year": int(season),
                "league_code": league_code,
                "league_name": league_name,
                "country": country,
                "matches": matches,
                "first_match": group["date"].min().date().isoformat(),
                "last_match": group["date"].max().date().isoformat(),
                "open_1x2_rows": int(open_1x2.sum()),
                "open_1x2_rate": float(open_1x2.mean()),
                "close_1x2_rows": int(close_1x2.sum()),
                "close_1x2_rate": float(close_1x2.mean()),
                "open_ou_2_5_rows": int(open_ou.sum()),
                "open_ou_2_5_rate": float(open_ou.mean()),
                "close_ou_2_5_rows": int(close_ou.sum()),
                "close_ou_2_5_rate": float(close_ou.mean()),
                "shot_stat_rows": int(shots_complete.sum()),
                "shot_stat_rate": float(shots_complete.mean()),
                "fully_comparable_rows": int(comparable.sum()),
                "fully_comparable_rate": float(comparable.mean()),
            }
        )
    return pd.DataFrame(rows)


def write_club_metadata(
    path: Path,
    *,
    matches: pd.DataFrame,
    features: pd.DataFrame,
    raw_paths: list[Path],
    leagues: tuple[LeagueConfig, ...],
    season_start_years: tuple[int, ...],
) -> None:
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "name": "Football-Data.co.uk",
            "base_url": FOOTBALL_DATA_BASE_URL,
            "raw_files": [str(path) for path in raw_paths],
            "odds_caveat": "The default odds columns use average closing 1X2 and over/under 2.5 odds when available, falling back to opening/pre-closing columns if closing columns are missing.",
        },
        "scope": {
            "leagues": [
                {"code": league.code, "name": league.name, "country": league.country}
                for league in leagues
            ],
            "season_start_years": list(season_start_years),
        },
        "processing": {
            "match_rows": int(len(matches)),
            "feature_rows": int(len(features)),
            "first_completed_match": matches["date"].min().date().isoformat(),
            "last_completed_match": matches["date"].max().date().isoformat(),
            "rows_with_1x2_odds": int(matches["has_1x2_odds"].sum()),
            "rows_with_ou_2_5_odds": int(matches["has_ou_2_5_odds"].sum()),
        },
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
