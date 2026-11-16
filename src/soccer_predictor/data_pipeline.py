"""Data ingestion and dataset-building helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from .elo import add_elo_features, fit_elo_state


INTERNATIONAL_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
SHOOTOUTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"

RAW_RESULTS_FILE = "international_results.csv"
RAW_SHOOTOUTS_FILE = "shootouts.csv"
SETTLEMENTS_FILE = "world_cup_settlements.csv"
PROCESSED_MATCHES_FILE = "matches_processed.csv"
FEATURE_DATASET_FILE = "matches_with_elo_features.csv"
METADATA_FILE = "dataset_metadata.json"

TRAIN_FILE = "train.csv"
VALIDATION_FILE = "validation.csv"
TEST_FILE = "test.csv"
HOLDOUT_DIR = "tournament_holdouts"
WORLD_CUP_HOLDOUT_YEARS = (2018, 2022)


@dataclass(frozen=True)
class DatasetSplitConfig:
    validation_start: str = "2019-01-01"
    test_start: str = "2023-01-01"

    @property
    def validation_start_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.validation_start)

    @property
    def test_start_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.test_start)


def download_international_results(raw_dir: Path, *, force: bool = False) -> Path:
    """Download the men's international results CSV."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / RAW_RESULTS_FILE
    if destination.exists() and not force:
        return destination

    urlretrieve(INTERNATIONAL_RESULTS_URL, destination)
    return destination


def download_shootouts(raw_dir: Path, *, force: bool = False) -> Path:
    """Download the international penalty shootouts CSV."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / RAW_SHOOTOUTS_FILE
    if destination.exists() and not force:
        return destination

    urlretrieve(SHOOTOUTS_URL, destination)
    return destination


def load_raw_results(path: Path) -> pd.DataFrame:
    """Load source results with parsed dates."""

    return pd.read_csv(path, parse_dates=["date"])


def load_shootouts(path: Path) -> pd.DataFrame:
    """Load source penalty shootout winners with parsed dates."""

    return pd.read_csv(path, parse_dates=["date"])


def process_international_results(results: pd.DataFrame) -> pd.DataFrame:
    """Normalize international results into the project's canonical match schema."""

    required_columns = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
    }
    missing = required_columns.difference(results.columns)
    if missing:
        raise ValueError(f"Missing required source columns: {sorted(missing)}")

    matches = results.loc[:, sorted(required_columns)].copy()
    matches = matches.rename(
        columns={
            "home_score": "home_goals",
            "away_score": "away_goals",
            "country": "venue_country",
        }
    )

    matches = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    matches["date"] = pd.to_datetime(matches["date"], errors="raise")
    matches["home_team"] = matches["home_team"].map(_clean_team_name)
    matches["away_team"] = matches["away_team"].map(_clean_team_name)
    matches["tournament"] = matches["tournament"].astype("string").str.strip()
    matches["city"] = matches["city"].astype("string").str.strip()
    matches["venue_country"] = matches["venue_country"].astype("string").str.strip()
    matches["neutral"] = matches["neutral"].astype(bool)
    matches["home_goals"] = matches["home_goals"].astype(int)
    matches["away_goals"] = matches["away_goals"].astype(int)

    matches = matches.sort_values(
        ["date", "home_team", "away_team", "tournament"],
        kind="mergesort",
    ).reset_index(drop=True)
    matches.insert(0, "match_id", matches.index + 1)

    matches["total_goals"] = matches["home_goals"] + matches["away_goals"]
    matches["goal_difference"] = matches["home_goals"] - matches["away_goals"]
    matches["result"] = matches["goal_difference"].map(_result_from_goal_difference)
    matches["both_teams_scored"] = (matches["home_goals"] > 0) & (matches["away_goals"] > 0)
    matches = add_world_cup_context(matches)
    matches = initialize_settlement_columns(matches)

    return matches[
        [
            "match_id",
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "result",
            "total_goals",
            "goal_difference",
            "both_teams_scored",
            "tournament",
            "city",
            "venue_country",
            "neutral",
            "is_world_cup",
            "world_cup_year",
            "world_cup_match_number",
            "world_cup_match_number_scaled",
            "world_cup_stage",
            "is_world_cup_group_stage",
            "is_world_cup_knockout_stage",
            "is_world_cup_round_of_16",
            "is_world_cup_quarter_final",
            "is_world_cup_semi_final",
            "is_world_cup_third_place",
            "is_world_cup_final",
            "world_cup_group_matchday",
            "is_world_cup_final_group_match",
            "is_clean_3way_result",
            "market_3way_home_goals",
            "market_3way_away_goals",
            "market_3way_result",
            "has_90_minute_settlement",
            "qualification_result",
            "settlement_notes",
        ]
    ]


def initialize_settlement_columns(matches: pd.DataFrame) -> pd.DataFrame:
    """Add default settlement columns before external overrides are applied."""

    enriched = matches.copy()
    enriched["market_3way_home_goals"] = enriched["home_goals"]
    enriched["market_3way_away_goals"] = enriched["away_goals"]
    enriched["market_3way_result"] = enriched["result"]
    enriched["has_90_minute_settlement"] = ~enriched["is_world_cup_knockout_stage"]
    enriched["qualification_result"] = ""
    enriched["settlement_notes"] = ""
    return enriched


def apply_world_cup_settlements(matches: pd.DataFrame, settlements: pd.DataFrame) -> pd.DataFrame:
    """Apply curated 90-minute and qualification settlements to World Cup rows."""

    enriched = matches.copy()
    if settlements.empty:
        return enriched

    required_columns = {
        "date",
        "home_team",
        "away_team",
        "market_3way_home_goals",
        "market_3way_away_goals",
        "qualification_result",
        "notes",
    }
    missing = required_columns.difference(settlements.columns)
    if missing:
        raise ValueError(f"Missing required settlement columns: {sorted(missing)}")

    cleaned = settlements.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="raise")
    cleaned["home_team"] = cleaned["home_team"].map(_clean_team_name)
    cleaned["away_team"] = cleaned["away_team"].map(_clean_team_name)

    for settlement in cleaned.itertuples(index=False):
        mask = (
            enriched["date"].eq(settlement.date)
            & enriched["home_team"].eq(settlement.home_team)
            & enriched["away_team"].eq(settlement.away_team)
        )
        if not mask.any():
            raise ValueError(
                "Settlement row did not match any processed match: "
                f"{settlement.date.date()} {settlement.home_team} vs {settlement.away_team}"
            )

        home_goals = int(settlement.market_3way_home_goals)
        away_goals = int(settlement.market_3way_away_goals)
        enriched.loc[mask, "market_3way_home_goals"] = home_goals
        enriched.loc[mask, "market_3way_away_goals"] = away_goals
        enriched.loc[mask, "market_3way_result"] = _result_from_goal_difference(home_goals - away_goals)
        enriched.loc[mask, "has_90_minute_settlement"] = True
        enriched.loc[mask, "is_clean_3way_result"] = True
        enriched.loc[mask, "qualification_result"] = str(settlement.qualification_result)
        enriched.loc[mask, "settlement_notes"] = str(settlement.notes)

    return enriched


def add_qualification_results(matches: pd.DataFrame, shootouts: pd.DataFrame) -> pd.DataFrame:
    """Derive home/away qualification labels for World Cup knockout matches."""

    enriched = matches.copy()
    shootout_lookup = _shootout_winner_lookup(shootouts)
    knockout_mask = enriched["is_world_cup_knockout_stage"].astype(bool)

    for row in enriched[knockout_mask].itertuples():
        if str(row.qualification_result) in {"1", "2"}:
            continue

        if row.home_goals > row.away_goals:
            qualification_result = "1"
        elif row.away_goals > row.home_goals:
            qualification_result = "2"
        else:
            winner = shootout_lookup.get((row.date, row.home_team, row.away_team), "")
            if winner == row.home_team:
                qualification_result = "1"
            elif winner == row.away_team:
                qualification_result = "2"
            else:
                qualification_result = ""

        enriched.at[row.Index, "qualification_result"] = qualification_result

    return enriched


def add_world_cup_context(matches: pd.DataFrame) -> pd.DataFrame:
    """Add World Cup stage/context features without changing source scores."""

    enriched = matches.copy()
    enriched["is_world_cup"] = enriched["tournament"].eq("FIFA World Cup")
    enriched["world_cup_year"] = 0
    enriched["world_cup_match_number"] = 0
    enriched["world_cup_match_number_scaled"] = 0.0
    enriched["world_cup_stage"] = "not_world_cup"
    enriched["is_world_cup_group_stage"] = False
    enriched["is_world_cup_knockout_stage"] = False
    enriched["is_world_cup_round_of_16"] = False
    enriched["is_world_cup_quarter_final"] = False
    enriched["is_world_cup_semi_final"] = False
    enriched["is_world_cup_third_place"] = False
    enriched["is_world_cup_final"] = False
    enriched["world_cup_group_matchday"] = 0
    enriched["is_world_cup_final_group_match"] = False

    world_cup_rows = enriched[enriched["is_world_cup"]]
    for year, index in world_cup_rows.groupby(world_cup_rows["date"].dt.year).groups.items():
        ordered_index = enriched.loc[index].sort_values(
            ["date", "match_id"],
            kind="mergesort",
        ).index
        match_count = _expected_world_cup_match_count(int(year), len(ordered_index))
        for position, row_index in enumerate(ordered_index, start=1):
            stage = _world_cup_stage(position, match_count)
            group_matchday = _world_cup_group_matchday(position, match_count, stage)
            enriched.at[row_index, "world_cup_year"] = int(year)
            enriched.at[row_index, "world_cup_match_number"] = position
            enriched.at[row_index, "world_cup_match_number_scaled"] = position / match_count
            enriched.at[row_index, "world_cup_stage"] = stage
            enriched.at[row_index, "is_world_cup_group_stage"] = stage == "group"
            enriched.at[row_index, "is_world_cup_knockout_stage"] = stage != "group"
            enriched.at[row_index, "is_world_cup_round_of_16"] = stage == "round_of_16"
            enriched.at[row_index, "is_world_cup_quarter_final"] = stage == "quarter_final"
            enriched.at[row_index, "is_world_cup_semi_final"] = stage == "semi_final"
            enriched.at[row_index, "is_world_cup_third_place"] = stage == "third_place"
            enriched.at[row_index, "is_world_cup_final"] = stage == "final"
            enriched.at[row_index, "world_cup_group_matchday"] = group_matchday
            enriched.at[row_index, "is_world_cup_final_group_match"] = group_matchday == 3

    enriched["is_clean_3way_result"] = ~enriched["is_world_cup_knockout_stage"]
    return enriched


def build_feature_dataset(matches: pd.DataFrame) -> pd.DataFrame:
    """Create the first model-ready dataset with pre-match Elo features."""

    return add_elo_features(matches)


def split_dataset(
    dataset: pd.DataFrame,
    config: DatasetSplitConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Create chronological train, validation, and test splits."""

    split_config = config or DatasetSplitConfig()
    data = dataset.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)

    train = data[data["date"] < split_config.validation_start_date].copy()
    validation = data[
        (data["date"] >= split_config.validation_start_date)
        & (data["date"] < split_config.test_start_date)
    ].copy()
    test = data[data["date"] >= split_config.test_start_date].copy()

    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def build_world_cup_holdouts(
    matches: pd.DataFrame,
    *,
    years: tuple[int, ...] = WORLD_CUP_HOLDOUT_YEARS,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Build strict World Cup holdouts with pre-tournament features.

    Each train split contains matches before that World Cup begins. Each test split
    contains only that year's FIFA World Cup finals matches. Test features are
    computed from the pre-tournament Elo state and are not updated with World Cup
    test results.
    """

    holdouts = {}
    sorted_matches = matches.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)

    for year in years:
        test_mask = (sorted_matches["tournament"] == "FIFA World Cup") & (
            sorted_matches["date"].dt.year == year
        )
        test_matches = sorted_matches[test_mask].copy()
        if test_matches.empty:
            continue

        tournament_start = test_matches["date"].min()
        train_matches = sorted_matches[sorted_matches["date"] < tournament_start].copy()

        train_features = add_elo_features(train_matches)
        pre_tournament_state = fit_elo_state(train_matches)
        test_features = add_elo_features(
            test_matches,
            initial_state=pre_tournament_state,
            update_ratings=False,
        )

        holdout_name = f"world_cup_{year}"
        holdouts[holdout_name] = {
            "train": train_features.reset_index(drop=True),
            "test": test_features.reset_index(drop=True),
        }

    return holdouts


def build_all_datasets(
    project_root: Path,
    *,
    force_download: bool = False,
    split_config: DatasetSplitConfig | None = None,
) -> dict[str, Path]:
    """Download raw data and write processed CSV outputs."""

    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_path = download_international_results(raw_dir, force=force_download)
    shootouts_path = download_shootouts(raw_dir, force=force_download)
    raw_results = load_raw_results(raw_path)
    shootouts = load_shootouts(shootouts_path)
    matches = process_international_results(raw_results)
    settlements_path = project_root / "data" / "external" / SETTLEMENTS_FILE
    if settlements_path.exists():
        settlements = pd.read_csv(settlements_path)
        matches = apply_world_cup_settlements(matches, settlements)
    matches = add_qualification_results(matches, shootouts)
    feature_dataset = build_feature_dataset(matches)
    splits = split_dataset(feature_dataset, split_config)
    holdouts = build_world_cup_holdouts(matches)

    output_paths = {
        "raw": raw_path,
        "shootouts": shootouts_path,
        "processed_matches": processed_dir / PROCESSED_MATCHES_FILE,
        "feature_dataset": processed_dir / FEATURE_DATASET_FILE,
        "metadata": processed_dir / METADATA_FILE,
        "train": processed_dir / TRAIN_FILE,
        "validation": processed_dir / VALIDATION_FILE,
        "test": processed_dir / TEST_FILE,
    }

    matches.to_csv(output_paths["processed_matches"], index=False)
    feature_dataset.to_csv(output_paths["feature_dataset"], index=False)
    splits["train"].to_csv(output_paths["train"], index=False)
    splits["validation"].to_csv(output_paths["validation"], index=False)
    splits["test"].to_csv(output_paths["test"], index=False)

    holdout_dir = processed_dir / HOLDOUT_DIR
    holdout_dir.mkdir(parents=True, exist_ok=True)
    for holdout_name, holdout_splits in holdouts.items():
        train_path = holdout_dir / f"{holdout_name}_train.csv"
        test_path = holdout_dir / f"{holdout_name}_test.csv"
        holdout_splits["train"].to_csv(train_path, index=False)
        holdout_splits["test"].to_csv(test_path, index=False)
        output_paths[f"{holdout_name}_train"] = train_path
        output_paths[f"{holdout_name}_test"] = test_path

    _write_metadata(
        output_paths["metadata"],
        raw_results,
        shootouts,
        matches,
        feature_dataset,
        splits,
        split_config,
        holdouts,
    )

    return output_paths


def summarize_dataset(dataset: pd.DataFrame) -> dict[str, object]:
    """Return a compact summary for CLI output and sanity checks."""

    return {
        "rows": int(len(dataset)),
        "teams": int(pd.unique(pd.concat([dataset["home_team"], dataset["away_team"]])).size),
        "first_match": dataset["date"].min().date().isoformat(),
        "last_match": dataset["date"].max().date().isoformat(),
        "tournaments": int(dataset["tournament"].nunique()),
    }


def _clean_team_name(value: object) -> str:
    return str(value).strip()


def _result_from_goal_difference(goal_difference: int) -> str:
    if goal_difference > 0:
        return "1"
    if goal_difference < 0:
        return "2"
    return "X"


def _write_metadata(
    path: Path,
    raw_results: pd.DataFrame,
    shootouts: pd.DataFrame,
    matches: pd.DataFrame,
    feature_dataset: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    split_config: DatasetSplitConfig | None,
    holdouts: dict[str, dict[str, pd.DataFrame]],
) -> None:
    config = split_config or DatasetSplitConfig()
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "name": "martj42/international_results",
            "url": INTERNATIONAL_RESULTS_URL,
            "shootouts_url": SHOOTOUTS_URL,
            "raw_rows": int(len(raw_results)),
            "shootout_rows": int(len(shootouts)),
            "score_caveat": "Some knockout match scores include extra time; use clean_3way filtering or add a 90-minute settlement source before 3-way betting backtests.",
        },
        "processing": {
            "completed_match_rows": int(len(matches)),
            "dropped_unplayed_or_unscored_rows": int(len(raw_results) - len(matches)),
            "feature_rows": int(len(feature_dataset)),
            "first_completed_match": matches["date"].min().date().isoformat(),
            "last_completed_match": matches["date"].max().date().isoformat(),
        },
        "splits": {
            "method": "chronological",
            "validation_start": config.validation_start,
            "test_start": config.test_start,
            "rows": {
                "train": int(len(splits["train"])),
                "validation": int(len(splits["validation"])),
                "test": int(len(splits["test"])),
            },
        },
        "tournament_holdouts": {
            name: {
                "method": "strict pre-tournament train, finals-only test, no test-result Elo updates",
                "train_rows": int(holdout_splits["train"].shape[0]),
                "test_rows": int(holdout_splits["test"].shape[0]),
                "train_last_date": holdout_splits["train"]["date"].max().date().isoformat(),
                "test_first_date": holdout_splits["test"]["date"].min().date().isoformat(),
                "test_last_date": holdout_splits["test"]["date"].max().date().isoformat(),
            }
            for name, holdout_splits in holdouts.items()
        },
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _world_cup_stage(match_number: int, match_count: int) -> str:
    if match_count >= 104:
        if match_number <= 72:
            return "group"
        if match_number <= 88:
            return "round_of_32"
        if match_number <= 96:
            return "round_of_16"
        if match_number <= 100:
            return "quarter_final"
        if match_number <= 102:
            return "semi_final"
        if match_number == 103:
            return "third_place"
        return "final"

    if match_count >= 64:
        if match_number <= 48:
            return "group"
        if match_number <= 56:
            return "round_of_16"
        if match_number <= 60:
            return "quarter_final"
        if match_number <= 62:
            return "semi_final"
        if match_number == 63:
            return "third_place"
        return "final"

    if match_count == 52:
        if match_number <= 36:
            return "group"
        if match_number <= 44:
            return "round_of_16"
        if match_number <= 48:
            return "quarter_final"
        if match_number <= 50:
            return "semi_final"
        if match_number == 51:
            return "third_place"
        return "final"

    if match_count <= 32:
        return "group" if match_number <= max(match_count - 4, 0) else "knockout"

    return "group"


def _expected_world_cup_match_count(year: int, completed_match_count: int) -> int:
    if year >= 2026 and completed_match_count > 64:
        return 104
    return completed_match_count


def _world_cup_group_matchday(match_number: int, match_count: int, stage: str) -> int:
    if stage != "group":
        return 0
    if match_count >= 104:
        return min(((match_number - 1) // 24) + 1, 3)
    if match_count >= 64:
        return min(((match_number - 1) // 16) + 1, 3)
    if match_count == 52:
        return min(((match_number - 1) // 12) + 1, 3)
    return 0


def _shootout_winner_lookup(shootouts: pd.DataFrame) -> dict[tuple[pd.Timestamp, str, str], str]:
    lookup = {}
    if shootouts.empty:
        return lookup

    required_columns = {"date", "home_team", "away_team", "winner"}
    missing = required_columns.difference(shootouts.columns)
    if missing:
        raise ValueError(f"Missing required shootout columns: {sorted(missing)}")

    cleaned = shootouts.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="raise")
    cleaned["home_team"] = cleaned["home_team"].map(_clean_team_name)
    cleaned["away_team"] = cleaned["away_team"].map(_clean_team_name)
    cleaned["winner"] = cleaned["winner"].map(_clean_team_name)

    for row in cleaned.itertuples(index=False):
        lookup[(row.date, row.home_team, row.away_team)] = row.winner
    return lookup
