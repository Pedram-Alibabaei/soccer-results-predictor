"""Confirmed-lineup rotation and dated player-strength features."""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


POSITION_GROUPS = ("goalkeeper", "defense", "midfield", "attack")

# These features are deliberately separate from the baseline. They are only
# valid once both starting XIs have been officially announced.
CONFIRMED_LINEUP_FEATURE_COLUMNS = (
    "has_confirmed_lineups",
    "home_previous_lineup_available_pre",
    "away_previous_lineup_available_pre",
    "home_lineup_changes_pre",
    "away_lineup_changes_pre",
    "lineup_changes_diff_pre",
    "home_goalkeeper_changed_pre",
    "away_goalkeeper_changed_pre",
    "home_defensive_continuity_pre",
    "away_defensive_continuity_pre",
    "defensive_continuity_diff_pre",
    "home_attacking_continuity_pre",
    "away_attacking_continuity_pre",
    "attacking_continuity_diff_pre",
    "home_goalkeeper_strength_change_pre",
    "away_goalkeeper_strength_change_pre",
    "home_defense_strength_change_pre",
    "away_defense_strength_change_pre",
    "home_midfield_strength_change_pre",
    "away_midfield_strength_change_pre",
    "home_attack_strength_change_pre",
    "away_attack_strength_change_pre",
    "home_lineup_strength_change_pre",
    "away_lineup_strength_change_pre",
    "min_player_strength_coverage_pre",
)

LINEUP_FEATURE_DEFAULTS = {
    "has_confirmed_lineups": 0.0,
    "home_previous_lineup_available_pre": 0.0,
    "away_previous_lineup_available_pre": 0.0,
    "home_lineup_changes_pre": 0.0,
    "away_lineup_changes_pre": 0.0,
    "lineup_changes_diff_pre": 0.0,
    "home_goalkeeper_changed_pre": 0.0,
    "away_goalkeeper_changed_pre": 0.0,
    "home_defensive_continuity_pre": 1.0,
    "away_defensive_continuity_pre": 1.0,
    "defensive_continuity_diff_pre": 0.0,
    "home_attacking_continuity_pre": 1.0,
    "away_attacking_continuity_pre": 1.0,
    "attacking_continuity_diff_pre": 0.0,
    "home_goalkeeper_strength_change_pre": 0.0,
    "away_goalkeeper_strength_change_pre": 0.0,
    "home_defense_strength_change_pre": 0.0,
    "away_defense_strength_change_pre": 0.0,
    "home_midfield_strength_change_pre": 0.0,
    "away_midfield_strength_change_pre": 0.0,
    "home_attack_strength_change_pre": 0.0,
    "away_attack_strength_change_pre": 0.0,
    "home_lineup_strength_change_pre": 0.0,
    "away_lineup_strength_change_pre": 0.0,
    "min_player_strength_coverage_pre": 0.0,
}

_TRANSLITERATION = str.maketrans(
    {
        "ł": "l",
        "Ł": "L",
        "ø": "o",
        "Ø": "O",
        "đ": "d",
        "Đ": "D",
        "ð": "d",
        "Ð": "D",
        "þ": "th",
        "Þ": "Th",
        "æ": "ae",
        "Æ": "Ae",
        "œ": "oe",
        "Œ": "Oe",
        "ı": "i",
        "ļ": "l",
        "Ļ": "L",
    }
)

_TEAM_NAME_ALIASES = {
    "iran islamic republic of": "iran",
    "korea republic": "south korea",
    "republic of korea": "south korea",
    "united states of america": "united states",
    "usa": "united states",
}

_CITIZENSHIP_ALIASES = {
    "south korea": {"south korea", "korea south"},
    "united states": {"united states", "united states of america"},
    "iran": {"iran"},
}


def normalize_name(value: object) -> str:
    """Normalize a person or team name for deterministic cross-source matching."""

    text = str(value or "").translate(_TRANSLITERATION)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def normalize_team_name(value: object) -> str:
    normalized = normalize_name(value)
    return _TEAM_NAME_ALIASES.get(normalized, normalized)


def position_group(position: object) -> str:
    """Map a StatsBomb position name to a compact tactical unit."""

    normalized = normalize_name(position)
    if "goalkeeper" in normalized:
        return "goalkeeper"
    if "back" in normalized:
        return "defense"
    if "midfield" in normalized:
        return "midfield"
    if "wing" in normalized or "forward" in normalized or "striker" in normalized:
        return "attack"
    return "unknown"


def parse_statsbomb_starters(
    match: dict[str, object],
    lineup_payload: list[dict[str, object]],
    *,
    competition_name: str,
    season_name: str,
) -> pd.DataFrame:
    """Return one normalized row per starting player from StatsBomb JSON."""

    match_id = int(match["match_id"])
    match_date = pd.Timestamp(match["match_date"])
    home_team = str(match["home_team"]["home_team_name"])
    away_team = str(match["away_team"]["away_team_name"])
    rows: list[dict[str, object]] = []

    for team_payload in lineup_payload:
        team = str(team_payload["team_name"])
        side = "home" if normalize_team_name(team) == normalize_team_name(home_team) else "away"
        for player in team_payload.get("lineup", []):
            starting_positions = [
                item
                for item in player.get("positions", [])
                if item.get("from") == "00:00" or item.get("start_reason") == "Starting XI"
            ]
            if not starting_positions:
                continue
            starting_position = starting_positions[0]
            rows.append(
                {
                    "statsbomb_match_id": match_id,
                    "date": match_date,
                    "competition_name": competition_name,
                    "season_name": season_name,
                    "competition_stage": match.get("competition_stage", {}).get("name", ""),
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": int(match["home_score"]),
                    "away_score": int(match["away_score"]),
                    "team": team,
                    "side": side,
                    "statsbomb_player_id": int(player["player_id"]),
                    "player_name": str(player["player_name"]),
                    "player_nickname": player.get("player_nickname"),
                    "position": str(starting_position["position"]),
                    "position_group": position_group(starting_position["position"]),
                }
            )

    return pd.DataFrame(rows)


def add_transfermarkt_player_strength(
    starters: pd.DataFrame,
    database_path: Path,
) -> pd.DataFrame:
    """Join the latest market value dated on or before each match."""

    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError("Install the project dependency 'duckdb' to add player values.") from exc

    data = starters.copy().reset_index(drop=True)
    data["_row_id"] = np.arange(len(data), dtype=int)
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        players = connection.sql(
            """
            SELECT player_id, name, first_name, last_name, position,
                   sub_position, country_of_citizenship
            FROM players
            """
        ).df()
        mapping = match_statsbomb_players(data, players)
        data = data.merge(
            mapping,
            on=["statsbomb_player_id", "player_name", "team"],
            how="left",
            validate="many_to_one",
        )
        lookup = data.loc[:, ["_row_id", "date", "transfermarkt_player_id"]].copy()
        lookup["transfermarkt_player_id"] = pd.to_numeric(
            lookup["transfermarkt_player_id"], errors="coerce"
        ).astype("Int64")
        connection.register("starter_lookup", lookup)
        values = connection.sql(
            """
            SELECT s._row_id,
                   v.date AS valuation_date,
                   v.market_value_in_eur
            FROM starter_lookup s
            LEFT JOIN LATERAL (
                SELECT date, market_value_in_eur
                FROM player_valuations pv
                WHERE pv.player_id = s.transfermarkt_player_id
                  AND pv.date <= CAST(s.date AS DATE)
                ORDER BY pv.date DESC
                LIMIT 1
            ) v ON TRUE
            """
        ).df()
    finally:
        connection.close()

    data = data.merge(values, on="_row_id", how="left", validate="one_to_one")
    data["valuation_date"] = pd.to_datetime(data["valuation_date"], errors="coerce")
    data["valuation_age_days"] = (data["date"] - data["valuation_date"]).dt.days
    return data.drop(columns="_row_id")


def match_statsbomb_players(starters: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Match unique StatsBomb players to Transfermarkt with an auditable score."""

    candidates = players.copy()
    candidates["_name"] = candidates["name"].map(normalize_name)
    candidates["_full_name"] = (
        candidates["first_name"].fillna("").astype(str)
        + " "
        + candidates["last_name"].fillna("").astype(str)
    ).map(normalize_name)
    candidates["_country"] = candidates["country_of_citizenship"].map(normalize_team_name)
    candidates["_position_group"] = candidates["position"].map(_transfermarkt_position_group)
    exact_index: dict[str, list[int]] = {}
    for index, candidate in candidates.iterrows():
        for candidate_name in {candidate["_name"], candidate["_full_name"]}:
            if candidate_name:
                exact_index.setdefault(candidate_name, []).append(index)
    country_groups = {
        country: group for country, group in candidates.groupby("_country", dropna=False)
    }

    nickname_column = (
        starters["player_nickname"]
        if "player_nickname" in starters.columns
        else pd.Series(None, index=starters.index)
    )
    position_column = (
        starters["position_group"]
        if "position_group" in starters.columns
        else pd.Series("unknown", index=starters.index)
    )
    unique_players = starters.assign(
        player_nickname=nickname_column,
        position_group=position_column,
    ).loc[
        :, ["statsbomb_player_id", "player_name", "player_nickname", "team", "position_group"]
    ]
    unique_players["_has_nickname"] = unique_players["player_nickname"].notna()
    unique_players = (
        unique_players.sort_values("_has_nickname", ascending=False)
        .drop_duplicates(["statsbomb_player_id", "player_name", "team"])
        .drop(columns="_has_nickname")
    )
    rows = []
    for row in unique_players.itertuples(index=False):
        targets = [normalize_name(row.player_name)]
        nickname = normalize_name(row.player_nickname)
        if nickname and nickname not in targets and nickname != "nan":
            targets.insert(0, nickname)
        team = normalize_team_name(row.team)
        target_position_group = str(row.position_group)
        country_names = _CITIZENSHIP_ALIASES.get(team, {team})
        match = None
        score = 0.0
        method = "unmatched"
        for target in targets:
            exact_candidates = candidates.loc[exact_index.get(target, [])]
            if len(exact_candidates) > 1:
                country_exact = exact_candidates[exact_candidates["_country"].isin(country_names)]
                exact_candidates = country_exact if not country_exact.empty else exact_candidates
            if not exact_candidates.empty:
                # Duplicate Transfermarkt rows occasionally represent the same
                # player. Older, lower IDs are the stable historical record.
                match, score, method = exact_candidates.sort_values("player_id").iloc[0], 1.0, (
                    "nickname_exact" if target == nickname else "exact"
                )
                break

        if match is None:
            country_candidates = (
                pd.concat(
                    [country_groups[name] for name in country_names if name in country_groups],
                    ignore_index=False,
                )
                if any(name in country_groups for name in country_names)
                else candidates.iloc[0:0]
            )
            position_candidates = country_candidates[
                country_candidates["_position_group"] == target_position_group
            ]
            if not position_candidates.empty:
                country_candidates = position_candidates
            possible_matches = [_best_player_match(target, country_candidates) for target in targets]
            match, score, method = max(possible_matches, key=lambda item: item[1])
            if match is not None and targets[0] == nickname:
                method = f"nickname_{method}"

        rows.append(
            {
                "statsbomb_player_id": row.statsbomb_player_id,
                "player_name": row.player_name,
                "team": row.team,
                "transfermarkt_player_id": None if match is None else int(match["player_id"]),
                "transfermarkt_player_name": None if match is None else str(match["name"]),
                "transfermarkt_position": None if match is None else str(match["position"]),
                "player_match_method": method,
                "player_match_score": score,
            }
        )
    return pd.DataFrame(rows)


def build_match_lineup_features(starters: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player rows into match features without using match outcomes."""

    required = {
        "statsbomb_match_id",
        "date",
        "competition_name",
        "season_name",
        "home_team",
        "away_team",
        "team",
        "side",
        "statsbomb_player_id",
        "position_group",
        "market_value_in_eur",
    }
    missing = required.difference(starters.columns)
    if missing:
        raise ValueError(f"Missing lineup columns: {sorted(missing)}")

    data = starters.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data.sort_values(["date", "statsbomb_match_id", "side", "statsbomb_player_id"])
    raw_values = pd.to_numeric(data["market_value_in_eur"], errors="coerce")
    data["_player_strength"] = np.log1p(raw_values / 1_000_000.0)
    position_medians = data.groupby(
        ["competition_name", "season_name", "position_group"], dropna=False
    )["_player_strength"].transform("median")
    tournament_medians = data.groupby(
        ["competition_name", "season_name"], dropna=False
    )["_player_strength"].transform("median")
    data["_player_strength_imputed"] = (
        data["_player_strength"].fillna(position_medians).fillna(tournament_medians).fillna(0.0)
    )
    previous_lineups: dict[tuple[str, str, str], pd.DataFrame] = {}
    rows: list[dict[str, object]] = []

    for _, match_rows in data.groupby("statsbomb_match_id", sort=False):
        first = match_rows.iloc[0]
        output = {
            column: first[column]
            for column in [
                "statsbomb_match_id",
                "date",
                "competition_name",
                "season_name",
                "home_team",
                "away_team",
            ]
        }
        for optional_column in ["competition_stage", "home_score", "away_score"]:
            if optional_column in match_rows.columns:
                output[optional_column] = first[optional_column]

        side_features = {}
        for side in ("home", "away"):
            current = match_rows[match_rows["side"] == side].copy()
            if current.empty:
                side_features[side] = _empty_team_features()
                continue
            team = str(current.iloc[0]["team"])
            history_key = (
                str(first["competition_name"]),
                str(first["season_name"]),
                normalize_team_name(team),
            )
            previous = previous_lineups.get(history_key)
            side_features[side] = _team_lineup_features(current, previous)
            previous_lineups[history_key] = current

        for side, values in side_features.items():
            output.update({f"{side}_{key}": value for key, value in values.items()})
        output["has_confirmed_lineups"] = float(
            side_features["home"]["starter_count"] == 11
            and side_features["away"]["starter_count"] == 11
        )
        _add_match_differences(output)
        rows.append(output)

    return pd.DataFrame(rows).sort_values(["date", "statsbomb_match_id"]).reset_index(drop=True)


def attach_lineup_features(matches: pd.DataFrame, lineup_features: pd.DataFrame) -> pd.DataFrame:
    """Attach confirmed-lineup features to an existing match frame."""

    data = matches.copy()
    features = lineup_features.copy()
    data["_lineup_date"] = pd.to_datetime(data["date"]).dt.normalize()
    features["_lineup_date"] = pd.to_datetime(features["date"]).dt.normalize()
    data["_lineup_home"] = data["home_team"].map(normalize_team_name)
    data["_lineup_away"] = data["away_team"].map(normalize_team_name)
    features["_lineup_home"] = features["home_team"].map(normalize_team_name)
    features["_lineup_away"] = features["away_team"].map(normalize_team_name)

    selected = features.loc[
        :, ["_lineup_date", "_lineup_home", "_lineup_away", *CONFIRMED_LINEUP_FEATURE_COLUMNS]
    ].drop_duplicates(["_lineup_date", "_lineup_home", "_lineup_away"])
    swapped = selected.copy()
    swapped["_lineup_home"] = selected["_lineup_away"]
    swapped["_lineup_away"] = selected["_lineup_home"]
    for column in CONFIRMED_LINEUP_FEATURE_COLUMNS:
        if column.startswith("home_"):
            counterpart = f"away_{column.removeprefix('home_')}"
            if counterpart in selected.columns:
                swapped[column] = selected[counterpart]
        elif column.startswith("away_"):
            counterpart = f"home_{column.removeprefix('away_')}"
            if counterpart in selected.columns:
                swapped[column] = selected[counterpart]
        elif column.endswith("_diff_pre"):
            swapped[column] = -pd.to_numeric(selected[column], errors="coerce")
    selected = pd.concat([selected, swapped], ignore_index=True).drop_duplicates(
        ["_lineup_date", "_lineup_home", "_lineup_away"]
    )
    data = data.merge(
        selected,
        on=["_lineup_date", "_lineup_home", "_lineup_away"],
        how="left",
        validate="many_to_one",
    )
    for column, default in LINEUP_FEATURE_DEFAULTS.items():
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(default)
    return data.drop(columns=["_lineup_date", "_lineup_home", "_lineup_away"])


def _best_player_match(
    target: str,
    candidates: pd.DataFrame,
    *,
    minimum_score: float = 0.92,
) -> tuple[object | None, float, str]:
    exact = candidates[(candidates["_name"] == target) | (candidates["_full_name"] == target)]
    if len(exact) == 1:
        return exact.iloc[0], 1.0, "exact"

    target_tokens = target.split()
    scored: list[tuple[float, object, str]] = []
    for _, candidate in candidates.iterrows():
        names = {candidate["_name"], candidate["_full_name"]}
        best_score = 0.0
        best_method = "fuzzy"
        for candidate_name in names:
            if not candidate_name:
                continue
            candidate_tokens = candidate_name.split()
            token_subset = set(candidate_tokens).issubset(target_tokens) or set(target_tokens).issubset(
                candidate_tokens
            )
            matching_boundaries = (
                candidate_tokens
                and target_tokens
                and candidate_tokens[0] == target_tokens[0]
                and candidate_tokens[-1] == target_tokens[-1]
            )
            shorter_tokens, longer_tokens = sorted(
                (candidate_tokens, target_tokens), key=len
            )
            contiguous_tokens = " ".join(shorter_tokens) in " ".join(longer_tokens)
            single_distinctive_token = (
                min(len(candidate_tokens), len(target_tokens)) == 1
                and any(
                    len(token) >= 5
                    for token in set(candidate_tokens).intersection(target_tokens)
                )
            )
            if token_subset and (
                matching_boundaries or contiguous_tokens or single_distinctive_token
            ):
                score = 0.96 - 0.01 * abs(len(candidate_tokens) - len(target_tokens))
                method = "token_subset"
            else:
                score = SequenceMatcher(None, target, candidate_name).ratio()
                if target_tokens and candidate_tokens and target_tokens[-1] == candidate_tokens[-1]:
                    score += 0.03
                method = "fuzzy"
            if score > best_score:
                best_score = score
                best_method = method
        scored.append((min(best_score, 1.0), candidate, best_method))

    if not scored:
        return None, 0.0, "unmatched"
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_match, method = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    if best_score < minimum_score or (best_score - runner_up < 0.02 and best_score < 0.95):
        return None, float(best_score), "ambiguous" if best_score >= minimum_score else "unmatched"
    return best_match, float(best_score), method


def _transfermarkt_position_group(position: object) -> str:
    normalized = normalize_name(position)
    if "goalkeeper" in normalized:
        return "goalkeeper"
    if "defender" in normalized:
        return "defense"
    if "midfield" in normalized:
        return "midfield"
    if "attack" in normalized:
        return "attack"
    return "unknown"


def _team_lineup_features(current: pd.DataFrame, previous: pd.DataFrame | None) -> dict[str, float]:
    current_ids = set(current["statsbomb_player_id"].astype(int))
    current_defense = _unit_ids(current, {"goalkeeper", "defense"})
    current_attack = _unit_ids(current, {"midfield", "attack"})
    current_goalkeeper = _unit_ids(current, {"goalkeeper"})

    if previous is None:
        previous_available = 0.0
        changes = 0.0
        goalkeeper_changed = 0.0
        defensive_continuity = 1.0
        attacking_continuity = 1.0
        overall_continuity = 1.0
    else:
        previous_available = 1.0
        previous_ids = set(previous["statsbomb_player_id"].astype(int))
        previous_defense = _unit_ids(previous, {"goalkeeper", "defense"})
        previous_attack = _unit_ids(previous, {"midfield", "attack"})
        previous_goalkeeper = _unit_ids(previous, {"goalkeeper"})
        changes = float(len(current_ids.difference(previous_ids)))
        goalkeeper_changed = float(bool(current_goalkeeper and previous_goalkeeper) and not bool(
            current_goalkeeper.intersection(previous_goalkeeper)
        ))
        defensive_continuity = _overlap_fraction(current_defense, previous_defense)
        attacking_continuity = _overlap_fraction(current_attack, previous_attack)
        overall_continuity = _overlap_fraction(current_ids, previous_ids)

    values = pd.to_numeric(current["market_value_in_eur"], errors="coerce")
    current_strengths = _lineup_strengths(current)
    previous_strengths = (
        current_strengths if previous is None else _lineup_strengths(previous)
    )
    strength_changes = {
        f"{name.removesuffix('_pre')}_change_pre": (
            current_strengths[name] - previous_strengths[name]
        )
        for name in [
            "goalkeeper_strength_pre",
            "defense_strength_pre",
            "midfield_strength_pre",
            "attack_strength_pre",
            "lineup_strength_pre",
        ]
    }

    return {
        "starter_count": float(len(current_ids)),
        "previous_lineup_available_pre": previous_available,
        "lineup_changes_pre": changes,
        "goalkeeper_changed_pre": goalkeeper_changed,
        "defensive_continuity_pre": defensive_continuity,
        "attacking_continuity_pre": attacking_continuity,
        "overall_continuity_pre": overall_continuity,
        "player_strength_coverage_pre": float(values.notna().mean()),
        **current_strengths,
        **strength_changes,
    }


def _empty_team_features() -> dict[str, float]:
    return {
        "starter_count": 0.0,
        "previous_lineup_available_pre": 0.0,
        "lineup_changes_pre": 0.0,
        "goalkeeper_changed_pre": 0.0,
        "defensive_continuity_pre": 1.0,
        "attacking_continuity_pre": 1.0,
        "overall_continuity_pre": 1.0,
        "lineup_strength_pre": 0.0,
        "lineup_strength_change_pre": 0.0,
        "player_strength_coverage_pre": 0.0,
        **{f"{group}_strength_pre": 0.0 for group in POSITION_GROUPS},
        **{f"{group}_strength_change_pre": 0.0 for group in POSITION_GROUPS},
    }


def _unit_ids(frame: pd.DataFrame, groups: set[str]) -> set[int]:
    return set(frame.loc[frame["position_group"].isin(groups), "statsbomb_player_id"].astype(int))


def _lineup_strengths(frame: pd.DataFrame) -> dict[str, float]:
    strengths = pd.to_numeric(frame["_player_strength_imputed"], errors="coerce")
    lineup_fallback = float(strengths.median()) if not strengths.empty else 0.0
    output = {
        "lineup_strength_pre": float(strengths.mean()) if not strengths.empty else 0.0
    }
    for group in POSITION_GROUPS:
        group_values = strengths[frame["position_group"] == group].dropna()
        output[f"{group}_strength_pre"] = (
            float(group_values.mean()) if not group_values.empty else lineup_fallback
        )
    return output


def _overlap_fraction(current: set[int], previous: set[int]) -> float:
    if not current:
        return 1.0
    return float(len(current.intersection(previous)) / len(current))


def _add_match_differences(output: dict[str, object]) -> None:
    for name in [
        "lineup_changes",
        "goalkeeper_changed",
        "defensive_continuity",
        "attacking_continuity",
        "overall_continuity",
        "goalkeeper_strength",
        "defense_strength",
        "midfield_strength",
        "attack_strength",
        "lineup_strength",
    ]:
        output[f"{name}_diff_pre"] = float(output[f"home_{name}_pre"]) - float(
            output[f"away_{name}_pre"]
        )
    output["min_player_strength_coverage_pre"] = min(
        float(output["home_player_strength_coverage_pre"]),
        float(output["away_player_strength_coverage_pre"]),
    )
