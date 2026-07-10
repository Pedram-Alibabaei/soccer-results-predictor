"""Pre-match rolling feature engineering for club soccer data."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import pandas as pd


ROLLING_WINDOW = 5
DEFAULT_REST_DAYS = 7.0
MAX_REST_DAYS = 30.0
DEFAULT_POINTS_PER_MATCH = 1.0
DEFAULT_GOALS = 1.25
DEFAULT_SHOTS = 10.0
DEFAULT_SHOTS_ON_TARGET = 4.0
DEFAULT_XG = 0.0


@dataclass
class TeamHistory:
    overall: Deque[dict[str, float]] = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    home: Deque[dict[str, float]] = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    away: Deque[dict[str, float]] = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    last_match_date: pd.Timestamp | None = None


def add_club_pre_match_features(
    matches: pd.DataFrame,
    *,
    window: int = ROLLING_WINDOW,
    default_rest_days: float = DEFAULT_REST_DAYS,
    max_rest_days: float = MAX_REST_DAYS,
) -> pd.DataFrame:
    """Attach rolling form, shot, and rest features available before kickoff."""

    data = matches.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)
    histories: dict[str, TeamHistory] = {}
    rows = []

    for row in data.itertuples(index=False):
        home_team = row.home_team
        away_team = row.away_team
        home_history = histories.setdefault(home_team, TeamHistory())
        away_history = histories.setdefault(away_team, TeamHistory())

        enriched = row._asdict()
        home_stats = summarize_history(home_history.overall)
        away_stats = summarize_history(away_history.overall)
        home_home_stats = summarize_history(home_history.home)
        away_away_stats = summarize_history(away_history.away)
        home_rest = rest_days(home_history.last_match_date, row.date, default_rest_days, max_rest_days)
        away_rest = rest_days(away_history.last_match_date, row.date, default_rest_days, max_rest_days)

        enriched.update(
            {
                "home_rest_days_pre": home_rest,
                "away_rest_days_pre": away_rest,
                "rest_days_diff_pre": home_rest - away_rest,
                "home_points_per_match_last5_pre": home_stats["points"],
                "away_points_per_match_last5_pre": away_stats["points"],
                "form_diff_pre": home_stats["points"] - away_stats["points"],
                "home_home_points_per_match_last5_pre": home_home_stats["points"],
                "away_away_points_per_match_last5_pre": away_away_stats["points"],
                "venue_form_diff_pre": home_home_stats["points"] - away_away_stats["points"],
                "home_goals_for_last5_pre": home_stats["goals_for"],
                "away_goals_for_last5_pre": away_stats["goals_for"],
                "home_goals_against_last5_pre": home_stats["goals_against"],
                "away_goals_against_last5_pre": away_stats["goals_against"],
                "goal_form_diff_pre": home_stats["goals_for"] - away_stats["goals_for"],
                "defensive_form_diff_pre": home_stats["goals_against"] - away_stats["goals_against"],
                "home_shots_for_last5_pre": home_stats["shots_for"],
                "away_shots_for_last5_pre": away_stats["shots_for"],
                "home_shots_against_last5_pre": home_stats["shots_against"],
                "away_shots_against_last5_pre": away_stats["shots_against"],
                "shot_form_diff_pre": home_stats["shots_for"] - away_stats["shots_for"],
                "shot_concession_diff_pre": home_stats["shots_against"] - away_stats["shots_against"],
                "home_sot_for_last5_pre": home_stats["sot_for"],
                "away_sot_for_last5_pre": away_stats["sot_for"],
                "home_sot_against_last5_pre": home_stats["sot_against"],
                "away_sot_against_last5_pre": away_stats["sot_against"],
                "sot_form_diff_pre": home_stats["sot_for"] - away_stats["sot_for"],
                "sot_concession_diff_pre": home_stats["sot_against"] - away_stats["sot_against"],
                "home_xg_for_last5_pre": home_stats["xg_for"],
                "away_xg_for_last5_pre": away_stats["xg_for"],
                "home_xg_against_last5_pre": home_stats["xg_against"],
                "away_xg_against_last5_pre": away_stats["xg_against"],
                "xg_form_diff_pre": home_stats["xg_for"] - away_stats["xg_for"],
                "xg_concession_diff_pre": home_stats["xg_against"] - away_stats["xg_against"],
            }
        )
        rows.append(enriched)

        home_points, away_points = match_points(row.home_goals, row.away_goals)
        home_record = match_record(
            points=home_points,
            goals_for=row.home_goals,
            goals_against=row.away_goals,
            shots_for=getattr(row, "home_shots", DEFAULT_SHOTS),
            shots_against=getattr(row, "away_shots", DEFAULT_SHOTS),
            sot_for=getattr(row, "home_shots_on_target", DEFAULT_SHOTS_ON_TARGET),
            sot_against=getattr(row, "away_shots_on_target", DEFAULT_SHOTS_ON_TARGET),
            xg_for=getattr(row, "home_xg", DEFAULT_XG),
            xg_against=getattr(row, "away_xg", DEFAULT_XG),
        )
        away_record = match_record(
            points=away_points,
            goals_for=row.away_goals,
            goals_against=row.home_goals,
            shots_for=getattr(row, "away_shots", DEFAULT_SHOTS),
            shots_against=getattr(row, "home_shots", DEFAULT_SHOTS),
            sot_for=getattr(row, "away_shots_on_target", DEFAULT_SHOTS_ON_TARGET),
            sot_against=getattr(row, "home_shots_on_target", DEFAULT_SHOTS_ON_TARGET),
            xg_for=getattr(row, "away_xg", DEFAULT_XG),
            xg_against=getattr(row, "home_xg", DEFAULT_XG),
        )
        home_history.overall.append(home_record)
        home_history.home.append(home_record)
        home_history.last_match_date = pd.Timestamp(row.date)
        away_history.overall.append(away_record)
        away_history.away.append(away_record)
        away_history.last_match_date = pd.Timestamp(row.date)

    return pd.DataFrame(rows)


def summarize_history(history: Deque[dict[str, float]]) -> dict[str, float]:
    return {
        "points": average(history, "points", DEFAULT_POINTS_PER_MATCH),
        "goals_for": average(history, "goals_for", DEFAULT_GOALS),
        "goals_against": average(history, "goals_against", DEFAULT_GOALS),
        "shots_for": average(history, "shots_for", DEFAULT_SHOTS),
        "shots_against": average(history, "shots_against", DEFAULT_SHOTS),
        "sot_for": average(history, "sot_for", DEFAULT_SHOTS_ON_TARGET),
        "sot_against": average(history, "sot_against", DEFAULT_SHOTS_ON_TARGET),
        "xg_for": average(history, "xg_for", DEFAULT_XG),
        "xg_against": average(history, "xg_against", DEFAULT_XG),
    }


def average(history: Deque[dict[str, float]], key: str, default: float) -> float:
    values = [record[key] for record in history if pd.notna(record[key])]
    if not values:
        return default
    return float(sum(values) / len(values))


def rest_days(
    last_match_date: pd.Timestamp | None,
    match_date: pd.Timestamp,
    default_rest_days: float,
    max_rest_days: float,
) -> float:
    if last_match_date is None:
        return default_rest_days
    days = (pd.Timestamp(match_date) - last_match_date).days
    return float(min(max(days, 0), max_rest_days))


def match_points(home_goals: int, away_goals: int) -> tuple[float, float]:
    if home_goals > away_goals:
        return 3.0, 0.0
    if away_goals > home_goals:
        return 0.0, 3.0
    return 1.0, 1.0


def match_record(
    *,
    points: float,
    goals_for: float,
    goals_against: float,
    shots_for: float,
    shots_against: float,
    sot_for: float,
    sot_against: float,
    xg_for: float,
    xg_against: float,
) -> dict[str, float]:
    return {
        "points": float(points),
        "goals_for": float(goals_for),
        "goals_against": float(goals_against),
        "shots_for": safe_float(shots_for, DEFAULT_SHOTS),
        "shots_against": safe_float(shots_against, DEFAULT_SHOTS),
        "sot_for": safe_float(sot_for, DEFAULT_SHOTS_ON_TARGET),
        "sot_against": safe_float(sot_against, DEFAULT_SHOTS_ON_TARGET),
        "xg_for": safe_float(xg_for, DEFAULT_XG),
        "xg_against": safe_float(xg_against, DEFAULT_XG),
    }


def safe_float(value: object, default: float) -> float:
    if pd.isna(value):
        return default
    return float(value)
