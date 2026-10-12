"""Leak-resistant chronological Elo walk-forward baseline.

Important: this is an Elo/rank-gap baseline, not the complete Tier-Mismatch V2.
Historical market lines are used only where both sides' moneylines exist.
"""
from __future__ import annotations

import math
import pandas as pd

RAW_GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

# Normalize common historical NFL franchise abbreviations to one franchise ID.
# These are identity mappings, not predictions or data enrichment.
TEAM_ALIASES = {
    "OAK": "LV",   # Oakland Raiders -> Las Vegas Raiders
    "SD": "LAC",   # San Diego Chargers -> Los Angeles Chargers
    "STL": "LA",   # St. Louis Rams -> Los Angeles Rams
    "JAC": "JAX",  # Jacksonville abbreviation variants
    "WSH": "WAS",  # Washington abbreviation variants
    "WFT": "WAS",  # Washington Football Team
}

def _team_id(value):
    if pd.isna(value):
        return value
    team = str(value).strip().upper()
    return TEAM_ALIASES.get(team, team)

def load_nflverse_games():
    return pd.read_csv(RAW_GAMES_URL, low_memory=False)

def american_to_implied(odds):
    try:
        x = float(odds)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or x == 0:
        return None
    return 100.0 / (x + 100.0) if x > 0 else abs(x) / (abs(x) + 100.0)

def validate_games_frame(frame):
    if frame is None or frame.empty:
        return False, ["CSV contains no rows."], pd.DataFrame()
    df = frame.copy()
    low = {str(c).lower(): c for c in df.columns}
    aliases = {
        "season": ["season", "year"],
        "gameday": ["gameday", "game_date", "date", "kickoff_date"],
        "home_team": ["home_team", "home", "home_abbr"],
        "away_team": ["away_team", "away", "away_abbr"],
        "home_score": ["home_score", "home_points", "score_home"],
        "away_score": ["away_score", "away_points", "score_away"],
        "home_moneyline": ["home_moneyline", "home_ml", "home_money_line"],
        "away_moneyline": ["away_moneyline", "away_ml", "away_money_line"],
        "game_type": ["game_type", "season_type", "type"],
    }
    rename = {}
    for target, names in aliases.items():
        for name in names:
            if name in low:
                if low[name] != target:
                    rename[low[name]] = target
                break
    df = df.rename(columns=rename)
    required = ["season", "gameday", "home_team", "away_team", "home_score", "away_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, ["Missing required column(s): " + ", ".join(missing)], pd.DataFrame()

    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce", utc=True).dt.date
    for col in ["home_score", "away_score", "home_moneyline", "away_moneyline"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["home_team", "away_team"]:
        df[col] = df[col].astype("string").str.upper().str.strip()
        df[col] = df[col].map(_team_id).astype("string")
    df = df.dropna(subset=required).copy()
    if df.empty:
        return False, ["No valid game rows remain."], df

    df["season"] = df["season"].astype(int)
    if "game_type" in df:
        df["game_type"] = df["game_type"].fillna("REG").astype(str).str.upper()
    else:
        df["game_type"] = "REG"

    df = df.drop_duplicates(
        ["season", "gameday", "home_team", "away_team"], keep="last"
    ).sort_values(["season", "gameday", "home_team", "away_team"]).reset_index(drop=True)

    if len(df) < 2:
        return False, ["At least two games are required."], df
    return True, [], df

def _p_home(home_elo, away_elo):
    return 1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + 55.0)) / 400.0))

def _profit(american_odds):
    return american_odds / 100.0 if american_odds > 0 else 100.0 / abs(american_odds)


def _dense_ranks(ratings, tolerance=1e-6):
    """Return dense ranks: teams with effectively equal Elo share a rank.

    Dense ranks prevent arbitrary alphabetical tie-breaks from manufacturing
    large rank gaps when many teams have identical preseason ratings. Distinct
    Elo levels receive consecutive ranks (1, 2, 3...) rather than positional
    ranks that skip numbers after ties.
    """
    ordered = sorted(ratings.items(), key=lambda item: (-float(item[1]), str(item[0])))
    ranks = {}
    current_rank = 0
    previous_rating = None
    for team, rating in ordered:
        rating = float(rating)
        if previous_rating is None or abs(previous_rating - rating) > tolerance:
            current_rank += 1
            previous_rating = rating
        ranks[team] = current_rank
    return ranks

def run_walk_forward_backtest(games, first_test_season, minimum_rank_gap=9, minimum_edge_pp=4.0):
    ok, issues, df = validate_games_frame(games)
    if not ok:
        raise ValueError("; ".join(issues))

    allowed_types = ["REG", "POST", "WC", "DIV", "CON", "SB"]
    df = df[df.game_type.isin(allowed_types)].copy()
    if df.empty:
        raise ValueError("No rows remain after filtering game types.")

    # A franchise is ranked only if it appears in the schedule for that season.
    # Team participation is schedule metadata; game outcomes remain strictly
    # chronological. This avoids ranking retired historical abbreviations as
    # extra teams and avoids arbitrary rank gaps from inactive franchises.
    teams_by_season = {}
    for season, season_df in df.groupby("season", sort=True):
        active = set(season_df["home_team"].astype(str)) | set(season_df["away_team"].astype(str))
        teams_by_season[int(season)] = active

    ratings = {}
    last_season = {}
    rows = []
    batches = 0
    seasons = sorted(teams_by_season)

    for season in seasons:
        active_teams = teams_by_season[season]

        # Initialize/regress every team in this season before the season's first
        # game. This makes preseason ranks internally consistent and applies
        # regression exactly once per team-season.
        for team in active_teams:
            if team not in ratings:
                ratings[team] = 1500.0
            elif team in last_season and season > last_season[team]:
                gap = min(season - last_season[team], 5)
                for _ in range(gap):
                    ratings[team] = 1500.0 + (ratings[team] - 1500.0) * 0.75

        season_df = df[df.season == season]
        for day, group in season_df.groupby("gameday", sort=True):
            batches += 1
            # Freeze all active teams' ratings for this date before any result
            # from this date is used.
            snapshot = {team: ratings.get(team, 1500.0) for team in active_teams}
            ranks = _dense_ranks(snapshot)
            updates = []

            for _, game in group.iterrows():
                home, away = str(game.home_team), str(game.away_team)
                home_elo = snapshot[home]
                away_elo = snapshot[away]
                home_prob = _p_home(home_elo, away_elo)
                home_rank, away_rank = ranks[home], ranks[away]

                favorite = "HOME" if home_prob >= 0.5 else "AWAY"
                favorite_prob = home_prob if favorite == "HOME" else 1.0 - home_prob
                rank_gap = (away_rank - home_rank) if favorite == "HOME" else (home_rank - away_rank)

                def num(value):
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return float("nan")

                home_ml = num(game.get("home_moneyline", float("nan")))
                away_ml = num(game.get("away_moneyline", float("nan")))
                market_available = (
                    math.isfinite(home_ml) and math.isfinite(away_ml)
                    and home_ml != 0 and away_ml != 0
                )
                favorite_ml = home_ml if favorite == "HOME" else away_ml
                underdog_ml = away_ml if favorite == "HOME" else home_ml
                p = american_to_implied(favorite_ml) if market_available else None
                q = american_to_implied(underdog_ml) if market_available else None
                no_vig = p / (p + q) if p is not None and q is not None else float("nan")
                edge = (favorite_prob - no_vig) * 100.0 if market_available else float("nan")
                eligible = rank_gap >= int(minimum_rank_gap)
                bet = bool(eligible and market_available and edge >= float(minimum_edge_pp))

                home_score, away_score = float(game.home_score), float(game.away_score)
                tie = home_score == away_score
                favorite_won = (home_score > away_score) if favorite == "HOME" else (away_score > home_score)
                if not bet:
                    net = 0.0
                elif tie or not favorite_won:
                    net = -1.0
                else:
                    net = _profit(favorite_ml)

                rows.append({
                    "season": int(season),
                    "gameday": str(day),
                    "game_type": str(game.game_type),
                    "home_team": home,
                    "away_team": away,
                    "home_score": home_score,
                    "away_score": away_score,
                    "home_elo_pregame": round(home_elo, 2),
                    "away_elo_pregame": round(away_elo, 2),
                    "home_rank_pregame": int(home_rank),
                    "away_rank_pregame": int(away_rank),
                    "model_favorite": favorite,
                    "favorite_rank_gap": int(rank_gap),
                    "model_favorite_win_probability": round(float(favorite_prob), 6),
                    "home_moneyline": home_ml if math.isfinite(home_ml) else None,
                    "away_moneyline": away_ml if math.isfinite(away_ml) else None,
                    "market_available": bool(market_available),
                    "favorite_no_vig_market_probability": round(float(no_vig), 6) if market_available else None,
                    "edge_percentage_points": round(float(edge), 4) if market_available else None,
                    "rank_gap_eligible": bool(eligible),
                    "bet_taken": bool(bet),
                    "bet_result": (int(favorite_won) if not tie else 0) if bet else None,
                    "risked_units": 1.0 if bet else 0.0,
                    "net_units": round(float(net), 4),
                    "actual_favorite_won": bool(favorite_won),
                    "season_team_count": len(active_teams),
                })

                outcome = 1.0 if home_score > away_score else (0.5 if home_score == away_score else 0.0)
                updates.append((home, away, home_elo, away_elo, outcome))

            # Update only after every game on this date has been scored.
            for home, away, home_elo, away_elo, outcome in updates:
                expected = _p_home(home_elo, away_elo)
                delta = 20.0 * (outcome - expected)
                ratings[home] = home_elo + delta
                ratings[away] = away_elo - delta
                last_season[home] = season
                last_season[away] = season

    out = pd.DataFrame(rows)
    out = out[out.season >= int(first_test_season)].copy()
    diagnostics = {
        "input_games": int(len(df)),
        "test_games_before_rank_filter": int(len(out)),
        "date_batches_processed": int(batches),
        "market_rows_before_rank_filter": int(out.market_available.sum()) if len(out) else 0,
        "rank_gap_eligible_games": int(out.rank_gap_eligible.sum()) if len(out) else 0,
        "rank_gap_filter_pass_rate_pct": round(float(out.rank_gap_eligible.mean() * 100.0), 2) if len(out) else 0.0,
        "test_games_after_rank_filter": int(out.rank_gap_eligible.sum()) if len(out) else 0,
        "max_rank": int(max(out.home_rank_pregame.max(), out.away_rank_pregame.max())) if len(out) else 0,
        "note": "Pregame ratings use only prior-date results; ranks include only franchises scheduled in that season. Equal Elo ratings share a dense rank, preventing alphabetical tie-breaks from creating artificial rank gaps."
    }
    out = out[out.rank_gap_eligible].reset_index(drop=True)
    return out, diagnostics

def summarize_backtest(results):
    if results is None or results.empty:
        return {
            "eligible_games": 0, "games_with_odds": 0, "bets": 0,
            "win_rate": "n/a", "net_units": "0.00", "roi": "n/a",
            "brier": "n/a", "odds_coverage": "n/a"
        }
    market = results[results.market_available]
    bets = results[results.bet_taken]
    wins = int((bets.bet_result == 1).sum()) if len(bets) else 0
    risk = float(bets.risked_units.sum())
    net = float(bets.net_units.sum())
    valid = results[results.home_score != results.away_score]
    brier = (
        ((valid.model_favorite_win_probability - valid.actual_favorite_won.astype(float)) ** 2).mean()
        if len(valid) else float("nan")
    )
    return {
        "eligible_games": len(results),
        "games_with_odds": len(market),
        "bets": len(bets),
        "win_rate": f"{wins / len(bets) * 100:.1f}%" if len(bets) else "n/a",
        "net_units": f"{net:+.2f}",
        "roi": f"{net / risk * 100:+.1f}%" if risk else "n/a",
        "brier": f"{brier:.4f}" if math.isfinite(brier) else "n/a",
        "odds_coverage": f"{len(market) / len(results) * 100:.1f}%" if len(results) else "n/a",
    }
