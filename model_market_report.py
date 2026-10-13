"""Same-game probability comparison for the Elo baseline and no-vig market."""
from __future__ import annotations

import math
import pandas as pd

_EPSILON = 1e-12
_EDGE_BINS = [-math.inf, -5.0, -2.0, 0.0, 2.0, 4.0, 6.0, math.inf]
_EDGE_LABELS = ["< -5 pp", "-5 to -2 pp", "-2 to 0 pp", "0 to 2 pp", "2 to 4 pp", "4 to 6 pp", ">= 6 pp"]


def _probability_metrics(probabilities, outcomes):
    p = pd.to_numeric(probabilities, errors="coerce")
    y = pd.to_numeric(outcomes, errors="coerce")
    valid = p.notna() & y.notna() & p.map(math.isfinite) & y.map(math.isfinite)
    p = p[valid].clip(_EPSILON, 1.0 - _EPSILON)
    y = y[valid]
    if p.empty:
        return {"n": 0, "brier": float("nan"), "log_loss": float("nan"), "mean_probability": float("nan"), "observed_rate": float("nan")}
    if not y.isin([0.0, 1.0]).all():
        raise ValueError("Outcomes must be binary (0 or 1) after tied games are excluded.")
    brier = float(((p - y) ** 2).mean())
    log_loss = float((-(y * p.map(math.log) + (1.0 - y) * (1.0 - p).map(math.log))).mean())
    return {"n": int(len(p)), "brier": brier, "log_loss": log_loss,
            "mean_probability": float(p.mean()), "observed_rate": float(y.mean())}


def _valid_scoring_rows(results):
    required = ["market_available", "home_score", "away_score",
                "model_favorite_win_probability", "favorite_no_vig_market_probability",
                "actual_favorite_won"]
    missing = [column for column in required if column not in results.columns]
    if missing:
        raise ValueError("Backtest results missing required column(s): " + ", ".join(missing))
    rows = results.copy()
    rows = rows[rows["market_available"].fillna(False).astype(bool)]
    rows = rows[rows["home_score"] != rows["away_score"]]
    for column in ["model_favorite_win_probability", "favorite_no_vig_market_probability"]:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows["actual_favorite_won"] = rows["actual_favorite_won"].map(
        lambda value: 1.0 if value is True or value == 1 else (0.0 if value is False or value == 0 else float("nan")))
    rows = rows.dropna(subset=["model_favorite_win_probability", "favorite_no_vig_market_probability", "actual_favorite_won"])
    for column in ["model_favorite_win_probability", "favorite_no_vig_market_probability"]:
        rows = rows[rows[column].map(lambda value: math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0)]
    return rows.copy()


def build_model_market_report(results):
    """Build overall, season, edge-band, and actual-bet comparisons.

    Pass the full audit output (include_ineligible=True) to avoid restricting
    the model/market comparison to games that pass the rank-gap filter.
    """
    if results is None or results.empty:
        empty = pd.DataFrame()
        return {"summary": empty, "by_season": empty, "by_edge": empty, "bets": empty}
    scored = _valid_scoring_rows(results)
    if scored.empty:
        empty = pd.DataFrame()
        return {"summary": empty, "by_season": empty, "by_edge": empty, "bets": empty}

    model = _probability_metrics(scored["model_favorite_win_probability"], scored["actual_favorite_won"])
    market = _probability_metrics(scored["favorite_no_vig_market_probability"], scored["actual_favorite_won"])
    summary = pd.DataFrame([{
        "Games scored": model["n"], "Elo Brier (lower better)": model["brier"],
        "Market Brier (lower better)": market["brier"],
        "Brier difference (Elo - market)": model["brier"] - market["brier"],
        "Elo log loss (lower better)": model["log_loss"], "Market log loss (lower better)": market["log_loss"],
        "Log-loss difference (Elo - market)": model["log_loss"] - market["log_loss"],
        "Mean Elo probability": model["mean_probability"], "Mean market probability": market["mean_probability"],
        "Observed favorite win rate": model["observed_rate"],
    }])

    season_rows = []
    for season, group in scored.groupby("season", sort=True):
        elo = _probability_metrics(group["model_favorite_win_probability"], group["actual_favorite_won"])
        mkt = _probability_metrics(group["favorite_no_vig_market_probability"], group["actual_favorite_won"])
        season_rows.append({
            "Season": int(season), "Games scored": elo["n"], "Elo Brier": elo["brier"],
            "Market Brier": mkt["brier"], "Brier difference (Elo - market)": elo["brier"] - mkt["brier"],
            "Elo log loss": elo["log_loss"], "Market log loss": mkt["log_loss"],
            "Log-loss difference (Elo - market)": elo["log_loss"] - mkt["log_loss"],
            "Elo mean probability": elo["mean_probability"], "Market mean probability": mkt["mean_probability"],
            "Observed win rate": elo["observed_rate"],
        })
    by_season = pd.DataFrame(season_rows)

    scored["edge_band"] = pd.cut(
        (scored["model_favorite_win_probability"] - scored["favorite_no_vig_market_probability"]) * 100.0,
        bins=_EDGE_BINS, labels=_EDGE_LABELS, right=False, ordered=True)
    edge_rows = []
    for band in _EDGE_LABELS:
        group = scored[scored["edge_band"] == band]
        if group.empty:
            continue
        elo = _probability_metrics(group["model_favorite_win_probability"], group["actual_favorite_won"])
        mkt = _probability_metrics(group["favorite_no_vig_market_probability"], group["actual_favorite_won"])
        edge_value = (group["model_favorite_win_probability"] - group["favorite_no_vig_market_probability"]) * 100.0
        edge_rows.append({
            "Model edge band": band, "Games": elo["n"], "Mean edge (pp)": float(edge_value.mean()),
            "Elo mean probability": elo["mean_probability"], "Market mean probability": mkt["mean_probability"],
            "Observed favorite win rate": elo["observed_rate"], "Elo Brier": elo["brier"],
            "Market Brier": mkt["brier"], "Brier difference (Elo - market)": elo["brier"] - mkt["brier"],
        })
    by_edge = pd.DataFrame(edge_rows)

    bet_mask = results.get("bet_taken", pd.Series(False, index=results.index)).fillna(False).astype(bool)
    bets = results[bet_mask].copy()
    if bets.empty:
        bet_summary = pd.DataFrame([{"Bets": 0, "Wins": 0, "Losses": 0, "Pushes": 0,
            "Win rate (settled bets)": float("nan"), "Net units": 0.0, "Risked units": 0.0,
            "ROI on risked units (%)": float("nan")}])
    else:
        pushes = int((bets["home_score"] == bets["away_score"]).sum())
        wins = int((bets["bet_result"] == 1).sum())
        losses = int((bets["bet_result"] == 0).sum())
        settled = wins + losses
        risked = float(pd.to_numeric(bets.get("risked_units", 1.0), errors="coerce").fillna(0).sum())
        net = float(pd.to_numeric(bets.get("net_units", 0.0), errors="coerce").fillna(0).sum())
        bet_summary = pd.DataFrame([{"Bets": int(len(bets)), "Wins": wins, "Losses": losses, "Pushes": pushes,
            "Win rate (settled bets)": wins / settled if settled else float("nan"),
            "Net units": net, "Risked units": risked,
            "ROI on risked units (%)": net / risked * 100.0 if risked else float("nan")}])
    return {"summary": summary, "by_season": by_season, "by_edge": by_edge, "bets": bet_summary}
