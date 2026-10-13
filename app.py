import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from backtest_engine import (
    american_to_implied,
    load_nflverse_games,
    run_walk_forward_backtest,
    summarize_backtest,
    validate_games_frame,
)

st.set_page_config(
    page_title="Tier-Mismatch Lab",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Product styling: compact sports-research workspace, not a generic admin panel.
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.35rem; padding-bottom: 2.5rem; max-width: 1500px;}
      [data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.18);}
      .hero {padding: 1.25rem 1.4rem; border: 1px solid rgba(128,128,128,.22);
             border-radius: 18px; margin-bottom: 1rem;
             background: linear-gradient(115deg, rgba(36,99,235,.12), rgba(16,185,129,.06));}
      .eyebrow {font-size:.75rem; letter-spacing:.12em; text-transform:uppercase;
                font-weight:700; opacity:.72; margin-bottom:.45rem;}
      .hero h1 {font-size:2rem; line-height:1.15; margin:0 0 .45rem 0;}
      .hero p {margin:0; opacity:.78; max-width:800px;}
      .section-note {opacity:.72; font-size:.9rem; margin-top:-.35rem;}
      .status-pill {display:inline-block; border:1px solid rgba(128,128,128,.3);
                    border-radius:999px; padding:.2rem .6rem; font-size:.75rem;}
      div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.18);
                    padding:.85rem .95rem; border-radius:14px; background:rgba(128,128,128,.035);}
      div[data-testid="stMetricLabel"] p {font-size:.8rem;}
      .small-muted {opacity:.68; font-size:.82rem;}
      .why-card {border:1px solid rgba(128,128,128,.2); border-radius:12px;
                 padding:.8rem 1rem; margin:.35rem 0;}
      .stButton button, .stDownloadButton button {border-radius:10px; font-weight:600;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">NFL · Research workspace</div>
      <h1>Tier-Mismatch Lab</h1>
      <p>Find, test, and audit historical mismatches. Every result should be traceable to data,
      assumptions, and a reproducible run — never a fabricated pick.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

PAGES = [
    "Overview",
    "Backtest Lab",
    "Game Explorer",
    "Manual Odds",
    "Data & Method",
]
with st.sidebar:
    st.markdown("## 🏈 Tier-Mismatch")
    st.caption("Research workspace · baseline model")
    page = st.radio("Workspace", PAGES, label_visibility="collapsed")
    st.divider()
    st.markdown("**Model status**")
    st.markdown('<span class="status-pill">Elo baseline</span> &nbsp; <span class="status-pill">Historical only</span>', unsafe_allow_html=True)
    st.caption("Not the complete Tier-Mismatch V2.")
    st.divider()
    st.caption("Free-first · No live picks · No invented data")

# Session state initialization
for key, default in [
    ("games", None), ("source", ""), ("result", None), ("audit_result", None), ("diag", None), ("meta", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def show_data_loader():
    st.subheader("1 · Choose your historical data")
    st.markdown('<p class="section-note">Load once, validate the columns, then reuse the same dataset across the workspace.</p>', unsafe_allow_html=True)
    mode = st.radio(
        "Data source",
        ["Public nflverse CSV", "Upload a CSV"],
        horizontal=True,
        key="data_source_mode",
    )
    if mode == "Public nflverse CSV":
        st.caption("Public historical games dataset. This is not a live odds feed.")
        if st.button("Load public historical data", type="primary", key="load_public"):
            try:
                with st.spinner("Fetching historical games…"):
                    st.session_state.games = load_nflverse_games()
                    st.session_state.source = "nflverse public games.csv"
                st.session_state.result = None
                st.session_state.audit_result = None
                st.session_state.meta = None
                st.success("Historical dataset loaded.")
            except Exception as exc:
                st.error(f"Could not fetch the public dataset: {exc}")
                st.info("Try uploading a compatible CSV. The app will not substitute sample results.")
    else:
        upload = st.file_uploader("Historical games CSV", type=["csv"], key="games_upload")
        if upload is not None:
            try:
                uploaded = pd.read_csv(upload)
                st.session_state.games = uploaded
                st.session_state.source = upload.name
                st.session_state.result = None
                st.session_state.audit_result = None
                st.session_state.meta = None
                st.success(f"Loaded {len(uploaded):,} rows from {upload.name}.")
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")

    if st.session_state.games is not None:
        ok, issues, normalized = validate_games_frame(st.session_state.games)
        if not ok:
            for issue in issues:
                st.error(issue)
            return None
        st.session_state.normalized_games = normalized
        seasons = sorted(normalized.season.unique().tolist())
        left, mid, right = st.columns([1.5, 1, 1])
        left.metric("Validated games", f"{len(normalized):,}")
        mid.metric("Seasons", f"{int(min(seasons))}–{int(max(seasons))}")
        right.metric("Rows with both moneylines", f"{int(normalized[['home_moneyline','away_moneyline']].notna().all(axis=1).sum()):,}" if {"home_moneyline","away_moneyline"}.issubset(normalized.columns) else "0")
        with st.expander("Preview normalized data", expanded=False):
            st.dataframe(normalized.head(12), use_container_width=True, hide_index=True)
        return normalized
    return None


def render_run_controls(normalized):
    if normalized is None:
        st.info("Load or upload a dataset above to configure a backtest.")
        return
    seasons = sorted(normalized.season.unique().tolist())
    st.subheader("2 · Configure the test")
    a, b, c = st.columns([1.2, 1, 1])
    default_index = max(0, len(seasons) - 3)
    start_season = a.selectbox("First test season", seasons, index=default_index, key="start_season")
    rank_gap = b.slider("Minimum rank gap", 1, 20, 9, help="A game must meet or exceed this pregame Elo-rank difference.", key="rank_gap")
    edge_pp = c.slider("Minimum edge (percentage points)", 0, 15, 4, help="Model probability minus the two-sided no-vig market probability.", key="edge_pp")
    st.caption("The engine scores each date using pregame ratings before updating Elo with results from that date.")
    if st.button("Run walk-forward backtest", type="primary", use_container_width=True, key="run_backtest"):
        try:
            with st.spinner("Scoring the historical schedule chronologically…"):
                # Keep strategy metrics on eligible games, while preserving the full
                # test-period population for the audit export.
                result, diag = run_walk_forward_backtest(
                    normalized, int(start_season), int(rank_gap), float(edge_pp)
                )
                audit_result, _ = run_walk_forward_backtest(
                    normalized, int(start_season), int(rank_gap), float(edge_pp),
                    include_ineligible=True,
                )
            meta = {
                "source": st.session_state.source,
                "first_test_season": int(start_season),
                "rank_gap": int(rank_gap),
                "edge_pp": float(edge_pp),
                "run_utc": datetime.now(timezone.utc).isoformat(),
                "model": "Elo/rank-gap baseline",
            }
            st.session_state.result = result
            st.session_state.audit_result = audit_result
            st.session_state.diag = diag
            st.session_state.meta = meta
            st.success("Backtest completed. Review the diagnostics before interpreting returns.")
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
            st.exception(exc)


def render_results():
    result = st.session_state.result
    if result is None:
        st.info("No completed run yet. Open Backtest Lab to run a historical test.")
        return
    summary = summarize_backtest(result)
    meta = st.session_state.meta or {}
    diag = st.session_state.diag or {}
    st.markdown("### Latest run")
    st.caption(f"Source: {meta.get('source','unknown')} · Test seasons from {meta.get('first_test_season','—')} · Run UTC: {meta.get('run_utc','—')}")
    if len(result) == 0:
        st.warning("No games met the selected rank-gap threshold in the test period. Strategy metrics are unavailable, but the full audit can still be downloaded.")
        audit_result = st.session_state.audit_result
        if audit_result is not None:
            st.download_button(
                "Download full audit CSV",
                audit_result.to_csv(index=False),
                "tier_mismatch_full_walk_forward_audit.csv",
                "text/csv",
            )
        st.download_button(
            "Download run metadata",
            json.dumps(meta, indent=2),
            "tier_mismatch_run_metadata.json",
            "application/json",
        )
        return
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rank-gap eligible", f"{summary['eligible_games']:,}")
    m2.metric("Games with odds", f"{summary['games_with_odds']:,}")
    m3.metric("Bets taken", f"{summary['bets']:,}")
    m4.metric("Win rate", summary["win_rate"])
    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Net units", summary["net_units"])
    m6.metric("ROI on risked units", summary["roi"])
    m7.metric("Brier score (eligible games)", summary["brier"])
    st.caption("Brier is calculated on eligible games with non-tied outcomes, using the model's favorite-win probability; it is not a market-relative score.")
    m8.metric("Odds coverage", summary["odds_coverage"])
    with st.expander("Run diagnostics", expanded=True):
        d1, d2, d3 = st.columns(3)
        d1.metric("Input games", f"{diag.get('input_games', '—'):,}" if isinstance(diag.get('input_games'), int) else "—")
        d2.metric("Date batches processed", f"{diag.get('date_batches_processed', '—'):,}" if isinstance(diag.get('date_batches_processed'), int) else "—")
        d3.metric("Max rank in returned audit", str(diag.get("max_rank", "—")))
        st.caption(diag.get("note", "Diagnostics not available."))
    st.warning("Research only: historical results are not proof of future profitability. Market odds may not have verified timestamps, and this baseline lacks the full strategy's contextual filters.")
    audit_result = st.session_state.audit_result
    if audit_result is not None:
        st.caption(
            f"Full audit export: {len(audit_result):,} test-period games, including games rejected "
            "by the rank-gap screen. Strategy metrics above remain based on rank-gap-eligible games."
        )
        st.download_button(
            "Download full audit CSV",
            audit_result.to_csv(index=False),
            "tier_mismatch_full_walk_forward_audit.csv",
            "text/csv",
        )
    else:
        st.download_button(
            "Download eligible-game CSV",
            result.to_csv(index=False),
            "tier_mismatch_eligible_walk_forward_audit.csv",
            "text/csv",
        )
    st.download_button("Download run metadata", json.dumps(meta, indent=2), "tier_mismatch_run_metadata.json", "application/json")
    st.markdown("### Performance by season")
    if "season" in result and "net_units" in result:
        season_stats = result.groupby("season", as_index=False).agg(
            bets=("bet_taken", "sum"),
            net_units=("net_units", "sum"),
            eligible_games=("season", "size"),
        )
        bet_counts = result[result.bet_taken].groupby("season").size()
        season_stats["bets"] = season_stats["season"].map(bet_counts).fillna(0).astype(int)
        season_stats["cumulative_net_units"] = season_stats["net_units"].cumsum()
        # Use HTML/CSS for these small charts rather than Streamlit's chart
        # helpers, which import Altair and can fail when the cloud environment
        # has an incompatible typing_extensions / Altair combination.
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Net units by season")
            max_abs = max(float(season_stats["net_units"].abs().max()), 1.0)
            rows = []
            for _, row in season_stats.iterrows():
                value = float(row["net_units"])
                width = min(abs(value) / max_abs * 100, 100)
                color = "#16a34a" if value >= 0 else "#dc2626"
                rows.append(
                    "<div style='display:grid;grid-template-columns:60px 1fr 72px;gap:10px;align-items:center;margin:9px 0;'>"
                    f"<span style='color:var(--text-color);'>{row['season']}</span>"
                    "<div style='height:14px;background:rgba(128,128,128,.15);border-radius:4px;overflow:hidden;'>"
                    f"<div style='height:14px;width:{width:.1f}%;background:{color};border-radius:4px;'></div></div>"
                    f"<span style='text-align:right;color:var(--text-color);font-variant-numeric:tabular-nums;'>{value:+.2f}</span>"
                    "</div>"
                )
            st.markdown("<div style='padding:4px 0;'>" + "".join(rows) + "</div>", unsafe_allow_html=True)
        with c2:
            st.caption("Cumulative net units across test seasons")
            max_abs_cum = max(float(season_stats["cumulative_net_units"].abs().max()), 1.0)
            rows = []
            for _, row in season_stats.iterrows():
                value = float(row["cumulative_net_units"])
                width = min(abs(value) / max_abs_cum * 100, 100)
                color = "#16a34a" if value >= 0 else "#dc2626"
                rows.append(
                    "<div style='display:grid;grid-template-columns:60px 1fr 72px;gap:10px;align-items:center;margin:9px 0;'>"
                    f"<span style='color:var(--text-color);'>{row['season']}</span>"
                    "<div style='height:14px;background:rgba(128,128,128,.15);border-radius:4px;overflow:hidden;'>"
                    f"<div style='height:14px;width:{width:.1f}%;background:{color};border-radius:4px;'></div></div>"
                    f"<span style='text-align:right;color:var(--text-color);font-variant-numeric:tabular-nums;'>{value:+.2f}</span>"
                    "</div>"
                )
            st.markdown("<div style='padding:4px 0;'>" + "".join(rows) + "</div>", unsafe_allow_html=True)
        st.dataframe(season_stats, use_container_width=True, hide_index=True)


if page == "Overview":
    st.markdown("### Research overview")
    if st.session_state.result is None:
        st.info("Start with a historical dataset, then run a backtest. This overview will populate from your latest completed run.")
        st.markdown("#### Your workflow")
        w1, w2, w3 = st.columns(3)
        with w1:
            st.markdown("**01 · Load data**")
            st.write("Use the public historical CSV or bring your own compatible dataset.")
        with w2:
            st.markdown("**02 · Run a test**")
            st.write("Choose the test period and the minimum rank-gap and market-edge thresholds.")
        with w3:
            st.markdown("**03 · Audit the evidence**")
            st.write("Review seasonal performance, odds coverage, and the game-level reasoning.")
    else:
        render_results()
    st.markdown("### Baseline model at a glance")
    a, b, c, d = st.columns(4)
    a.metric("Starting Elo", "1500")
    b.metric("K-factor", "20")
    c.metric("Home advantage", "55 Elo")
    d.metric("Season regression", "25%")
    st.caption("These are fixed engine assumptions, not optimized settings or evidence of predictive advantage.")

elif page == "Backtest Lab":
    st.markdown("### Backtest Lab")
    st.write("Set the data and thresholds, run the test, then review performance and diagnostics below.")
    normalized = show_data_loader()
    render_run_controls(normalized)
    if st.session_state.result is not None:
        st.divider()
        render_results()

elif page == "Game Explorer":
    st.markdown("### Game Explorer")
    result = st.session_state.result
    if result is None or result.empty:
        st.info("Run a backtest first. The explorer uses the eligible-game audit from that run.")
    else:
        st.caption("This table contains games that passed the rank-gap screen, not only bets taken.")
        col_a, col_b, col_c = st.columns([1.2, 1, 1])
        seasons = sorted(result.season.unique().tolist())
        chosen_seasons = col_a.multiselect("Season", seasons, default=seasons, key="explorer_seasons")
        bet_filter = col_b.selectbox("Bet status", ["All eligible games", "Bets taken", "No bet"], key="explorer_bet_status")
        market_filter = col_c.selectbox("Odds", ["All", "Odds available", "Odds missing"], key="explorer_market_status")
        view = result[result.season.isin(chosen_seasons)].copy()
        if bet_filter == "Bets taken":
            view = view[view.bet_taken]
        elif bet_filter == "No bet":
            view = view[~view.bet_taken]
        if market_filter == "Odds available":
            view = view[view.market_available]
        elif market_filter == "Odds missing":
            view = view[~view.market_available]
        search = st.text_input("Find a team", placeholder="Type a team abbreviation…", key="team_search").strip().upper()
        if search:
            view = view[view.home_team.str.contains(search, na=False) | view.away_team.str.contains(search, na=False)]
        display_cols = [c for c in [
            "season", "gameday", "away_team", "home_team", "model_favorite",
            "favorite_rank_gap", "model_favorite_win_probability",
            "favorite_no_vig_market_probability", "edge_percentage_points",
            "market_available", "rank_gap_eligible", "bet_taken", "bet_decision_reason", "bet_result",
            "home_moneyline", "away_moneyline", "net_units",
        ] if c in view.columns]
        st.caption(f"{len(view):,} games shown")
        st.dataframe(view[display_cols], use_container_width=True, hide_index=True)
        st.download_button("Download filtered games", view.to_csv(index=False), "tier_mismatch_filtered_games.csv", "text/csv")

elif page == "Manual Odds":
    st.markdown("### Manual Odds Calculator")
    st.write("Compare a hypothetical model probability against a two-sided market price. This tool does not generate a prediction.")
    left, right = st.columns(2)
    with left:
        team = st.text_input("Team", "Team A")
        ml = st.number_input("Team American moneyline", min_value=-10000, max_value=10000, value=120, step=5)
    with right:
        opponent = st.text_input("Opponent", "Team B")
        opponent_ml = st.number_input("Opponent American moneyline", min_value=-10000, max_value=10000, value=-140, step=5)
    model_pct = st.slider("Hypothetical model probability", 1, 99, 50, format="%d%%")
    threshold = st.slider("Minimum edge threshold", 0, 15, 4, format="%d pp")
    p = american_to_implied(ml)
    q = american_to_implied(opponent_ml)
    if p is None or q is None:
        st.error("American moneylines cannot be zero.")
    else:
        no_vig = p / (p + q)
        edge_pp = (model_pct / 100 - no_vig) * 100
        a, b, c = st.columns(3)
        a.metric(f"{team} no-vig probability", f"{no_vig * 100:.2f}%")
        b.metric("Hypothetical probability", f"{model_pct:.1f}%")
        c.metric("Edge vs market", f"{edge_pp:+.2f} pp")
        if edge_pp >= threshold:
            st.warning(f"Threshold passed for {team}, but this is not a pick: the probability was manually entered.")
        else:
            st.info("The hypothetical probability does not pass the selected edge threshold.")
        with st.expander("How this is calculated"):
            st.write("Each American moneyline is converted to an implied probability. The two implied probabilities are normalized to remove the displayed overround, then compared with your manually entered probability.")
            st.code("no_vig = p_team / (p_team + p_opponent)\nedge_pp = (model_probability - no_vig) * 100")

elif page == "Data & Method":
    st.markdown("### Data & Method")
    st.markdown("#### Data source")
    st.markdown("[nflverse historical games CSV](https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv)")
    st.write("The app fetches the public file only when requested. It is a historical game dataset, not a live odds service.")
    st.markdown("#### Required columns")
    st.code("season, gameday, home_team, away_team, home_score, away_score")
    st.markdown("Optional moneyline columns: `home_moneyline`, `away_moneyline`.")
    st.markdown("#### Walk-forward safeguards")
    st.markdown("""
    - Games on the same date are scored from the same pre-update rating snapshot.
    - Elo updates occur after all games on that date are scored.
    - Missing moneylines remain missing and cannot trigger a bet.
    - The returned audit contains games passing the rank-gap threshold; `bet_taken` is a stricter additional filter.
    - Tied Elo ratings share a rank in the patched engine; review the deployed engine version before relying on this behavior.
    """)
    st.markdown("#### Fixed baseline assumptions")
    st.markdown("""
    - Starting Elo: 1500
    - K-factor: 20
    - Home advantage: 55 Elo
    - Season regression: 25%
    - Probability scale: 400
    """)
    st.markdown("#### Not implemented")
    missing = [
        "Original turnover lookback and penalty",
        "Timestamped quarterback/injury status adjustments",
        "EPA and success-rate composite",
        "Relative rest and divisional-road adjustments",
        "Separate ATS probability and ATS backtest",
        "Closing-line-value tracking and odds timestamp validation",
        "Live odds or live picks",
    ]
    for item in missing:
        st.markdown(f"- {item}")
    st.warning("Research only. No profitability claim is made. Treat every backtest as provisional until data timing, calibration, and out-of-sample stability have been independently checked.")

st.divider()
st.caption("Tier-Mismatch Lab · Sports research workspace · Baseline model only · Not betting advice")
