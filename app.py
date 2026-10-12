import json
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
from backtest_engine import load_nflverse_games, validate_games_frame, run_walk_forward_backtest, summarize_backtest, american_to_implied

st.set_page_config(page_title='Tier-Mismatch Lab', page_icon='🏈', layout='wide')
st.title('🏈 Tier-Mismatch Lab')
st.caption('Free-first research workspace · walk-forward Elo baseline · no fabricated picks')
page=st.sidebar.radio('Workspace',['Walk-Forward Backtest','Manual Odds Calculator','Data & Limitations'])
if page=='Walk-Forward Backtest':
    st.warning('This is an Elo baseline, not the full Tier-Mismatch V2. Turnover, timestamped QB/injury, EPA/success-rate, rest, divisional-road, ATS, and CLV features are not implemented.')
    mode=st.radio('Historical data',['Load public nflverse CSV','Upload CSV'],horizontal=True)
    games=None; source=''
    if mode=='Load public nflverse CSV':
        if st.button('Load public historical data',type='primary'):
            try:
                with st.spinner('Fetching public historical data…'):
                    st.session_state.games=load_nflverse_games(); st.session_state.source='nflverse public games.csv'
            except Exception as e: st.error(f'Fetch failed: {e}. Upload a CSV instead; no sample results will be substituted.')
        games=st.session_state.get('games'); source=st.session_state.get('source','nflverse public games.csv')
        st.caption('Public historical source: https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv. Not a live odds feed.')
    else:
        up=st.file_uploader('Upload historical games CSV',type=['csv'])
        if up:
            try: games=pd.read_csv(up); source=up.name
            except Exception as e: st.error(str(e))
    if games is not None:
        ok,issues,norm=validate_games_frame(games)
        if not ok:
            for issue in issues: st.error(issue)
        else:
            st.success(f'Validated {len(norm):,} game rows · source: {source}')
            seasons=sorted(norm.season.unique().tolist())
            a,b,c=st.columns(3)
            start=a.selectbox('First test season',seasons,index=max(0,len(seasons)-3))
            gap=b.slider('Minimum pregame rank gap',1,20,9)
            edge=c.slider('Minimum edge vs no-vig market (pp)',0,15,4)
            if st.button('Run walk-forward backtest',type='primary'):
                try:
                    with st.spinner('Scoring games chronologically…'):
                        result,diag=run_walk_forward_backtest(norm,int(start),int(gap),float(edge))
                    st.session_state.result=result; st.session_state.diag=diag
                    st.session_state.meta={'source':source,'first_test_season':int(start),'rank_gap':int(gap),'edge_pp':float(edge),'run_utc':datetime.now(timezone.utc).isoformat()}
                except Exception as e: st.exception(e)
    if 'result' in st.session_state:
        result=st.session_state.result; s=summarize_backtest(result); st.subheader('Latest run')
        st.caption(f"Source: {st.session_state.meta['source']} · run UTC: {st.session_state.meta['run_utc']}")
        cols=st.columns(4)
        for col,label,key in zip(cols,['Eligible games','Games with odds','Bets taken','Bet win rate'],['eligible_games','games_with_odds','bets','win_rate']): col.metric(label,s[key])
        cols=st.columns(4)
        for col,label,key in zip(cols,['Net units','ROI on risked units','Brier score','Odds coverage'],['net_units','roi','brier','odds_coverage']): col.metric(label,s[key])
        st.info('Historical results are not proof of future profitability. Check sample size, odds coverage, season stability, market timestamps and closing-line value.')
        st.subheader('Game-level audit trail'); st.dataframe(result,use_container_width=True,hide_index=True)
        st.download_button('Download audit CSV',result.to_csv(index=False),'tier_mismatch_walk_forward_audit.csv','text/csv')
        st.download_button('Download run metadata',json.dumps(st.session_state.meta,indent=2),'tier_mismatch_run_metadata.json','application/json')
elif page=='Manual Odds Calculator':
    st.write('Enter both sides from the same sportsbook and timestamp. The probability below is a hypothetical manual input, not a generated prediction.')
    a,b=st.columns(2)
    with a: team=st.text_input('Team','Team A'); ml=st.number_input('Team American moneyline',-10000,10000,120,5)
    with b: opp=st.text_input('Opponent','Team B'); oml=st.number_input('Opponent American moneyline',-10000,10000,-140,5)
    model=st.slider('Hypothetical model probability (%)',1,99,50); threshold=st.slider('Minimum edge (pp)',0,15,4)
    p,q=american_to_implied(ml),american_to_implied(oml)
    if p is None or q is None: st.error('American odds cannot be zero.')
    else:
        nv=p/(p+q); edge=model/100-nv
        x,y,z=st.columns(3); x.metric(f'{team} no-vig probability',f'{nv*100:.2f}%'); y.metric('Hypothetical probability',f'{model:.1f}%'); z.metric('Edge',f'{edge*100:+.2f} pp')
        st.warning('Threshold screen only. This is not a pick because the model probability was manually entered.') if edge*100>=threshold else st.info('Does not pass the selected threshold.')
else:
    st.subheader('Sources and method')
    st.markdown('''**Public source:** [nflverse historical games CSV](https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv). The app only fetches it when requested. If unavailable, upload your own CSV.\n\n**Required columns:** `season, gameday, home_team, away_team, home_score, away_score`. Optional odds columns: `home_moneyline, away_moneyline`.\n\n**Walk-forward safeguards:** all games on the same date are scored using pregame ratings before any same-date results update Elo. Missing odds remain missing and cannot trigger a bet.\n\n**Fixed baseline:** starting Elo 1500, K=20, home advantage 55, season regression 25%, logistic scale 400.\n\n**Not included:** live picks, timestamped odds, historical QB/injury status, turnover lookback, EPA/success rate, divisional-road/rest filters, ATS model, closing-line value. No claim of validated profitability.''')
st.divider(); st.caption('Tier-Mismatch Lab v0.2 · Research only, not financial advice.')
