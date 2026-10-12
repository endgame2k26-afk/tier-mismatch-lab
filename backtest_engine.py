"""Leak-resistant chronological Elo walk-forward baseline. Not the full Tier-Mismatch V2."""
import math
import pandas as pd
RAW_GAMES_URL='https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'

def load_nflverse_games():
    return pd.read_csv(RAW_GAMES_URL, low_memory=False)

def american_to_implied(odds):
    try: x=float(odds)
    except (TypeError,ValueError): return None
    if not math.isfinite(x) or x==0: return None
    return 100/(x+100) if x>0 else abs(x)/(abs(x)+100)

def validate_games_frame(frame):
    if frame is None or frame.empty: return False,['CSV contains no rows.'],pd.DataFrame()
    df=frame.copy(); low={str(c).lower():c for c in df.columns}
    aliases={'season':['season','year'],'gameday':['gameday','game_date','date','kickoff_date'],'home_team':['home_team','home','home_abbr'],'away_team':['away_team','away','away_abbr'],'home_score':['home_score','home_points','score_home'],'away_score':['away_score','away_points','score_away'],'home_moneyline':['home_moneyline','home_ml','home_money_line'],'away_moneyline':['away_moneyline','away_ml','away_money_line'],'game_type':['game_type','season_type','type']}
    rename={}
    for target,names in aliases.items():
        for name in names:
            if name in low:
                if low[name]!=target: rename[low[name]]=target
                break
    df=df.rename(columns=rename); required=['season','gameday','home_team','away_team','home_score','away_score']; missing=[c for c in required if c not in df]
    if missing: return False,['Missing required column(s): '+', '.join(missing)],pd.DataFrame()
    df['season']=pd.to_numeric(df['season'],errors='coerce'); df['gameday']=pd.to_datetime(df['gameday'],errors='coerce',utc=True).dt.date
    for c in ['home_score','away_score','home_moneyline','away_moneyline']:
        if c in df: df[c]=pd.to_numeric(df[c],errors='coerce')
    for c in ['home_team','away_team']: df[c]=df[c].astype('string').str.upper().str.strip()
    df=df.dropna(subset=required).copy()
    if df.empty: return False,['No valid game rows remain.'],df
    df['season']=df['season'].astype(int); df['game_type']=df['game_type'].fillna('REG').astype(str).str.upper() if 'game_type' in df else 'REG'
    df=df.drop_duplicates(['season','gameday','home_team','away_team']).sort_values(['gameday','season','home_team','away_team']).reset_index(drop=True)
    return len(df)>=2,([] if len(df)>=2 else ['At least two games are required.']),df

def _p_home(h,a): return 1/(1+10**((a-(h+55))/400))
def _profit(odds): return odds/100 if odds>0 else 100/abs(odds)

def run_walk_forward_backtest(games,first_test_season,minimum_rank_gap=9,minimum_edge_pp=4.0):
    ok,issues,df=validate_games_frame(games)
    if not ok: raise ValueError('; '.join(issues))
    df=df[df.game_type.isin(['REG','POST','WC','DIV','CON','SB'])].copy(); ratings={}; last_season={}; rows=[]; batches=0
    for day,group in df.groupby('gameday',sort=True):
        batches+=1; season=int(group.season.min()); teams=set(group.home_team.astype(str))|set(group.away_team.astype(str))
        for t in teams:
            ratings.setdefault(t,1500.0)
            if t in last_season and season>last_season[t]:
                for _ in range(min(season-last_season[t],5)): ratings[t]=1500+(ratings[t]-1500)*.75
        snap=dict(ratings); ordered=sorted(ratings.items(),key=lambda z:(-z[1],z[0])); ranks={t:i+1 for i,(t,_) in enumerate(ordered)}; updates=[]
        for _,g in group.iterrows():
            h,a=str(g.home_team),str(g.away_team); he,ae=snap.get(h,1500),snap.get(a,1500); hp=_p_home(he,ae); hr,ar=ranks.get(h,len(ranks)+1),ranks.get(a,len(ranks)+1)
            fav='HOME' if hp>=.5 else 'AWAY'; fp=hp if fav=='HOME' else 1-hp; gap=(ar-hr) if fav=='HOME' else (hr-ar)
            def num(v):
                try: return float(v)
                except (TypeError,ValueError): return float('nan')
            hml=num(g.get('home_moneyline',float('nan'))); aml=num(g.get('away_moneyline',float('nan'))); available=math.isfinite(hml) and math.isfinite(aml) and hml!=0 and aml!=0
            fml=hml if fav=='HOME' else aml; dml=aml if fav=='HOME' else hml; p=american_to_implied(fml) if available else None; q=american_to_implied(dml) if available else None
            nv=p/(p+q) if p is not None and q is not None else float('nan'); edge=(fp-nv)*100 if available else float('nan'); eligible=gap>=int(minimum_rank_gap); bet=bool(eligible and available and edge>=float(minimum_edge_pp))
            hs,as_=float(g.home_score),float(g.away_score); tie=hs==as_; won=(hs>as_) if fav=='HOME' else (as_>hs_); net=(_profit(fml) if won else -1.0) if bet and not tie else (-1.0 if bet else 0.0)
            rows.append({'season':int(g.season),'gameday':str(day),'home_team':h,'away_team':a,'home_score':hs,'away_score':as_,'home_elo_pregame':round(he,2),'away_elo_pregame':round(ae,2),'home_rank_pregame':hr,'away_rank_pregame':ar,'model_favorite':fav,'favorite_rank_gap':gap,'model_favorite_win_probability':round(fp,6),'home_moneyline':hml if math.isfinite(hml) else None,'away_moneyline':aml if math.isfinite(aml) else None,'market_available':bool(available),'favorite_no_vig_market_probability':round(nv,6) if available else None,'edge_percentage_points':round(edge,4) if available else None,'rank_gap_eligible':bool(eligible),'bet_taken':bet,'bet_result':(int(won) if not tie else 0) if bet else None,'risked_units':1.0 if bet else 0.0,'net_units':round(net,4),'actual_favorite_won':bool(won)})
            outcome=1.0 if hs>as_ else (.5 if hs==as_ else 0.0); updates.append((h,a,he,ae,outcome))
        # All games on this date were scored against frozen pregame ratings above.
        for h,a,he,ae,outcome in updates:
            delta=20*(outcome-_p_home(he,ae)); ratings[h]=he+delta; ratings[a]=ae-delta; last_season[h]=max(season,last_season.get(h,season)); last_season[a]=max(season,last_season.get(a,season))
    out=pd.DataFrame(rows); out=out[out.season>=int(first_test_season)].copy(); diag={'input_games':len(df),'test_games_before_rank_filter':len(out),'date_batches_processed':batches,'market_rows':int(out.market_available.sum()) if len(out) else 0}; out=out[out.rank_gap_eligible].reset_index(drop=True); return out,diag

def summarize_backtest(r):
    if r is None or r.empty: return {'eligible_games':0,'games_with_odds':0,'bets':0,'win_rate':'n/a','net_units':'0.00','roi':'n/a','brier':'n/a','odds_coverage':'n/a'}
    market=r[r.market_available]; bets=r[r.bet_taken]; wins=int((bets.bet_result==1).sum()) if len(bets) else 0; risk=float(bets.risked_units.sum()); net=float(bets.net_units.sum()); valid=r[r.home_score!=r.away_score]; brier=((valid.model_favorite_win_probability-valid.actual_favorite_won.astype(float))**2).mean() if len(valid) else float('nan')
    return {'eligible_games':len(r),'games_with_odds':len(market),'bets':len(bets),'win_rate':f'{wins/len(bets)*100:.1f}%' if len(bets) else 'n/a','net_units':f'{net:+.2f}','roi':f'{net/risk*100:+.1f}%' if risk else 'n/a','brier':f'{brier:.4f}' if math.isfinite(brier) else 'n/a','odds_coverage':f'{len(market)/len(r)*100:.1f}%'}
