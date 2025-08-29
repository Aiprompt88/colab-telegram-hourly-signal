import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import ccxt
from dateutil import tz
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import AverageTrueRange
from ta.trend import EMAIndicator

AFRICA_LAGOS = tz.gettz('Africa/Lagos')

TF_TO_SEC = {
    "1m":60, "3m":180, "5m":300, "15m":900, "30m":1800,
    "1h":3600, "2h":7200, "4h":14400, "6h":21600, "8h":28800, "12h":43200,
    "1d":86400
}

def parse_percent_from_val(v):
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return None
    if isinstance(v, str):
        s = v.strip().lower().replace(",", "")
        if s.endswith("bps"):
            num = float(s[:-3].strip()); return num/10_000.0
        has_pct = s.endswith("%")
        if has_pct: s = s[:-1].strip()
        num = float(s); return num/100.0 if has_pct or (num > 1) else num
    num = float(v); return (num/100.0) if num > 1 else num

def fetch_quote_volume_usdt(tkr):
    if not tkr: return None
    if 'quoteVolume' in tkr and tkr['quoteVolume'] is not None:
        try: return float(tkr['quoteVolume'])
        except: pass
    base = tkr.get('baseVolume')
    last = tkr.get('last') or tkr.get('close')
    if base is not None and last is not None:
        try: return float(base) * float(last)
        except: pass
    info = tkr.get('info', {}) or {}
    for k in ['quoteVolume','turnover','amount','quote_volume','quoteVol','volValue']:
        if k in info:
            try: return float(info[k])
            except: continue
    return None

def add_features(df, short_history=False):
    c = df["close"]; h = df["high"]; l = df["low"]; o = df["open"]
    df["r1"] = c.pct_change(1)
    for n in [2,4,8,12,24]:
        df[f"r{n}"] = c.pct_change(n)
    rng = (h - l).replace(0, np.nan)
    df["hlr1"] = rng / c
    df["cbr1"] = (c - o) / (rng + 1e-9)
    df["u_wick"] = (h - np.maximum(o, c)) / (rng + 1e-9)
    df["l_wick"] = (np.minimum(o, c) - l) / (rng + 1e-9)
    df["atr7"]  = AverageTrueRange(high=h, low=l, close=c, window=7).average_true_range()
    df["atr14"] = AverageTrueRange(high=h, low=l, close=c, window=14).average_true_range()
    df["rv7"]   = df["r1"].rolling(7).std()
    df["rv14"]  = df["r1"].rolling(14).std()
    df["ema10"] = EMAIndicator(c, 10).ema_indicator()
    df["ema20"] = EMAIndicator(c, 20).ema_indicator()
    df["ema50"] = EMAIndicator(c, 50).ema_indicator()
    if not short_history:
        df["ema100"] = EMAIndicator(c, 100).ema_indicator()
    df["ema10_slope"] = df["ema10"].diff()
    df["ema20_slope"] = df["ema20"].diff()
    df["ema50_slope"] = df["ema50"].diff()
    roll20_max = h.rolling(20).max(); roll20_min = l.rolling(20).min()
    df["ppos20"] = (c - roll20_min) / (roll20_max - roll20_min + 1e-9)
    df["rsi14"]  = RSIIndicator(c, 14).rsi()
    st = StochasticOscillator(high=h, low=l, close=c, window=14, smooth_window=3)
    df["stochK"] = st.stoch(); df["stochD"] = st.stoch_signal()
    df["willr"]  = WilliamsRIndicator(h, l, c, lbp=14).williams_r()
    prev_close = c.shift(1); prev_open = o.shift(1)
    df["engulf_bull"] = ((c>o) & (o<=prev_close) & (c>=prev_open)).astype(int)
    df["pinbar_bull"] = (((np.minimum(o,c)-l)/(rng+1e-9) > 0.6) & (df["cbr1"]>-0.1)).astype(int)
    df["three_up"]    = ((c>prev_close) & (prev_close>prev_open)).astype(int)
    df["doji"]        = (np.abs(c-o)/(rng+1e-9) < 0.1).astype(int)
    for n in ([3,6,12,24] if not short_history else [3,6,12]):
        df[f"mean_cbr1_{n}"] = df["cbr1"].rolling(n).mean()
        df[f"mean_r1_{n}"]   = df["r1"].rolling(n).mean()
        df[f"max_uw_{n}"]    = df["u_wick"].rolling(n).max()
        df[f"max_lw_{n}"]    = df["l_wick"].rolling(n).max()
        df[f"mean_hlr_{n}"]  = df["hlr1"].rolling(n).mean()
        df[f"zclose_{n}"]    = (c - c.rolling(n).mean())/(c.rolling(n).std()+1e-9)
    df["vol_ratio"] = df["atr14"] / c
    df["trend_ok"]  = (df["ema20"] > df["ema50"]).astype(int)
    return df

def triple_barrier_labels(df, tp=0.05, sl=0.03, horizon=12, tie_policy="sl_first"):
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    n = len(df); y = np.zeros(n, dtype=np.int8)
    for i in range(n - 1 - horizon):
        entry  = c[i]; tp_lvl = entry*(1+tp); sl_lvl = entry*(1-sl)
        hit = 0
        for j in range(1, horizon+1):
            hi = h[i+j]; lo = l[i+j]
            tp_hit = hi >= tp_lvl; sl_hit = lo <= sl_lvl
            if tp_hit and sl_hit:
                hit = 1 if tie_policy=="tp_first" else (-1 if tie_policy=="sl_first" else 0); break
            elif tp_hit:
                hit = 1;  break
            elif sl_hit:
                hit = -1; break
        y[i] = 1 if hit==1 else 0
    df["y"] = y
    return df

def fetch_ohlcv_paged(ex, symbol, timeframe="1h", start_ms=None, end_ms=None, max_req=200):
    limit = 1000; out = []; since = start_ms
    step = TF_TO_SEC[timeframe]*1000
    for _ in range(max_req):
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch: break
        out.extend(batch)
        if len(batch) < limit: break
        last_ts = batch[-1][0]
        since = last_ts + step
        if end_ms and since > end_ms + step: break
        time.sleep(ex.rateLimit/1000.0 if hasattr(ex,'rateLimit') else 0.2)
    if not out: return None
    df = pd.DataFrame(out, columns=['ts','open','high','low','close','volume'])
    df['open_time'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    if start_ms is not None: df = df[df['ts'] >= start_ms]
    if end_ms   is not None: df = df[df['ts'] <= end_ms]
    return df.reset_index(drop=True) if len(df) else None

def compute_top_signals(
    quote_ccy="USDT",
    timeframe="1h",
    lookback=900,
    horizon=12,
    vol_min=2_000_000,
    vol_max=5_000_000_000,
    scan_all=False,
    top_n=200,
    tp_pct="5%",
    sl_pct="3%",
    fee_pct="0.06%",
    prob_floor=0.60,
    require_trend=True,
    use_vol=True,
    vol_thresh="1.5%",
    tie_policy="sl_first",
    short_history=False,
    max_symbols_in_message=8,
):
    tp = parse_percent_from_val(tp_pct); sl = parse_percent_from_val(sl_pct)
    fees = parse_percent_from_val(fee_pct) or 0.0006
    vth  = parse_percent_from_val(vol_thresh) if use_vol else None

    ex = ccxt.mexc({'enableRateLimit': True})
    markets = ex.load_markets()
    qc = (quote_ccy or "USDT").upper()
    syms = [s for s,m in markets.items() if m.get('spot') and s.endswith('/'+qc) and (m.get('active') in [True,None])]
    tks = ex.fetch_tickers()

    filt=[]
    for s in syms:
        qv=fetch_quote_volume_usdt(tks.get(s,{}))
        if qv is not None and vol_min<=qv<=vol_max:
            filt.append((s,qv))
    if not filt:
        return {"status":"no_markets","table":pd.DataFrame(), "message":"No markets matched 24h volume range."}

    filt.sort(key=lambda x:x[1], reverse=True)
    if not scan_all:
        n = int(top_n or 0)
        if n < 1:
            return {"status":"bad_topn","table":pd.DataFrame(), "message":"Top-N must be ≥ 1 or enable Scan ALL."}
        filt = filt[:n]
    symbols = [s for s,_ in filt]

    frames=[]
    lb=int(lookback); hz=int(horizon)
    for sym in symbols:
        df=fetch_ohlcv_paged(ex, sym, timeframe=timeframe)
        if df is None or len(df)<max(120, hz+2, lb):
            continue
        df=df.iloc[-max(150, hz+120):]
        df=add_features(df, short_history=bool(short_history)).dropna().reset_index(drop=True)
        df=triple_barrier_labels(df, tp=tp, sl=sl, horizon=hz, tie_policy=tie_policy)
        df["symbol"]=sym; frames.append(df); time.sleep(0.01)
    if not frames:
        return {"status":"no_data","table":pd.DataFrame(), "message":"Not enough data after features."}

    data=pd.concat(frames, ignore_index=True)
    nonf=set(["ts","open_time","open","high","low","close","volume","y","symbol"])
    feats=[c for c in data.columns if c not in nonf and np.issubdtype(data[c].dtype, np.number)]
    data.replace([np.inf,-np.inf],np.nan,inplace=True)
    data.dropna(subset=feats+["y"], inplace=True)
    data=data.sort_values("ts")
    cut=int(0.8*len(data)); tr,va=data.iloc[:cut], data.iloc[cut:]

    model=XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=4,
        gamma=0.2, reg_alpha=0.5, reg_lambda=1.5,
        objective="binary:logistic", eval_metric="logloss",
        n_jobs=4, tree_method="hist", random_state=42
    )
    model.fit(tr[feats].values, tr["y"].values, eval_set=[(va[feats].values, va["y"].values)], verbose=False)
    calib=IsotonicRegression(out_of_bounds='clip').fit(model.predict_proba(va[feats].values)[:,1], va["y"].values)

    latest=[]
    for sym in symbols:
        df=fetch_ohlcv_paged(ex, sym, timeframe=timeframe)
        if df is None or len(df)<max(120, hz+2, lb):
            continue
        df=add_features(df, short_history=bool(short_history)).dropna().reset_index(drop=True)
        row=df.iloc[-1:].copy(); row["symbol"]=sym; latest.append(row)
    if not latest:
        return {"status":"no_latest","table":pd.DataFrame(), "message":"No latest rows."}

    live=pd.concat(latest, ignore_index=True)
    if require_trend:
        live=live[live["ema20"]>live["ema50"]]
    if use_vol:
        live=live[(live["atr14"]/live["close"])<=vth]
    if live.empty:
        return {"status":"filtered_out","table":pd.DataFrame(), "message":"After filters, nothing remains."}

    X=live[[c for c in live.columns if c in feats]].values
    proba=calib.predict(model.predict_proba(X)[:,1])
    live["prob_TP_first"]=proba
    live["vol_ratio"]=live["atr14"]/live["close"]
    net_edge=(tp*live["prob_TP_first"]) - (sl*(1-live["prob_TP_first"]))
    live["net_edge_%"]=100.0*(net_edge - (parse_percent_from_val(fee_pct) or 0.0006))

    if prob_floor is not None:
        live=live[live["prob_TP_first"]>=float(prob_floor)]
    if live.empty:
        return {"status":"below_floor","table":pd.DataFrame(), "message":"No symbols meet probability floor."}

    live=live.sort_values(["prob_TP_first","net_edge_%"], ascending=[False,False]).reset_index(drop=True)
    tbl=live[["symbol","prob_TP_first","net_edge_%","ema20","ema50","atr14","vol_ratio"]].copy()
    tbl["prob_TP_first"]=(tbl["prob_TP_first"]*100).round(2)
    tbl["net_edge_%"]=tbl["net_edge_%"].round(2)
    for col in ["ema20","ema50","atr14","vol_ratio"]:
        tbl[col]=tbl[col].astype(float).round(6 if col!="vol_ratio" else 4)

    now = datetime.now(timezone.utc).astimezone(AFRICA_LAGOS).strftime("%Y-%m-%d %H:%M")
    head = f"<b>Hourly Scan</b> • {timeframe} • {now} (Africa/Lagos)"
    lines = [head, "Top candidates:"]
    for i,(sym,prob,edge) in enumerate(tbl[["symbol","prob_TP_first","net_edge_%"]].head(max_symbols_in_message).itertuples(index=False), start=1):
        lines.append(f"{i}. <code>{sym}</code> • P={prob:.2f}% • Edge={edge:.2f}%")
    msg = "\n".join(lines)

    return {"status":"ok","table":tbl, "message":msg}
  
