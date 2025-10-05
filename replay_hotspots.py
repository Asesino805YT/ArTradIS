"""Replay de hotspots: para cada hotspot seleccionado (both confirmers OK prioritarios),
reproduce el snapshot y ejecuta todas las estrategias para ver qué señal devuelven.
Escribe `hotspot_replay.txt` con el detalle por hotspot.

Uso: python replay_hotspots.py --rsi-period 7
"""
from __future__ import annotations
import os
import pandas as pd
from typing import List, Dict, Any

DATA_DIR = 'data'
OUT = 'hotspot_replay.txt'


def load_df(tf: str):
    p = os.path.join(DATA_DIR, f'{tf}.csv')
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    return df


def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -delta.clip(upper=0).rolling(period).mean()
    rs = up / down
    return 100 - (100 / (1 + rs))


def find_candidates(df1m, df15, df1h, rsi_period=14):
    df1m = df1m.copy()
    df1m['rsi'] = compute_rsi(df1m['close'], rsi_period)
    if df15 is not None:
        df15 = df15.copy()
        df15['ema50'] = df15['close'].ewm(span=50, adjust=False).mean()
    if df1h is not None:
        df1h = df1h.copy()
        df1h['ema200'] = df1h['close'].ewm(span=200, adjust=False).mean()

    candidates = []
    n = len(df1m)
    for i in range(2, n):
        try:
            r2 = float(df1m['rsi'].iat[i-2])
            r1 = float(df1m['rsi'].iat[i-1])
            r0 = float(df1m['rsi'].iat[i])
        except Exception:
            continue
        if r2 < 30 and r1 < 30 and r0 > 30:
            conf15 = 'na'
            conf1h = 'na'
            idx15 = min(int(i/15), len(df15)-1) if df15 is not None and len(df15)>0 else None
            idx1h = min(int(i/60), len(df1h)-1) if df1h is not None and len(df1h)>0 else None
            try:
                if idx15 is not None:
                    price15 = float(df15['close'].iat[idx15]); ema50_15 = float(df15['ema50'].iat[idx15])
                    conf15 = 'ok' if price15 < ema50_15 else 'fail'
                if idx1h is not None:
                    price1h = float(df1h['close'].iat[idx1h]); ema200_1h = float(df1h['ema200'].iat[idx1h])
                    conf1h = 'ok' if price1h > ema200_1h else 'fail'
            except Exception:
                pass
            candidates.append({'index': i, 'type': 'buy', 'conf15': conf15, 'conf1h': conf1h})
        if r2 > 70 and r1 > 70 and r0 < 70:
            conf15 = 'na'
            conf1h = 'na'
            idx15 = min(int(i/15), len(df15)-1) if df15 is not None and len(df15)>0 else None
            idx1h = min(int(i/60), len(df1h)-1) if df1h is not None and len(df1h)>0 else None
            try:
                if idx15 is not None:
                    price15 = float(df15['close'].iat[idx15]); ema50_15 = float(df15['ema50'].iat[idx15])
                    conf15 = 'ok' if price15 > ema50_15 else 'fail'
                if idx1h is not None:
                    price1h = float(df1h['close'].iat[idx1h]); ema200_1h = float(df1h['ema200'].iat[idx1h])
                    conf1h = 'ok' if price1h < ema200_1h else 'fail'
            except Exception:
                pass
            candidates.append({'index': i, 'type': 'sell', 'conf15': conf15, 'conf1h': conf1h})
    return candidates


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--rsi-period', type=int, default=14, help='RSI period to use when locating hotspots')
    p.add_argument('--history', type=int, default=200, help='Number of 1m bars to include before candidate index (snapshot window)')
    args = p.parse_args()

    df1m = load_df('1m')
    # load 5m if present; create 15m approximation if no explicit 15m file
    df5m = load_df('5m')
    df15 = load_df('15m')
    if df15 is None and df5m is not None:
        try:
            # crude aggregation: take every 3rd 5m bar as 15m snapshot
            df15 = df5m.iloc[::3].reset_index(drop=True)
        except Exception:
            df15 = None
    df1h = load_df('1h')
    if df1m is None:
        print('data/1m.csv no encontrado, corre backtest_pipeline.py --download primero')
        raise SystemExit(1)

    candidates = find_candidates(df1m, df15, df1h, rsi_period=args.rsi_period)
    # select hotspots (both ok first)
    both_ok = [c for c in candidates if c['conf15']=='ok' and c['conf1h']=='ok']
    selected = both_ok[:12] if len(both_ok)>=12 else (both_ok + [c for c in candidates if (c['conf15']=='ok' or c['conf1h']=='ok')][:12-len(both_ok)])

    # import strategies
    try:
        from ArTradIS import StrategyFactory, TrendFollowingAdvanced, RangeTradingAdvanced, ScalpingAdvanced, MultipleTimeFrameStrategy, MovingAverageStrategy, BollingerBandsStrategy, RSIStrategy, MACDStrategy, Config
    except Exception as e:
        print('No se pudieron importar estrategias desde ArTradIS.py:', e)
        raise

    cfg = Config()
    # align config rsi_period with CLI
    try:
        cfg.rsi_period = args.rsi_period
    except Exception:
        pass
    strategies = {
        **{name: cls(cfg) for name, cls in StrategyFactory.items()},
        'trend_adv': TrendFollowingAdvanced(cfg),
        'range_adv': RangeTradingAdvanced(cfg),
        'scalp_adv': ScalpingAdvanced(cfg),
        'mtf_adv': MultipleTimeFrameStrategy(cfg),
        'mtf': MultipleTimeFrameStrategy(cfg),
        'ma_adv': MovingAverageStrategy(cfg),
        'bb_adv': BollingerBandsStrategy(cfg),
        'rsi_adv': RSIStrategy(cfg),
        'macd_adv': MACDStrategy(cfg),
    }

    # prepare report
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('Hotspot replay report\n\n')
        f.write(f'RSI period: {args.rsi_period}\n')
        f.write(f'Total candidates: {len(candidates)} selected: {len(selected)}\n\n')
        for c in selected:
            i = c['index']
            f.write(f"=== Candidate index {i} type={c.get('type')} conf15={c.get('conf15')} conf1h={c.get('conf1h')} ===\n")
            # build snapshot (include history bars before index)
            snap = {}
            for tf, df in [('1m', df1m), ('5m', df15), ('1h', df1h)]:
                if df is None:
                    snap[tf] = None
                    continue
                try:
                    lo = max(0, i - args.history + 1)
                    snap[tf] = df.iloc[lo:i+1].reset_index(drop=True)
                except Exception:
                    snap[tf] = df
            # preprocess snapshots so strategies see indicators
            try:
                from ArTradIS import Indicators
                for tf in ('1m', '5m', '1h'):
                    if snap.get(tf) is not None:
                        try:
                            snap[tf] = Indicators.compute(snap[tf])
                        except Exception:
                            pass
            except Exception:
                pass
            # run each strategy
            f.write('Strategy results:\n')
            from ArTradIS import SignalAggregator
            counts = {'buy': 0, 'sell': 0, 'none': 0}
            for name, strat in strategies.items():
                try:
                    sig = strat.analyze(snap)
                    f.write(f"{name}: {sig}\n")
                except Exception as e:
                    sig = None
                    import traceback
                    tb = traceback.format_exc()
                    f.write(f"{name}: ERROR -> {e}\n")
                    f.write(tb + "\n")
                counts['none' if sig is None else sig] = counts.get('none' if sig is None else sig, 0) + 1
            # aggregator decision
            try:
                agg = SignalAggregator(strategies)
                agg_sig, conf, confirmers = agg.aggregate(snap)
            except Exception:
                agg_sig, conf, confirmers = (None, 0.0, [])
            f.write(f"Aggregator: {agg_sig} conf={conf:.2f} confirmers={confirmers}\n")
            f.write('\n')

    print('Wrote', OUT)
    print('Selected candidates:', len(selected))
