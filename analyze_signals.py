"""Analizador de señales: lee data/1m.csv, 15m.csv y 1h.csv, calcula indicadores y busca las barras donde
la condición de entrada de TrendFollowingAdvanced (RSI en 1m: dip <30 then rise >30 para buy; >70 then drop <70 para sell)
se cumple. Genera reportes indicando qué confirmers (1h EMA200 y 15m EMA50) se cumplen o faltan.

Salida: signal_analysis_report.txt y resumen por pantalla.
"""
from __future__ import annotations
import os
import pandas as pd
import numpy as np
from typing import Dict

DATA_DIR = 'data'
REPORT = 'signal_analysis_report.txt'


def load_csv(tf: str):
    p = os.path.join(DATA_DIR, f'{tf}.csv')
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    return df


def compute_indicators(df: pd.DataFrame, rsi_period: int = 14):
    out = df.copy()
    out['close'] = out['close'].astype(float)
    out['rsi'] = None
    try:
        # simple RSI implementation
        delta = out['close'].diff()
        up = delta.clip(lower=0).rolling(rsi_period).mean()
        down = -delta.clip(upper=0).rolling(rsi_period).mean()
        rs = up / down
        out['rsi'] = 100 - (100 / (1 + rs))
    except Exception:
        out['rsi'] = 50.0
    try:
        out['ema20'] = out['close'].ewm(span=20, adjust=False).mean()
        out['ema50'] = out['close'].ewm(span=50, adjust=False).mean()
        out['ema200'] = out['close'].ewm(span=200, adjust=False).mean()
    except Exception:
        pass
    # MACD
    try:
        ema12 = out['close'].ewm(span=12, adjust=False).mean()
        ema26 = out['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        out['macd'] = macd
        out['macd_signal'] = signal
        out['macd_hist'] = macd - signal
    except Exception:
        out['macd'] = 0.0
        out['macd_signal'] = 0.0
        out['macd_hist'] = 0.0
    return out


def find_rsi_crosses(df1m: pd.DataFrame, df15: pd.DataFrame, df1h: pd.DataFrame):
    """Return list of dicts describing candidate signals found in 1m where RSI dip->rise or rise->drop occurs.
    For each candidate, evaluate 15m EMA50 and 1h EMA200 confirmers using the closest prior bar.
    """
    candidates = []
    # ensure indices are integer positions
    n = len(df1m)
    for i in range(2, n):
        try:
            r2 = float(df1m['rsi'].iat[i-2])
            r1 = float(df1m['rsi'].iat[i-1])
            r0 = float(df1m['rsi'].iat[i])
        except Exception:
            continue
        # buy condition
        if r2 < 30 and r1 < 30 and r0 > 30:
            ts = df1m.index[i]
            price = float(df1m['close'].iat[i])
            # find nearest 15m bar (approx every 15 rows) by timestamp if present, else nearest index
            conf15 = 'unknown'
            conf1h = 'unknown'
            try:
                # if timestamp column exists, try to map by timestamp
                if 'timestamp' in df1m.columns and 'timestamp' in df15.columns and 'timestamp' in df1h.columns:
                    t = pd.to_datetime(df1m['timestamp'].iat[i])
                    # find last 15m <= t
                    t15 = pd.to_datetime(df15['timestamp'])
                    t1h = pd.to_datetime(df1h['timestamp'])
                    idx15 = t15[t15 <= t]
                    idx1h = t1h[t1h <= t]
                    if len(idx15):
                        last15 = idx15.index[-1]
                        ema50_15 = float(df15['ema50'].iat[last15]) if 'ema50' in df15.columns else None
                        price15 = float(df15['close'].iat[last15])
                        conf15 = 'ok' if (price15 < ema50_15) else 'fail' if ema50_15 is not None else 'na'
                    if len(idx1h):
                        last1h = idx1h.index[-1]
                        ema200_1h = float(df1h['ema200'].iat[last1h]) if 'ema200' in df1h.columns else None
                        price1h = float(df1h['close'].iat[last1h])
                        conf1h = 'ok' if (price1h > ema200_1h) else 'fail' if ema200_1h is not None else 'na'
                else:
                    # fallback: use nearest indexes by scaling
                    idx15 = int(i / 15)
                    idx1h = int(i / 60)
                    ema50_15 = float(df15['ema50'].iat[min(idx15, len(df15)-1)]) if 'ema50' in df15.columns else None
                    price15 = float(df15['close'].iat[min(idx15, len(df15)-1)])
                    conf15 = 'ok' if (price15 < ema50_15) else 'fail' if ema50_15 is not None else 'na'
                    ema200_1h = float(df1h['ema200'].iat[min(idx1h, len(df1h)-1)]) if 'ema200' in df1h.columns else None
                    price1h = float(df1h['close'].iat[min(idx1h, len(df1h)-1)])
                    conf1h = 'ok' if (price1h > ema200_1h) else 'fail' if ema200_1h is not None else 'na'
            except Exception:
                pass
            candidates.append({'index': i, 'type': 'buy', 'rsi': (r2, r1, r0), 'price': price, 'conf15': conf15, 'conf1h': conf1h})
        # sell condition
        if r2 > 70 and r1 > 70 and r0 < 70:
            ts = df1m.index[i]
            price = float(df1m['close'].iat[i])
            conf15 = 'unknown'
            conf1h = 'unknown'
            try:
                if 'timestamp' in df1m.columns and 'timestamp' in df15.columns and 'timestamp' in df1h.columns:
                    t = pd.to_datetime(df1m['timestamp'].iat[i])
                    t15 = pd.to_datetime(df15['timestamp'])
                    t1h = pd.to_datetime(df1h['timestamp'])
                    idx15 = t15[t15 <= t]
                    idx1h = t1h[t1h <= t]
                    if len(idx15):
                        last15 = idx15.index[-1]
                        ema50_15 = float(df15['ema50'].iat[last15]) if 'ema50' in df15.columns else None
                        price15 = float(df15['close'].iat[last15])
                        conf15 = 'ok' if (price15 > ema50_15) else 'fail' if ema50_15 is not None else 'na'
                    if len(idx1h):
                        last1h = idx1h.index[-1]
                        ema200_1h = float(df1h['ema200'].iat[last1h]) if 'ema200' in df1h.columns else None
                        price1h = float(df1h['close'].iat[last1h])
                        conf1h = 'ok' if (price1h < ema200_1h) else 'fail' if ema200_1h is not None else 'na'
                else:
                    idx15 = int(i / 15)
                    idx1h = int(i / 60)
                    ema50_15 = float(df15['ema50'].iat[min(idx15, len(df15)-1)]) if 'ema50' in df15.columns else None
                    price15 = float(df15['close'].iat[min(idx15, len(df15)-1)])
                    conf15 = 'ok' if (price15 > ema50_15) else 'fail' if ema50_15 is not None else 'na'
                    ema200_1h = float(df1h['ema200'].iat[min(idx1h, len(df1h)-1)]) if 'ema200' in df1h.columns else None
                    price1h = float(df1h['close'].iat[min(idx1h, len(df1h)-1)])
                    conf1h = 'ok' if (price1h < ema200_1h) else 'fail' if ema200_1h is not None else 'na'
            except Exception:
                pass
            candidates.append({'index': i, 'type': 'sell', 'rsi': (r2, r1, r0), 'price': price, 'conf15': conf15, 'conf1h': conf1h})
    return candidates


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--rsi-period', type=int, default=14, help='RSI period to use in 1m analysis')
    args = p.parse_args()

    df1m = load_csv('1m')
    df15 = load_csv('5m')
    df1h = load_csv('1h')
    if df1m is None:
        print('No existe data/1m.csv — ejecuta backtest_pipeline.py --download primero')
        raise SystemExit(1)
    # compute indicators where missing
    df1m = compute_indicators(df1m, rsi_period=args.rsi_period)
    if df15 is not None:
        df15 = compute_indicators(df15, rsi_period=args.rsi_period)
    if df1h is not None:
        df1h = compute_indicators(df1h, rsi_period=args.rsi_period)

    candidates = find_rsi_crosses(df1m, df15 if df15 is not None else pd.DataFrame(), df1h if df1h is not None else pd.DataFrame())

    # write report
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('Signal analysis report\n')
        f.write('\nTotal candidates found: %d\n' % len(candidates))
        f.write('\nFirst 50 candidates:\n')
        for c in candidates[:50]:
            f.write(str(c) + '\n')

    print('Analysis complete. Candidates found:', len(candidates))
    print('Wrote', REPORT)
