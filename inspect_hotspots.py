"""Inspecciona las 'zonas calientes' encontradas por analyze_signals.py.
Selecciona candidatos donde ambos confirmers ('conf15' y 'conf1h') sean 'ok' y extrae
contexto de 10 barras antes y después en 1m, además del último bar 15m y 1h previo.
Genera `signal_hotspots.txt` con la información.
"""
from __future__ import annotations
import os
import pandas as pd
import numpy as np
from typing import List

DATA_DIR = 'data'
OUT = 'signal_hotspots.txt'


def load_and_prepare(tf: str):
    p = os.path.join(DATA_DIR, f'{tf}.csv')
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    # coerce types
    df['close'] = df['close'].astype(float)
    try:
        df['rsi'] = df['close'].diff().clip(lower=0).rolling(14).mean()
    except Exception:
        pass
    try:
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    except Exception:
        pass
    return df


def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -delta.clip(upper=0).rolling(period).mean()
    rs = up / down
    return 100 - (100 / (1 + rs))


def find_candidates(df1m, df15, df1h):
    candidates = []
    df1m = df1m.copy()
    df1m['rsi'] = compute_rsi(df1m['close'])
    if df15 is not None:
        df15 = df15.copy()
        df15['ema50'] = df15['close'].ewm(span=50, adjust=False).mean()
    if df1h is not None:
        df1h = df1h.copy()
        df1h['ema200'] = df1h['close'].ewm(span=200, adjust=False).mean()

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
            # nearest by index fallback
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
            candidates.append({'index': i, 'type': 'buy', 'rsi': (r2,r1,r0), 'conf15': conf15, 'conf1h': conf1h})
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
            candidates.append({'index': i, 'type': 'sell', 'rsi': (r2,r1,r0), 'conf15': conf15, 'conf1h': conf1h})
    return candidates


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--rsi-period', type=int, default=14, help='RSI period to use in 1m')
    args = p.parse_args()

    df1m = load_and_prepare('1m')
    df15 = load_and_prepare('5m')
    df1h = load_and_prepare('1h')
    if df1m is None:
        print('data/1m.csv no encontrado. Ejecuta backtest_pipeline.py --download primero')
        raise SystemExit(1)
    candidates = find_candidates(df1m, df15, df1h)
    # filter those with both ok
    both_ok = [c for c in candidates if c['conf15']=='ok' and c['conf1h']=='ok']
    # if fewer than 6, relax to those with at least one ok
    selected = both_ok[:6] if len(both_ok)>=6 else (both_ok + [c for c in candidates if (c['conf15']=='ok' or c['conf1h']=='ok')][:6-len(both_ok)])

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('Hotspots report\n\n')
        f.write(f'Total candidates: {len(candidates)}\n')
        f.write(f'Both confirmers OK: {len(both_ok)}\n\n')
        for c in selected:
            i = c['index']
            f.write(f"Candidate index {i} type={c['type']} rsi={c['rsi']} conf15={c['conf15']} conf1h={c['conf1h']}\n")
            lo = max(0, i-10)
            hi = min(len(df1m)-1, i+10)
            f.write('1m context (index, close, rsi, ema20, macd, macd_signal):\n')
            for k in range(lo, hi+1):
                row = df1m.iloc[k]
                close_v = row.get('close')
                rsi_v = row.get('rsi')
                ema20_v = row.get('ema20') if 'ema20' in row else None
                macd_v = row.get('macd') if 'macd' in row else None
                macd_sig_v = row.get('macd_signal') if 'macd_signal' in row else None
                close_s = f"{close_v:.6f}" if close_v is not None else 'nan'
                rsi_s = f"{rsi_v:.4f}" if (rsi_v is not None and pd.notna(rsi_v)) else 'nan'
                ema20_s = f"{ema20_v:.6f}" if ema20_v is not None else 'na'
                macd_s = f"{macd_v:.6f}" if macd_v is not None else 'na'
                macd_sig_s = f"{macd_sig_v:.6f}" if macd_sig_v is not None else 'na'
                f.write(f"{k}, {close_s}, {rsi_s}, {ema20_s}, {macd_s}, {macd_sig_s}\n")
            # 15m and 1h last bar info
            try:
                idx15 = min(int(i/15), len(df15)-1) if df15 is not None else None
                idx1h = min(int(i/60), len(df1h)-1) if df1h is not None else None
                if idx15 is not None:
                    r15 = df15.iloc[idx15]
                    f.write(f"15m last at idx {idx15}: close={r15.get('close'):.6f}, ema50={r15.get('ema50'):.6f}\n")
                if idx1h is not None:
                    r1h = df1h.iloc[idx1h]
                    f.write(f"1h last at idx {idx1h}: close={r1h.get('close'):.6f}, ema200={r1h.get('ema200'):.6f}\n")
            except Exception:
                pass
            f.write('\n---\n\n')

    print('Wrote', OUT)
    print('selected candidates:', len(selected))
    for c in selected:
        print(c)
