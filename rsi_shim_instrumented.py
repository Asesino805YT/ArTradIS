"""Instrumented shim that replicates RSIStrategy.analyze logic but prints reasons.

Usage: python rsi_shim_instrumented.py --rsi-period 7 --history 200
"""
from __future__ import annotations
import os
import pandas as pd
import argparse
from typing import Any, Dict

DATA_DIR = 'data'
IN = 'signal_hotspots.txt'
OUT = 'rsi_shim_report.txt'


def load_df(tf: str):
    p = os.path.join(DATA_DIR, f'{tf}.csv')
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -delta.clip(upper=0).rolling(period).mean()
    rs = up / down
    return 100 - (100 / (1 + rs))


def parse_candidates(path: str):
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('Candidate index'):
                parts = line.split()
                try:
                    idx = int(parts[2])
                except Exception:
                    continue
                typ = 'unknown'
                conf15 = 'na'
                conf1h = 'na'
                for tok in parts[3:]:
                    if tok.startswith('type='):
                        typ = tok.split('=', 1)[1]
                    if tok.startswith('conf15='):
                        conf15 = tok.split('=', 1)[1]
                    if tok.startswith('conf1h='):
                        conf1h = tok.split('=', 1)[1]
                out.append({'index': idx, 'type': typ, 'conf15': conf15, 'conf1h': conf1h})
    return out


def shim_analyze(snap: Dict[str, Any], rsi_period: int):
    """Replicate RSIStrategy.analyze and return (decision, reasons dict)."""
    reasons = []
    df = snap.get('1m') if snap.get('1m') is not None else next((v for v in snap.values() if v is not None), None)
    if df is None:
        reasons.append('no data')
        return None, reasons
    close = df['close']
    # compute rsi
    try:
        import talib
        rsi = talib.RSI(close.astype(float), timeperiod=rsi_period)
        reasons.append('used talib')
    except Exception:
        rsi = compute_rsi(close, rsi_period)
        reasons.append('used fallback safe_rsi')
    if len(rsi) < 3:
        reasons.append('rsi length < 3')
        return None, reasons
    r0 = float(rsi.iat[-1] if hasattr(rsi, 'iat') else rsi[-1])
    r1 = float(rsi.iat[-2] if hasattr(rsi, 'iat') else rsi[-2])
    r2 = float(rsi.iat[-3] if hasattr(rsi, 'iat') else rsi[-3])
    reasons.append(f'r2={r2:.4f} r1={r1:.4f} r0={r0:.4f}')

    # bias from 1h
    bias = None
    df1h = snap.get('1h')
    if df1h is not None and len(df1h)>0:
        try:
            close1h = df1h['close']
            try:
                import talib
                ema200 = talib.EMA(close1h.astype(float), timeperiod=200)
            except Exception:
                ema200 = close1h.rolling(200).mean()
            if ema200 is not None:
                ema200_last = float(ema200.iat[-1] if hasattr(ema200, 'iat') else ema200[-1])
                price1h = float(df1h.iloc[-1]['close'])
                bias = 'bull' if price1h > ema200_last else 'bear'
                reasons.append(f'bias={bias} price1h={price1h:.6f} ema200_last={ema200_last:.6f}')
        except Exception:
            reasons.append('bias_check_failed')

    # reversal detection
    if r2 < 30 and r1 < 30 and r0 > 30:
        if bias is None or bias == 'bull' or r0 > 50:
            reasons.append('buy_condition_met')
            return 'buy', reasons
        else:
            reasons.append('buy_condition_met_but_bias_failed')
            return None, reasons
    if r2 > 70 and r1 > 70 and r0 < 70:
        if bias is None or bias == 'bear' or r0 < 50:
            reasons.append('sell_condition_met')
            return 'sell', reasons
        else:
            reasons.append('sell_condition_met_but_bias_failed')
            return None, reasons

    reasons.append('no_reversal_detected')
    return None, reasons


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rsi-period', type=int, default=14)
    p.add_argument('--history', type=int, default=200)
    args = p.parse_args()

    df1m = load_df('1m')
    df5m = load_df('5m')
    df15 = load_df('15m')
    if df15 is None and df5m is not None:
        try:
            df15 = df5m.iloc[::3].reset_index(drop=True)
        except Exception:
            df15 = None
    df1h = load_df('1h')
    if df1m is None:
        print('data/1m.csv missing')
        return

    candidates = parse_candidates(IN)
    reports = []
    for c in candidates:
        idx = c['index']
        lo = max(0, idx - args.history + 1)
        snap = {
            '1m': df1m.iloc[lo:idx+1].reset_index(drop=True),
            '15m': (df15.iloc[:min(int(idx/15)+1, len(df15))].reset_index(drop=True) if df15 is not None and len(df15)>0 else None),
            '1h': (df1h.iloc[:min(int(idx/60)+1, len(df1h))].reset_index(drop=True) if df1h is not None and len(df1h)>0 else None)
        }
        # compute indicators via ArTradIS if available
        try:
            from ArTradIS import Indicators
            for tf in ('1m','15m','1h'):
                if snap.get(tf) is not None:
                    try:
                        snap[tf] = Indicators.compute(snap[tf])
                    except Exception:
                        pass
        except Exception:
            pass

        decision, reasons = shim_analyze(snap, args.rsi_period)
        reports.append({'index': idx, 'decision': decision, 'reasons': reasons, 'conf15': c.get('conf15'), 'conf1h': c.get('conf1h')})

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('RSI shim instrumented report\n\n')
        f.write(f'RSI period: {args.rsi_period} history: {args.history}\n')
        for r in reports:
            f.write(f"Index {r['index']} decision={r['decision']} conf15={r['conf15']} conf1h={r['conf1h']}\n")
            for reason in r['reasons']:
                f.write(f"  - {reason}\n")
            f.write('\n')

    print('Wrote', OUT)


if __name__ == '__main__':
    main()
