"""Proxy RSI-only probe.

Reproduce los hotspots listados en `signal_hotspots.txt`, reconstruye snapshots
como hace `replay_hotspots.py`, ejecuta un checker RSI-only (dip->rise y rise->drop)
y compara con la salida de `RSIStrategy` desde `ArTradIS.py` si está disponible.

Genera `hotspot_probe_report.txt` con el resultado por candidato.

Uso: python probe_rsi_strategy.py --rsi-period 7
"""
from __future__ import annotations
import os
import pandas as pd
import argparse
from typing import Optional

DATA_DIR = 'data'
IN = 'signal_hotspots.txt'
OUT = 'hotspot_probe_report.txt'


def load_df(tf: str) -> Optional[pd.DataFrame]:
    p = os.path.join(DATA_DIR, f'{tf}.csv')
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -delta.clip(upper=0).rolling(period).mean()
    rs = up / down
    return 100 - (100 / (1 + rs))


def parse_signal_hotspots(path: str):
    """Extrae los índices de candidate del archivo `signal_hotspots.txt`.
    Devuelve una lista de dicts con keys: index, type, conf15, conf1h
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Candidate index'):
                # formato: Candidate index 61 type=buy rsi=(...), conf15=ok conf1h=ok
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


def rsi_proxy_check(df1m: pd.DataFrame, idx: int, period: int) -> Optional[str]:
    """Aplica la regla dip->rise y rise->drop sobre df1m hasta idx (inclusive).
    Devuelve 'buy'|'sell'|None
    """
    if idx < 2 or len(df1m) <= idx:
        return None
    rsi = compute_rsi(df1m['close'], period)
    try:
        r2 = float(rsi.iat[idx-2])
        r1 = float(rsi.iat[idx-1])
        r0 = float(rsi.iat[idx])
    except Exception:
        return None
    if r2 < 30 and r1 < 30 and r0 > 30:
        return 'buy'
    if r2 > 70 and r1 > 70 and r0 < 70:
        return 'sell'
    return None


def diagnostic_values(snap: dict, period: int):
    """Return dict with r0,r1,r2 and 1h bias computed similarly to RSIStrategy."""
    res = {'r0': None, 'r1': None, 'r2': None, 'bias': None}
    df1m = snap.get('1m')
    if df1m is not None and len(df1m) >= 3:
        try:
            # try to use ArTradIS Indicators.safe_rsi if available
            try:
                from ArTradIS import Indicators
                rsi = Indicators.safe_rsi(df1m['close'], period)
            except Exception:
                rsi = compute_rsi(df1m['close'], period)
            res['r0'] = float(rsi.iat[-1] if hasattr(rsi, 'iat') else rsi[-1])
            res['r1'] = float(rsi.iat[-2] if hasattr(rsi, 'iat') else rsi[-2])
            res['r2'] = float(rsi.iat[-3] if hasattr(rsi, 'iat') else rsi[-3])
        except Exception:
            pass
    # bias from 1h
    df1h = snap.get('1h')
    if df1h is not None and len(df1h) > 0:
        try:
            close1h = df1h['close']
            # compute ema200 (or rolling mean fallback)
            ema200 = None
            try:
                import talib
                ema200 = talib.EMA(close1h.astype(float), timeperiod=200)
            except Exception:
                try:
                    ema200 = close1h.rolling(200).mean()
                except Exception:
                    ema200 = None
            if ema200 is not None:
                ema200_last = float(ema200.iat[-1] if hasattr(ema200, 'iat') else ema200[-1])
                price1h = float(df1h.iloc[-1]['close'])
                res['bias'] = 'bull' if price1h > ema200_last else 'bear'
        except Exception:
            pass
    return res


def build_snapshot(df: pd.DataFrame, idx: int):
    try:
        return df.iloc[:idx+1].reset_index(drop=True)
    except Exception:
        return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rsi-period', type=int, default=14)
    p.add_argument('--history', type=int, default=200, help='Number of 1m bars to include before candidate index')
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
        print('data/1m.csv no encontrado. Ejecuta backtest_pipeline.py --download primero')
        raise SystemExit(1)

    candidates = parse_signal_hotspots(IN)
    # try import RSIStrategy
    RSIStrategy = None
    try:
        from ArTradIS import RSIStrategy, Config, Indicators
    except Exception:
        RSIStrategy = None
    cfg = None
    if RSIStrategy is not None:
        try:
            cfg = Config()
            cfg.rsi_period = args.rsi_period
        except Exception:
            cfg = None

    reports = []
    for c in candidates:
        idx = c['index']
        proxy = rsi_proxy_check(df1m, idx, args.rsi_period)
        # build snapshot with history window
        snap = {}
        lo = max(0, idx - args.history + 1)
        snap['1m'] = df1m.iloc[lo:idx+1].reset_index(drop=True)
        snap['15m'] = (df15.iloc[:min(int(idx/15)+1, len(df15))].reset_index(drop=True) if df15 is not None and len(df15)>0 else None)
        snap['1h'] = (df1h.iloc[:min(int(idx/60)+1, len(df1h))].reset_index(drop=True) if df1h is not None and len(df1h)>0 else None)
        # enrich indicators
        try:
            from ArTradIS import Indicators as _Indicators
            for tf in ('1m', '15m', '1h'):
                if snap.get(tf) is not None:
                    try:
                        snap[tf] = _Indicators.compute(snap[tf])
                    except Exception:
                        pass
        except Exception:
            pass

        # diagnostic values (r0,r1,r2,bias)
        diag = diagnostic_values(snap, args.rsi_period)

        rsi_strategy_out = None
        if RSIStrategy is not None:
            try:
                inst = RSIStrategy(cfg)
                rsi_strategy_out = inst.analyze(snap)
            except Exception:
                rsi_strategy_out = None

        reports.append({'index': idx, 'proxy': proxy, 'rsi_strategy': rsi_strategy_out, 'conf15': c.get('conf15'), 'conf1h': c.get('conf1h'), 'diag': diag})

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('Hotspot RSI proxy probe report\n\n')
        f.write(f'RSI period: {args.rsi_period}\n')
        f.write(f'Total candidates parsed: {len(candidates)}\n\n')
        for r in reports:
            d = r.get('diag', {})
            f.write(f"Index {r['index']}: proxy={r['proxy']} rsi_strategy={r['rsi_strategy']} conf15={r['conf15']} conf1h={r['conf1h']}\n")
            f.write(f"  diag: r2={d.get('r2')} r1={d.get('r1')} r0={d.get('r0')} bias={d.get('bias')}\n")

    print('Wrote', OUT)


if __name__ == '__main__':
    main()
