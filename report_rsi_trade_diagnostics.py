"""Report de diagnostics para RSIStrategy por trade.

Uso: ejecutar desde la raíz del repo con el virtualenv:
    python report_rsi_trade_diagnostics.py --rsi-period 7 --trades trades_rsi7.csv --out report_rsi7.txt

Genera un fichero de texto con, por cada trade registrado, el periodo RSI usado,
los valores r2,r1,r0 calculados en 1m y la decisión final retornada por RSIStrategy.
"""
from __future__ import annotations
import argparse
import os
import json
from typing import Optional
import pandas as pd

# intentar importar ArTradIS helpers
try:
    from ArTradIS import Indicators, RSIStrategy, Config
except Exception:
    Indicators = None
    RSIStrategy = None
    Config = None


def load_df(tf: str) -> Optional[pd.DataFrame]:
    p = os.path.join('data', f'{tf}.csv')
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


def find_index_for_timestamp(df: pd.DataFrame, ts_value: str) -> Optional[int]:
    # try exact match on 'timestamp' column
    if 'timestamp' in df.columns:
        try:
            matched = df.index[df['timestamp'].astype(str) == str(ts_value)]
            if len(matched) > 0:
                return int(matched[0])
        except Exception:
            pass
        # try parseable datetimes (fallback)
        try:
            ser = pd.to_datetime(df['timestamp'], errors='coerce')
            t = pd.to_datetime(ts_value, errors='coerce')
            if pd.notna(t):
                # find nearest index with timestamp <= t
                le = ser[ser <= t]
                if len(le) > 0:
                    return int(le.index[-1])
        except Exception:
            pass
    # try interpret ts_value as int index
    try:
        ii = int(float(ts_value))
        if 0 <= ii < len(df):
            return ii
    except Exception:
        pass
    # fallback: try to match by close price approx (less robust)
    try:
        # maybe ts_value is entry price included in trades file? Not used here.
        return None
    except Exception:
        return None


def compute_r_values(series_close: pd.Series, period: int):
    # use Indicators.safe_rsi if available, else naive implementation
    try:
        if Indicators is not None:
            rsi = Indicators.safe_rsi(series_close, period)
        else:
            # naive compute
            delta = series_close.diff()
            up = delta.clip(lower=0).rolling(period).mean()
            down = -delta.clip(upper=0).rolling(period).mean()
            rs = up / down
            rsi = 100 - (100 / (1 + rs))
    except Exception:
        # fallback
        delta = series_close.diff()
        up = delta.clip(lower=0).rolling(period).mean()
        down = -delta.clip(upper=0).rolling(period).mean()
        rs = up / down
        rsi = 100 - (100 / (1 + rs))
    # ensure length
    if len(rsi) < 3:
        return None, None, None
    try:
        r0 = float(rsi.iat[-1])
        r1 = float(rsi.iat[-2])
        r2 = float(rsi.iat[-3])
    except Exception:
        try:
            r0 = float(rsi.iloc[-1])
            r1 = float(rsi.iloc[-2])
            r2 = float(rsi.iloc[-3])
        except Exception:
            return None, None, None
    return r2, r1, r0


def compute_1h_bias(df1h: Optional[pd.DataFrame], idx_1m: int):
    # build a 1h slice aligned to 1m index (approx): row count for 1h is int(idx/60)+1
    if df1h is None or len(df1h) == 0:
        return None
    try:
        up_to = min(int(idx_1m / 60) + 1, len(df1h))
        slice1h = df1h.iloc[:up_to]
        close1h = slice1h['close']
    except Exception:
        return None
    # compute ema200
    try:
        import talib
        ema200 = talib.EMA(close1h.astype(float), timeperiod=200)
    except Exception:
        try:
            if len(close1h) >= 200:
                ema200 = close1h.rolling(200).mean()
            else:
                return None
        except Exception:
            return None
    try:
        ema_last = float(ema200.iat[-1] if hasattr(ema200, 'iat') else ema200[-1])
    except Exception:
        return None
    try:
        price1h = float(slice1h.iloc[-1]['close'])
    except Exception:
        return None
    if ema_last != ema_last:  # NaN
        return None
    return 'bull' if price1h > ema_last else 'bear'


def analyze_trades(trades_file: str, rsi_period: int, out_file: str, max_trades: int = 10):
    df1m = load_df('1m')
    df15 = load_df('15m')
    df1h = load_df('1h')
    if df15 is None and df1m is not None:
        try:
            df15 = df1m.iloc[::15].reset_index(drop=True)
        except Exception:
            df15 = None
    if df1h is None and df1m is not None:
        try:
            df1h = df1m.iloc[::60].reset_index(drop=True)
        except Exception:
            df1h = None

    if df1m is None:
        print('data/1m.csv no encontrado; no puedo reconstruir snapshots')
        return

    trades = pd.read_csv(trades_file)
    out = []
    count = 0
    for _, row in trades.iterrows():
        if count >= max_trades:
            break
        ts = row.get('timestamp')
        idx = find_index_for_timestamp(df1m, ts)
        if idx is None:
            # try to match by entry price
            entry = row.get('entry')
            try:
                cand = df1m.index[(df1m['close'].astype(float) - float(entry)).abs() < 1e-6]
                if len(cand) > 0:
                    idx = int(cand[0])
            except Exception:
                pass
        if idx is None:
            # skip if cannot map
            out.append({'timestamp': ts, 'note': 'no index mapping'})
            count += 1
            continue
        # build snapshot up to idx
        snap = {}
        snap['1m'] = df1m.iloc[:idx+1].reset_index(drop=True)
        snap['15m'] = (df15.iloc[:min(int(idx/15)+1, len(df15))].reset_index(drop=True) if df15 is not None and len(df15)>0 else None)
        snap['1h'] = (df1h.iloc[:min(int(idx/60)+1, len(df1h))].reset_index(drop=True) if df1h is not None and len(df1h)>0 else None)

        r2, r1, r0 = compute_r_values(snap['1m']['close'], rsi_period)
        bias = compute_1h_bias(snap['1h'], idx)
        # decision from RSIStrategy if available
        decision = None
        try:
            if RSIStrategy is not None and Config is not None:
                cfg = Config()
                cfg.rsi_period = rsi_period
                inst = RSIStrategy(cfg)
                decision = inst.analyze(snap)
        except Exception:
            decision = None

        out.append({'timestamp': ts, 'index_1m': idx, 'rsi_period': rsi_period, 'r2': r2, 'r1': r1, 'r0': r0, 'bias': bias, 'decision': decision, 'entry': row.get('entry')})
        count += 1

    # save report
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('Wrote', out_file)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rsi-period', type=int, required=True)
    p.add_argument('--trades', type=str, required=True)
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--max-trades', type=int, default=4)
    args = p.parse_args()
    analyze_trades(args.trades, args.rsi_period, args.out, max_trades=args.max_trades)
