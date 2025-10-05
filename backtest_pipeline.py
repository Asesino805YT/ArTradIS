"""Backtest pipeline rápido para ArTradIS.
Genera datos sintéticos (reproducibles) y ejecuta StrategyEvaluator sobre las estrategias
registradas en `ArTradIS.py`. También prueba una pequeña malla de parámetros iniciales.

Uso: ejecutar desde la raíz del repo usando el Python del virtualenv.
"""
from __future__ import annotations
import os
import sys
import time
from types import SimpleNamespace
from typing import Any
import logging
import csv
from datetime import datetime

# intentar importar pandas, si no, abortar con mensaje claro
try:
    import pandas as pd
    import numpy as np
except Exception as e:
    print('Este script requiere pandas/numpy. Instala con pip install pandas numpy')
    raise

# Importa las clases desde ArTradIS
try:
    from ArTradIS import Config, StrategyEvaluator, StrategyFactory, TrendFollowingAdvanced, RangeTradingAdvanced, ScalpingAdvanced, MultipleTimeFrameStrategy, MovingAverageStrategy, BollingerBandsStrategy, RSIStrategy, MACDStrategy, Indicators, DerivClient
except Exception as e:
    print('No fue posible importar módulos desde ArTradIS.py:', e)
    raise

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('backtest_pipeline')


def make_synthetic_price_series(n=1000, seed=42, start=1000.0, vol=0.002):
    rng = np.random.RandomState(seed)
    returns = rng.normal(loc=0.0, scale=vol, size=n)
    price = start * np.exp(np.cumsum(returns))
    return price


def build_df_from_prices(prices):
    # build OHLC simple bars where open=prev close, high/low add tiny noise
    closes = prices
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    highs = np.maximum(opens, closes) * (1.0 + 0.0005)
    lows = np.minimum(opens, closes) * (1.0 - 0.0005)
    df = pd.DataFrame({'open': opens, 'high': highs, 'low': lows, 'close': closes})
    # add simple indicators used by strategies
    try:
        df['rsi'] = Indicators.safe_rsi(df['close'], 14)
    except Exception:
        # fallback naive rsi if Indicators fails
        df['rsi'] = 50.0
    try:
        df['sma50'] = df['close'].rolling(50).mean()
        df['sma200'] = df['close'].rolling(200).mean()
        df['sma'] = df['close'].rolling(20).mean()
    except Exception:
        pass
    return df


def _load_csv_df(path: str):
    try:
        df = pd.read_csv(path)
        # ensure required columns
        for c in ('open', 'high', 'low', 'close'):
            if c not in df.columns:
                raise ValueError(f'Missing column {c} in {path}')
        return df
    except Exception as e:
        raise


def prepare_multi(timeframes=('1h', '15m', '1m'), n=1000, data_dir: str = None, cfg: Any = None):
    """Prepare multi-timeframe data.

    Priority:
    - If `data_dir` provided and CSV files exist (data/1h.csv, data/15m.csv, data/1m.csv), load them.
    - Else if DERIV_API_TOKEN or DerivClient available, try to fetch historical candles via DerivClient.fetch_candles.
    - Else fall back to synthetic data.
    """
    multi = {}

    # try CSVs first
    if data_dir:
        for tf in timeframes:
            fname = os.path.join(data_dir, f'{tf}.csv')
            if os.path.exists(fname):
                try:
                    multi[tf] = _load_csv_df(fname)
                except Exception:
                    multi[tf] = None
            else:
                multi[tf] = None
        # if we found at least one real df, return (leave missing as None)
        if any(v is not None for v in multi.values()):
            return multi

    # try to fetch via DerivClient if possible
    token = os.getenv('DERIV_API_TOKEN')
    if token and cfg is not None:
        client = DerivClient(token=token, dry_run=True)
        # mapping of timeframe string to granularity used by client if needed
        for tf in timeframes:
            try:
                # many Deriv clients accept the timeframe as string; adapt if yours uses minutes
                df = client.fetch_candles(cfg.symbol, tf, cfg.history_size)
                multi[tf] = df
            except Exception:
                multi[tf] = None
        if any(v is not None for v in multi.values()):
            return multi

    # fallback to synthetic data
    base_seed = 123
    for tf in timeframes:
        if tf == '1h':
            vol = 0.001
            nrows = n
            seed = base_seed + 1
        elif tf == '15m':
            vol = 0.0015
            nrows = n
            seed = base_seed + 2
        else:  # 1m
            vol = 0.0025
            nrows = n
            seed = base_seed + 3
        prices = make_synthetic_price_series(n=nrows, seed=seed, start=1000.0, vol=vol)
        df = build_df_from_prices(prices)
        multi[tf] = df
    return multi


def run_backtest_once(cfg, strategies, multi):
    ev = StrategyEvaluator(strategies, cfg)
    results = ev.evaluate_on(multi)
    return results


def evaluate_and_record(cfg, strategies, multi, commission_pct: float = 0.0005, slippage_pct: float = 0.0005, trades_path: str = 'trades.csv'):
    """Evaluate strategies like StrategyEvaluator but record individual trades to CSV,
    applying commission (fraction) and slippage (fraction). Returns summary dict per strategy.

    Commission and slippage are interpreted as fraction of price (e.g. 0.0005 = 0.05%).
    """
    results: dict = {}
    base_tf = cfg.timeframes[0]
    base = multi.get(base_tf)
    if base is None:
        return results
    length = len(base) if not hasattr(base, 'shape') else base.shape[0]
    window_size = min(cfg.history_size, length)

    # Prepare CSV (add strategy column)
    # add margin/leverage/notional/pnl columns so we can simulate leveraged PnL
    fields = ['timestamp', 'symbol', 'strategy', 'side', 'amount', 'entry', 'exit', 'stop_loss', 'take_profit', 'status', 'reason', 'atr', 'pnl', 'commission', 'margin_allocated', 'leverage', 'notional', 'pnl_usd', 'pnl_pct_on_margin', 'liquidated']
    with open(trades_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for name, strat in strategies.items():
            trades = 0
            wins = 0
            pnl_sum = 0.0
            pnl_usd_sum = 0.0
            # sizing / leverage defaults - read from cfg which should be injected by caller
            initial_capital = float(getattr(cfg, 'initial_capital', 1000.0))
            default_leverage = float(getattr(cfg, 'default_leverage', 1.0))
            # risk_per_trade_pct is expected as decimal (e.g. 0.02 == 2%)
            raw_risk = float(getattr(cfg, 'risk_per_trade_pct', 0.02))
            # If older configs used percentage numbers >=1, detect and convert
            if raw_risk >= 1.0:
                risk_per_trade_pct = raw_risk / 100.0
            else:
                risk_per_trade_pct = raw_risk
            # iterate through history as StrategyEvaluator
            for i in range(length - window_size, length - 1):
                snap = {}
                for tf in cfg.timeframes:
                    v = multi.get(tf)
                    try:
                        snap[tf] = v.iloc[:i + 1]
                    except Exception:
                        snap[tf] = v
                try:
                    sig = strat.analyze(snap)
                except Exception:
                    sig = None
                if sig is None:
                    continue
                trades += 1
                try:
                    entry = float(base.iloc[i]['close'])
                except Exception:
                    continue
                atr = float(base.iloc[i].get('atr') or 1.0)
                sl = entry - atr * 1.5
                tp = entry + atr * 3.0

                hit = None
                exit_price = None
                for j in range(1, min(30, length - i - 1) + 1):
                    fut = base.iloc[i + j]
                    if sig == 'buy':
                        if float(fut['high']) >= tp:
                            hit = 'tp'
                            # apply adverse slippage on exit
                            exit_price = tp * (1.0 - slippage_pct)
                            break
                        if float(fut['low']) <= sl:
                            hit = 'sl'
                            exit_price = sl * (1.0 + slippage_pct)
                            break
                    else:
                        if float(fut['low']) <= tp:
                            hit = 'tp'
                            exit_price = tp * (1.0 + slippage_pct)
                            break
                        if float(fut['high']) >= sl:
                            hit = 'sl'
                            exit_price = sl * (1.0 - slippage_pct)
                            break

                # If no hit found within lookahead, skip
                if hit is None or exit_price is None:
                    continue

                # apply slippage on entry (adverse)
                if sig == 'buy':
                    entry_eff = entry * (1.0 + slippage_pct)
                    exit_eff = float(exit_price)
                    pnl_price = exit_eff - entry_eff
                else:
                    entry_eff = entry * (1.0 - slippage_pct)
                    exit_eff = float(exit_price)
                    pnl_price = entry_eff - exit_eff


                # compute sizing and notional based on margin/leverage
                margin_allocated = max(initial_capital * risk_per_trade_pct, 0.0)
                leverage = float(getattr(cfg, 'default_leverage', default_leverage))
                notional = margin_allocated * leverage

                # return percent relative to entry (price return)
                return_pct = (pnl_price) / entry_eff if entry_eff != 0 else 0.0

                # commission in USD approximated over notional
                commission_usd = commission_pct * notional

                # pnl in USD using leveraged exposure: pnl_pct_on_margin = return_pct * leverage
                pnl_pct_on_margin = return_pct * leverage * 100.0  # in percent
                pnl_usd = margin_allocated * (pnl_pct_on_margin / 100.0) - commission_usd

                # --- NUEVA LÓGICA DE LIQUIDACIÓN ---
                # If configured, apply a simple full-margin liquidation: if pnl_usd <= -margin_allocated,
                # mark the trade as liquidated and cap the loss to the margin allocated.
                try:
                    liq_mode = getattr(cfg, 'liquidation_mode', 'simple')
                except Exception:
                    liq_mode = 'simple'

                liquidated_flag = False
                if str(liq_mode) == 'simple':
                    # compare raw pnl_usd to negative margin; if loss exceeds margin, force-close at -margin
                    if pnl_usd <= -margin_allocated:
                        liquidated_flag = True
                        pnl_usd = -margin_allocated
                        pnl_pct_on_margin = -100.0
                else:
                    liquidated_flag = False

                # keep a legacy 'pnl' column for backward compatibility (price units/net)
                commission = commission_pct * (abs(entry_eff) + abs(exit_eff))
                pnl_net = pnl_price - commission

                if pnl_usd > 0:
                    wins += 1

                pnl_sum += pnl_net
                pnl_usd_sum += pnl_usd

                # timestamp: try to use timestamp column or index
                ts = ''
                try:
                    row = base.iloc[i]
                    if isinstance(row, (dict,)) and 'timestamp' in row:
                        ts = row.get('timestamp')
                    else:
                        # pandas Series
                        try:
                            ts = str(row.get('timestamp')) if 'timestamp' in row.index else str(base.index[i])
                        except Exception:
                            ts = datetime.utcnow().isoformat()
                except Exception:
                    ts = datetime.utcnow().isoformat()

                # write extended record including sizing and USD pnl
                writer.writerow({
                    'timestamp': ts,
                    'symbol': cfg.symbol,
                    'strategy': name,
                    'side': sig,
                    'amount': 1.0,
                    'entry': '%.6f' % entry_eff,
                    'exit': '%.6f' % exit_eff,
                    'stop_loss': '%.6f' % sl,
                    'take_profit': '%.6f' % tp,
                    'status': 'closed',
                    'reason': hit,
                    'atr': '%.6f' % atr,
                    'pnl': '%.6f' % pnl_net,
                    'commission': '%.6f' % commission,
                    'margin_allocated': '%.6f' % margin_allocated,
                    'leverage': '%.2f' % leverage,
                    'notional': '%.6f' % notional,
                    'pnl_usd': '%.6f' % pnl_usd,
                    'pnl_pct_on_margin': '%.4f' % pnl_pct_on_margin,
                    'liquidated': bool(liquidated_flag),
                })

            winrate = wins / trades if trades else 0.0
            # return both pnl (legacy) and pnl_usd as summary
            results[name] = {'trades': trades, 'wins': wins, 'winrate': winrate, 'pnl': pnl_sum, 'pnl_usd': pnl_usd_sum}
    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--download', action='store_true', help='Descargar datos desde Deriv y guardarlos en data/')
    p.add_argument('--symbol', default='R_50', help='Símbolo a descargar (ej: R_50, R_100)')
    p.add_argument('--data-dir', default='data', help='Directorio para leer/escribir CSVs por timeframe')
    p.add_argument('--history-size', type=int, default=500, help='Número de velas a solicitar')
    p.add_argument('--rsi-period', type=int, default=14, help='RSI period to use when annotating data for backtest')
    p.add_argument('--enable-trend-filter', action='store_true', help='Enable EMA200 trend filter for RSIStrategy (only when strategy checks cfg.enable_trend_filter)')
    p.add_argument('--matrix', action='store_true', help='Run comparative matrix of backtests (RSI/min_confirmers combos)')
    p.add_argument('--exclude-strategies', default='', help='Comma-separated list of strategy keys to exclude from the backtest')
    p.add_argument('--only-strategies', default='', help='Comma-separated list of strategy keys to run exclusively (overrides exclude)')
    args = p.parse_args()

    # load config if available
    cfg = None
    try:
        cfg = Config.load_from('config.json')
    except Exception:
        cfg = None
    if cfg is None:
        # build minimal config-like object
        cfg = SimpleNamespace()
        cfg.timeframes = ['1h', '15m', '1m']
        cfg.history_size = args.history_size
        cfg.rsi_period = args.rsi_period
        cfg.enable_trend_filter = bool(args.enable_trend_filter)
        cfg.risk_per_trade_pct = 1.0
        cfg.atr_multiplier_sl = 1.5
        cfg.atr_multiplier_tp = 3.0
        cfg.symbol = args.symbol

    # build strategies mapping (same as CLI live mapping)
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

    # Filter strategies based on CLI flags
    if args.only_strategies:
        requested = {s.strip() for s in args.only_strategies.split(',') if s.strip()}
        strategies = {k: v for k, v in strategies.items() if k in requested}
        logger.info('Running only strategies: %s', ','.join(strategies.keys()))
    else:
        if args.exclude_strategies:
            excluded = {s.strip() for s in args.exclude_strategies.split(',') if s.strip()}
            for e in excluded:
                if e in strategies:
                    strategies.pop(e, None)
            logger.info('Excluded strategies: %s', ','.join(excluded))

    # If download requested, try to fetch and save CSVs
    if args.download:
        token = os.getenv('DERIV_API_TOKEN')
        if not token:
            logger.error('DESCARGA SOLICITADA pero no se encontró DERIV_API_TOKEN en el entorno')
        else:
            client = DerivClient(token=token, dry_run=True)
            os.makedirs(args.data_dir, exist_ok=True)
            for tf in cfg.timeframes:
                try:
                    df = client.fetch_candles(cfg.symbol, tf, cfg.history_size)
                    if df is None:
                        logger.warning('No data returned for %s %s', cfg.symbol, tf)
                        continue
                    outp = os.path.join(args.data_dir, f'{tf}.csv')
                    # try to write DataFrame or dict/list
                    try:
                        if hasattr(df, 'to_csv'):
                            df.to_csv(outp, index=False)
                        else:
                            pd.DataFrame(df).to_csv(outp, index=False)
                        logger.info('Guardado %s', outp)
                    except Exception:
                        # fallback naive writer
                        pd.DataFrame(df).to_csv(outp, index=False)
                except Exception as e:
                    logger.exception('Fallo al descargar %s %s: %s', cfg.symbol, tf, e)

    logger.info('Preparando datos (CSV / API / sintético)...')
    multi = prepare_multi(timeframes=cfg.timeframes, n=cfg.history_size, data_dir=args.data_dir, cfg=cfg)

    # annotate RSI with chosen period
    try:
        for tf, df in multi.items():
            if df is None:
                continue
            # if pandas DataFrame, compute RSI using Indicators.safe_rsi from ArTradIS
            try:
                import pandas as _pd
                if hasattr(df, 'loc') or hasattr(df, 'iloc'):
                    # create/replace rsi column
                    from ArTradIS import Indicators
                    df['rsi'] = Indicators.safe_rsi(df['close'], args.rsi_period)
                    multi[tf] = df
            except Exception:
                pass
    except Exception:
        pass
    # If matrix requested, run comparative scenarios
    def run_aggregated_backtest(cfg, strategies, multi, min_confirmers=2, commission_pct=0.0005, slippage_pct=0.0005, trades_path='trades_agg.csv'):
        """Aggregated backtest: at each timestep collect signals from all strategies and execute when
        one side has at least min_confirmers more votes than the opposite side (or >= min_confirmers absolute).
        Records trades to CSV and returns summary per scenario."""
        import csv
        from datetime import datetime
        results = {'trades': 0, 'wins': 0, 'pnl': 0.0}
        base_tf = cfg.timeframes[0]
        base = multi.get(base_tf)
        if base is None:
            return results
        length = len(base) if not hasattr(base, 'shape') else base.shape[0]
        window_size = min(cfg.history_size, length)

        fields = ['timestamp', 'symbol', 'strategy', 'side', 'entry', 'exit', 'stop_loss', 'take_profit', 'status', 'reason', 'atr', 'pnl', 'commission']
        with open(trades_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            for i in range(length - window_size, length - 1):
                snap = {}
                for tf in cfg.timeframes:
                    v = multi.get(tf)
                    try:
                        snap[tf] = v.iloc[:i + 1]
                    except Exception:
                        snap[tf] = v
                # collect signals
                counts = {'buy': 0, 'sell': 0}
                sigs = {}
                for name, strat in strategies.items():
                    try:
                        s = strat.analyze(snap)
                    except Exception:
                        s = None
                    sigs[name] = s
                    if s in counts:
                        counts[s] += 1
                total = counts['buy'] + counts['sell']
                if total == 0:
                    continue
                # decide side if one side has at least min_confirmers votes
                side = None
                if counts['buy'] >= min_confirmers and counts['buy'] > counts['sell']:
                    side = 'buy'
                elif counts['sell'] >= min_confirmers and counts['sell'] > counts['buy']:
                    side = 'sell'
                if side is None:
                    continue

                # create trade at base close price
                try:
                    entry = float(base.iloc[i]['close'])
                except Exception:
                    continue
                atr = float(base.iloc[i].get('atr') or 1.0)
                sl = entry - atr * 1.5
                tp = entry + atr * 3.0
                hit = None
                exit_price = None
                for j in range(1, min(30, length - i - 1) + 1):
                    fut = base.iloc[i + j]
                    if side == 'buy':
                        if float(fut['high']) >= tp:
                            hit = 'tp'
                            exit_price = tp * (1.0 - slippage_pct)
                            break
                        if float(fut['low']) <= sl:
                            hit = 'sl'
                            exit_price = sl * (1.0 + slippage_pct)
                            break
                    else:
                        if float(fut['low']) <= tp:
                            hit = 'tp'
                            exit_price = tp * (1.0 + slippage_pct)
                            break
                        if float(fut['high']) >= sl:
                            hit = 'sl'
                            exit_price = sl * (1.0 - slippage_pct)
                            break
                if hit is None or exit_price is None:
                    continue
                if side == 'buy':
                    entry_eff = entry * (1.0 + slippage_pct)
                    pnl_price = float(exit_price) - entry_eff
                else:
                    entry_eff = entry * (1.0 - slippage_pct)
                    pnl_price = entry_eff - float(exit_price)
                commission = commission_pct * (abs(entry_eff) + abs(exit_price))
                pnl_net = pnl_price - commission
                results['trades'] += 1
                if pnl_net > 0:
                    results['wins'] += 1
                results['pnl'] += pnl_net
                ts = ''
                try:
                    row = base.iloc[i]
                    ts = str(row.get('timestamp')) if 'timestamp' in row.index else str(base.index[i])
                except Exception:
                    ts = datetime.utcnow().isoformat()
                writer.writerow({'timestamp': ts, 'symbol': cfg.symbol, 'strategy': 'aggregated', 'side': side, 'entry': '%.6f' % entry_eff, 'exit': '%.6f' % float(exit_price), 'stop_loss': '%.6f' % sl, 'take_profit': '%.6f' % tp, 'status': 'closed', 'reason': hit, 'atr': '%.6f' % atr, 'pnl': '%.6f' % pnl_net, 'commission': '%.6f' % commission})

        return results

    if args.matrix:
        logger.info('Running comparative matrix of backtests...')
        scenarios = [
            ('rsi7_min2', 7, 2),
            ('rsi7_min1', 7, 1),
            ('rsi14_min1', 14, 1),
        ]
        matrix_results = {}
        for name, rsi_p, minc in scenarios:
            # annotate rsi
            for tf, df in multi.items():
                try:
                    if df is None:
                        continue
                    from ArTradIS import Indicators
                    df['rsi'] = Indicators.safe_rsi(df['close'], rsi_p)
                except Exception:
                    pass
            out_csv = f'trades_{name}.csv'
            res_s = run_aggregated_backtest(cfg, strategies, multi, min_confirmers=minc, commission_pct=0.0005, slippage_pct=0.0005, trades_path=out_csv)
            matrix_results[name] = res_s
            print(f"Scenario {name}: trades={res_s['trades']} wins={res_s['wins']} pnl={res_s['pnl']:.4f} (saved {out_csv})")
        print('\nMatrix complete')
        # write matrix summary
        with open('matrix_summary.txt', 'w', encoding='utf-8') as f:
            for k, v in matrix_results.items():
                f.write(f"{k}: {v}\n")
        raise SystemExit(0)

    logger.info('Ejecutando backtest base (con registro de trades, comisiones y slippage)...')
    # record trades to trades.csv applying small commission/slippage by default
    res = evaluate_and_record(cfg, strategies, multi, commission_pct=0.0005, slippage_pct=0.0005, trades_path='trades.csv')
    print('\nBacktest baseline (per strategy):')
    for k, v in res.items():
        print(f"{k}: trades={v['trades']} wins={v['wins']} winrate={v['winrate']:.3f} pnl={v['pnl']:.3f}")

    # small grid on risk_per_trade_pct
    grid = [0.25, 0.5, 1.0]
    agg_results = []
    logger.info('Probando malla de parámetros iniciales (risk_per_trade_pct)...')
    for r in grid:
        cfg.risk_per_trade_pct = r
        res = run_backtest_once(cfg, strategies, multi)
        total_pnl = sum(v['pnl'] for v in res.values())
        agg_results.append((r, total_pnl))
        print(f'Risk%={r} -> total_pnl={total_pnl:.3f}')

    print('\nResumen malla:')
    for r, p in agg_results:
        print(f'  risk%={r} -> pnl={p:.3f}')

    print('\nBacktest pipeline finalizado.')
