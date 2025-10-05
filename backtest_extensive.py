"""
backtest_extensive.py

Script para ejecutar backtests extensivos (3-6 meses) en R_50, R_75, R_100.
- Descarga datos (si se solicita y se tiene token)
- Ejecuta backtests para todas las estrategias configuradas
- Genera reportes por símbolo/estrategia (JSON + CSV de trades)
- Selecciona automáticamente la mejor estrategia por PnL USD

USO (ejemplo):
  .venv\Scripts\python backtest_extensive.py --months 6 --symbols R_50,R_100 --download

AVISO: Este script sólo prepara y ejecuta backtests en modo local/sintético/API.
No realiza órdenes live.
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import logging
from types import SimpleNamespace
from datetime import datetime

# dependencias
try:
    import pandas as pd
    import numpy as np
except Exception:
    print('Instala pandas/numpy en la venv antes de ejecutar: pip install pandas numpy')
    raise

# importar utilidades del pipeline
try:
    import backtest_pipeline as bkp
    from ArTradIS import Config, StrategyFactory, TrendFollowingAdvanced, RangeTradingAdvanced, ScalpingAdvanced, MultipleTimeFrameStrategy, MovingAverageStrategy, BollingerBandsStrategy, RSIStrategy, MACDStrategy, Indicators, DerivClient
except Exception as e:
    print('Error importando módulos internos:', e)
    raise

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('backtest_extensive')


def ensure_dirs(path: str):
    os.makedirs(path, exist_ok=True)


def build_cfg_for_symbol(symbol: str, history_size: int, months: int, enable_trend_filter: bool = False) -> SimpleNamespace:
    cfg = SimpleNamespace()
    cfg.symbol = symbol
    cfg.timeframes = ['1h', '15m', '1m']
    cfg.history_size = int(history_size)
    cfg.rsi_period = 14
    cfg.enable_trend_filter = enable_trend_filter
    # risk as decimal (e.g. 0.02 = 2%) default conservative
    cfg.risk_per_trade_pct = 0.02
    cfg.initial_capital = 1000.0
    cfg.default_leverage = 1.0
    cfg.liquidation_mode = 'simple'
    return cfg


def compute_max_drawdown_from_trades(df_trades: pd.DataFrame, metric_col: str = 'pnl_usd') -> float:
    if df_trades is None or df_trades.empty:
        return 0.0
    # ensure chronological
    try:
        if 'timestamp' in df_trades.columns:
            df_trades['ts'] = pd.to_datetime(df_trades['timestamp'], errors='coerce')
            df_trades = df_trades.sort_values('ts')
            series = df_trades[metric_col].cumsum().fillna(method='ffill').fillna(0).to_list()
        else:
            series = df_trades[metric_col].cumsum().fillna(0).to_list()
    except Exception:
        # fallback simple cumulative
        series = df_trades[metric_col].cumsum().fillna(0).to_list()
    # use Indicators.max_drawdown if available
    try:
        return float(Indicators.max_drawdown(series))
    except Exception:
        # manual computation
        peak = -float('inf')
        max_dd = 0.0
        for v in series:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak and peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd


def monthly_breakdown(df_trades: pd.DataFrame) -> dict:
    out = {}
    if df_trades is None or df_trades.empty:
        return out
    try:
        df = df_trades.copy()
        df['ts'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['month'] = df['ts'].dt.to_period('M')
        grouped = df.groupby('month')
        for name, g in grouped:
            out[str(name)] = {'trades': len(g), 'pnl_usd': float(g['pnl_usd'].sum())}
    except Exception:
        pass
    return out


def run_for_symbol(symbol: str, history_size: int, months: int, download: bool, data_dir: str, out_dir: str, enable_trend_filter: bool = False, leverage_levels=None, capital=15.0):
    logger.info('Running extensive backtest for %s (months=%s history_size=%d)', symbol, months, history_size)
    ensure_dirs(data_dir)
    ensure_dirs(out_dir)

    cfg = build_cfg_for_symbol(symbol, history_size, months, enable_trend_filter=enable_trend_filter)

    # attempt to download CSVs if requested
    if download:
        token = os.getenv('DERIV_API_TOKEN')
        if not token:
            logger.warning('Download requested but DERIV_API_TOKEN not set; skipping download')
        else:
            client = DerivClient(token=token, dry_run=True)
            for tf in cfg.timeframes:
                try:
                    df = client.fetch_candles(cfg.symbol, tf, cfg.history_size)
                    if df is None:
                        logger.warning('No data returned for %s %s', cfg.symbol, tf)
                        continue
                    outp = os.path.join(data_dir, f'{tf}.csv')
                    try:
                        if hasattr(df, 'to_csv'):
                            df.to_csv(outp, index=False)
                        else:
                            pd.DataFrame(df).to_csv(outp, index=False)
                        logger.info('Saved %s', outp)
                    except Exception:
                        pd.DataFrame(df).to_csv(outp, index=False)
                except Exception as e:
                    logger.exception('Failed downloading %s %s: %s', cfg.symbol, tf, e)

    # prepare multi timeframe data (this will load CSVs if present or fallback to synthetic)
    multi = bkp.prepare_multi(timeframes=cfg.timeframes, n=cfg.history_size, data_dir=data_dir, cfg=cfg)

    # create strategies mapping
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

    # run evaluate_and_record which will write trades.csv and return per-strategy summary
    trades_csv = os.path.join(out_dir, f'trades_{symbol}.csv')
    summary = bkp.evaluate_and_record(cfg, strategies, multi, commission_pct=0.0005, slippage_pct=0.0005, trades_path=trades_csv)

    # load trades CSV to compute per-strategy perfromance and drawdown
    try:
        df_trades = pd.read_csv(trades_csv)
    except Exception:
        df_trades = pd.DataFrame()

    # aggregate per-strategy details
    detailed = {}
    for strat_name, metrics in summary.items():
        # filter trades for this strategy
        try:
            df_s = df_trades[df_trades['strategy'] == strat_name].copy()
            # ensure numeric
            if 'pnl_usd' in df_s.columns:
                df_s['pnl_usd'] = pd.to_numeric(df_s['pnl_usd'], errors='coerce').fillna(0.0)
            else:
                df_s['pnl_usd'] = 0.0
            max_dd = compute_max_drawdown_from_trades(df_s, metric_col='pnl_usd')
            month = monthly_breakdown(df_s)
        except Exception:
            max_dd = 0.0
            month = {}
        detailed[strat_name] = {
            'summary': metrics,
            'max_drawdown': max_dd,
            'monthly': month,
        }

    # pick best by pnl_usd
    best = None
    best_pnl = -float('inf')
    for k, v in detailed.items():
        pnl_usd = float(v['summary'].get('pnl_usd', 0.0))
        if pnl_usd > best_pnl:
            best_pnl = pnl_usd
            best = k

    # Now evaluate different leverage levels: compute how many trades would have liquidated
    leverage_levels = list(leverage_levels) if leverage_levels is not None else [1]
    leverage_summary = {}
    try:
        df_trades_all = pd.read_csv(trades_csv)
    except Exception:
        df_trades_all = pd.DataFrame()

    for lev in leverage_levels:
        lev = float(lev)
        # compute liquidation threshold: with capital C and leverage L, position notional = C * L -> liquidation move approx = C / (C*L) = 1/L in price terms
        # We approximate the percent move that would wipe margin: 1/lev (e.g., lev=200 -> 0.5%)
        liq_pct = 1.0 / lev
        # count trades where absolute return (entry->exit) in percent would have exceeded liq_pct against the position
        liquidations = 0
        total_trades = 0
        wins = 0
        pnl_sum = 0.0
        pnl_usd_sum = 0.0
        max_dd = 0.0
        if not df_trades_all.empty:
            # ensure numeric columns exist
            df = df_trades_all.copy()
            for col in ('entry','exit','pnl_usd'):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            # compute per-row percentage move relative to entry
            def would_liquidate(row):
                try:
                    e = float(row.get('entry') or 0.0)
                    x = float(row.get('exit') or 0.0)
                    side = row.get('side','buy')
                    if e == 0:
                        return False
                    if side.lower() == 'buy':
                        move = (e - x) / e
                    else:
                        move = (x - e) / e
                    return abs(move) >= liq_pct
                except Exception:
                    return False

            for _, r in df.iterrows():
                total_trades += 1
                if float(r.get('pnl_usd', 0.0)) > 0:
                    wins += 1
                pnl_sum += float(r.get('pnl', 0.0) or 0.0)
                pnl_usd_sum += float(r.get('pnl_usd', 0.0) or 0.0)
                if would_liquidate(r):
                    liquidations += 1
            # approximate max drawdown from cumulative pnl_usd
            try:
                series = df['pnl_usd'].cumsum().fillna(0).tolist()
                max_dd = float(Indicators.max_drawdown(series))
            except Exception:
                max_dd = 0.0

        leverage_summary[int(lev)] = {
            'liquidations': int(liquidations),
            'total_trades': int(total_trades),
            'wins': int(wins),
            'winrate': (wins / total_trades) if total_trades else 0.0,
            'pnl': float(pnl_sum),
            'pnl_usd': float(pnl_usd_sum),
            'max_drawdown': float(max_dd),
            'liq_pct': liq_pct,
        }

    out_report = {
        'symbol': symbol,
        'months': months,
        'history_size': history_size,
        'generated_at': datetime.utcnow().isoformat(),
        'strategies': detailed,
        'best_strategy': best,
        'best_pnl_usd': best_pnl,
        'leverage_summary': leverage_summary,
    }

    # save JSON report
    out_json = os.path.join(out_dir, f'backtest_extensive_{symbol}.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out_report, f, indent=2)
    logger.info('Saved report %s', out_json)

    return out_report
    


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--months', type=int, default=6, help='Months of history to test (approx uses 30 days/month)')
    p.add_argument('--symbols', type=str, default='R_50,R_75,R_100', help='Comma separated symbols to test')
    p.add_argument('--download', action='store_true', help='Attempt to download candles from Deriv (requires DERIV_API_TOKEN)')
    p.add_argument('--data-dir', default='data', help='Directory with timeframe CSVs or where to store downloaded CSVs')
    p.add_argument('--out-dir', default='results', help='Output directory for reports and trades CSVs')
    p.add_argument('--enable-trend-filter', action='store_true', help='Enable EMA200 trend filter in strategies that honor it')
    p.add_argument('--history-size', type=int, default=None, help='Override history size in ticks (e.g. 150000)')
    p.add_argument('--leverage-levels', nargs='+', type=float, default=[1.0], help='List of leverage levels to evaluate (e.g. 20 50 100 200)')
    p.add_argument('--capital', type=float, default=15.0, help='Capital per trial in USD')
    p.add_argument('--output', type=str, default=None, help='Optional CSV output path to aggregate results')
    p.add_argument('--use-existing', type=str, default=None, help='Path to existing CSV to use (e.g. data/1m.csv)')
    p.add_argument('--max-records', type=int, default=100000, help='Max number of records to use from existing CSV')
    args = p.parse_args()

    months = max(1, int(args.months))
    # choose history_size as hourly bars to keep size reasonable: months * 30 days * 24 hours
    history_size = args.history_size if args.history_size is not None else months * 30 * 24

    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
    all_reports = {}

    # If use_existing CSV specified, load it and run targeted RSI strategy per leverage
    if getattr(args, 'use_existing', None):
        filepath = args.use_existing
        max_records = int(getattr(args, 'max_records', 100000))
        print(f'Loading existing data from {filepath} (limit {max_records})')
        df = pd.read_csv(filepath)
        if len(df) > max_records:
            df = df.tail(max_records)
        print(f'Loaded {len(df)} rows from {filepath}')

        # single-symbol run for each leverage
        out_rows = []
        for sym in symbols:
            for lev in args.leverage_levels:
                cfg = build_cfg_for_symbol(sym, history_size, months, enable_trend_filter=True)
                # set strategy specific params: RSI10 + EMA9 + EMA200
                cfg.rsi_period = 10
                cfg.momentum_ema = 9
                cfg.enable_trend_filter = True
                cfg.initial_capital = float(args.capital)
                cfg.default_leverage = float(lev)
                # When using an existing single timeframe CSV (1m) ensure the cfg's base timeframe
                # matches the provided data so evaluate_and_record can find the base series.
                cfg.timeframes = ['1m']
                # Align history_size to the available rows to avoid window issues
                try:
                    cfg.history_size = int(len(df))
                except Exception:
                    cfg.history_size = int(history_size)

                strategies = {
                    'rsi10_ema9_ema200': RSIStrategy(cfg),
                }

                multi = {'1m': df}
                trades_path = os.path.join(args.out_dir, f'trades_{sym}_lev{int(lev)}.csv')
                ensure_dirs(args.out_dir)
                print(f'Running backtest for {sym} leverage={lev}...')
                summary = bkp.evaluate_and_record(cfg, strategies, multi, commission_pct=0.0005, slippage_pct=0.0005, trades_path=trades_path)

                # load trades and compute metrics
                try:
                    df_tr = pd.read_csv(trades_path)
                except Exception:
                    df_tr = pd.DataFrame()

                total_trades = len(df_tr)
                wins = int((df_tr['pnl_usd'].astype(float) > 0).sum()) if 'pnl_usd' in df_tr.columns else 0
                winrate = (wins / total_trades) if total_trades else 0.0
                pnl_usd = float(df_tr['pnl_usd'].astype(float).sum()) if 'pnl_usd' in df_tr.columns else 0.0
                max_dd = compute_max_drawdown_from_trades(df_tr, metric_col='pnl_usd') if not df_tr.empty else 0.0

                # liquidation count: approximate threshold = 1/lev
                liq_pct = 1.0 / float(lev)
                liquidations = 0
                if not df_tr.empty and 'entry' in df_tr.columns and 'exit' in df_tr.columns:
                    df_tr['entry'] = pd.to_numeric(df_tr['entry'], errors='coerce')
                    df_tr['exit'] = pd.to_numeric(df_tr['exit'], errors='coerce')
                    for _, r in df_tr.iterrows():
                        e = r.get('entry') or 0.0
                        x = r.get('exit') or 0.0
                        side = str(r.get('side','buy')).lower()
                        if e == 0:
                            continue
                        if side == 'buy':
                            move = (e - x) / e
                        else:
                            move = (x - e) / e
                        if abs(move) >= liq_pct:
                            liquidations += 1

                # profit factor
                pf = None
                try:
                    gains = df_tr[df_tr['pnl_usd'].astype(float) > 0]['pnl_usd'].astype(float).sum() if 'pnl_usd' in df_tr.columns else 0.0
                    losses = abs(df_tr[df_tr['pnl_usd'].astype(float) < 0]['pnl_usd'].astype(float).sum()) if 'pnl_usd' in df_tr.columns else 0.0
                    pf = (gains / losses) if losses > 0 else (float('inf') if gains > 0 else None)
                except Exception:
                    pf = None

                out_rows.append({
                    'symbol': sym,
                    'leverage': int(lev),
                    'total_trades': int(total_trades),
                    'wins': int(wins),
                    'winrate': float(winrate),
                    'pnl_usd': float(pnl_usd),
                    'max_drawdown': float(max_dd),
                    'liquidations': int(liquidations),
                    'liq_pct': float(liq_pct),
                    'profit_factor': pf,
                })

        # write aggregated CSV
        if args.output:
            ensure_dirs(os.path.dirname(args.output) or '.')
            try:
                pd.DataFrame(out_rows).to_csv(args.output, index=False)
                print(f'Wrote aggregated CSV {args.output}')
            except Exception:
                print('Failed to write output CSV')

        print('\nBacktest extensivo (existing data) completado.')
        return

    # default flow: run per symbol using run_for_symbol
    for sym in symbols:
        r = run_for_symbol(sym, history_size=history_size, months=months, download=args.download, data_dir=args.data_dir, out_dir=args.out_dir, enable_trend_filter=args.enable_trend_filter, leverage_levels=args.leverage_levels, capital=args.capital)
        all_reports[sym] = r

    # save global summary
    now = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_all = os.path.join(args.out_dir, f'backtest_extensive_summary_{now}.json')
    with open(out_all, 'w', encoding='utf-8') as f:
        json.dump(all_reports, f, indent=2)
    logger.info('Saved aggregated summary %s', out_all)

    # optionally write aggregated CSV
    if args.output:
        ensure_dirs(os.path.dirname(args.output) or '.')
        rows = []
        for sym, rpt in all_reports.items():
            lev_sum = rpt.get('leverage_summary', {})
            for lev, metrics in lev_sum.items():
                rows.append({
                    'symbol': sym,
                    'leverage': lev,
                    'total_trades': metrics.get('total_trades', 0),
                    'liquidations': metrics.get('liquidations', 0),
                    'winrate': metrics.get('winrate', 0.0),
                    'pnl_usd': metrics.get('pnl_usd', 0.0),
                    'max_drawdown': metrics.get('max_drawdown', 0.0),
                    'liq_pct': metrics.get('liq_pct', 0.0),
                })
        try:
            pd.DataFrame(rows).to_csv(args.output, index=False)
            logger.info('Wrote aggregated CSV %s', args.output)
        except Exception:
            logger.exception('Failed to write aggregated CSV')

    print('\nBacktest extensivo completado. Revisa la carpeta results/ para CSVs y JSONs.')


if __name__ == '__main__':
    main()
