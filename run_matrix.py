"""Script para ejecutar una matriz de experimentos sobre RSIStrategy.

Genera o actualiza results/matrix_results.csv con una línea por combinación probada.
Ejemplo de uso:
    python run_matrix.py --rsi-periods 5 7 10 --overbought 65 70 --oversold 25 30 --take-profit 2.0 --stop-loss 1.0
"""
from __future__ import annotations
import argparse
import itertools
import os
import csv
import time
from typing import List
from logger_config import get_logger
import json
import datetime
import sys
import hashlib
import subprocess

try:
    import pandas as _pd
except Exception:
    _pd = None
try:
    import numpy as _np
except Exception:
    _np = None

logger = get_logger('matrix')
ROOT = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
VERBOSE_LOGS_DIR = os.path.join(RESULTS_DIR, 'verbose_logs')
os.makedirs(VERBOSE_LOGS_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(RESULTS_DIR, 'matrix_results.csv')

# Helper: append header if file not exists
def ensure_csv_header(path: str, header: List[str]):
    if not os.path.exists(path):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)


def save_metadata(run_id: str, params: dict, results: dict, artifacts: dict, root_dir: str = ROOT):
    """Escribe results/meta_<run_id>.json con metadata del experimento.

    No lanza excepciones hacia el llamador; solo registra un warning en stdout si falla.
    """
    try:
        meta = {
            "run_id": str(run_id),
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "parameters": params,
            "results": results,
            "artifacts": artifacts,
        }

        # Extras opcionales
        # trades_hash
        trades_path = artifacts.get('trades_csv', '')
        trades_hash = None
        try:
            if trades_path:
                abs_trades = os.path.join(root_dir, trades_path) if not os.path.isabs(trades_path) else trades_path
                if os.path.exists(abs_trades):
                    h = hashlib.sha256()
                    with open(abs_trades, 'rb') as tf:
                        for chunk in iter(lambda: tf.read(8192), b''):
                            h.update(chunk)
                    trades_hash = h.hexdigest()
        except Exception:
            trades_hash = None
        meta['trades_hash'] = trades_hash

        # Python and libs
        meta['python_version'] = sys.version
        try:
            meta['pandas_version'] = _pd.__version__ if _pd is not None else None
        except Exception:
            meta['pandas_version'] = None
        try:
            meta['numpy_version'] = _np.__version__ if _np is not None else None
        except Exception:
            meta['numpy_version'] = None

        # Git commit (best-effort)
        try:
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root_dir, stderr=subprocess.DEVNULL).decode().strip()
            meta['git_commit'] = git_commit
        except Exception:
            meta['git_commit'] = None
        # Git branch and remote (best-effort)
        try:
            git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=root_dir, stderr=subprocess.DEVNULL).decode().strip()
            meta['git_branch'] = git_branch
        except Exception:
            meta['git_branch'] = None
        try:
            git_remote = subprocess.check_output(['git', 'remote', 'get-url', 'origin'], cwd=root_dir, stderr=subprocess.DEVNULL).decode().strip()
            meta['git_remote'] = git_remote
        except Exception:
            meta['git_remote'] = None

        # Ensure results dir exists
        out_dir = os.path.join(root_dir, 'results')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"meta_{run_id}.json")
        # write json with indent=2 utf-8
        with open(out_path, 'w', encoding='utf-8') as jf:
            json.dump(meta, jf, indent=2, ensure_ascii=False)
        # success; log minimal info to stdout
        print(f"[meta] Saved metadata to {out_path}")
    except Exception as e:
        # Non-fatal: warn and continue
        try:
            print(f"[meta] Warning: failed to write meta file for {run_id}: {e}")
        except Exception:
            pass


def run_backtest_and_collect(cfg_args: dict) -> dict:
    """Llamar a backtest_pipeline.py (como process) y parsear resultado simple.
    Para simplicidad invocaremos la función interna importando y llamando `backtest_pipeline.evaluate_and_record`.
    """
    try:
        # import funciones desde backtest_pipeline
        import importlib
        bp = importlib.import_module('backtest_pipeline')
        # construir cfg similar
        try:
            from ArTradIS import Config
            cfg = Config.load_from('config.json') if hasattr(Config, 'load_from') else None
        except Exception:
            cfg = None
        from types import SimpleNamespace
        if cfg is None:
            cfg = SimpleNamespace()
            cfg.timeframes = ['1h','15m','1m']
            cfg.history_size = cfg_args.get('history_size', 500)
            cfg.rsi_period = cfg_args.get('rsi_period', 14)
            cfg.symbol = cfg_args.get('symbol', 'R_100')
        # Ensure CLI-supplied cfg_args override any loaded Config values
        try:
            cfg.rsi_period = cfg_args.get('rsi_period', getattr(cfg, 'rsi_period', 14))
            cfg.history_size = cfg_args.get('history_size', getattr(cfg, 'history_size', 500))
            cfg.overbought = cfg_args.get('overbought', getattr(cfg, 'overbought', getattr(cfg, 'rsi_overbought', None)))
            cfg.oversold = cfg_args.get('oversold', getattr(cfg, 'oversold', getattr(cfg, 'rsi_oversold', None)))
            # TP/SL as percent
            cfg.take_profit_pct = cfg_args.get('take_profit_pct', getattr(cfg, 'take_profit_pct', getattr(cfg, 'take_profit', None)))
            cfg.stop_loss_pct = cfg_args.get('stop_loss_pct', getattr(cfg, 'stop_loss_pct', getattr(cfg, 'stop_loss', None)))
            # momentum-related flags (optional)
            cfg.enable_momentum = cfg_args.get('enable_momentum', getattr(cfg, 'enable_momentum', False))
            cfg.momentum_ema = cfg_args.get('momentum_ema', getattr(cfg, 'momentum_ema', 9))
            cfg.momentum_threshold = cfg_args.get('momentum_threshold', getattr(cfg, 'momentum_threshold', 0.0))
            cfg.momentum_required = cfg_args.get('momentum_required', getattr(cfg, 'momentum_required', False))
            cfg.verbose_strategy = cfg_args.get('verbose_strategy', getattr(cfg, 'verbose_strategy', False))
            cfg.enable_trend_filter = cfg_args.get('enable_trend_filter', getattr(cfg, 'enable_trend_filter', False))
            # sizing / leverage / capital - ensure these are present on cfg so backtest uses them
            cfg.initial_capital = cfg_args.get('initial_capital', getattr(cfg, 'initial_capital', 15.0))
            cfg.default_leverage = cfg_args.get('default_leverage', getattr(cfg, 'default_leverage', 1.0))
            # risk_per_trade_pct is expected as a percent (e.g. 0.5 means 0.5%), keep as provided
            cfg.risk_per_trade_pct = cfg_args.get('risk_per_trade_pct', getattr(cfg, 'risk_per_trade_pct', 1.0))
        except Exception:
            # best-effort: if cfg is a plain object without attributes, ignore
            pass

        # Instantiate the chosen RSIStrategy implementation (verbose wrapper if requested)
        try:
            use_verbose = getattr(cfg, 'verbose_strategy', False)
        except Exception:
            use_verbose = False

        if use_verbose:
            try:
                from rsistrategy_verbose_patch import VerboseRSIStrategy
            except Exception:
                VerboseRSIStrategy = None
            if VerboseRSIStrategy is not None:
                strategies = {'rsi_adv': VerboseRSIStrategy(cfg, logger)}
            else:
                from ArTradIS import RSIStrategy
                strategies = {'rsi_adv': RSIStrategy(cfg)}
        else:
            from ArTradIS import RSIStrategy
            strategies = {'rsi_adv': RSIStrategy(cfg)}
        multi = bp.prepare_multi(timeframes=cfg.timeframes, n=cfg.history_size, data_dir='data', cfg=cfg)
        run_id = cfg_args.get('run_id', f"rsi{cfg.rsi_period}_{int(time.time())}")
        trades_path = os.path.join(RESULTS_DIR, f"trades_{run_id}.csv")
        # ensure verbose log file exists (empty) so evaluator/VerboseRSIStrategy can append
        verbose_log_path = cfg_args.get('verbose_log_path')
        if verbose_log_path:
            try:
                os.makedirs(os.path.dirname(verbose_log_path), exist_ok=True)
                open(verbose_log_path, 'a', encoding='utf-8').close()
            except Exception:
                pass
        # attach verbose path into cfg so wrappers can use it
        try:
            cfg.verbose_log_path = verbose_log_path
            cfg.run_id = run_id
        except Exception:
            pass
        # determine slippage: prefer cfg attribute, else default 0.0005
        slippage_val = float(getattr(cfg, 'slippage_pct', cfg_args.get('slippage_pct', 0.0005))) if isinstance(cfg, object) else cfg_args.get('slippage_pct', 0.0005)
        res = bp.evaluate_and_record(cfg, strategies, multi, commission_pct=0.0005, slippage_pct=slippage_val, trades_path=trades_path)
        summary = res.get('rsi_adv', {'trades':0, 'wins':0, 'pnl':0.0})
        # compute max drawdown from trades file if possible
        max_dd = 0.0
        try:
            if os.path.exists(trades_path):
                dftr = _pd.read_csv(trades_path)
                if 'pnl' in dftr.columns:
                    # cumulative pnl over trades
                    cum = dftr['pnl'].astype(float).cumsum()
                    running_max = cum.cummax()
                    dd = (running_max - cum).fillna(0.0)
                    max_dd = float(dd.max()) if len(dd) > 0 else 0.0
        except Exception:
            max_dd = 0.0

        # count alerts heuristically from verbose log if available
        alerts = 0
        try:
            if verbose_log_path and os.path.exists(verbose_log_path):
                with open(verbose_log_path, 'r', encoding='utf-8') as vf:
                    for L in vf:
                        if 'skip' in L.lower() or 'ignored' in L.lower() or 'unavailable' in L.lower():
                            alerts += 1
        except Exception:
            alerts = 0

        return {'trades_path': trades_path, 'trades': summary['trades'], 'wins': summary['wins'], 'pnl': summary['pnl'], 'max_drawdown': max_dd, 'alerts': alerts}
    except Exception as e:
        logger.exception('Failed to run backtest: %s', e)
        return {'trades_path': '', 'trades': 0, 'wins': 0, 'pnl': 0.0}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rsi-periods', type=int, nargs='+', default=[5,7,10,14,21])
    p.add_argument('--overbought', type=int, nargs='+', default=[65,70,75])
    p.add_argument('--oversold', type=int, nargs='+', default=[25,30,35])
    p.add_argument('--take-profit', type=float, nargs='*', default=[3.0])
    p.add_argument('--stop-loss', type=float, nargs='*', default=[1.5])
    p.add_argument('--history-size', type=int, default=500)
    p.add_argument('--initial-capital', type=float, default=15.0, help='Initial capital in USD for equity simulation')
    p.add_argument('--default-leverage', type=float, default=1.0, help='Default leverage to apply to positions (e.g. 200)')
    p.add_argument('--risk-per-trade-pct', type=float, default=1.0, help='Percent of capital to risk per trade (e.g. 1.0 means 1%)')
    p.add_argument('--slippage-pct', type=float, default=0.002, help='Slippage porcentual (default: 0.002 = 0.2%)')
    p.add_argument('--liquidation-mode', type=str, default='simple', choices=['none', 'simple'],
                    help='Define how liquidation is handled (none or simple full-margin liquidation)')
    p.add_argument('--enable-momentum', action='store_true', help='Enable momentum EMA confirmation in RSIStrategy')
    p.add_argument('--momentum-ema', type=int, default=9)
    p.add_argument('--momentum-threshold', type=float, default=0.0)
    p.add_argument('--momentum-required', action='store_true', help='Require momentum confirmation for entries')
    p.add_argument('--verbose-strategy', action='store_true', help='Use VerboseRSIStrategy wrapper to log r2/r1/r0 and decisions')
    p.add_argument('--enable-trend-filter', action='store_true', help='Enable EMA200 trend filter in RSIStrategy when running experiments')
    args = p.parse_args()

    header = ['rsi_period','rsi_overbought','rsi_oversold','take_profit','stop_loss','profit','win_rate','trades','max_drawdown','trades_path','momentum_flags','trend_filter_enabled','alerts']
    ensure_csv_header(RESULTS_CSV, header)

    combinations = list(itertools.product(args.rsi_periods, args.overbought, args.oversold, args.take_profit, args.stop_loss))
    logger.info('Running %d experiments', len(combinations))
    for rsi_p, ob, osd, tp, sl in combinations:
        logger.info('Experiment rsi=%s ob=%s os=%s tp=%s sl=%s', rsi_p, ob, osd, tp, sl)
        # create a run id for this experiment to avoid overwriting files
        run_ts = int(time.time())
        run_id = f"rsi{rsi_p}_{run_ts}"
        verbose_log_path = os.path.join(VERBOSE_LOGS_DIR, f"verbose_{run_id}.log")

        cfg_args = {
            'rsi_period': rsi_p,
            'history_size': args.history_size,
            'overbought': ob,
            'oversold': osd,
            'take_profit_pct': tp,
            'stop_loss_pct': sl,
            'enable_momentum': args.enable_momentum,
            'momentum_ema': args.momentum_ema,
            'momentum_threshold': args.momentum_threshold,
            'momentum_required': args.momentum_required,
            'enable_trend_filter': getattr(args, 'enable_trend_filter', False),
            'verbose_strategy': getattr(args, 'verbose_strategy', False),
            'verbose_log_path': verbose_log_path,
            'run_id': run_id,
            'initial_capital': float(getattr(args, 'initial_capital', 15.0)),
            'default_leverage': float(getattr(args, 'default_leverage', 1.0)),
            'risk_per_trade_pct': float(getattr(args, 'risk_per_trade_pct', 1.0)),
            'liquidation_mode': getattr(args, 'liquidation_mode', 'simple'),
            'slippage_pct': float(getattr(args, 'slippage_pct', 0.002)),
        }
        res = run_backtest_and_collect(cfg_args)
        # compute win rate
        trades = res.get('trades',0)
        wins = res.get('wins',0)
        pnl = res.get('pnl',0.0)
        win_rate = (wins / trades) if trades>0 else 0.0
        # write to results CSV (append)
        # compute USD metrics from trades CSV for live print and meta
        def compute_usd_metrics(trades_path, initial_capital):
             metrics = {'total_pnl_usd': None, 'total_liquidations': None, 'capital_final_usd': None, 'max_drawdown_usd': None}
             try:
                 if _pd is None:
                     return metrics
                 if trades_path and os.path.exists(trades_path):
                     tdf = _pd.read_csv(trades_path)
                     if 'pnl_usd' in tdf.columns:
                         total_pnl = float(tdf['pnl_usd'].fillna(0).sum())
                         metrics['total_pnl_usd'] = total_pnl
                     if 'liquidated' in tdf.columns:
                         metrics['total_liquidations'] = int(tdf['liquidated'].astype(bool).sum())
                     if 'pnl_usd' in tdf.columns:
                         # equity curve and max drawdown
                         try:
                             tdf_sorted = tdf.sort_values('timestamp') if 'timestamp' in tdf.columns else tdf
                             equity = tdf_sorted['pnl_usd'].fillna(0).cumsum()
                             running_max = equity.cummax()
                             dd = (running_max - equity).fillna(0.0)
                             metrics['max_drawdown_usd'] = float(dd.max()) if len(dd)>0 else 0.0
                             # percent drawdown relative to initial capital
                             try:
                                metrics['max_drawdown_pct'] = float(metrics['max_drawdown_usd']) / float(initial_capital) if initial_capital and float(initial_capital) != 0 else None
                             except Exception:
                                metrics['max_drawdown_pct'] = None
                         except Exception:
                             metrics['max_drawdown_usd'] = None
                     if initial_capital is not None and metrics.get('total_pnl_usd') is not None:
                         metrics['capital_final_usd'] = float(initial_capital) + float(metrics['total_pnl_usd'])
             except Exception:
                 pass
             return metrics

        usd_metrics = compute_usd_metrics(res.get('trades_path',''), cfg_args.get('initial_capital', 15.0))

        # write to results CSV (append) with extra metadata
        momentum_flags = 'on' if args.enable_momentum else 'off'
        trend_on = 'on' if getattr(args, 'enable_trend_filter', False) else 'off'
        alerts = res.get('alerts', 0)
        with open(RESULTS_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([rsi_p, ob, osd, tp, sl, pnl, '%.4f' % win_rate, trades, res.get('max_drawdown', 0.0), res.get('trades_path',''), momentum_flags, trend_on, alerts])
        logger.info('Experiment finished: pnl=%s trades=%s win_rate=%.4f', pnl, trades, win_rate)
        # Print live USD metrics summary for the experiment
        try:
            tpu = usd_metrics.get('total_pnl_usd')
            tlq = usd_metrics.get('total_liquidations')
            cf = usd_metrics.get('capital_final_usd')
            mdd = usd_metrics.get('max_drawdown_usd')
            mdd_pct = usd_metrics.get('max_drawdown_pct')
            if tpu is not None:
                print(f"[meta] {run_id}: total_pnl_usd={tpu:.6f} total_liquidations={int(tlq) if tlq is not None else 0} capital_final_usd={cf:.6f} max_drawdown_usd={mdd:.6f} max_drawdown_pct={(mdd_pct if mdd_pct is not None else 'NA')}")
        except Exception:
            pass

        # Prepare metadata and save to results/meta_<run_id>.json
        try:
            params = {
                'rsi_period': int(rsi_p),
                'overbought': int(ob),
                'oversold': int(osd),
                'take_profit': float(tp),
                'stop_loss': float(sl),
                'history_size': int(args.history_size),
                'momentum_enabled': bool(args.enable_momentum),
                'momentum_ema': int(args.momentum_ema),
                'momentum_threshold': float(args.momentum_threshold),
                'momentum_required': bool(args.momentum_required),
                'trend_filter_enabled': bool(getattr(args, 'enable_trend_filter', False)),
                'verbose': bool(getattr(args, 'verbose_strategy', False)),
                'initial_capital': float(getattr(args, 'initial_capital', 15.0)),
                'default_leverage': float(getattr(args, 'default_leverage', 1.0)),
                'risk_per_trade_pct': float(getattr(args, 'risk_per_trade_pct', 1.0)),
                'liquidation_mode': str(getattr(args, 'liquidation_mode', 'simple')),
                    'slippage_pct': float(getattr(args, 'slippage_pct', 0.002)),
            }
            results_meta = {
                'profit': float(pnl),
                'win_rate': float(round(win_rate, 4)),
                'trades': int(trades),
                'alerts': [] if alerts == 0 else ['heuristic_alerts_present']
            }
            # attach USD metrics into results for summarize_meta and future use
            try:
                if usd_metrics.get('total_pnl_usd') is not None:
                    results_meta['total_pnl_usd'] = usd_metrics.get('total_pnl_usd')
                if usd_metrics.get('capital_final_usd') is not None:
                    results_meta['capital_final_usd'] = usd_metrics.get('capital_final_usd')
                if usd_metrics.get('max_drawdown_usd') is not None:
                    results_meta['max_drawdown_usd'] = usd_metrics.get('max_drawdown_usd')
                if usd_metrics.get('max_drawdown_pct') is not None:
                    results_meta['max_drawdown_pct'] = usd_metrics.get('max_drawdown_pct')
                if usd_metrics.get('total_liquidations') is not None:
                    results_meta['total_liquidations'] = int(usd_metrics.get('total_liquidations'))
            except Exception:
                pass
            # artifacts: use relative paths
            artifacts = {
                'matrix_results_csv': os.path.relpath(RESULTS_CSV, ROOT).replace('\\', '/'),
                'trades_csv': os.path.relpath(res.get('trades_path',''), ROOT).replace('\\', '/'),
                'verbose_log': os.path.relpath(verbose_log_path, ROOT).replace('\\', '/') if verbose_log_path else None,
                'alerts_summary': os.path.relpath(os.path.join(RESULTS_DIR, 'alerts_summary.csv'), ROOT).replace('\\', '/'),
            }
            save_metadata(run_id, params, results_meta, artifacts, root_dir=ROOT)
        except Exception:
            # ensure non-fatal
            print(f"[meta] Warning: failed preparing metadata for run {run_id}")

    logger.info('Matrix complete. Results saved to %s', RESULTS_CSV)
