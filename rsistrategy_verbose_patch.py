"""Parche externo que envuelve `RSIStrategy` para añadir logging detallado sin modificar la fuente original.

Ofrece `VerboseRSIStrategy(cfg, logger)` con la misma interfaz analyze(multi) que registra entradas/salidas.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import traceback

try:
    from ArTradIS import RSIStrategy
except Exception:
    RSIStrategy = None


class VerboseRSIStrategy:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        # wrap original if available
        self._impl = RSIStrategy(cfg) if RSIStrategy is not None else None
        # optional per-experiment verbose log path (file)
        self.verbose_log_path = getattr(cfg, 'verbose_log_path', None)

    def analyze(self, multi: Dict[str, Any]) -> Optional[str]:
        try:
            # call underlying strategy
            if self._impl is None:
                self.logger.error('RSIStrategy implementation not found')
                return None
            # compute RSI values using underlying implementation by calling analyze after instrumenting
            decision = self._impl.analyze(multi)
            # attempt to calculate r0,r1,r2 for logging
            try:
                df = multi.get('1m')
                close = df['close'] if df is not None else None
                period = getattr(self.cfg, 'rsi_period', 14)
                from ArTradIS import Indicators
                rsi = None
                try:
                    import talib
                    rsi = talib.RSI(close.astype(float), timeperiod=period)
                except Exception:
                    rsi = Indicators.safe_rsi(close, period)
                r0 = float(rsi[-1] if hasattr(rsi, '__len__') and len(rsi)>0 else None)
                r1 = float(rsi[-2] if hasattr(rsi, '__len__') and len(rsi)>1 else None)
                r2 = float(rsi[-3] if hasattr(rsi, '__len__') and len(rsi)>2 else None)
            except Exception:
                r0 = r1 = r2 = None
            # Compose extended log line with available context
            try:
                # attempt to fetch ema200 and bias if annotated by RSIStrategy
                df1 = multi.get('1m')
                ema200 = None
                bias = None
                if df1 is not None:
                    try:
                        ema200 = df1.iloc[-1].get('ema200') if 'ema200' in df1.columns else None
                    except Exception:
                        ema200 = None
                # momentum value if present in cfg
                momentum_val = getattr(self.cfg, 'momentum_value', None)
            except Exception:
                ema200 = bias = momentum_val = None

            log_msg = {
                'ts': None,
                'timeframe': '1m',
                'r2': r2,
                'r1': r1,
                'r0': r0,
                'ema200': ema200,
                'bias': bias,
                'momentum': momentum_val,
                'decision': decision,
                'reason': getattr(self._impl, 'last_reason', None) if hasattr(self._impl, 'last_reason') else None,
            }
            # logger line
            self.logger.info('RSI debug: period=%s r2=%s r1=%s r0=%s decision=%s ema200=%s momentum=%s', period, r2, r1, r0, decision, ema200, momentum_val)

            # also append to per-experiment verbose log if configured
            if self.verbose_log_path:
                try:
                    import json
                    line = json.dumps(log_msg, default=str)
                    with open(self.verbose_log_path, 'a', encoding='utf-8') as vf:
                        vf.write(line + '\n')
                except Exception:
                    # don't break strategy on logging failure
                    pass
            return decision
        except Exception as e:
            self.logger.exception('Error in VerboseRSIStrategy.analyze: %s', e)
            return None
