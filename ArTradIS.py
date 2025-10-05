"""ArTradIS - minimal, clean demo-first trading orchestrator

Versión compacta y coherente. Dry-run por defecto. Evita referencias externas
no definidas y mantiene un CLI simple con modos: --teach, --explain,
--backtest, --optimize.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

try:
    import pandas as pd
    import numpy as np
except Exception:
    pd = None
    np = None

try:
    import talib  # type: ignore[reportMissingImports]
except Exception:
    talib = None

try:
    import ta
except Exception:
    ta = None

logger = logging.getLogger("ArTradIS")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class Config:
    symbol: str = "R_100"
    timeframes: List[str] = None
    history_size: int = 300
    dry_run: bool = True
    # expressed as decimal (0.02 == 2%)
    risk_per_trade_pct: float = 0.02
    atr_multiplier_sl: float = 1.5
    atr_multiplier_tp: float = 3.0
    api_token: Optional[str] = None
    chunk_usdt: float = 3.0
    rsi_period: int = 14
    eval_lookback: int = 200
    eval_min_trades: int = 3
    eval_min_winrate: float = 0.4
    eval_enabled: bool = True
    eval_double_confirm: bool = False
    eval_top_n: int = 2
    strategy_name: str = "default"
    # Strategy parameters (added for RSIStrategy tuning)
    overbought: int = 70
    oversold: int = 30
    # percentages expressed as decimals: 0.05 == 5%, 0.02 == 2%
    take_profit_pct: float = 0.05
    stop_loss_pct: float = 0.02
    enable_momentum: bool = False
    momentum_ema: int = 9
    momentum_threshold: float = 0.0
    momentum_required: bool = False
    enable_trend_filter: bool = False
    # Execution modes: 'DRY_RUN', 'DEMO_LIVE', 'PRODUCTION'
    EXECUTION_MODE: str = 'DRY_RUN'

    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ["1m", "5m", "1h"]
        # default multiplier for MULTIPLIER contracts
        if not hasattr(self, 'multiplier'):
            self.multiplier = 100

    @staticmethod
    def load_from(path: str) -> "Config":
        cfg = Config()
        if not os.path.exists(path):
            return cfg
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        except Exception:
            pass
        return cfg


def choose_multi_df(multi: dict, *preferred_keys):
    """Return the first dataframe from multi matching preferred_keys or any non-None value.

    Avoids relying on the truthiness of pandas objects which raises a ValueError.
    """
    if not isinstance(multi, dict):
        return None
    for k in preferred_keys:
        v = multi.get(k)
        if v is not None:
            return v
    for v in multi.values():
        if v is not None:
            return v
    return None


class DerivClient:
    def __init__(self, token: Optional[str] = None, dry_run: bool = True):
        self.token = token
        self.dry_run = dry_run
        # websocket retry controls
        self.ws_connection = None
        self.max_retries = 3
        self.retry_delay = 2

    def fetch_candles(self, symbol: str, timeframe: str, count: int):
        # Synthetic candles (dry-run). If pandas/numpy available return DataFrame.
        if pd is None or np is None:
            from random import gauss
            close = [1000.0 + gauss(0, 1) * (i * 0.01) for i in range(count)]
            openp = [c + gauss(0, 0.1) for c in close]
            high = [max(o, c) + abs(gauss(0, 0.5)) for o, c in zip(openp, close)]
            low = [min(o, c) - abs(gauss(0, 0.5)) for o, c in zip(openp, close)]
            return {'open': openp, 'high': high, 'low': low, 'close': close}
        now = pd.Timestamp.now()
        close = 1000 + np.cumsum(np.random.randn(count))
        openp = close + np.random.randn(count) * 0.5
        high = np.maximum(openp, close) + np.abs(np.random.randn(count))
        low = np.minimum(openp, close) - np.abs(np.random.randn(count))
        vol = np.random.randint(1, 100, size=count)
        df = pd.DataFrame({'timestamp': pd.date_range(end=now, periods=count, freq='T'), 'open': openp, 'high': high, 'low': low, 'close': close, 'volume': vol})
        return df

    def place_order(self, symbol: str, side: str, amount: float, stop_loss_pct: Optional[float] = None, take_profit_pct: Optional[float] = None):
        # Respect execution mode from Config
        mode = getattr(Config, 'EXECUTION_MODE', 'DRY_RUN')
        if mode == 'DRY_RUN' or self.dry_run:
            logger.info('DRY-RUN order: %s %s %.4f SL_pct=%s TP_pct=%s', symbol, side, amount, stop_loss_pct, take_profit_pct)
            return {'status': 'simulated', 'symbol': symbol, 'side': side, 'amount': amount}
        # MULTIPLIER contract: MULTUP (buy) or MULTDOWN (sell)
        contract_type = 'MULTUP' if side.lower() in ('buy', 'up', 'long') else 'MULTDOWN'

        # Get current price (required to compute absolute stop/take prices)
        current_price = self._get_current_price(symbol)
        if not current_price:
            logger.error('No current price available for %s; aborting order', symbol)
            return {'status': 'error', 'error': 'no_price'}

        # Convert percent inputs into absolute prices (stop_loss_pct = 0.01 => 1%)
        stop_loss_price = None
        take_profit_price = None
        try:
            if stop_loss_pct is not None:
                if side.lower() in ('buy', 'up', 'long'):
                    stop_loss_price = current_price * (1 - float(stop_loss_pct))
                else:
                    stop_loss_price = current_price * (1 + float(stop_loss_pct))
            if take_profit_pct is not None:
                if side.lower() in ('buy', 'up', 'long'):
                    take_profit_price = current_price * (1 + float(take_profit_pct))
                else:
                    take_profit_price = current_price * (1 - float(take_profit_pct))
        except Exception:
            stop_loss_price = None
            take_profit_price = None

        multiplier_val = getattr(Config, 'multiplier', 100)

        payload = {
            'buy': 1,
            'price': float(amount),
            'parameters': {
                'amount': float(amount),
                'basis': 'stake',
                'contract_type': contract_type,
                'currency': 'USD',
                'multiplier': multiplier_val,
                'symbol': symbol,
            }
        }

        # Deriv expects stop_loss/take_profit as absolute prices inside limit_order
        if stop_loss_price is not None or take_profit_price is not None:
            payload['parameters']['limit_order'] = {}
            if stop_loss_price is not None:
                payload['parameters']['limit_order']['stop_loss'] = round(stop_loss_price, 5)
            if take_profit_price is not None:
                payload['parameters']['limit_order']['take_profit'] = round(take_profit_price, 5)

        # Logging for verbose tracing
        logger.info('Preparing MULTIPLIER order %s %s', side, symbol)
        logger.info('  entry_price=%s stop=%s tp=%s multiplier=%s stake=%s', current_price, stop_loss_price, take_profit_price, multiplier_val, amount)
        try:
            resp = self._send_ws_request(payload)
            if mode == 'DEMO_LIVE':
                logger.info('DEMO_LIVE order placed (resp): %s', resp)
            else:
                logger.warning('PRODUCTION order placed (resp): %s', resp)
            # Log execution to file for tracking
            try:
                if isinstance(resp, dict) and resp.get('buy'):
                    self._log_trade_execution(resp, symbol, side, amount, current_price)
            except Exception:
                logger.exception('Failed to log trade execution')
            return resp
        except Exception as e:
            logger.exception('Failed to place order (mode=%s): %s', mode, e)
            return {'status': 'error', 'error': str(e), 'payload': payload}

    def place_order_with_killswitch(self, symbol: str, side: str, amount: float, stop_loss_pct: Optional[float] = None, take_profit_pct: Optional[float] = None, kill_switch: Optional[KillSwitch] = None, balance_usd: float = 1000.0, pct_limit: float = 5.0, virtual: bool = True):
        """Place an order but check KillSwitch and virtual flag first.

        If virtual is True or client.dry_run is True then only simulate.
        If a KillSwitch is provided and reports blocked, do not send order.
        """
        if virtual or self.dry_run:
            logger.info('VIRTUAL/DRY-RUN order (not sent): %s %s %.4f', symbol, side, amount)
            return {'status': 'simulated', 'symbol': symbol, 'side': side, 'amount': amount}
        if kill_switch is not None:
            try:
                if kill_switch.is_blocked(balance_usd, pct_limit):
                    logger.warning('KillSwitch engaged: refusing to place order')
                    return {'status': 'blocked', 'reason': 'killswitch'}
            except Exception:
                logger.exception('Failed to evaluate KillSwitch; aborting order')
                return {'status': 'error', 'reason': 'killswitch_error'}
        # In a real integration here we'd call the Deriv REST/ws API to place the order.
        logger.info('Placing order: %s %s %.4f SL_pct=%s TP_pct=%s', symbol, side, amount, stop_loss_pct, take_profit_pct)
        return self.place_order(symbol, side, amount, stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct)

    def fetch_candles_live(self, symbol: str, granularity: int, count: int = 200, timeout: int = 10):
        """Fetch recent candles from Deriv via websocket ticks_history (style=candles).

        This method opens a short-lived websocket connection, authorizes with token,
        requests ticks_history and returns a DataFrame (if pandas available) or dict.
        """
        if not self.token:
            raise RuntimeError('No API token provided for live fetch')
        import websocket as wsclient

        url = 'wss://ws.binaryws.com/websockets/v3?app_id=1089'
        conn = wsclient.create_connection(url, timeout=timeout)
        raw = None
        try:
            conn.send(json.dumps({'authorize': self.token}))
            _ = conn.recv()
            req = {'ticks_history': symbol, 'end': 'latest', 'count': count, 'granularity': granularity, 'style': 'candles'}
            conn.send(json.dumps(req))
            raw = conn.recv()
        finally:
            try:
                conn.close()
            except Exception:
                pass

        try:
            data = json.loads(raw)
        except Exception:
            # save raw payload for debugging
            with open('live_raw_response_error.json', 'w', encoding='utf-8') as f:
                f.write(raw if isinstance(raw, str) else str(raw))
            raise RuntimeError('Failed to parse JSON response from market; raw saved to live_raw_response_error.json')

        # Prefer top-level 'candles'
        candles = []
        if isinstance(data, dict):
            if 'candles' in data and isinstance(data['candles'], list):
                candles = data['candles']
            elif 'history' in data and isinstance(data['history'], dict) and 'candles' in data['history']:
                candles = data['history']['candles']
            elif 'candles' in data and isinstance(data['candles'], dict) and 'candles' in data['candles']:
                candles = data['candles']['candles']

        if not candles:
            with open('live_raw_response_error.json', 'w', encoding='utf-8') as f:
                f.write(json.dumps(data))
            raise RuntimeError('No candles found in market response; full payload saved to live_raw_response_error.json')

        # Build normalized list
        normalized = []
        for c in candles:
            try:
                normalized.append({'open': float(c.get('open')), 'high': float(c.get('high')), 'low': float(c.get('low')), 'close': float(c.get('close'))})
            except Exception:
                # skip invalid
                continue

        if not normalized:
            with open('live_raw_response_error.json', 'w', encoding='utf-8') as f:
                f.write(json.dumps(data))
            raise RuntimeError('Candles present but could not be normalized; payload saved to live_raw_response_error.json')

        if pd is not None:
            return pd.DataFrame(normalized)
        return {'open': [c['open'] for c in normalized], 'high': [c['high'] for c in normalized], 'low': [c['low'] for c in normalized], 'close': [c['close'] for c in normalized]}

    def _send_ws_request(self, payload: dict, timeout: int = 10):
        """Helper: enviar petición por websocket y devolver JSON con retries/backoff.

        Implementa reintentos simples para mejorar robustez en demo/live.
        """
        if not self.token:
            raise RuntimeError('No API token provided for ws requests')
        import websocket as wsclient
        import socket

        url = 'wss://ws.binaryws.com/websockets/v3?app_id=1089'

        attempt = 0
        last_err = None
        while attempt < getattr(self, 'max_retries', 3):
            attempt += 1
            try:
                conn = wsclient.create_connection(url, timeout=timeout)
                try:
                    conn.send(json.dumps({'authorize': self.token}))
                    _ = conn.recv()
                    conn.send(json.dumps(payload))
                    raw = conn.recv()
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass
                try:
                    return json.loads(raw)
                except Exception:
                    return {'error_raw': raw}
            except (socket.error, Exception) as e:
                last_err = e
                logger.warning('WebSocket error (attempt %d/%d): %s', attempt, getattr(self, 'max_retries', 3), e)
                # backoff
                time.sleep(getattr(self, 'retry_delay', 2) * attempt)
                continue
        # if we exit loop
        logger.error('WebSocket request failed after %d attempts: %s', getattr(self, 'max_retries', 3), last_err)
        raise last_err

    def get_balance(self):
        """Obtener balance de cuenta (USD) mediante petición ws simple.

        Retorna float(balance) o None si no disponible.
        """
        try:
            payload = {'balance': 1, 'subscribe': 0}
            resp = self._send_ws_request(payload)
            if isinstance(resp, dict):
                if 'error' in resp:
                    return None
                if 'balance' in resp and isinstance(resp['balance'], dict):
                    val = resp['balance'].get('balance') or resp['balance'].get('amount')
                    try:
                        return float(val)
                    except Exception:
                        return None
                for v in resp.values():
                    try:
                        return float(v)
                    except Exception:
                        continue
            return None
        except Exception:
            return None

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Return latest tick price for symbol or None."""
        try:
            payload = {'ticks': symbol, 'subscribe': 0}
            resp = self._send_ws_request(payload)
            if isinstance(resp, dict) and 'tick' in resp:
                q = resp['tick'].get('quote') or resp['tick'].get('price')
                try:
                    return float(q)
                except Exception:
                    return None
        except Exception:
            return None
        return None

    def _log_trade_execution(self, response, symbol, side, amount, entry_price):
        try:
            import json
            from datetime import datetime
            trade_log = {
                'timestamp': datetime.utcnow().isoformat(),
                'contract_id': None,
                'symbol': symbol,
                'side': side.upper(),
                'entry_price': float(entry_price) if entry_price is not None else None,
                'stake': float(amount),
                'status': 'opened',
                'response': response
            }
            # response['buy'] can be dict with contract_id or numeric id; attempt best-effort
            try:
                buy = response.get('buy') if isinstance(response, dict) else None
                if isinstance(buy, dict):
                    trade_log['contract_id'] = buy.get('contract_id') or buy.get('id')
                elif isinstance(buy, (int, float, str)):
                    trade_log['contract_id'] = str(buy)
            except Exception:
                pass

            os.makedirs('logs', exist_ok=True)
            with open(os.path.join('logs', 'trade_executions.jsonl'), 'a', encoding='utf-8') as f:
                f.write(json.dumps(trade_log) + '\n')
        except Exception:
            logger.exception('Failed to write trade execution log')

    def get_open_positions(self) -> list:
        """Versión robusta: pide portfolio y parsea defensivamente.

        Retorna lista de posiciones con esquema:
        symbol, contract_id, side, entry_price, current_price, profit_usd, profit_pct, status, opened_at, stake
        """
        if getattr(Config, 'EXECUTION_MODE', 'DRY_RUN') == 'DRY_RUN':
            return []

        try:
            payload = {"portfolio": 1}
            response = self._send_ws_request(payload)

            if not response:
                logger.warning("Response vacío en get_open_positions")
                return []

            if 'portfolio' not in response:
                logger.warning(f"Respuesta sin 'portfolio': {response}")
                return []

            contracts = response.get('portfolio', {}).get('contracts', [])
            if not isinstance(contracts, list):
                logger.error(f"'contracts' no es lista: {type(contracts)}")
                return []

            positions = []
            for contract in contracts:
                try:
                    contract_type = contract.get('contract_type', '')
                    if contract_type not in ['MULTUP', 'MULTDOWN']:
                        continue

                    pos = {
                        'contract_id': str(contract.get('contract_id', '')),
                        'symbol': contract.get('underlying') or contract.get('symbol', 'UNKNOWN'),
                        'side': 'BUY' if contract_type == 'MULTUP' else 'SELL',
                        'entry_price': float(contract.get('buy_price', 0)),
                        'current_price': float(contract.get('bid_price', 0)),
                        'profit_usd': float(contract.get('profit', 0)),
                        'profit_pct': float(contract.get('profit_percentage', 0)),
                        'status': 'open',
                        'opened_at': contract.get('date_start', ''),
                        'stake': float(contract.get('payout', 0))
                    }

                    positions.append(pos)

                except (ValueError, TypeError) as e:
                    logger.warning(f"Error parseando contrato {contract.get('contract_id')}: {e}")
                    continue

            return positions

        except Exception as e:
            logger.error(f"Error crítico en get_open_positions: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_today_trades(self) -> list:
        """Return simple list of today's trades by requesting history (demo)."""
        try:
            payload = {'transactions': 1, 'description': 1}
            resp = self._send_ws_request(payload)
            out = []
            if not isinstance(resp, dict):
                return out
            txs = resp.get('transactions') or []
            from datetime import datetime
            today = datetime.utcnow().date()
            for t in txs:
                try:
                    dt = t.get('transaction_time') or t.get('date') or t.get('time')
                    if not dt:
                        continue
                    # transaction_time often like '2025-09-30 12:34:56'
                    if isinstance(dt, str) and dt.startswith(str(today)):
                        out.append(t)
                except Exception:
                    continue
            return out
        except Exception:
            return []


class Indicators:
    @staticmethod
    def safe_rsi(series, period: int = 14):
        try:
            if pd is not None and isinstance(series, pd.Series):
                delta = series.diff()
                up = delta.clip(lower=0).rolling(period).mean()
                down = -delta.clip(upper=0).rolling(period).mean()
                rs = up / down
                return 100 - (100 / (1 + rs))
        except Exception:
            pass
        vals = list(series)
        if len(vals) < period + 1:
            return [50] * len(vals)
        deltas = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
        rsis = [50]
        up = 0.0
        down = 0.0
        for i in range(len(deltas)):
            d = deltas[i]
            up = (up * (period - 1) + max(d, 0)) / period if i >= period - 1 else up + max(d, 0)
            down = (down * (period - 1) + max(-d, 0)) / period if i >= period - 1 else down + max(-d, 0)
            if down == 0:
                rsis.append(100.0)
            else:
                rs = up / down
                rsis.append(100 - (100 / (1 + rs)))
        return rsis

    @staticmethod
    def compute(df):
        if pd is not None and hasattr(df, 'copy'):
            out = df.copy()
            try:
                if talib is not None:
                    out['rsi'] = talib.RSI(out['close'].astype(float), timeperiod=14)
                    out['sma50'] = talib.SMA(out['close'].astype(float), timeperiod=50)
                    out['sma200'] = talib.SMA(out['close'].astype(float), timeperiod=200)
                    out['atr'] = talib.ATR(out['high'].astype(float), out['low'].astype(float), out['close'].astype(float), timeperiod=14)
                else:
                    out['rsi'] = Indicators.safe_rsi(out['close'], 14)
                    out['sma50'] = out['close'].rolling(50).mean()
                    out['sma200'] = out['close'].rolling(200).mean()
                    out['atr'] = out['high'] - out['low']
                if ta is not None:
                    try:
                        from ta.trend import EMAIndicator
                        from ta.volatility import BollingerBands
                        out['ema20'] = EMAIndicator(out['close'].astype(float), window=20).ema_indicator()
                        out['ema200'] = EMAIndicator(out['close'].astype(float), window=200).ema_indicator()
                        bb = BollingerBands(out['close'].astype(float), window=20, window_dev=2)
                        out['bb_mid'] = bb.bollinger_mavg()
                        out['bb_high'] = bb.bollinger_hband()
                        out['bb_low'] = bb.bollinger_lband()
                    except Exception:
                        pass
            except Exception:
                out['rsi'] = 50
                out['sma50'] = out['close'].rolling(50).mean() if pd is not None else None
                out['atr'] = out['high'] - out['low']
            return out
        if isinstance(df, dict):
            close = df.get('close', [])
            rsi = Indicators.safe_rsi(close, 14)
            sma50 = []
            for i in range(len(close)):
                window = close[max(0, i - 49):i + 1]
                sma50.append(sum(window) / len(window) if window else 0)
            atr = [h - l for h, l in zip(df.get('high', []), df.get('low', []))]
            out = dict(df)
            out['rsi'] = rsi
            out['sma50'] = sma50
            out['atr'] = atr
            return out
        raise RuntimeError('Unsupported df type for indicators')

    # --- Additional quantitative utilities requested by the user ---
    @staticmethod
    def sma(series, period: int):
        """Simple Moving Average for a sequence or pandas Series."""
        if pd is not None and hasattr(series, 'rolling'):
            return series.rolling(window=period).mean()
        vals = list(series)
        out = []
        for i in range(len(vals)):
            window = vals[max(0, i - period + 1):i + 1]
            out.append(sum(window) / len(window) if window else 0.0)
        return out

    @staticmethod
    def ema(series, period: int):
        """Exponential Moving Average calculated iteratively."""
        alpha = 2.0 / (period + 1)
        if pd is not None and hasattr(series, 'ewm'):
            return series.ewm(alpha=alpha, adjust=False).mean()
        vals = list(series)
        out = []
        ema_prev = None
        for v in vals:
            if ema_prev is None:
                ema_prev = v
            else:
                ema_prev = (v * alpha) + (ema_prev * (1 - alpha))
            out.append(ema_prev)
        return out

    @staticmethod
    def rsi(series, period: int = 14):
        """Calculate RSI following the standard Wilder smoothing method."""
        # prefer talib if available
        try:
            if talib is not None:
                return talib.RSI(series.astype(float), timeperiod=period)
        except Exception:
            pass
        return Indicators.safe_rsi(series, period)

    @staticmethod
    def macd(series):
        """Return (macd_line, signal_line, hist) using 12/26/9 default periods."""
        try:
            if talib is not None:
                macd, signal, hist = talib.MACD(series.astype(float), fastperiod=12, slowperiod=26, signalperiod=9)
                return macd, signal, hist
            if pd is not None:
                ema12 = series.ewm(span=12, adjust=False).mean()
                ema26 = series.ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9, adjust=False).mean()
                hist = macd - signal
                return macd, signal, hist
        except Exception:
            pass
        # fallback
        n = len(series)
        return [0] * n, [0] * n, [0] * n

    # --- Risk and position sizing ---
    @staticmethod
    def position_size_by_risk(balance: float, risk_pct: float, entry_price: float, stop_loss: float, min_chunk: float = 1.0) -> float:
        """Calculate position size (units) given balance, % risk and distance to SL.

        Formula: (Balance * risk_pct) / |entry - stop_loss|
        Returns a size rounded down to a multiple of min_chunk.
        """
        if balance <= 0 or risk_pct <= 0:
            return 0.0
        sl_dist = abs(entry_price - stop_loss)
        if sl_dist <= 0:
            return 0.0
        risk_amount = balance * (risk_pct / 100.0)
        raw_size = risk_amount / sl_dist
        # quantize to min_chunk
        chunks = int(raw_size // min_chunk)
        if chunks <= 0:
            return min(raw_size, min_chunk)
        return float(chunks * min_chunk)

    @staticmethod
    def rr_ratio(entry_price: float, exit_price: float, stop_loss: float) -> Optional[float]:
        """Return Reward-to-Risk ratio (R:R) for a trade."""
        denom = abs(entry_price - stop_loss)
        if denom == 0:
            return None
        return (abs(exit_price - entry_price) / denom)

    # --- Backtest metrics ---
    @staticmethod
    def net_profit(trades: List[Dict[str, Any]]) -> float:
        """Sum of PnL values in trades list. Each trade dict expected to have 'pnl'."""
        s = 0.0
        for t in trades:
            try:
                s += float(t.get('pnl', 0.0))
            except Exception:
                continue
        return s

    @staticmethod
    def profit_factor(trades: List[Dict[str, Any]]) -> Optional[float]:
        """Profit Factor = sum(wins) / abs(sum(losses))"""
        wins = 0.0
        losses = 0.0
        for t in trades:
            try:
                pnl = float(t.get('pnl', 0.0))
            except Exception:
                continue
            if pnl > 0:
                wins += pnl
            elif pnl < 0:
                losses += pnl
        if losses == 0:
            return None if wins == 0 else float('inf')
        return wins / abs(losses)

    @staticmethod
    def max_drawdown(equity_curve: List[float]) -> float:
        """Compute maximum drawdown given a list of equity values over time."""
        peak = float('-inf')
        max_dd = 0.0
        for v in equity_curve:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak and peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @staticmethod
    def true_range(high_series, low_series, close_series):
        """Compute True Range series for arrays or pandas Series."""
        # Accept pandas Series
        if pd is not None and hasattr(high_series, 'iloc'):
            h = high_series.astype(float)
            l = low_series.astype(float)
            c = close_series.astype(float)
            prev_close = c.shift(1).fillna(method='bfill')
            tr1 = h - l
            tr2 = (h - prev_close).abs()
            tr3 = (l - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr
        # fallback lists
        hs = list(high_series)
        ls = list(low_series)
        cs = list(close_series)
        tr = []
        for i in range(len(hs)):
            prev = cs[i-1] if i > 0 else cs[0]
            tr1 = hs[i] - ls[i]
            tr2 = abs(hs[i] - prev)
            tr3 = abs(ls[i] - prev)
            tr.append(max(tr1, tr2, tr3))
        return tr

    @staticmethod
    def atr(high_series, low_series, close_series, period: int = 14):
        """Average True Range using Wilder smoothing (period default 14)."""
        try:
            tr = Indicators.true_range(high_series, low_series, close_series)
            if pd is not None and hasattr(tr, 'rolling'):
                # Wilder smoothing: first ATR = SMA(TR, period), then ATR_t = (ATR_{t-1}*(period-1) + TR_t)/period
                first = tr.rolling(window=period).mean()
                atr_series = first.copy()
                for i in range(period, len(tr)):
                    prev_atr = atr_series.iloc[i-1]
                    atr_series.iloc[i] = (prev_atr * (period - 1) + tr.iloc[i]) / period
                return atr_series
            # fallback list
            trs = list(tr)
            atrs = []
            if len(trs) < period:
                # simple average
                avg = sum(trs) / len(trs) if trs else 0.0
                return [avg] * len(trs)
            # initial ATR as SMA
            init = sum(trs[:period]) / period
            atrs = [init] * period
            prev = init
            for i in range(period, len(trs)):
                cur = (prev * (period - 1) + trs[i]) / period
                atrs.append(cur)
                prev = cur
            return atrs
        except Exception:
            # safe fallback: return zeros
            n = len(close_series) if hasattr(close_series, '__len__') else 0
            return [0.0] * n


# --- Incremental / low-latency helpers ---
def update_ema(prev_ema: Optional[float], price: float, period: int) -> float:
    """Update EMA incrementally for a new price point."""
    alpha = 2.0 / (period + 1)
    if prev_ema is None:
        return price
    return price * alpha + prev_ema * (1 - alpha)


def wilder_rsi_update(prev_avg_gain: Optional[float], prev_avg_loss: Optional[float], gain: float, loss: float, period: int):
    """Update Wilder RSI averages incrementally and return rsi value and new averages."""
    if prev_avg_gain is None or prev_avg_loss is None:
        # initial seed
        avg_gain = gain
        avg_loss = loss
    else:
        avg_gain = (prev_avg_gain * (period - 1) + gain) / period
        avg_loss = (prev_avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi, avg_gain, avg_loss


def incremental_macd(prev_ema12: Optional[float], prev_ema26: Optional[float], price: float):
    """Given previous EMA12 and EMA26, update them and return macd line value.
    Note: caller should maintain prev_ema12/26 using update_ema.
    """
    # caller is responsible for periods; this helper just computes macd from provided emas
    if prev_ema12 is None or prev_ema26 is None:
        return 0.0
    return float(prev_ema12 - prev_ema26)


class RealTimeProcessor:
    """Lightweight processor that keeps fixed-length deques per timeframe and computes incremental indicators.

    Purpose: avoid recreating big DataFrames on each tick and allow near-instant signal evaluation.
    """
    def __init__(self, cfg: Config, strategies: Dict[str, StrategyBase], maxlen: int = 500):
        self.cfg = cfg
        self.buffers: Dict[str, deque] = {tf: deque(maxlen=maxlen) for tf in cfg.timeframes}
        self.strategies = strategies
        self.ema_state: Dict[str, Dict[str, Optional[float]]] = {}
        # store per-tf EMA12/26 for macd
        for tf in cfg.timeframes:
            self.ema_state[tf] = {'ema12': None, 'ema26': None, 'ema9': None}

    def push_tick(self, timeframe: str, o: float, h: float, l: float, c: float):
        buf = self.buffers.get(timeframe)
        if buf is None:
            return
        buf.append({'open': o, 'high': h, 'low': l, 'close': c})

    def _to_light_df(self, timeframe: str):
        """Return a minimal pandas-like object or dict expected by strategies (close series and last row)."""
        buf = self.buffers.get(timeframe, None)
        if not buf:
            return None
        closes = [b['close'] for b in buf]
        highs = [b['high'] for b in buf]
        lows = [b['low'] for b in buf]
        # prefer pandas Series if available
        if pd is not None:
            try:
                import pandas as _pd
                df = _pd.DataFrame({'open': [b['open'] for b in buf], 'high': highs, 'low': lows, 'close': closes})
                return Indicators.compute(df)
            except Exception:
                pass
        # fallback dict with lists and last-row access via index
        return {'open': [b['open'] for b in buf], 'high': highs, 'low': lows, 'close': closes}

    def evaluate(self) -> Tuple[Optional[str], float, List[str]]:
        # build lightweight multi dict
        multi = {}
        for tf in self.buffers.keys():
            v = self._to_light_df(tf)
            multi[tf] = v
        aggregator = SignalAggregator(self.strategies)
        return aggregator.aggregate(multi)


class RiskManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.daily_loss = 0.0
        self.trades_today = 0
        self.daily_loss_limit_pct = 5.0

    def compute_position_size(self, balance: float, sl_dist: float, chunk_usdt: float = 3.0) -> float:
        if sl_dist <= 0 or balance <= 0:
            return 0.0
        # risk_per_trade_pct is decimal (0.02 == 2%)
        max_risk = balance * (self.cfg.risk_per_trade_pct)
        size = max_risk / sl_dist if sl_dist > 0 else 0
        chunks = int(size // chunk_usdt)
        if chunks <= 0:
            chosen = min(chunk_usdt, size)
        else:
            chosen = chunks * chunk_usdt
        return float(min(chosen, balance))

    def dynamic_sl_tp(self, price: float, atr: float, direction: str) -> Tuple[float, float]:
        sl = atr * self.cfg.atr_multiplier_sl
        tp = atr * self.cfg.atr_multiplier_tp
        if direction == 'buy':
            return price - sl, price + tp
        return price + sl, price - tp


class KillSwitch:
    """Persistente y sencillo: guarda en JSON pérdidas del día y bloqueo."""
    def __init__(self, path: str = 'kill_switch.json'):
        self.path = Path(path)
        if not self.path.exists():
            self._write({'date': self._today_str(), 'daily_loss_usd': 0.0, 'enabled': True})

    def _today_str(self) -> str:
        return time.strftime('%Y-%m-%d')

    def _read(self) -> Dict[str, Any]:
        try:
            with self.path.open('r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {'date': self._today_str(), 'daily_loss_usd': 0.0, 'enabled': True}

    def _write(self, data: Dict[str, Any]):
        with self.path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def record_loss(self, amount: float):
        d = self._read()
        if d.get('date') != self._today_str():
            d = {'date': self._today_str(), 'daily_loss_usd': 0.0, 'enabled': True}
        d['daily_loss_usd'] = float(d.get('daily_loss_usd', 0.0)) + float(amount)
        self._write(d)

    def get_status(self) -> Dict[str, Any]:
        d = self._read()
        if d.get('date') != self._today_str():
            d = {'date': self._today_str(), 'daily_loss_usd': 0.0, 'enabled': True}
            self._write(d)
        return d

    def is_blocked(self, balance_usd: float, pct_limit: float = 5.0) -> bool:
        d = self.get_status()
        if not d.get('enabled', True):
            return True
        limit = balance_usd * (pct_limit / 100.0)
        return float(d.get('daily_loss_usd', 0.0)) >= limit


class TradeRecorder:
    def __init__(self, path: str = 'trades.csv'):
        self.path = Path(path)
        self.fields = ['timestamp', 'symbol', 'side', 'amount', 'stop_loss', 'take_profit', 'status', 'reason', 'atr', 'pnl']
        if not self.path.exists():
            with self.path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()

    def record(self, row: Dict[str, Any]):
        # ensure default fields exist
        clean = {k: row.get(k, '') for k in self.fields}
        with self.path.open('a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(clean)


def generate_daily_report(trades_csv: str = 'trades.csv', out_csv: str = None):
    """Generate a daily summary report (CSV) with success rate, net profit, profit factor and max drawdown.

    If out_csv is None, write to trades_daily_report.csv
    """
    if out_csv is None:
        out_csv = 'trades_daily_report.csv'
    if not os.path.exists(trades_csv):
        raise RuntimeError('No trades file found: %s' % trades_csv)
    rows = []
    with open(trades_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        raise RuntimeError('No trades to report')
    # group by date
    from collections import defaultdict
    by_date = defaultdict(list)
    for r in rows:
        ts = r.get('timestamp', '')
        date = ts.split(' ')[0] if ts else 'unknown'
        try:
            pnl = float(r.get('pnl') or 0.0)
        except Exception:
            pnl = 0.0
        by_date[date].append(pnl)

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'trades', 'wins', 'winrate', 'net_profit', 'profit_factor', 'max_drawdown'])
        for d, pnls in sorted(by_date.items()):
            trades = len(pnls)
            wins = sum(1 for p in pnls if p > 0)
            winrate = wins / trades if trades else 0.0
            net = sum(pnls)
            wins_sum = sum(p for p in pnls if p > 0)
            losses_sum = sum(p for p in pnls if p < 0)
            pf = (wins_sum / abs(losses_sum)) if losses_sum != 0 else (float('inf') if wins_sum != 0 else None)
            # approximate equity curve for drawdown
            eq = []
            cum = 0.0
            for p in pnls:
                cum += p
                eq.append(cum)
            dd = Indicators.max_drawdown(eq) if eq else 0.0
            writer.writerow([d, trades, wins, '%.4f' % winrate, '%.4f' % net, (pf if pf is None or pf == float('inf') else '%.4f' % pf), '%.4f' % dd])


class StrategyBase:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def analyze(self, multi: Dict[str, Any]) -> Optional[str]:
        raise NotImplementedError()


class TrendFollowingAdvanced(StrategyBase):
    """Multi-timeframe trend-following as described by the user.

    - 1h: EMA200 to determine primary trend
    - 15m: EMA50 for pullback detection
    - 1m: RSI confirmation (cross below 30 then back above for buy; above 70 then back below for sell)
    """
    def analyze(self, multi: Dict[str, Any]) -> Optional[str]:
        # expect pandas DataFrames in multi
        try:
            df1h = multi.get('1h')
            df15 = multi.get('15m')
            df1 = multi.get('1m')
        except Exception:
            return None
        if df1h is None or df15 is None or df1 is None:
            return None
        try:
            last1h = df1h.iloc[-1]
            ema200_1h = float(last1h.get('ema200') or last1h.get('sma200') or 0)
            price1h = float(last1h['close'])
        except Exception:
            return None
        # Determine trend
        if price1h > ema200_1h:
            primary = 'bull'
        elif price1h < ema200_1h:
            primary = 'bear'
        else:
            return None

        # Pullback on 15m
        try:
            last15 = df15.iloc[-1]
            ema50_15 = float(last15.get('ema50') or last15.get('sma50') or 0)
            price15 = float(last15['close'])
        except Exception:
            return None

        if primary == 'bull' and price15 >= ema50_15:
            # not a pullback
            return None
        if primary == 'bear' and price15 <= ema50_15:
            return None

        # RSI confirmation on 1m: need a recent cross
        try:
            rsi_series = df1['rsi'] if 'rsi' in df1.columns else df1['rsi']
        except Exception:
            return None
        if len(rsi_series) < 3:
            return None
        r0 = float(rsi_series.iloc[-1])
        r1 = float(rsi_series.iloc[-2])
        r2 = float(rsi_series.iloc[-3])

        # Buy condition: RSI dipped below 30 then rose above
        if primary == 'bull' and r2 < 30 and r1 < 30 and r0 > 30:
            return 'buy'
        # Sell condition: RSI went above 70 then dropped below
        if primary == 'bear' and r2 > 70 and r1 > 70 and r0 < 70:
            return 'sell'
        return None


class SignalAggregator:
    """Combine multiple strategy signals to produce a final action.

    The aggregator will count confirmations (number of strategies that signal buy or sell).
    If more strategies agree on the same direction, return that direction and a confidence score.
    """
    def __init__(self, strategies: Dict[str, StrategyBase]):
        self.strategies = strategies

    def aggregate(self, multi: Dict[str, Any]) -> Tuple[Optional[str], float, List[str]]:
        counts = {'buy': 0, 'sell': 0}
        confirmers: List[str] = []
        for name, strat in self.strategies.items():
            try:
                sig = strat.analyze(multi)
            except Exception:
                sig = None
            if sig in counts:
                counts[sig] += 1
                confirmers.append(name)
        total = sum(counts.values())
        if total == 0:
            return None, 0.0, []
        if counts['buy'] > counts['sell']:
            return 'buy', counts['buy'] / total, confirmers
        if counts['sell'] > counts['buy']:
            return 'sell', counts['sell'] / total, confirmers
        return None, 0.0, confirmers


class TrendFollowingStrategy(StrategyBase):
    def analyze(self, multi: Dict[str, Any]) -> Optional[str]:
        df = choose_multi_df(multi, '1h', '5m')
        if df is None:
            return None
        try:
            last = df.iloc[-1]
            sma50 = float(last.get('sma50') or 0)
            sma200 = float(last.get('sma200') or 0)
        except Exception:
            return None
        if sma50 > sma200:
            return 'buy'
        if sma50 < sma200:
            return 'sell'
        return None


class RangeTradingStrategy(StrategyBase):
    def analyze(self, multi: Dict[str, Any]) -> Optional[str]:
        df = choose_multi_df(multi, '5m')
        if df is None:
            return None
        try:
            last = df.iloc[-1]
            rsi = float(last.get('rsi') or 50)
        except Exception:
            return None
        if rsi < 35:
            return 'buy'
        if rsi > 65:
            return 'sell'
        return None


class ScalpingStrategy(StrategyBase):
    def analyze(self, multi: Dict[str, Any]) -> Optional[str]:
        df = choose_multi_df(multi, '1m')
        if df is None or len(df) < 3:
            return None
        try:
            c0 = float(df.iloc[-1]['close'])
            c1 = float(df.iloc[-2]['close'])
            c2 = float(df.iloc[-3]['close'])
        except Exception:
            return None
        if c0 > c1 > c2:
            return 'buy'
        if c0 < c1 < c2:
            return 'sell'
        return None


StrategyFactory = {
    'trend': TrendFollowingStrategy,
    'range': RangeTradingStrategy,
    'scalp': ScalpingStrategy,
}

# Note: MultipleTimeFrameStrategy is defined later and is registered in the live strategies mapping.


class MovingAverageStrategy(StrategyBase):
    """Estrategia de medias móviles: cruces EMA50/EMA200 con confirmación MACD/RSI.

    - Señal de compra cuando EMA50 cruza por encima de EMA200 (golden cross)
    - Señal de venta cuando EMA50 cruza por debajo de EMA200 (death cross)
    - Requiere confirmación de MACD (histograma) o RSI para evitar falsas señales
    """
    def __init__(self, cfg: Config):
        super().__init__(cfg)

    def _ema(self, series, period: int):
        try:
            if talib is not None:
                return talib.EMA(series.astype(float), timeperiod=period)
            if pd is not None:
                return series.ewm(span=period, adjust=False).mean()
        except Exception:
            pass
        vals = list(series)
        out = []
        for i in range(len(vals)):
            window = vals[max(0, i - period + 1):i + 1]
            out.append(sum(window) / len(window) if window else vals[i])
        return out

    def _macd_hist(self, series):
        try:
            if talib is not None:
                macd, macdsignal, macdhist = talib.MACD(series.astype(float))
                return macdhist
            if pd is not None:
                ema12 = series.ewm(span=12, adjust=False).mean()
                ema26 = series.ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9, adjust=False).mean()
                return macd - signal
        except Exception:
            pass
        return [0] * len(series)

    def analyze(self, multi: Dict[str, Any]) -> Optional[str]:
        # Prefer 1h for trend detection
        df = choose_multi_df(multi, '1h')
        if df is None:
            return None
        try:
            close = df['close']
            ema50 = self._ema(close, 50)
            ema200 = self._ema(close, 200)
            if len(ema50) < 2 or len(ema200) < 2:
                return None
            e0 = float(ema50[-1] if isinstance(ema50, list) else ema50.iloc[-1])
            e1 = float(ema50[-2] if isinstance(ema50, list) else ema50.iloc[-2])
            E0 = float(ema200[-1] if isinstance(ema200, list) else ema200.iloc[-1])
            E1 = float(ema200[-2] if isinstance(ema200, list) else ema200.iloc[-2])

            # detect cross
            # golden cross: previous e1 <= E1 and current e0 > E0
            if e1 <= E1 and e0 > E0:
                # confirm with MACD or RSI
                macdh = self._macd_hist(close)
                macdh_last = float(macdh[-1] if isinstance(macdh, list) else macdh.iloc[-1])
                try:
                    rsi = talib.RSI(close.astype(float), timeperiod=14) if talib is not None else Indicators.safe_rsi(close, 14)
                    r0 = float(rsi[-1] if isinstance(rsi, list) else rsi.iloc[-1])
                except Exception:
                    r0 = 50
                if macdh_last > 0 or r0 > 50:
                    return 'buy'
            # death cross
            if e1 >= E1 and e0 < E0:
                macdh = self._macd_hist(close)
                macdh_last = float(macdh[-1] if isinstance(macdh, list) else macdh.iloc[-1])
                try:
                    rsi = talib.RSI(close.astype(float), timeperiod=14) if talib is not None else Indicators.safe_rsi(close, 14)
                    r0 = float(rsi[-1] if isinstance(rsi, list) else rsi.iloc[-1])
                except Exception:
                    r0 = 50
                if macdh_last < 0 or r0 < 50:
                    return 'sell'
        except Exception:
            return None
        return None


class BollingerBandsStrategy(StrategyBase):
    """Bollinger Bands strategy with RSI confirmation.

    - Entry buy when price touches or crosses below lower band and RSI < 30
    - Entry sell when price touches or crosses above upper band and RSI > 70
    - TP: target opposite band
    - SL: placed outside the opposite band (simple heuristic)
    """
    def __init__(self, cfg: Config, window: int = 20, dev: float = 2.0):
        super().__init__(cfg)
        self.window = window
        self.dev = dev

    def _bollinger(self, series):
        if pd is not None:
            mid = series.rolling(self.window).mean()
            std = series.rolling(self.window).std()
            return mid - self.dev * std, mid + self.dev * std, mid
        vals = list(series)
        mid = []
        lower = []
        upper = []
        for i in range(len(vals)):
            window = vals[max(0, i - self.window + 1):i + 1]
            m = sum(window) / len(window) if window else vals[i]
            s = (sum((x - m) ** 2 for x in window) / len(window)) ** 0.5 if window else 0
            mid.append(m)
            lower.append(m - self.dev * s)
            upper.append(m + self.dev * s)
        return lower, upper, mid

    def analyze(self, multi: Dict[str, Any]) -> Optional[str]:
        # prefer 15m or 5m for range entries
        df = choose_multi_df(multi, '15m', '5m')
        if df is None:
            return None
        try:
            close = df['close']
            high = df.get('high', close)
            low = df.get('low', close)
            lower, upper, mid = self._bollinger(close)
            if not lower or not upper:
                return None
            price = float(df.iloc[-1]['close']) if hasattr(df, 'iloc') else close[-1]
            rsi_series = None
            try:
                if talib is not None:
                    rsi_series = talib.RSI(close.astype(float), timeperiod=14)
                else:
                    rsi_series = Indicators.safe_rsi(close, 14)
            except Exception:
                rsi_series = Indicators.safe_rsi(close, 14)
            r0 = float(rsi_series[-1] if isinstance(rsi_series, list) else rsi_series.iloc[-1])

            lb = lower[-1] if isinstance(lower, list) else lower.iloc[-1]
            ub = upper[-1] if isinstance(upper, list) else upper.iloc[-1]

            # Buy: price <= lower band and RSI < 30
            if price <= lb and r0 < 30:
                return 'buy'
            # Sell: price >= upper band and RSI > 70
            if price >= ub and r0 > 70:
                return 'sell'
        except Exception:
            return None
        return None


class RSIStrategy(StrategyBase):
    """RSI based strategy.

    - Reversal entries: RSI crosses above 30 (from below) -> buy; RSI crosses below 70 (from above) -> sell
    - Confirmation: if 1h EMA200 available, use it to determine bias and require RSI>50 for bull confirmation or RSI<50 for bear confirmation
    """
    def __init__(self, cfg: Config):
        super().__init__(cfg)

    def analyze(self, multi: Dict[str, Any]) -> Optional[str]:
        # prefer 1m or fallback
        df = choose_multi_df(multi, '1m')
        if df is None:
            return None
        try:
            close = df['close']
            period = getattr(self.cfg, 'rsi_period', 14) if hasattr(self, 'cfg') else 14
            ob_level = getattr(self.cfg, 'overbought', 70)
            os_level = getattr(self.cfg, 'oversold', 30)
            try:
                if talib is not None:
                    rsi = talib.RSI(close.astype(float), timeperiod=period)
                else:
                    rsi = Indicators.safe_rsi(close, period)
            except Exception:
                rsi = Indicators.safe_rsi(close, period)
            if len(rsi) < 3:
                return None
            r0 = float(rsi[-1] if isinstance(rsi, list) else rsi.iloc[-1])
            r1 = float(rsi[-2] if isinstance(rsi, list) else rsi.iloc[-2])
            r2 = float(rsi[-3] if isinstance(rsi, list) else rsi.iloc[-3])
            # --- Trend filter: compute EMA200 on the same timeframe used by this strategy ---
            bias = None
            ema200_last = None
            try:
                # compute EMA200 series and store in df['ema200'] if pandas DataFrame
                ema_series = None
                if talib is not None:
                    try:
                        ema_series = talib.EMA(close.astype(float), timeperiod=200)
                    except Exception:
                        ema_series = None
                if ema_series is None and pd is not None:
                    if len(close) >= 200:
                        try:
                            ema_series = close.rolling(window=200).mean()
                        except Exception:
                            try:
                                ema_series = close.ewm(span=200, adjust=False).mean()
                            except Exception:
                                ema_series = None
                    else:
                        ema_series = None

                if ema_series is not None:
                    # annotate df in-place for later inspection
                    try:
                        if hasattr(df, 'loc'):
                            # use option_context to suppress SettingWithCopyWarning when df is a slice/view
                            try:
                                with pd.option_context('mode.chained_assignment', None):
                                    df.loc[:, 'ema200'] = (ema_series if hasattr(ema_series, 'astype') else pd.Series(list(ema_series)))
                            except Exception:
                                # last-resort: assign directly
                                df['ema200'] = (ema_series if hasattr(ema_series, 'astype') else pd.Series(list(ema_series)))
                    except Exception:
                        pass
                    # extract last value if available and finite
                    try:
                        ema200_raw = (ema_series[-1] if isinstance(ema_series, list) else ema_series.iloc[-1])
                        ema200_last = float(ema200_raw)
                        if ema200_last != ema200_last:
                            ema200_last = None
                    except Exception:
                        ema200_last = None

                # determine bias only when ema200_last available and finite
                if ema200_last is not None:
                    price_now = float(df.iloc[-1]['close']) if hasattr(df, 'iloc') else float(close[-1])
                    bias = 'bull' if price_now > ema200_last else 'bear'
                else:
                    bias = None
            except Exception:
                bias = None

            # optional momentum confirmation (short EMA slope on 1m)
            momentum_ok = True
            try:
                if getattr(self.cfg, 'enable_momentum', False):
                    mperiod = getattr(self.cfg, 'momentum_ema', 9)
                    # prefer talib if available
                    ema = None
                    if talib is not None:
                        ema = talib.EMA(close.astype(float), timeperiod=mperiod)
                        ema_last = (ema[-1] if isinstance(ema, list) else ema.iloc[-1])
                        ema_prev = (ema[-3] if isinstance(ema, list) and len(ema) > 2 else (ema.iloc[-3] if hasattr(ema, 'iloc') and len(ema) > 2 else None))
                    else:
                        if pd is not None and len(close) >= mperiod:
                            ema_series = close.ewm(span=mperiod, adjust=False).mean()
                            ema_last = ema_series.iat[-1]
                            ema_prev = ema_series.iat[-3] if len(ema_series) > 2 else None
                        else:
                            ema_last = None
                            ema_prev = None
                    if ema_last is None or ema_prev is None:
                        # if momentum required and we can't compute it, fail confirmation
                        if getattr(self.cfg, 'momentum_required', False):
                            momentum_ok = False
                    else:
                        slope = (float(ema_last) - float(ema_prev)) / (abs(float(ema_prev)) + 1e-9)
                        thr = float(getattr(self.cfg, 'momentum_threshold', 0.0))
                        if slope > thr:
                            momentum_dir = 'bull'
                        elif slope < -thr:
                            momentum_dir = 'bear'
                        else:
                            momentum_dir = None
                        # if required, enforce direction alignment with candidate later
                        # store as attribute to inspect below
                        _momentum_dir = momentum_dir
                else:
                    _momentum_dir = None
            except Exception:
                momentum_ok = True

            # Reversal detection
            # buy candidate: r2 < os_level and r1 < os_level and r0 > os_level (cross up)
            decision = None
            reason = ''
            if r2 < os_level and r1 < os_level and r0 > os_level:
                # trend filter: if enabled, only allow buys when price > EMA200
                if getattr(self.cfg, 'enable_trend_filter', False):
                    if ema200_last is None:
                        # not enough history - skip filter for this iteration
                        reason = 'RSI buy but EMA200 unavailable -> allow'
                        decision = 'buy'
                    else:
                        price_now = float(df.iloc[-1]['close']) if hasattr(df, 'iloc') else float(close[-1])
                        if price_now > ema200_last:
                            decision = 'buy'
                            reason = 'RSI buy and price > EMA200'
                        else:
                            decision = None
                            reason = 'RSI OK but price <= EMA200 -> skip'
                else:
                    decision = 'buy'
                    reason = 'RSI buy (trend filter disabled)'
            # sell: r2 > 70 and r1 > 70 and r0 < 70 (cross down)
            if r2 > ob_level and r1 > ob_level and r0 < ob_level:
                # trend filter: if enabled, only allow sells when price < EMA200
                if getattr(self.cfg, 'enable_trend_filter', False):
                    if ema200_last is None:
                        reason = 'RSI sell but EMA200 unavailable -> allow'
                        decision = 'sell'
                    else:
                        price_now = float(df.iloc[-1]['close']) if hasattr(df, 'iloc') else float(close[-1])
                        if price_now < ema200_last:
                            decision = 'sell'
                            reason = 'RSI sell and price < EMA200'
                        else:
                            decision = None
                            reason = 'RSI OK but price >= EMA200 -> skip'
                else:
                    decision = 'sell'
                    reason = 'RSI sell (trend filter disabled)'

            # Logging of decision and context (verbose)
            try:
                ts = ''
                try:
                    row = df.iloc[-1]
                    ts = str(row.get('timestamp')) if 'timestamp' in row.index else str(df.index[-1])
                except Exception:
                    ts = ''
                logger.info('RSIStrategy decision: ts=%s r2=%.4f r1=%.4f r0=%.4f ema200=%s bias=%s decision=%s reason=%s', ts, r2, r1, r0, ('%.6f' % ema200_last) if ema200_last is not None else 'NA', bias, decision, reason)
            except Exception:
                pass

            # expose last reason and ema for external wrappers/inspection
            try:
                self.last_reason = reason
                self.last_ema200 = ema200_last
            except Exception:
                pass

            # enforce momentum direction alignment if required
            if decision is not None:
                if getattr(self.cfg, 'enable_momentum', False):
                    # if momentum required but momentum_ok False -> skip
                    if getattr(self.cfg, 'momentum_required', False) and not momentum_ok:
                        return None
                    # if momentum_dir is set and contradicts decision -> skip
                    try:
                        if _momentum_dir is not None:
                            if decision == 'buy' and _momentum_dir != 'bull':
                                return None
                            if decision == 'sell' and _momentum_dir != 'bear':
                                return None
                    except Exception:
                        pass
                return decision
        except Exception:
            return None
        return None


class MACDStrategy(StrategyBase):
    """MACD based strategy: signal line crosses and simple divergence detection.

    - Buy when MACD line crosses above signal line
    - Sell when MACD line crosses below signal line
    - Detect simple divergence: price makes lower low but MACD histogram makes higher low -> bullish divergence
    """
    def __init__(self, cfg: Config):
        super().__init__(cfg)

    def _macd(self, series):
        try:
            if talib is not None:
                macd, signal, hist = talib.MACD(series.astype(float))
                return macd, signal, hist
            if pd is not None:
                ema12 = series.ewm(span=12, adjust=False).mean()
                ema26 = series.ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9, adjust=False).mean()
                hist = macd - signal
                return macd, signal, hist
        except Exception:
            pass
        # fallback zeros
        n = len(series)
        return [0]*n, [0]*n, [0]*n

    def _simple_divergence(self, prices, hist, lookback=10):
        # naive detection: compare last two local lows/highs
        try:
            ps = list(prices)
            hs = list(hist)
            if len(ps) < lookback + 2:
                return None
            # find two recent lows
            lows = []
            for i in range(len(ps)-lookback, len(ps)):
                lows.append((ps[i], i))
            lows_sorted = sorted(lows, key=lambda x: x[0])[:2]
            if len(lows_sorted) < 2:
                return None
            low1, i1 = lows_sorted[0]
            low2, i2 = lows_sorted[1]
            # bullish divergence: prices low2 < low1 but hist at i2 > hist at i1
            if low2 < low1 and hs[i2] > hs[i1]:
                return 'bull'
            # bearish divergence: price highs - symmetric check
            highs = []
            for i in range(len(ps)-lookback, len(ps)):
                highs.append((ps[i], i))
            highs_sorted = sorted(highs, key=lambda x: x[0], reverse=True)[:2]
            if len(highs_sorted) >= 2:
                high1, j1 = highs_sorted[0]
                high2, j2 = highs_sorted[1]
                if high2 > high1 and hs[j2] < hs[j1]:
                    return 'bear'
        except Exception:
            return None
        return None

    def analyze(self, multi: Dict[str, Any]) -> Optional[str]:
        df = choose_multi_df(multi, '5m', '1m')
        if df is None:
            return None
        try:
            close = df['close']
            macd, signal, hist = self._macd(close)
            if len(macd) < 2:
                return None
            m0 = float(macd[-1] if isinstance(macd, list) else macd.iloc[-1])
            m1 = float(macd[-2] if isinstance(macd, list) else macd.iloc[-2])
            s0 = float(signal[-1] if isinstance(signal, list) else signal.iloc[-1])
            s1 = float(signal[-2] if isinstance(signal, list) else signal.iloc[-2])
            # cross detection
            if m1 <= s1 and m0 > s0:
                return 'buy'
            if m1 >= s1 and m0 < s0:
                return 'sell'
            # divergence
            div = self._simple_divergence(close, hist, lookback=20)
            if div == 'bull':
                return 'buy'
            if div == 'bear':
                return 'sell'
        except Exception:
            return None
        return None



class ScalpingAdvanced(StrategyBase):
    """Scalping avanzado: usa RSI + Estocástico en pequeña ventana.

    Fases:
    - Detección: RSI < 30 y Stoch < 20  => candidate buy
      o RSI > 70 y Stoch > 80 => candidate sell
    - Confirmación: esperar cruce del Estocástico por encima de 20 (buy) o por debajo de 80 (sell)
    - Ejecución: devolver 'buy' o 'sell' cuando se cumpla la confirmación.
    """
    def __init__(self, cfg: Config, rsi_period: int = 7, stoch_k: int = 5, stoch_d: int = 3):
        super().__init__(cfg)
        self.rsi_period = rsi_period
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d

    def _get_series(self, df, name: str):
        try:
            if pd is not None and hasattr(df, 'iloc'):
                return df[name]
            return df.get(name, [])
        except Exception:
            return []

    def _stochastic(self, close_series, high_series, low_series):
        # %K = (C - LowestLow)/(HighestHigh - LowestLow) * 100
        vals_c = list(close_series)
        vals_h = list(high_series)
        vals_l = list(low_series)
        k = []
        for i in range(len(vals_c)):
            start = max(0, i - self.stoch_k + 1)
            window_h = vals_h[start:i+1]
            window_l = vals_l[start:i+1]
            hh = max(window_h) if window_h else vals_h[i]
            ll = min(window_l) if window_l else vals_l[i]
            denom = hh - ll if hh != ll else 1e-9
            k.append((vals_c[i] - ll) / denom * 100.0)
        # simple %D smoothing
        d = []
        for i in range(len(k)):
            start = max(0, i - self.stoch_d + 1)
            window = k[start:i+1]
            d.append(sum(window) / len(window) if window else k[i])
        return k, d

    def analyze(self, multi: Dict[str, Any]) -> Optional[str]:
        # Prefer 1m or fallback to any available
        df = choose_multi_df(multi, '1m')
        if df is None:
            return None
        try:
            close = self._get_series(df, 'close')
            high = self._get_series(df, 'high')
            low = self._get_series(df, 'low')
            # compute RSI
            if pd is not None and hasattr(df, 'copy'):
                series = close if isinstance(close, (list,)) else close
                try:
                    rsi_vals = talib.RSI(close.astype(float), timeperiod=self.rsi_period) if talib is not None else Indicators.safe_rsi(close, self.rsi_period)
                except Exception:
                    rsi_vals = Indicators.safe_rsi(close, self.rsi_period)
            else:
                rsi_vals = Indicators.safe_rsi(close, self.rsi_period)
            k_vals, d_vals = self._stochastic(close, high, low)
            if not k_vals or not rsi_vals:
                return None
            # current and previous
            idx = len(k_vals) - 1
            if idx < 2:
                return None
            k0, k1 = k_vals[idx], k_vals[idx-1]
            r0, r1 = (rsi_vals[idx] if isinstance(rsi_vals, list) else float(rsi_vals.iloc[idx])), (rsi_vals[idx-1] if isinstance(rsi_vals, list) else float(rsi_vals.iloc[idx-1]))

            # detection
            # buy candidate
            if (r1 < 30 or r0 < 30) and (k1 < 20 or k0 < 20):
                # confirmation: stochastic crossing above 20
                if k1 < 20 and k0 >= 20:
                    return 'buy'
            # sell candidate
            if (r1 > 70 or r0 > 70) and (k1 > 80 or k0 > 80):
                if k1 > 80 and k0 <= 80:
                    return 'sell'
        except Exception:
            return None
        return None


class RangeTradingAdvanced(StrategyBase):
    """Range trading using Bollinger Bands + RSI confirmation.

    - Use 15m to detect range (bands narrow and price within bands)
    - Use 5m for entry timing: touch lower band + RSI<30 then RSI rising -> buy
      touch upper band + RSI>70 then RSI falling -> sell
    """
    def __init__(self, cfg: Config, bb_window: int = 20, bb_dev: float = 2.0):
        super().__init__(cfg)
        self.bb_window = bb_window
        self.bb_dev = bb_dev

    def _bband(self, series):
        # simple bollinger calculation
        if pd is not None:
            mid = series.rolling(self.bb_window).mean()
            std = series.rolling(self.bb_window).std()
            return mid - self.bb_dev * std, mid + self.bb_dev * std, mid
        vals = list(series)
        mid = []
        lower = []
        upper = []
        for i in range(len(vals)):
            window = vals[max(0, i - self.bb_window + 1):i + 1]
            m = sum(window) / len(window) if window else vals[i]
            s = (sum((x - m) ** 2 for x in window) / len(window)) ** 0.5 if window else 0
            mid.append(m)
            lower.append(m - self.bb_dev * s)
            upper.append(m + self.bb_dev * s)
        return lower, upper, mid

    def analyze(self, multi: Dict[str, Any]) -> Optional[str]:
        df15 = multi.get('15m')
        df5 = multi.get('5m')
        if df15 is None or df5 is None:
            return None
        try:
            close15 = df15['close']
            lower15, upper15, mid15 = self._bband(close15)
            last15 = df15.iloc[-1]
            price15 = float(last15['close'])
            # check price within bands
            if price15 <= (lower15[-1] if isinstance(lower15, list) else lower15.iloc[-1]) or price15 >= (upper15[-1] if isinstance(upper15, list) else upper15.iloc[-1]):
                # Price touching or outside bands: possible breakout => abort
                return None
        except Exception:
            return None

        # Entry logic on 5m
        try:
            close5 = df5['close']
            lower5, upper5, mid5 = self._bband(close5)
            last5 = df5.iloc[-1]
            price5 = float(last5['close'])
            rsi5 = float(last5.get('rsi') or 50)
        except Exception:
            return None

        # Buy: touch or cross below lower band + RSI<30
        lb = lower5[-1] if isinstance(lower5, list) else lower5.iloc[-1]
        ub = upper5[-1] if isinstance(upper5, list) else upper5.iloc[-1]
        if price5 <= lb and rsi5 < 30:
            return 'buy'
        if price5 >= ub and rsi5 > 70:
            return 'sell'
        return None


class MultipleTimeFrameStrategy(StrategyBase):
    """Multiple Time Frame Analysis strategy.

    - 1H: EMA200 decides bias (only trade in that direction)
    - 15M: EMA50 + MACD for confirmation and pullback detection
    - 1M: RSI for timing (RSI cross as entry)
    """
    def __init__(self, cfg: Config):
        super().__init__(cfg)

    def _ema(self, series, period: int):
        try:
            if talib is not None:
                return talib.EMA(series.astype(float), timeperiod=period)
            if pd is not None:
                # use exponential weighting for EMA when pandas is available
                return series.ewm(span=period, adjust=False).mean()
        except Exception:
            pass
        # fallback simple SMA
        vals = list(series)
        out = []
        for i in range(len(vals)):
            window = vals[max(0, i - period + 1):i + 1]
            out.append(sum(window) / len(window) if window else vals[i])
        return out

    def _macd_hist(self, series):
        try:
            if talib is not None:
                macd, macdsignal, macdhist = talib.MACD(series.astype(float))
                return macdhist
            if pd is not None:
                ema12 = series.ewm(span=12, adjust=False).mean()
                ema26 = series.ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9, adjust=False).mean()
                return macd - signal
        except Exception:
            pass
        return [0] * len(series)

    def analyze(self, multi: Dict[str, Any]) -> Optional[str]:
        df1h = multi.get('1h')
        df15 = multi.get('15m')
        df1 = multi.get('1m')
        if df1h is None or df15 is None or df1 is None:
            return None
        try:
            # bias from 1h
            close1h = df1h['close']
            ema200 = self._ema(close1h, 200)
            price1h = float(df1h.iloc[-1]['close']) if hasattr(df1h, 'iloc') else close1h[-1]
            ema200_last = float(ema200[-1] if isinstance(ema200, (list,)) else ema200.iloc[-1])
            bias = 'bull' if price1h > ema200_last else 'bear'

            # confirmation on 15m: price relative to EMA50 and MACD histogram sign
            close15 = df15['close']
            ema50_15 = self._ema(close15, 50)
            ema50_last = float(ema50_15[-1] if isinstance(ema50_15, (list,)) else ema50_15.iloc[-1])
            price15 = float(df15.iloc[-1]['close'])
            macdh = self._macd_hist(close15)
            macdh_last = float(macdh[-1] if isinstance(macdh, (list,)) else macdh.iloc[-1])

            if bias == 'bull':
                if not (price15 > ema50_last and macdh_last > 0):
                    return None
            else:
                if not (price15 < ema50_last and macdh_last < 0):
                    return None

            # look for pullback: price near EMA50 on 15m
            if bias == 'bull' and price15 > ema50_last * 1.02:
                # not near a pullback
                return None
            if bias == 'bear' and price15 < ema50_last * 0.98:
                return None

            # timing on 1m using RSI
            close1 = df1['close']
            try:
                if talib is not None:
                    rsi1 = talib.RSI(close1.astype(float), timeperiod=14)
                else:
                    rsi1 = Indicators.safe_rsi(close1, 14)
            except Exception:
                rsi1 = Indicators.safe_rsi(close1, 14)
            if len(rsi1) < 3:
                return None
            r0 = float(rsi1[-1] if isinstance(rsi1, list) else rsi1.iloc[-1])
            r1 = float(rsi1[-2] if isinstance(rsi1, list) else rsi1.iloc[-2])
            r2 = float(rsi1[-3] if isinstance(rsi1, list) else rsi1.iloc[-3])

            if bias == 'bull' and r2 < 30 and r1 < 30 and r0 > 30:
                return 'buy'
            if bias == 'bear' and r2 > 70 and r1 > 70 and r0 < 70:
                return 'sell'
        except Exception:
            return None
        return None



class StrategyEvaluator:
    def __init__(self, strategies: Dict[str, StrategyBase], cfg: Config):
        self.strategies = strategies
        self.cfg = cfg

    def evaluate_on(self, multi: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        base_tf = self.cfg.timeframes[0]
        base = multi.get(base_tf)
        if base is None:
            return results
        length = len(base) if not hasattr(base, 'shape') else base.shape[0]
        window_size = min(self.cfg.history_size, length)
        for name, strat in self.strategies.items():
            trades = 0
            wins = 0
            pnl = 0.0
            for i in range(length - window_size, length - 1):
                snap = {}
                for tf in self.cfg.timeframes:
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
                # take_profit_pct and stop_loss_pct are decimals (0.05 == 5%)
                tp_pct = float(getattr(self.cfg, 'take_profit_pct', 0.05))
                sl_pct = float(getattr(self.cfg, 'stop_loss_pct', 0.02))
                if sig == 'buy':
                    tp = entry * (1.0 + tp_pct)
                    sl = entry * (1.0 - sl_pct)
                else:
                    # for sell (short), tp is lower price, sl is higher price
                    tp = entry * (1.0 - tp_pct)
                    sl = entry * (1.0 + sl_pct)
                hit = None
                for j in range(1, min(30, length - i - 1) + 1):
                    fut = base.iloc[i + j]
                    if sig == 'buy':
                        if float(fut['high']) >= tp:
                            hit = 'tp'
                            break
                        if float(fut['low']) <= sl:
                            hit = 'sl'
                            break
                    else:
                        if float(fut['low']) <= tp:
                            hit = 'tp'
                            break
                        if float(fut['high']) >= sl:
                            hit = 'sl'
                            break
                tp_dist = abs(tp - entry)
                sl_dist = abs(entry - sl)
                # For percent-based pnl we can express as relative move or absolute price move; use relative % of entry
                if hit == 'tp':
                    wins += 1
                    pnl += tp_dist / entry
                elif hit == 'sl':
                    pnl -= sl_dist / entry
            winrate = wins / trades if trades else 0.0
            results[name] = {'trades': trades, 'wins': wins, 'winrate': winrate, 'pnl': pnl}
        return results


class GridSearchOptimizer:
    def __init__(self, grid: Dict[str, List[Any]]):
        self.grid = grid

    def search(self, evaluator):
        import itertools
        keys = list(self.grid.keys())
        best = None
        best_score = float('-inf')
        for vals in itertools.product(*[self.grid[k] for k in keys]):
            params = dict(zip(keys, vals))
            score = evaluator(params)
            if score > best_score:
                best_score = score
                best = (params, score)
        return best


def load_persistent_strategies(path: str = 'strategies.json') -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def save_persistent_strategy(spec: Dict[str, Any], path: str = 'strategies.json'):
    all_specs = load_persistent_strategies(path)
    name = spec.get('name') or f'manual_{int(time.time())}'
    all_specs[name] = spec
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(all_specs, f, indent=2, ensure_ascii=False)


def cli_entry():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.json')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--teach')
    p.add_argument('--explain', action='store_true')
    p.add_argument('--backtest', action='store_true')
    p.add_argument('--optimize', action='store_true')
    p.add_argument('--live', action='store_true', help='run live mode (requires DERIV_API_TOKEN)')
    p.add_argument('--virtual', action='store_true', help='in live mode, operate in virtual (no real orders)')
    p.add_argument('--interval', type=int, default=60, help='seconds between live fetches when not --once')
    p.add_argument('--duration', type=int, default=0, help='total seconds to run live loop (0 = forever)')
    p.add_argument('--once', action='store_true', help='in live mode, fetch once and exit')
    args = p.parse_args()

    if args.teach:
        try:
            with open(args.teach, 'r', encoding='utf-8') as f:
                spec = json.load(f)
        except Exception as e:
            logger.error('Could not read strategy spec: %s', e)
            raise SystemExit(1)
        save_persistent_strategy(spec)
        logger.info('Strategy saved')
        raise SystemExit(0)

    cfg = Config.load_from(args.config)
    cfg.dry_run = args.dry_run or cfg.dry_run
    token = os.getenv('DERIV_API_TOKEN') or cfg.api_token
    client = DerivClient(token=token, dry_run=cfg.dry_run)

    multi: Dict[str, Any] = {}
    for tf in cfg.timeframes:
        try:
            df = client.fetch_candles(cfg.symbol, tf, cfg.history_size)
            if pd is not None and hasattr(df, 'copy'):
                df = Indicators.compute(df)
            multi[tf] = df
        except Exception:
            multi[tf] = None

    strategies = {name: cls(cfg) for name, cls in StrategyFactory.items()}

    if args.explain or args.backtest:
        ev = StrategyEvaluator(strategies, cfg)
        scores = ev.evaluate_on(multi)
        for k, v in scores.items():
            logger.info('%s => trades=%s winrate=%.2f pnl=%.2f', k, v['trades'], v['winrate'], v['pnl'])
        raise SystemExit(0)

    if args.optimize:
        grid = {'risk_per_trade_pct': [0.25, 0.5, 1.0, 2.0]}
        def eval_fn(params):
            cfg.risk_per_trade_pct = params['risk_per_trade_pct']
            ev = StrategyEvaluator(strategies, cfg)
            scores = ev.evaluate_on(multi)
            return sum(v['pnl'] for v in scores.values())
        opt = GridSearchOptimizer(grid).search(eval_fn)
        logger.info('Optimization result: %s', opt)
        raise SystemExit(0)

    if args.live:
        if not client.token and not cfg.api_token:
            logger.error('Live mode requires DERIV_API_TOKEN in env or config')
            raise SystemExit(1)
        logger.info('Live mode: virtual=%s once=%s', args.virtual, args.once)

        # Build strategies including advanced ones
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
        aggregator = SignalAggregator(strategies)

        # Simple signal recorder
        class SignalRecorder:
            def __init__(self, path: str = 'signals.csv'):
                self.path = Path(path)
                self.fields = ['timestamp', 'symbol', 'signal', 'confidence', 'virtual', 'confirmers', 'provenance']
                if not self.path.exists():
                    with self.path.open('w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=self.fields)
                        writer.writeheader()

            def record(self, signal_or_row, confidence: Optional[float] = None, virtual: bool = False, symbol: Optional[str] = None, confirmers: Optional[List[str]] = None, provenance: Optional[Dict[str, Any]] = None):
                """Accept either a row dict or positional params: (signal, confidence, virtual, symbol).

                This makes calling sites simpler: recorder.record(sig, conf, virtual=args.virtual, symbol=cfg.symbol, confirmers=confirmers)
                """
                confirmers = confirmers or []
                provenance = provenance or {}
                if isinstance(signal_or_row, dict):
                    row = signal_or_row
                    signal = row.get('signal') or ''
                    conf = float(row.get('confidence') or 0.0)
                    virt = bool(row.get('virtual'))
                    sym = row.get('symbol')
                    ts = row.get('timestamp')
                else:
                    signal = signal_or_row or ''
                    conf = float(confidence or 0.0)
                    virt = bool(virtual)
                    sym = symbol or ''
                    ts = time.strftime('%Y-%m-%d %H:%M:%S')

                out = {
                    'timestamp': ts,
                    'symbol': sym,
                    'signal': signal,
                    'confidence': '%.4f' % (conf,),
                    'virtual': str(virt),
                    'confirmers': '|'.join(confirmers),
                    'provenance': json.dumps(provenance, ensure_ascii=False),
                }
                with self.path.open('a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.fields)
                    writer.writerow(out)

        recorder = SignalRecorder()

        # Map timeframe strings to granularity seconds for Deriv
        tf_to_min = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600}

        try:
            gran = tf_to_min.get(cfg.timeframes[0], 60)
            df_live = client.fetch_candles_live(cfg.symbol, granularity=gran, count=cfg.history_size)
            if pd is not None and hasattr(df_live, 'copy'):
                df_live = Indicators.compute(df_live)

            # Build multi dict with copies for simplicity
            multi_live = {cfg.timeframes[0]: df_live}

            # Also try to fetch other configured timeframes
            for tf in ['5m', '15m', '1m', '1h']:
                if tf in cfg.timeframes and tf != cfg.timeframes[0]:
                    mn = tf_to_min.get(tf, None)
                    if mn:
                        try:
                            dfx = client.fetch_candles_live(cfg.symbol, granularity=mn, count=cfg.history_size)
                            if pd is not None and hasattr(dfx, 'copy'):
                                dfx = Indicators.compute(dfx)
                            multi_live[tf] = dfx
                        except Exception:
                            multi_live[tf] = None

            sig, conf, confirmers = aggregator.aggregate(multi_live)
            logger.info('Live signal: %s confidence=%.2f confirmers=%s', sig, conf, confirmers)

            # record
            try:
                recorder.record(sig, conf, virtual=args.virtual, symbol=cfg.symbol, confirmers=confirmers)
            except Exception:
                logger.exception('Failed to record signal')

            if args.once:
                raise SystemExit(0)

            # Continuous loop
            start = time.time()
            interval = max(1, int(args.interval or 60))
            duration = int(args.duration or 0)

            while True:
                # check duration
                if duration > 0 and (time.time() - start) >= duration:
                    logger.info('Live loop duration reached, exiting')
                    raise SystemExit(0)

                time.sleep(interval)

                try:
                    multi_live = {}
                    for tf in cfg.timeframes:
                        mn = tf_to_min.get(tf, None)
                        if mn:
                            try:
                                dfx = client.fetch_candles_live(cfg.symbol, granularity=mn, count=cfg.history_size)
                                if pd is not None and hasattr(dfx, 'copy'):
                                    dfx = Indicators.compute(dfx)
                                multi_live[tf] = dfx
                            except Exception:
                                multi_live[tf] = None

                    sig, conf, confirmers = aggregator.aggregate(multi_live)
                    logger.info('Live signal: %s confidence=%.2f confirmers=%s', sig, conf, confirmers)

                    try:
                        recorder.record(sig, conf, virtual=args.virtual, symbol=cfg.symbol, confirmers=confirmers)
                    except Exception:
                        logger.exception('Failed to record signal')

                except Exception as e:
                    logger.error('Live loop iteration failed: %s', e)
                    # continue looping unless critical

        except Exception as e:
            logger.error('Live fetch failed: %s', e)
            raise SystemExit(1)

    logger.info('No action requested. Use --explain, --backtest, --optimize or --teach')


if __name__ == '__main__':
    cli_entry()
