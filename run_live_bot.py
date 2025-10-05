"""Runner simple para ejecutar el bot en modo DEMO_LIVE.

Uso: .venv\Scripts\python run_live_bot.py

Antes de ejecutar: configura Config.EXECUTION_MODE = 'DEMO_LIVE' o establece en runtime.
"""
from ArTradIS import DerivClient, Config, KillSwitch
import time
from datetime import datetime

# helper básico de estrategia (RSI10) - usa ArTradIS Indicators si disponible
from ArTradIS import Indicators


def apply_rsi_strategy(candles_df, period=10):
    try:
        if hasattr(candles_df, 'copy'):
            df = candles_df.copy()
            # close series
            close = df['close']
            rsi = Indicators.safe_rsi(close, period)
            # simple rule: last rsi < 30 -> BUY, >70 -> SELL
            last = rsi[-1] if len(rsi) > 0 else 50
            if last < 30:
                return 'BUY'
            if last > 70:
                return 'SELL'
    except Exception:
        pass
    return None


def main():
    # set mode
    Config.EXECUTION_MODE = 'DEMO_LIVE'
    token = None
    import os
    token = os.getenv('DERIV_API_TOKEN') or (Config.api_token if hasattr(Config,'api_token') else None)
    client = DerivClient(token=token, dry_run=False)
    ks = KillSwitch()
    print('Starting bot in DEMO_LIVE mode -', datetime.utcnow().isoformat())
    bal = client.get_balance()
    print('Balance:', bal)
    while True:
        try:
            # fetch recent candles; 1-minute granularity (use 60s mapping as granularity)
            df = client.fetch_candles_live('R_50', granularity=60, count=100)
            sig = apply_rsi_strategy(df, period=10)
            if sig in ('BUY','SELL'):
                amount = Config.chunk_usdt
                resp = client.place_order_with_killswitch(symbol='R_50', side=sig.lower(), amount=amount, stop_loss=None, take_profit=None, kill_switch=ks, balance_usd=bal, pct_limit=5.0, virtual=False)
                print('Order response:', resp)
        except Exception as e:
            print('Loop error:', e)
        time.sleep(5)


if __name__ == '__main__':
    main()
