"""
Validación rápida de estrategia RSI
Sin dependencias complejas, solo pandas y numpy
"""

import pandas as pd
import numpy as np


def calculate_rsi(prices, period=10):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss.replace(0, np.nan))
    rs = rs.fillna(0)
    return 100 - (100 / (1 + rs))


def calculate_ema(prices, period):
    return prices.ewm(span=period, adjust=False).mean()


def backtest_simple(df, capital=15.0, leverage=50.0, sl_pct=0.02, tp_pct=0.05):
    df = df.copy().reset_index(drop=True)
    df['rsi'] = calculate_rsi(df['close'], 10)
    df['ema9'] = calculate_ema(df['close'], 9)
    df['ema200'] = calculate_ema(df['close'], 200)

    # Señales
    df['signal'] = 0
    df.loc[(df['rsi'] < 30) & (df['close'] > df['ema9']) & (df['ema9'] > df['ema200']), 'signal'] = 1
    df.loc[(df['rsi'] > 70) & (df['close'] < df['ema9']) & (df['ema9'] < df['ema200']), 'signal'] = -1

    trades = []
    in_position = False
    entry_price = 0.0
    position_type = 0

    # iterate rows
    for i in range(len(df)):
        price = float(df.at[i, 'close'])
        sig = int(df.at[i, 'signal'])

        if in_position:
            # evaluate current PnL
            if position_type == 1:
                pnl_pct = (price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - price) / entry_price

            pnl_pct_leveraged = pnl_pct * leverage

            # liquidation: loss >= 100% margin
            if pnl_pct_leveraged <= -1.0:
                trades.append({'pnl_pct': -1.0, 'pnl_usd': -capital, 'liquidated': True})
                in_position = False
                continue

            # SL/TP thresholds are on leveraged pnl_pct
            if pnl_pct_leveraged <= -sl_pct or pnl_pct_leveraged >= tp_pct:
                pnl_usd = capital * pnl_pct_leveraged
                trades.append({'pnl_pct': pnl_pct_leveraged, 'pnl_usd': pnl_usd, 'liquidated': False})
                in_position = False
                # do not open new trade same candle
                continue

            # otherwise hold
            continue

        else:
            if sig != 0:
                in_position = True
                entry_price = price
                position_type = sig
                continue

    if not trades:
        return {'total_trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0, 'max_drawdown': 0.0, 'liquidations': 0, 'avg_win': 0.0, 'avg_loss': 0.0}

    tdf = pd.DataFrame(trades)
    wins = (tdf['pnl_usd'] > 0).sum()
    total = len(tdf)
    pnl_sum = tdf['pnl_usd'].sum()
    cum = (capital + tdf['pnl_usd']).cumsum()
    # max drawdown in USD
    peak = capital
    max_dd = 0.0
    running = capital
    for v in (capital + tdf['pnl_usd']).cumsum():
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd

    return {
        'total_trades': int(total),
        'win_rate': float(wins) / float(total) * 100.0,
        'total_pnl': float(pnl_sum),
        'max_drawdown': float(max_dd),
        'liquidations': int(tdf['liquidated'].sum()),
        'avg_win': float(tdf[tdf['pnl_usd'] > 0]['pnl_usd'].mean()) if (tdf['pnl_usd'] > 0).any() else 0.0,
        'avg_loss': float(tdf[tdf['pnl_usd'] < 0]['pnl_usd'].mean()) if (tdf['pnl_usd'] < 0).any() else 0.0
    }


if __name__ == '__main__':
    import time
    start = time.time()
    df = pd.read_csv('data/1m.csv')
    df = df.tail(20000)
    print('='*60)
    print('VALIDACIÓN RÁPIDA DE ESTRATEGIA')
    print('='*60)
    print(f'Filas cargadas: {len(df)}')

    for lev in [50]:
        res = backtest_simple(df, capital=15.0, leverage=float(lev), sl_pct=0.02, tp_pct=0.05)
        print('\n' + '='*60)
        print(f'LEVERAGE {lev}x')
        print('='*60)
        print(f"Trades: {res['total_trades']}")
        print(f"Win Rate: {res['win_rate']:.1f}%")
        print(f"PnL Total: ${res['total_pnl']:.2f}")
        print(f"Max Drawdown: ${res['max_drawdown']:.2f}")
        print(f"Liquidaciones: {res['liquidations']}")
        print(f"Avg Win: ${res['avg_win']:.2f}")
        print(f"Avg Loss: ${res['avg_loss']:.2f}")

        # quick evaluation
        if res['total_trades'] < 10:
            print('⚠️ Muestra muy pequeña')
        elif res['liquidations'] > res['total_trades'] * 0.1:
            print('❌ Demasiadas liquidaciones')
        elif res['win_rate'] < 60:
            print('❌ Win rate insuficiente')
        elif res['max_drawdown'] < -15.0:  # drawdown more than capital-15 USD
            print('❌ Drawdown muy alto')
        else:
            print('✅ Configuración potencialmente viable')

    end = time.time()
    print(f'\nTiempo total: {end - start:.1f}s')
