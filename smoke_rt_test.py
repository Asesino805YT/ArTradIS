import math
import random
import time
from ArTradIS import Indicators, RealTimeProcessor, Config, TrendFollowingAdvanced, ScalpingAdvanced, MultipleTimeFrameStrategy

print('Running smoke test for ArTradIS components')
cfg = Config()
# ensure timeframe includes '1m' for the test
cfg.timeframes = ['1m']
strategies = {
    'trend_adv': TrendFollowingAdvanced(cfg),
    'scalp_adv': ScalpingAdvanced(cfg),
    'mtf_adv': MultipleTimeFrameStrategy(cfg),
}
rt = RealTimeProcessor(cfg, strategies, maxlen=200)

# generate synthetic ticks: gentle sine wave + noise
for i in range(200):
    base = 1000.0 + math.sin(i / 10.0) * 5.0
    price = base + random.gauss(0, 0.5)
    o = price + random.gauss(0, 0.2)
    h = max(o, price) + abs(random.gauss(0, 0.2))
    l = min(o, price) - abs(random.gauss(0, 0.2))
    c = price
    rt.push_tick('1m', o, h, l, c)

print('Buffer length for 1m:', len(rt.buffers['1m']))
# compute indicators on a small series
buf = rt.buffers['1m']
closes = [b['close'] for b in buf]
highs = [b['high'] for b in buf]
lows = [b['low'] for b in buf]

print('Last SMA(10):', Indicators.sma(closes, 10)[-1])
print('Last EMA(10):', Indicators.ema(closes, 10)[-1])
print('Last RSI(14):', Indicators.rsi(closes, 14)[-1] if len(closes)>=14 else 'n/a')
print('Last ATR(14):', Indicators.atr(highs, lows, closes, period=14)[-1] if len(closes)>=14 else 'n/a')

sig, conf, confirmers = rt.evaluate()
print('RealTimeProcessor.evaluate =>', sig, conf, confirmers)

print('Smoke test complete')
