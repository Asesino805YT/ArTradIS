"""Configuración de logging compartida para el pipeline de backtests.

Provee una función get_logger(name) que devuelve un logger con TimedRotatingFileHandler
(escrito en logs/backtest.log) y un StreamHandler para consola.
"""
from __future__ import annotations
import logging
from logging.handlers import TimedRotatingFileHandler
import os

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

LOG_PATH = os.path.join(LOG_DIR, 'backtest.log')

DEFAULT_FORMAT = '%(asctime)s %(levelname)s [%(name)s] %(message)s'


def get_logger(name: str = 'backtest', level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if getattr(logger, '_configured', False):
        return logger
    logger.setLevel(level)
    # Timed rotating file handler (daily) with 7 days backup
    fh = TimedRotatingFileHandler(LOG_PATH, when='midnight', backupCount=7, encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(DEFAULT_FORMAT))
    logger.addHandler(fh)
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(DEFAULT_FORMAT))
    logger.addHandler(ch)
    logger.propagate = False
    logger._configured = True
    return logger
