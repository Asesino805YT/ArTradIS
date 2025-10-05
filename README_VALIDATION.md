# Fase 1: Validación de Estrategias

## Ejecutar Backtest con Slippage

```powershell
.python .venv\Scripts\python scripts\run_matrix.py --slippage-pct 0.002
```

Analizar Resultados

```powershell
.venv\Scripts\python scripts\summarize_meta.py
```

Seleccionar Top 3

```powershell
.venv\Scripts\python scripts\select_top_strategies.py
```

Métricas Calculadas

- Sharpe Ratio: rendimiento ajustado por volatilidad total
- Sortino Ratio: rendimiento ajustado por volatilidad negativa
- Calmar Ratio: PnL / Max Drawdown
- Win Rate: % de trades ganadores
- Profit Factor: ganancia bruta / pérdida bruta

Próximos Pasos

- Ejecutar backtest de 1 semana (prueba rápida)
- Validar que las métricas se calculan correctamente
- Ejecutar backtest de 6 meses completo
- Seleccionar top 3 estrategias para paper trading

## Checklist

1. Estructura de archivos

```powershell
ls scripts/metrics.py
ls scripts/select_top_strategies.py
ls README_VALIDATION.md
```

2. Test rápido de métricas

```powershell
python -c "from scripts.metrics import calculate_sharpe_ratio; print(calculate_sharpe_ratio([0.01, 0.02, -0.01]))"
```

3. Ejecutar summarize (debe añadir nuevas columnas)

```powershell
.venv\Scripts\python scripts\summarize_meta.py
head -n 1 results\matrix_results.csv
```

4. Test de selección

```powershell
.venv\Scripts\python scripts\select_top_strategies.py
```
