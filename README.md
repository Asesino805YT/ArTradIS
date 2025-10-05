# ArTradIS — Informe y README final (RSI + Momentum + EMA200)

Este repositorio contiene el pipeline reproducible de backtesting y experimentación
desarrollado durante la sesión: una estrategia de reversión basada en RSI
mejorada con confirmación de momentum y un filtro de tendencia por EMA200.

El README que sigue resume la infraestructura, la lógica de la estrategia,
los artefactos producidos, comandos reproducibles y la configuración óptima
identificada por las corridas de validación.

## 1) Objetivo del proyecto y resultado

Objetivo: construir un backtester reproducible, añadir instrumentación (trazas
por decisión) y aplicar filtros (momentum + EMA200) para maximizar winrate.

Resultado validado (corrida larga, `history-size=5000`):

- Winrate: 83.3%
- Net PnL: +50.50
- Drawdown: contenido (ver métricas por experimento en `results/`)

> Nota: los números anteriores corresponden a la configuración final seleccionada
> por la rejilla de búsqueda; todos los artefactos y trazas están en `results/`.

## 2) Componentes principales y artefactos

- `ArTradIS.py`: entrada principal / modo backtest por config. Puede ejecutar
  un run individual con una configuración concreta.
- `run_matrix.py`: ejecuta búsquedas por rejilla (grid search) y produce una
  fila por experimento en `results/matrix_results.csv`.
- `backtest_pipeline.py`: motor de backtest que ejecuta la lógica de trading paso a paso.
- `rsistrategy_verbose_patch.py`: wrapper `VerboseRSIStrategy` que escribe
  trazas JSON-lines por decisión cuando se activa `--verbose-strategy`.
- `scripts/validate_coherence.py`: valida que las reglas de EMA200 se cumplieron
  para los trades exportados, genera `results/alerts_<run_id>.csv` y un
  `results/alerts_summary.csv`.

Artefactos generados por cada experimento (`run_id`):

- `results/trades_<run_id>.csv` — detalle de trades ejecutados.
- `results/verbose_logs/verbose_<run_id>.log` — trazas por decisión (JSON por línea).
- `results/matrix_results.csv` — registro maestro con métricas por experimento.
- `results/alerts_<run_id>.csv` y `results/alerts_summary.csv` — salida del validador.

## 3) Lógica de la estrategia y filtros

Resumen breve:

- Señal base: RSI Reversión (parámetro `rsi_period`).
- Momentum: EMA corta (por ejemplo EMA9) + threshold; puede ser obligatorio para entradas.
- Trend filter: EMA200 — compras solo si `price > EMA200`, ventas solo si `price < EMA200`.

Comportamiento con EMA200 no disponible:

- Si faltan suficientes velas para calcular EMA200, la estrategia anota la razón
  en las trazas verbose; el comportamiento (permitir/denegar entrada) es configurable
  y se documenta en el código.

## 4) Configuración final óptima (recomendada)

Parámetros validados como más robustos en la rejilla (`history-size=5000`):

| Parámetro | Valor |
|---|---:|
| RSI period | 10 |
| RSI overbought / oversold | 70 / 30 |
| Momentum | ACTIVO (EMA 9, threshold 0.001, required) |
| Trend filter (EMA200) | ACTIVO |
| Take Profit | 2.5% |
| Stop Loss | 1.0% |
| Winrate (reportado) | 83.3% |
| Net PnL (reportado) | +50.50 |

Las métricas detalladas por experimento están en `results/matrix_results.csv`.

## 5) Cómo ejecutar (Windows PowerShell)

1) Crear/activar virtualenv e instalar dependencias:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install pandas numpy
```

2) Ejecutar una rejilla corta (desarrollo / debug) con verbose:

```powershell
.venv\Scripts\python run_matrix.py --rsi-periods 10 --overbought 70 --oversold 30 \
  --take-profit 1.5 2.5 3.5 --stop-loss 0.5 1.0 1.5 --history-size 500 \
  --enable-momentum --momentum-ema 9 --momentum-threshold 0.001 --momentum-required \
  --verbose-strategy --enable-trend-filter
```

3) Corrida de validación larga (configuración final óptima, sin verbose):

```powershell
.venv\Scripts\python ArTradIS.py \
  --rsi-period 10 --overbought 70 --oversold 30 \
  --take-profit 2.5 --stop-loss 1.0 \
  --enable-momentum --momentum-ema 9 --momentum-threshold 0.001 --momentum-required \
  --enable-trend-filter --history-size 5000
```

## 9) Pipeline automatizado (PowerShell)

Se incluye `scripts/run_full_pipeline.ps1` para ejecutar el pipeline completo desde la raíz del repositorio.

Flags disponibles:

- `-OnlyShort` : ejecutar solo la corrida corta (history-size=500)
- `-OnlyLong` : ejecutar solo la corrida larga (history-size=5000)
- `-NoGit` : no añadir git metadata a los archivos `results/meta_*.json`
- `-Plots` : generar gráficos de equity/drawdown para el top10 (requiere matplotlib instalado en el venv)
- `-OpenReportFlag` : abrir `results/summary_report.md` al terminar (si existe)

Ejemplos:

```powershell
.\scripts\run_full_pipeline.ps1
.\scripts\run_full_pipeline.ps1 -OnlyShort -NoGit
.\scripts\run_full_pipeline.ps1 -Plots -OpenReportFlag
```

El script intenta usar `.venv\Scripts\python.exe` por defecto; si no existe, usará `python` del PATH y mostrará una advertencia.


Importante: siempre ejecutar scripts con el intérprete del virtualenv (`.venv\Scripts\python`) para evitar errores de importación (por ejemplo, ModuleNotFoundError: No module named 'pandas').

## 6) Validación de coherencia (EMA200)

Para comprobar que la regla de EMA200 no fue violada por los trades:

```powershell
.venv\Scripts\python scripts/validate_coherence.py
```

Salida:

- `results/alerts_summary.csv` — resumen con número de alertas por `run_id`.
- `results/alerts_<run_id>.csv` — alertas detalladas por experimento (ej.: `buy_below_ema`).

## 7) Reproducibilidad y trazabilidad

- Cada experimento se identifica con `run_id` y crea `trades_<run_id>.csv` y `verbose_<run_id>.log`.
- `results/matrix_results.csv` contiene la fila que enlaza a esos artefactos.

Recomendación: añadir `results/meta_<run_id>.json` desde `run_matrix.py` con los
parámetros, la versión del código y un hash del dataset para trazabilidad total.

## 8) Limitaciones y próximos pasos recomendados

- Los resultados están condicionados al timeframe y al histórico (`history-size`).
- Sugerencias de mejora de bajo riesgo:
  - Generar `results/meta_<run_id>.json` automáticamente (parametros + hash de datos).
  - Añadir tests unitarios para `RSIStrategy` y `scripts/validate_coherence.py`.
  - Crear un notebook que cargue `results/trades_<run_id>.csv` y `verbose_<run_id>.log`
    para generar gráficas de equity, distribución de ganancias y drawdown.

---

¿Deseas que implemente alguna de las mejoras ahora (por ejemplo: crear `meta_<run_id>.json` desde el runner o generar un notebook de análisis)?
