# ArTradIS (demo-first)

Este repositorio contiene una versión demo del bot de trading para índices sintéticos Deriv.

Archivos principales:
- `ArTradIS.py` - Orquestador principal (dry-run por defecto).
- `test_deriv_ws.py` - Script de prueba para conexión WebSocket con Deriv.
- `teacher_prompt.txt` - Prompt IA en español (persistido verbatim).

Instalación rápida (PowerShell):

```powershell
python -m pip install -r requirements.txt
```

Notas:
- `TA-Lib` es opcional y puede requerir instalar binarios en Windows. Se incluye en `requirements.txt` como referencia.
- El bot funciona en modo dry-run por defecto y usa datos sintéticos si no hay dependencias instaladas.
ArTradIS - Estado actual y guía rápida

Resumen rápido:
- Archivo principal: `ArTradIS.py` (se han hecho múltiples reescrituras).
- Comportamiento por defecto: dry-run (simulación). No usa la API real de Deriv a menos que el usuario habilite e implemente `DerivAPI`.
- Implementaciones presentes (parciales o completas):
  - Cliente dry-run que genera velas sintéticas.
  - Indicadores (talib/ta fallback), incluyendo RSI, MACD, SMA/EMA, ATR y Bollinger.
  - RiskManager con position sizing básico y chunking.
  - TradeRecorder que escribe en `trades.csv`.
  - Strategy base, varias estrategias ejemplo (ScalpStrategy, SwingStrategy, TrendFollowingStrategy parcial, ScalpingImpulseStrategy), y `SimpleRuleStrategy` para reglas JSON.
  - Backtester simple (Backtester) y GridSearchOptimizer.
  - Persistence: `strategies.json` y `knowledge.json` via helpers.
  - CLI flags: `--teach`, `--explain`, `--teach-fundamentals`, `--show-fundamentals`, `--dry-run`.

Qué falta o necesita pulido:
- El archivo `ArTradIS.py` todavía contiene fragmentos duplicados y secciones mezcladas por ediciones previas. Hay que limpiar y consolidar definitivamente (he intentado varias limpiezas, pero aún quedan restos).
- `TrendFollowingStrategy` y otras clases avanzadas están parcialmente anidadas en lugares incorrectos y requieren reubicación/consolidación.
- Integración real con deriv-api (WebSocket/REST) no implementada.
- Optimizador genético no implementado (solo GridSearch básico).
- Kill-switch diario persistente no guardado en disco (solo en memoria).
- Tests unitarios y harness para backtesting no incluidos.

Siguientes pasos recomendados (priorizados):
1. Consolidar `ArTradIS.py`: mover/limpiar definiciones duplicadas, asegurar que sólo haya una definición por clase, y que el flujo `main` sea lineal. (Alta prioridad)
2. Añadir pruebas unitarias básicas para `Indicators.compute`, `RiskManager.compute_position_size`, y `SimpleRuleStrategy.evaluate_rule`. (Media)
3. Implementar persistencia de kill-switch diario (archivo JSON) y cargar al inicio. (Media)
4. Implementar fetch real en `DerivClient` usando deriv-api (opcional, realizar sólo si el usuario quiere ejecutar en real). (Baja hasta que el dry-run sea estable)
5. Añadir README de uso y comandos para ejecutar en Windows PowerShell.

Comandos útiles para probar (PowerShell):

```powershell
python .\ArTradIS.py --explain
python .\ArTradIS.py --teach .\example_strategy.json
python .\ArTradIS.py --dry-run
```

Notas finales:
- He guardado tu prompt educativo en `teacher_prompt.txt` para que puedas usarlo como base cuando quieras que el bot aprenda.
- Puedo continuar y realizar la consolidación final de `ArTradIS.py` (recomendado). ¿Procedo con eso ahora?
