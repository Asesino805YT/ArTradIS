"""Prueba simple de conexión y balance a Deriv usando DerivClient.get_balance().

Ejecutar:
    .venv\Scripts\python test_connection.py
"""
from ArTradIS import DerivClient, Config

if __name__ == '__main__':
    # Intentar leer token desde entorno/config
    import os
    token = os.getenv('DERIV_API_TOKEN') or (Config.api_token if hasattr(Config, 'api_token') else None)
    client = DerivClient(token=token, dry_run=True)
    print('Token provided:', bool(token))
    bal = None
    try:
        bal = client.get_balance()
    except Exception as e:
        print('get_balance failed:', e)
    print('Balance:', bal)
