import os, json, time

try:
    import websocket  # type: ignore[reportMissingImports]
except Exception as e:
    print('websocket-client no está instalado:', e)
    raise SystemExit(1)
except SystemExit:
    raise
except Exception as e:
    print('Error al intentar importar websocket-client:', e)
    raise SystemExit(1)

token = os.getenv('DERIV_API_TOKEN')
if not token:
    print('No se encontró DERIV_API_TOKEN en el entorno. Asegúrate de setx o .env cargado y reinicia la terminal.')
    raise SystemExit(1)

url = 'wss://ws.binaryws.com/websockets/v3?app_id=1089'
print('Conectando a', url)

try:
    ws = websocket.create_connection(url, timeout=10)
    payload = json.dumps({"authorize": token})
    print('Enviando authorize...')
    ws.send(payload)
    resp = ws.recv()
    print('Respuesta del servidor:')
    print(resp)
    # opcional: esperar un poco y cerrar
    time.sleep(0.5)
    ws.close()
except Exception as e:
    print('Error durante la conexión o autorización:', e)
    raise
