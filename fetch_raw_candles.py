import os, json
import websocket

token = os.getenv('DERIV_API_TOKEN')
if not token:
    print('No DERIV_API_TOKEN en entorno')
    raise SystemExit(1)

url = 'wss://ws.binaryws.com/websockets/v3?app_id=1089'
print('Conectando', url)
ws = websocket.create_connection(url, timeout=10)
try:
    ws.send(json.dumps({'authorize': token}))
    auth = ws.recv()
    print('auth received')
    req = {'ticks_history': 'R_100', 'end': 'latest', 'count': 50, 'granularity': 60, 'style': 'candles'}
    ws.send(json.dumps(req))
    resp = ws.recv()
    print('resp length', len(resp))
    with open('live_raw_response.json', 'w', encoding='utf-8') as f:
        f.write(resp)
    print('Saved to live_raw_response.json')
finally:
    try:
        ws.close()
    except Exception:
        pass
