"""
Diagnóstico detallado de conexión WebSocket con Deriv
Muestra exactamente qué está fallando
"""

import json
import websocket
import os
import sys
import traceback


def test_websocket_connection():
    print("="*70)
    print("🔍 DIAGNÓSTICO DE CONEXIÓN DERIV WEBSOCKET")
    print("="*70)
    
    # 1. Verificar token
    print("\n1️⃣ Verificando token...")
    token = os.getenv('DERIV_API_TOKEN')
    
    if not token:
        print("❌ DERIV_API_TOKEN no encontrado en variables de entorno")
        print("\n📝 SOLUCIÓN:")
        print("   PowerShell: $env:DERIV_API_TOKEN = 'tu_token_aqui'")
        print("   CMD:        set DERIV_API_TOKEN=tu_token_aqui")
        print("   Linux/Mac:  export DERIV_API_TOKEN='tu_token_aqui'")
        return False
    
    print(f"✅ Token encontrado: {token[:10]}...{token[-4:]}")
    
    # 2. Test de conectividad básica
    print("\n2️⃣ Testeando conectividad a Deriv...")
    
    try:
        ws_url = "wss://ws.derivws.com/websockets/v3?app_id=1089"
        print(f"   Conectando a: {ws_url}")
        
        ws = websocket.create_connection(ws_url, timeout=10)
        print("✅ Conexión WebSocket establecida")
        
        # 3. Test de autorización
        print("\n3️⃣ Testeando autorización...")
        
        auth_payload = {
            "authorize": token
        }
        
        ws.send(json.dumps(auth_payload))
        print(f"   Payload enviado: {auth_payload}")
        
        response_raw = ws.recv()
        print(f"   Respuesta recibida (raw): {str(response_raw)[:200]}...")
        
        response = json.loads(response_raw)
        print(f"   Respuesta parseada: {json.dumps(response, indent=2)[:500]}...")
        
        if 'error' in response:
            print(f"\n❌ ERROR DE AUTORIZACIÓN:")
            print(f"   Código: {response['error'].get('code')}")
            print(f"   Mensaje: {response['error'].get('message')}")
            print(f"\n📝 POSIBLES CAUSAS:")
            print("   - Token inválido o expirado")
            print("   - Token de cuenta real en vez de demo (o viceversa)")
            print("   - Permisos insuficientes en el token")
            return False
        
        if 'authorize' in response:
            print("✅ Autorización exitosa")
            account_info = response['authorize']
            print(f"\n📊 INFO DE CUENTA:")
            print(f"   Currency: {account_info.get('currency', 'N/A')}")
            print(f"   Account type: {account_info.get('account_type', 'N/A')}")
            print(f"   Login ID: {account_info.get('loginid', 'N/A')}")
        
        # 4. Test de balance
        print("\n4️⃣ Solicitando balance...")
        
        balance_payload = {
            "balance": 1,
            "subscribe": 1
        }
        
        ws.send(json.dumps(balance_payload))
        print(f"   Payload enviado: {balance_payload}")
        
        balance_response_raw = ws.recv()
        print(f"   Respuesta recibida: {str(balance_response_raw)[:200]}...")
        
        balance_response = json.loads(balance_response_raw)
        print(f"   Respuesta parseada: {json.dumps(balance_response, indent=2)}")
        
        if 'error' in balance_response:
            print(f"\n❌ ERROR AL OBTENER BALANCE:")
            print(f"   {balance_response['error']}")
            return False
        
        if 'balance' in balance_response:
            balance_value = balance_response['balance'].get('balance')
            currency = balance_response['balance'].get('currency')
            print(f"\n✅ Balance obtenido: {balance_value} {currency}")
            
            # 5. Verificar si es demo o real
            print("\n5️⃣ Verificando tipo de cuenta...")
            loginid = balance_response['balance'].get('loginid', '')
            
            if isinstance(loginid, str) and loginid.startswith('VR'):
                print("✅ Cuenta VIRTUAL (Demo) - Perfecto para pruebas")
            elif isinstance(loginid, str) and loginid.startswith('CR'):
                print("⚠️ Cuenta REAL - CUIDADO con ejecutar órdenes")
            else:
                print(f"ℹ️ Cuenta tipo: {loginid}")
        
        ws.close()
        
        print("\n" + "="*70)
        print("✅ DIAGNÓSTICO COMPLETO - CONEXIÓN OK")
        print("="*70)
        return True
        
    except websocket.WebSocketTimeoutException:
        print("\n❌ TIMEOUT - No se pudo conectar a Deriv")
        print("📝 POSIBLES CAUSAS:")
        print("   - Firewall bloqueando WebSocket")
        print("   - Proxy/VPN interfiriendo")
        print("   - Deriv API temporalmente caído")
        return False
        
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO:")
        print(f"   Tipo: {type(e).__name__}")
        print(f"   Mensaje: {str(e)}")
        print(f"\n📋 TRACEBACK COMPLETO:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_websocket_connection()
    
    if success:
        print("\n🚀 LISTO PARA CONTINUAR")
        print("   Siguiente paso: python pre_launch_check.py")
        sys.exit(0)
    else:
        print("\n🛑 CORRIJE LOS ERRORES ANTES DE CONTINUAR")
        sys.exit(1)
