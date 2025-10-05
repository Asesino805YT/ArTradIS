import os, pkgutil

token = os.getenv('DERIV_API_TOKEN')
print('DERIV_API_TOKEN found:', bool(token))
print('DERIV_API_TOKEN length:', len(token) if token else 0)

try:
    loader = pkgutil.find_loader('deriv_api')
    print('deriv_api installed:', loader is not None)
except Exception as e:
    print('deriv_api installed: error checking', e)
