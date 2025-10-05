"""Pre-launch checklist for DEMO_LIVE activation

Checks performed:
 - DERIV_API_TOKEN present
 - Can fetch balance
 - Config.EXECUTION_MODE == 'DEMO_LIVE' recommended
 - Logs and results directories writable
"""
from ArTradIS import DerivClient, Config
import os


def run_checks():
    ok = True
    token = os.getenv('DERIV_API_TOKEN')
    print('Checking environment...')
    print(f' - DERIV_API_TOKEN set: {bool(token)}')
    if not token:
        print('FAIL: DERIV_API_TOKEN not set')
        ok = False

    client = DerivClient(token=token, dry_run=True)
    print(' - Trying to fetch balance...')
    bal = client.get_balance()
    if bal is None:
        print('FAIL: could not fetch balance')
        ok = False
    else:
        print(f'   Balance: {bal}')

    print(' - Checking Config execution mode...')
    mode = getattr(Config, 'EXECUTION_MODE', 'DRY_RUN')
    print(f'   Config.EXECUTION_MODE = {mode}')
    if mode != 'DEMO_LIVE':
        print('   Warning: not DEMO_LIVE (recommended to set for demo activation)')

    # writable dirs
    print(' - Checking logs/results directories...')
    try:
        os.makedirs('logs', exist_ok=True)
        with open(os.path.join('logs', 'prelaunch_test.txt'), 'w', encoding='utf-8') as f:
            f.write('ok')
        os.remove(os.path.join('logs', 'prelaunch_test.txt'))
        print('   logs OK')
    except Exception as e:
        print('FAIL: logs not writable', e)
        ok = False

    try:
        os.makedirs('results', exist_ok=True)
        with open(os.path.join('results', 'prelaunch_test.txt'), 'w', encoding='utf-8') as f:
            f.write('ok')
        os.remove(os.path.join('results', 'prelaunch_test.txt'))
        print('   results OK')
    except Exception as e:
        print('FAIL: results not writable', e)
        ok = False

    print('\nSUMMARY:')
    if ok:
        print('PRE-LAUNCH CHECKS PASSED')
        return 0
    else:
        print('PRE-LAUNCH CHECKS FAILED')
        return 2


if __name__ == '__main__':
    import sys
    code = run_checks()
    sys.exit(code)
