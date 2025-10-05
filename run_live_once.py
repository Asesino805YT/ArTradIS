import subprocess, sys, os
ROOT = r'c:\Users\Ariel\Desktop\moneymoney'
SCRIPT = os.path.join(ROOT, 'ArTradIS.py')
cmd = [sys.executable, SCRIPT, '--live', '--virtual', '--once']
print('Running:', cmd)
proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
print('Returncode:', proc.returncode)
print('Stdout:', proc.stdout[:2000])
print('Stderr:', proc.stderr[:2000])
signals = os.path.join(ROOT, 'signals.csv')
print('signals exists?', os.path.exists(signals))
if os.path.exists(signals):
    with open(signals,'r',encoding='utf-8') as f:
        print('Header:', f.readline().strip())
