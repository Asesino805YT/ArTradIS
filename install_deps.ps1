# Script para instalar dependencias en Windows (PowerShell)
# Uso: abrir PowerShell, activar el venv y ejecutar: .\install_deps.ps1

param(
    [switch]$withTalib
)

Write-Host "Actualizando pip..."
python -m pip install --upgrade pip setuptools wheel

if ($withTalib) {
    Write-Host "Intentando instalar TA-Lib vía pipwin..."
    pip install pipwin
    pipwin install TA-Lib
}

Write-Host "Instalando dependencias del requirements.txt (usa 'ta' si no instalaste TA-Lib)..."
pip install -r requirements.txt

# Instalar python-dotenv para permitir carga de .env automáticamente
Write-Host "Instalando python-dotenv (para .env support)..."
pip install python-dotenv
Write-Host "Instalación finalizada. Si hubo errores, lee la salida y prueba la alternativa sin TA-Lib." 
