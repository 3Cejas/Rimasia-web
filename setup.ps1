$ErrorActionPreference = 'Stop'

Set-Location $PSScriptRoot

if (-not (Test-Path '.venv')) {
    py -3.14 -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pip install --no-deps pyverse==1.1.0

@"
from rhyme_engine import ensure_engine_assets, get_constellation_payload

ensure_engine_assets()
get_constellation_payload()
"@ | & .\.venv\Scripts\python.exe -

Write-Host ''
Write-Host 'Entorno listo.'
Write-Host 'Arranque: .\\.venv\\Scripts\\python.exe run.py'
