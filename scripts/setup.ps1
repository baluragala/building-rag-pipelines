# ===========================================================================
# Building RAG Pipelines - one-shot environment setup for Windows (PowerShell)
# ===========================================================================
# Creates a Python virtual environment in .venv, installs dependencies, and
# registers a Jupyter kernel so the notebooks pick up this environment.
#
# Usage (from the project root, in PowerShell):
#     powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
# Then activate:
#     .\.venv\Scripts\Activate.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"

# Move to the project root (this script lives in scripts\).
Set-Location (Join-Path $PSScriptRoot "..")

# Pick a python launcher: prefer the 'py' launcher, fall back to 'python'.
$py = $null
if (Get-Command py -ErrorAction SilentlyContinue)     { $py = "py -3" }
elseif (Get-Command python -ErrorAction SilentlyContinue) { $py = "python" }
else {
  Write-Error "Python 3 was not found. Install Python 3.9+ from python.org (tick 'Add to PATH') and re-run."
  exit 1
}
Write-Host "Using launcher: $py"

Write-Host "==> Creating virtual environment in .venv"
Invoke-Expression "$py -m venv .venv"

Write-Host "==> Activating environment"
& .\.venv\Scripts\Activate.ps1

Write-Host "==> Upgrading pip"
python -m pip install --upgrade pip | Out-Null

Write-Host "==> Installing project dependencies (requirements.txt)"
pip install -r requirements.txt

Write-Host "==> Installing JupyterLab + kernel"
pip install jupyterlab ipykernel
python -m ipykernel install --user --name rag-pipelines --display-name "Python (RAG Pipelines)"

Write-Host ""
Write-Host "============================================================"
Write-Host " Setup complete."
Write-Host " Activate the environment:   .\.venv\Scripts\Activate.ps1"
Write-Host " Launch the notebooks:       jupyter lab"
Write-Host "   (in a notebook, choose the 'Python (RAG Pipelines)' kernel)"
Write-Host " Deactivate when done:       deactivate"
Write-Host "============================================================"
