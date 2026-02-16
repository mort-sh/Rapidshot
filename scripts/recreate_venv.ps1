param(
    [string]$VenvPath = ".venv",
    [switch]$All
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "Recreating venv at $VenvPath..."

if (Test-Path $VenvPath) {
    Write-Host "Removing existing venv..."
    Remove-Item -Recurse -Force $VenvPath
}

Write-Host "Creating venv with uv..."
uv venv $VenvPath

$activateScript = Join-Path $VenvPath "Scripts\\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    throw "Activation script not found at $activateScript"
}

Write-Host "Activating venv..."
. $activateScript

if ($All) {
    Write-Host "Installing all groups + extras..."
    uv sync --all-groups --all-extras
} else {
    Write-Host "Installing default dependencies..."
    uv sync
}

Write-Host "Done."
