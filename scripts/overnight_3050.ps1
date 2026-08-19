param(
    [string]$Config = "config/config_3050.yaml",
    [string[]]$Methods = @("finetune", "etf_anchor"),
    [int[]]$Seeds = @(42)
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Virtual environment not found. Run: python -m venv .venv"
}

Write-Host "Checking CUDA visibility..."
$cudaCheck = .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
$cudaCheck
if ($cudaCheck -notcontains "True") {
    throw "CUDA is not visible to PyTorch. Do not start overnight training yet."
}

Write-Host "Running tests..."
.\.venv\Scripts\python.exe -m pytest

Write-Host "Running smoke check..."
.\.venv\Scripts\python.exe experiments\run_all.py --config config\smoke.yaml --seeds 42 --methods finetune etf_anchor --skip-plots

Write-Host "Starting overnight run..."
.\.venv\Scripts\python.exe experiments\run_all.py --config $Config --seeds $Seeds --methods $Methods

Write-Host "Overnight run finished."

