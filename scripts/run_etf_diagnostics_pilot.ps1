$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$configs = @(
    "config/cifar_pilot.yaml",
    "config/diagnostic_centered_pilot.yaml",
    "config/diagnostic_normalized_pilot.yaml",
    "config/diagnostic_normalized_sparse_pilot.yaml",
    "config/diagnostic_warmup_pilot.yaml",
    "config/diagnostic_sparse_anchor_pilot.yaml"
)

foreach ($config in $configs) {
    Write-Host "Running ETF diagnostic pilot: $config"
    .\.venv\Scripts\python.exe experiments\run_all.py --config $config --seeds 42 --methods etf_anchor --skip-plots
}

Write-Host "Diagnostic pilots finished."
