param(
    [string]$Config = "config/config.yaml"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

python experiments\run_ablation.py --config $Config --vary alpha --values 0.1 0.5 1.0 5.0 10.0

