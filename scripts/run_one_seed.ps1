param(
    [string]$Method = "etf_anchor",
    [int]$Seed = 42,
    [string]$Config = "config/config.yaml"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

python experiments\run_all.py --config $Config --seeds $Seed --methods $Method

