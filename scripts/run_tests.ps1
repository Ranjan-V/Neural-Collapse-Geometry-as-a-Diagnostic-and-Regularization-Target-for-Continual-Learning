param(
    [switch]$Install
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if ($Install) {
    python -m pip install -r requirements.txt
}

python scripts\smoke_check.py
python -m pytest

