<#
.SYNOPSIS
    Runs a golden tool from tools/goldens in LRM_DATA/venv-golden, isolated and offline (ReferenceOracles contract).

.DESCRIPTION
    Resolves the venv from tools/bootstrap/toolchain.json (goldenVenv.path under LRM_DATA), sets the offline and
    hub-kernel switches, and runs `python -I <tool> <args>`. The tools check these guards again themselves
    (tools/goldens/common.py), so running python directly with -I works too.

.EXAMPLE
    pwsh tools/goldens/run.ps1 make_tiny.py
    pwsh tools/goldens/run.ps1 make_tiny.py --check
    pwsh tools/goldens/run.ps1 make_tokenizer.py --check
    pwsh tools/goldens/run.ps1 selftest.py
    pwsh tools/goldens/run.ps1 verify_goldens.py
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [string]$Tool,
    [Parameter(ValueFromRemainingArguments)]
    [string[]]$ToolArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$toolchain = Get-Content -Raw (Join-Path $repo 'tools/bootstrap/toolchain.json') | ConvertFrom-Json
$data = if ($env:LRM_DATA -and $env:LRM_DATA.Trim()) { $env:LRM_DATA } else { $toolchain.dataRoot }
$python = Join-Path $data (Join-Path $toolchain.goldenVenv.path 'Scripts/python.exe')
if (-not (Test-Path $python)) { throw "run.ps1: the golden venv is missing: $python" }

$script = Join-Path $PSScriptRoot $Tool
if (-not (Test-Path $script)) { throw "run.ps1: no such tool: $script" }

$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:USE_HUB_KERNELS = '0'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:HF_HUB_DISABLE_TELEMETRY = '1'

& $python -I $script @ToolArgs
exit $LASTEXITCODE
