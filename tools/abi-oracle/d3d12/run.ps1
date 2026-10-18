<#
.SYNOPSIS
    The D3D12 ABI oracle (D3D12ComIdiom contract): checks the runtime layout of Lrm.Gpu.D3D12.Interop with cl.exe.

.DESCRIPTION
    Runs `Lrm.Gpu.D3D12.Interop.Tests abi-oracle`, which
      1. dumps the runtime layout of every bound item (struct sizes and offsets, enum values, vtable slots, delegate
         and P/Invoke signatures, HRESULTs, IIDs) as d3d12_abi.g.c and d3d12_sig.g.cpp;
      2. fails when the committed copies in this folder differ (-Update rewrites them instead);
      3. compiles them with cl.exe (VS 2022 Build Tools through vswhere and vcvars64.bat, Windows SDK 10.0.26100.0
         when installed) against the SDK headers, links oracle_main.c and runs it: the IIDs are compared at run time.
    Exit codes: 0 passed; 1 failed or stale; 2 no Visual Studio C++ toolchain.
    The cpu tier runs the same checks (Lrm.Gpu.D3D12.Interop.Tests), including mutants that must fail.

.EXAMPLE
    pwsh tools/abi-oracle/d3d12/run.ps1             # build, check the committed sources, compile and run
    pwsh tools/abi-oracle/d3d12/run.ps1 -Update     # after changing the bindings: rewrite the committed sources
    pwsh tools/abi-oracle/d3d12/run.ps1 -NoBuild    # what CI runs after its Release build
#>
[CmdletBinding()]
param(
    [switch]$Update,
    [switch]$NoBuild,
    [switch]$NoCompile,
    [ValidateSet('Release', 'Debug')]
    [string]$Configuration = 'Release'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$project = Join-Path $repo 'tests/Lrm.Gpu.D3D12.Interop.Tests'

if (-not $NoBuild) {
    & dotnet build $project -c $Configuration --nologo -v quiet
    if ($LASTEXITCODE -ne 0) { Write-Host 'abi-oracle: build failed' -ForegroundColor Red; exit 1 }
}

$dll = Join-Path $repo "artifacts/bin/Lrm.Gpu.D3D12.Interop.Tests/$($Configuration.ToLowerInvariant())/Lrm.Gpu.D3D12.Interop.Tests.dll"
if (-not (Test-Path -LiteralPath $dll)) { Write-Host "abi-oracle: $dll is missing (build first)" -ForegroundColor Red; exit 1 }

$arguments = @('abi-oracle')
if ($Update) { $arguments += '--update' }
if ($NoCompile) { $arguments += '--no-compile' }

$env:LRM_REPO_ROOT = $repo
& dotnet $dll @arguments
exit $LASTEXITCODE
