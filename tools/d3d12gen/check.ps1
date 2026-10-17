<#
.SYNOPSIS
    Codegen diff for the D3D12 interop: regenerates in memory and fails when the committed files are stale.

.DESCRIPTION
    Runs tools/d3d12gen (Lrm.Tools.D3D12Gen) with --check against the Windows SDK headers pinned in
    tools/d3d12gen/bindings.json (version and SHA-256 per header). Exit codes: 0 current; 1 stale (regenerate with
    `dotnet run --project tools/d3d12gen/Lrm.Tools.D3D12Gen -c Release`); 2 generator error; 4 the pinned headers are
    missing or differ. With -SkipIfHeadersDiffer, exit 4 becomes a warning and exit 0: a runner with another
    servicing build of SDK 10.0.26100 cannot regenerate byte for byte, while the ABI oracle still checks the result.

.EXAMPLE
    pwsh tools/d3d12gen/check.ps1
    pwsh tools/d3d12gen/check.ps1 -NoBuild -SkipIfHeadersDiffer    # CI
#>
[CmdletBinding()]
param(
    [switch]$NoBuild,
    [switch]$SkipIfHeadersDiffer,
    [ValidateSet('Release', 'Debug')]
    [string]$Configuration = 'Release'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$project = Join-Path $repo 'tools/d3d12gen/Lrm.Tools.D3D12Gen'

if (-not $NoBuild) {
    & dotnet build $project -c $Configuration --nologo -v quiet
    if ($LASTEXITCODE -ne 0) { Write-Host 'd3d12gen: build failed' -ForegroundColor Red; exit 2 }
}

$dll = Join-Path $repo "artifacts/bin/Lrm.Tools.D3D12Gen/$($Configuration.ToLowerInvariant())/Lrm.Tools.D3D12Gen.dll"
& dotnet $dll --check --repo $repo
$code = $LASTEXITCODE

if ($code -eq 4 -and $SkipIfHeadersDiffer) {
    $message = 'd3d12gen: the pinned Windows SDK headers are not available here; codegen diff skipped'
    if ($env:GITHUB_ACTIONS -eq 'true') { Write-Host "::warning::$message" } else { Write-Warning $message }
    exit 0
}
exit $code
