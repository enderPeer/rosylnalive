<#
.SYNOPSIS
    Size check: per-file limits, budgets and forbidden files over a commit's tree (tools/ci/hygiene.json).

.DESCRIPTION
    Sizes are the committed blob sizes from `git ls-tree -r -l`, so the result does not depend on line-ending
    conversion in a checkout. Each blob is held to the first matching fileLimits rule, else defaultMaxFileBytes; each
    budget bounds the total of the blobs its glob matches; forbidden rules keep build output, toolchains, registries,
    headers, weights and the agent allowlist out of git.

    Exit code: 0 passed, 1 findings, 2 usage or git error.

.EXAMPLE
    pwsh tools/ci/check-sizes.ps1
    pwsh tools/ci/check-sizes.ps1 -Rev origin/ws/O1-oracles
#>
[CmdletBinding()]
param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path,
    [string]$Rev = 'HEAD',
    [string]$Config,
    [string]$ResultPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'Hygiene.psm1') -Force
$clock = [Diagnostics.Stopwatch]::StartNew()
try {
    if (-not $Config) { $Config = Join-Path $RepoRoot 'tools/ci/hygiene.json' }
    $r = Test-Sizes -Config (Read-HygieneConfig $Config) -Entries @(Get-GitTreeEntries $RepoRoot $Rev)
}
catch {
    Write-Host "error: check-sizes: $($_.Exception.Message)" -ForegroundColor Red
    exit 2
}
exit (Complete-HygieneCheck $r $ResultPath $clock.Elapsed.TotalMilliseconds)
