<#
.SYNOPSIS
    Layering check (ProjectGraph contract): runs tools/ci/gen-project-graph.ps1 -Check against a repo root.

.DESCRIPTION
    The generator is the one validator of docs/project-graph.json: the layering rules on the transitive closure of
    refs and buildRefs, the fsproj shape (file items only, so a hand-written ProjectReference fails), registration of
    every fsproj, stray .lrm.targets files, and generated files that are out of date. This script turns its output
    into hygiene findings.

    Exit code: 0 passed, 1 findings, 2 usage error.

.EXAMPLE
    pwsh tools/ci/check-layering.ps1
#>
[CmdletBinding()]
param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path,
    [string]$ResultPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'Hygiene.psm1') -Force
$clock = [Diagnostics.Stopwatch]::StartNew()
try {
    $r = Test-Layering -RepoRoot ([IO.Path]::GetFullPath($RepoRoot))
}
catch {
    Write-Host "error: check-layering: $($_.Exception.Message)" -ForegroundColor Red
    exit 2
}
exit (Complete-HygieneCheck $r $ResultPath $clock.Elapsed.TotalMilliseconds)
