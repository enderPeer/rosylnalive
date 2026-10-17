<#
.SYNOPSIS
    Ownership check (OwnershipAndContractChange contract): map coverage and out-of-path edits.

.DESCRIPTION
    1. Every file in the tree of -Rev is owned by exactly one family of docs/ownership.json.
    2. The branch name follows ws/<id>-<slug>, ws/<id>-fix-<slug> or int/<slug>, and <id> is a known slice.
    3. On a ws/ branch, every path changed between the merge base of -Base and -Head, and -Head, is owned by the
       branch's family, or is a shared append-only path (docs/contracts/CHANGELOG.md) that the change only appends to.
       With the exception label (set only by the integrator) such paths are warnings instead of errors.
    int/ branches may touch any path, and main is checked for coverage only.

    Exit code: 0 passed, 1 findings, 2 usage or git error.

.EXAMPLE
    pwsh tools/ci/check-ownership.ps1 -Base origin/main
    pwsh tools/ci/check-ownership.ps1 -Branch ws/O1-oracles -Base origin/main -Head origin/ws/O1-oracles
#>
[CmdletBinding()]
param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path,
    # Default: the current branch.
    [string]$Branch,
    # Base commit or ref of the branch diff (the diff starts at its merge base with -Head).
    [string]$Base = 'origin/main',
    [string]$Head = 'HEAD',
    # Tree whose files must all be owned. Default: -Head.
    [string]$Rev,
    # PR labels. Leave out when unknown; pass '' for a PR without labels.
    [string[]]$Labels,
    [string]$Map,
    [string]$ResultPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'Hygiene.psm1') -Force
$clock = [Diagnostics.Stopwatch]::StartNew()
try {
    if (-not $Map) { $Map = Join-Path $RepoRoot 'docs/ownership.json' }
    if (-not $Rev) { $Rev = $Head }
    $m = Read-OwnershipMap $Map
    if (-not $Branch) { $Branch = Get-GitBranch $RepoRoot }
    if (-not $Branch) { throw 'HEAD is detached; pass -Branch' }
    $tracked = @(Get-GitTreeEntries $RepoRoot $Rev | ForEach-Object Path)
    $changes = @()
    if ((Resolve-Branch $m $Branch).Kind -ceq 'workstream') { $changes = @(Get-GitChanges $RepoRoot $Base $Head) }
    $known = $PSBoundParameters.ContainsKey('Labels')
    $r = Test-Ownership -Map $m -Tracked $tracked -Branch $Branch -Changes $changes -Labels (Split-Labels $Labels) -LabelsKnown $known
}
catch {
    Write-Host "error: check-ownership: $($_.Exception.Message)" -ForegroundColor Red
    exit 2
}
exit (Complete-HygieneCheck $r $ResultPath $clock.Elapsed.TotalMilliseconds)
