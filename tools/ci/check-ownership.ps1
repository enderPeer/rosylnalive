<#
.SYNOPSIS
    Ownership check (OwnershipAndContractChange contract): map coverage and out-of-path edits.

.DESCRIPTION
    1. Every file in the tree of -Rev is owned by exactly one family of docs/ownership.json.
    2. The branch name follows ws/<id>-<slug>, ws/<id>-fix-<slug> or int/<slug>, and <id> is a known slice.
    3. On a ws/ branch, every path changed between the merge base of -Base and -Head, and -Head, is owned by the
       branch's family, or is a shared append-only path (docs/contracts/CHANGELOG.md) that the change only appends to,
       at the end. With the exception label (set only by the integrator) such paths are warnings instead of errors.
    4. Outside family S0, no governance path (docs/ownership.json, tools/ci/**, .github/**) changes, whatever the map
       and the labels say.
    Steps 2 to 4 use docs/ownership.json at the merge base (or -PolicyMap), never the branch's own copy, so a branch
    cannot grant itself paths. int/ branches may touch any path (reported as a warning), and main is checked for
    coverage only. Who set the exception label, and who opened an int/ PR, are verified by hygiene.ps1 -Ci only.

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
    # The map for coverage. Default: docs/ownership.json in the working tree.
    [string]$Map,
    # The map that judges the branch. Default: docs/ownership.json at the merge base.
    [string]$PolicyMap,
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
    $policy = $m
    $source = 'the checked-out map'
    if ($Branch -cne 'main') {
        $mergeBase = Get-GitMergeBase $RepoRoot $Base $Head
        if ($PolicyMap) { $policy = Read-OwnershipMap $PolicyMap -Partial; $source = $PolicyMap }
        else { $policy = Get-GitOwnershipMap $RepoRoot $mergeBase; $source = "the merge base $($mergeBase.Substring(0, 12))" }
        $changes = @(Get-GitChanges $RepoRoot $Base $Head)
        Add-GitAppendInfo $RepoRoot $Base $Head $mergeBase $changes $policy
    }
    $known = $PSBoundParameters.ContainsKey('Labels')
    $r = Test-Ownership -Map $m -PolicyMap $policy -PolicySource $source -Tracked $tracked -Branch $Branch -Changes $changes -Labels (Split-Labels $Labels) -LabelsKnown $known
}
catch {
    Write-Host "error: check-ownership: $($_.Exception.Message)" -ForegroundColor Red
    exit 2
}
exit (Complete-HygieneCheck $r $ResultPath $clock.Elapsed.TotalMilliseconds)
