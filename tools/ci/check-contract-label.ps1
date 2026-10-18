<#
.SYNOPSIS
    Contract-label check (OwnershipAndContractChange contract).

.DESCRIPTION
    Contract files are the globs under `contract.files` in docs/ownership.json at the merge base (the branch's own copy
    cannot drop a file or rename the tag). Once the freeze tag (contracts-v1) exists, a PR that changes one needs the
    contract-change label. A labelled PR must change a contract file, must add
    lines to docs/contracts/CHANGELOG.md, and every contract text it touches must be named in those lines. Without
    labels (not a PR) the label rules are reported, not enforced.

    Exit code: 0 passed, 1 findings, 2 usage or git error.

.EXAMPLE
    pwsh tools/ci/check-contract-label.ps1 -Base origin/main -Labels contract-change
#>
[CmdletBinding()]
param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path,
    [string]$Base = 'origin/main',
    [string]$Head = 'HEAD',
    # PR labels. Leave out when unknown; pass '' for a PR without labels.
    [string[]]$Labels,
    # auto: frozen when the freeze tag exists in the repository.
    [ValidateSet('auto', 'yes', 'no')]
    [string]$Frozen = 'auto',
    # The map whose contract section applies. Default: docs/ownership.json at the merge base, else the working tree's.
    [string]$Map,
    [string]$ResultPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'Hygiene.psm1') -Force
$clock = [Diagnostics.Stopwatch]::StartNew()
try {
    if ($Map) { $m = Read-OwnershipMap $Map }
    else {
        $mergeBase = Get-GitMergeBase $RepoRoot $Base $Head
        $m = (Select-ContractMap (Get-GitOwnershipMap $RepoRoot $mergeBase) (Read-OwnershipMap (Join-Path $RepoRoot 'docs/ownership.json') -Partial)).Map
        if ($null -eq $m.Contract) { throw 'neither the merge base nor the working tree has a docs/ownership.json with a contract section' }
    }
    $isFrozen = switch ($Frozen) { 'yes' { $true } 'no' { $false } default { Test-GitTag $RepoRoot $m.Contract.FreezeTag } }
    $changes = @(Get-GitChanges $RepoRoot $Base $Head)
    $added = @(Get-GitAddedLines $RepoRoot $Base $Head 'docs/contracts/CHANGELOG.md')
    $known = $PSBoundParameters.ContainsKey('Labels')
    $r = Test-ContractLabel -Map $m -Changes $changes -Labels (Split-Labels $Labels) -LabelsKnown $known -Frozen $isFrozen -ChangelogAdded $added
}
catch {
    Write-Host "error: check-contract-label: $($_.Exception.Message)" -ForegroundColor Red
    exit 2
}
exit (Complete-HygieneCheck $r $ResultPath $clock.Elapsed.TotalMilliseconds)
