<#
.SYNOPSIS
    Runs the five hygiene checks (ownership, layering, sizes, EOL, contract label) and writes a summary.

.DESCRIPTION
    The hygiene CI job (ubuntu-latest) runs this with -Ci after tools/ci/test-hygiene.ps1. Locally, run it before
    pushing; on a stacked branch pass the parent branch as -Base.

    Context:
      - Branch: -Branch, else (with -Ci) the PR head ref or GITHUB_REF_NAME, else the current branch.
      - Diff: from the merge base of -Base and -Head to -Head. With -Ci on a pull request, Base and Head are the
        event's base and head SHAs. Tree checks (coverage, sizes, EOL, layering) use the checked-out tree.
      - Labels: -Labels, else (with -Ci on a pull request) read live through `gh api` (so a re-run sees labels added
        after the push), falling back to the event payload. Outside a PR, labels are unknown and the label rules are
        reported, not enforced.

    Output: one block per check, then a table. artifacts/hygiene/summary.json (schema lrm.hygiene-summary/1, one
    record per check, named hygiene.<check>) and summary.md; with -Ci the table is also appended to
    GITHUB_STEP_SUMMARY.

    Exit code: 0 when every check passes, 1 when any check has an error, 2 for a usage or git error.

.EXAMPLE
    pwsh tools/ci/hygiene.ps1                              # a branch based on main
    pwsh tools/ci/hygiene.ps1 -Base origin/ws/S1-core      # a stacked branch
    pwsh tools/ci/hygiene.ps1 -Labels contract-change      # as a PR with that label
    pwsh tools/ci/hygiene.ps1 -Ci                          # GitHub Actions
#>
[CmdletBinding()]
param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path,
    [string]$Branch,
    [string]$Base = 'origin/main',
    [string]$Head = 'HEAD',
    # PR labels. Leave out when unknown; pass '' for a PR without labels.
    [string[]]$Labels,
    [ValidateSet('ownership', 'layering', 'sizes', 'eol', 'contract-label')]
    [string[]]$Only,
    [switch]$Ci,
    [string]$OutDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'Hygiene.psm1') -Force

$RepoRoot = [IO.Path]::GetFullPath($RepoRoot)
if (-not $OutDir) { $OutDir = Join-Path $RepoRoot 'artifacts/hygiene' }
$checks = if ($Only) { @($Only | ForEach-Object { $_ -split ',' } | ForEach-Object Trim) } else { @('ownership', 'layering', 'sizes', 'eol', 'contract-label') }

# ---- context ------------------------------------------------------------------------------------

$labelsKnown = $PSBoundParameters.ContainsKey('Labels')
$labelList = @(Split-Labels $Labels)
$labelSource = if ($labelsKnown) { 'argument' } else { 'unknown' }
$event = $null
try {
    if ($Ci) {
        $eventName = $env:GITHUB_EVENT_NAME
        if ($env:GITHUB_EVENT_PATH -and (Test-Path -LiteralPath $env:GITHUB_EVENT_PATH)) { $event = Get-Content -Raw -LiteralPath $env:GITHUB_EVENT_PATH | ConvertFrom-Json }
        $pr = if ($event) { Get-Prop $event 'pull_request' } else { $null }
        if ($pr) {
            if (-not $PSBoundParameters.ContainsKey('Branch')) { $Branch = [string]$pr.head.ref }
            if (-not $PSBoundParameters.ContainsKey('Base')) { $Base = [string]$pr.base.sha }
            if (-not $PSBoundParameters.ContainsKey('Head')) { $Head = [string]$pr.head.sha }
            if (-not $labelsKnown) {
                $live = $null
                if (Get-Command gh -ErrorAction SilentlyContinue) {
                    try {
                        $out = & gh api "repos/$($env:GITHUB_REPOSITORY)/issues/$($pr.number)/labels" --jq '.[].name' 2>&1
                        $code = $LASTEXITCODE
                    }
                    catch { $out = $_.Exception.Message; $code = 1 }
                    if ($code -eq 0) { $live = @($out | ForEach-Object { "$_".Trim() } | Where-Object { $_ }) }
                    else { Write-Host "warning: gh api could not read the PR labels ($out); using the event payload" -ForegroundColor Yellow }
                }
                if ($null -ne $live) { $labelList = $live; $labelSource = 'GitHub API' }
                else { $labelList = @($pr.labels | ForEach-Object { [string]$_.name }); $labelSource = 'event payload' }
                $labelsKnown = $true
            }
        }
        elseif (-not $PSBoundParameters.ContainsKey('Branch')) { $Branch = $env:GITHUB_REF_NAME }
        Write-Host "hygiene: GitHub event '$eventName'"
    }
    if (-not $Branch) { $Branch = Get-GitBranch $RepoRoot }
    if (-not $Branch) { throw 'HEAD is detached; pass -Branch' }
    $map = Read-OwnershipMap (Join-Path $RepoRoot 'docs/ownership.json')
    $kind = (Resolve-Branch $map $Branch).Kind
    $needDiff = ($checks -contains 'contract-label') -or (($checks -contains 'ownership') -and $kind -ceq 'workstream')
    $changes = @()
    $mergeBase = $null
    if ($needDiff -and $kind -cne 'main') {
        $mergeBase = Get-GitMergeBase $RepoRoot $Base $Head
        $changes = @(Get-GitChanges $RepoRoot $Base $Head)
    }
}
catch {
    Write-Host "error: hygiene: $($_.Exception.Message)" -ForegroundColor Red
    exit 2
}

$labelText = if ($labelsKnown) { if ($labelList.Count) { $labelList -join ', ' } else { '(none)' } } else { '(unknown)' }
Write-Host "hygiene: repo $RepoRoot"
Write-Host "hygiene: branch $Branch ($kind); diff $(if ($mergeBase) { "$($mergeBase.Substring(0, 12))..$Head, $($changes.Count) file(s)" } else { 'not needed' }); labels $labelText [$labelSource]"

# ---- checks -------------------------------------------------------------------------------------

$records = @()
$failed = @()
foreach ($check in $checks) {
    Write-Host ''
    Write-Host "== $check" -ForegroundColor Cyan
    $clock = [Diagnostics.Stopwatch]::StartNew()
    try {
        $r = switch ($check) {
            'ownership' {
                $tracked = @(Get-GitTreeEntries $RepoRoot 'HEAD' | ForEach-Object Path)
                Test-Ownership -Map $map -Tracked $tracked -Branch $Branch -Changes $changes -Labels $labelList -LabelsKnown $labelsKnown
            }
            'layering' { Test-Layering -RepoRoot $RepoRoot }
            'sizes' { Test-Sizes -Config (Read-HygieneConfig (Join-Path $RepoRoot 'tools/ci/hygiene.json')) -Entries @(Get-GitTreeEntries $RepoRoot 'HEAD') }
            'eol' { Test-Eol -Entries @(Get-GitEolEntries $RepoRoot) }
            'contract-label' {
                $frozen = Test-GitTag $RepoRoot $map.Contract.FreezeTag
                $added = if ($kind -ceq 'main') { @() } else { @(Get-GitAddedLines $RepoRoot $Base $Head 'docs/contracts/CHANGELOG.md') }
                $res = Test-ContractLabel -Map $map -Changes $changes -Labels $labelList -LabelsKnown $labelsKnown -Frozen $frozen -ChangelogAdded $added
                if ($kind -ceq 'main') { Add-Finding $res note 'main: the label applies to pull requests only' }
                $res
            }
        }
    }
    catch {
        $r = New-HygieneResult $check
        Add-Finding $r error "the check itself failed: $($_.Exception.Message)"
    }
    $clock.Stop()
    Write-HygieneResult $r
    $records += , (ConvertTo-HygieneRecord $r $clock.Elapsed.TotalMilliseconds)
    if ((Get-FindingCount $r 'error') -gt 0) { $failed += $check }
}

# ---- summary ------------------------------------------------------------------------------------

$head = (Invoke-Git $RepoRoot @('rev-parse', '--short=12', 'HEAD')).Out.Trim()
$labelsValue = $null
if ($labelsKnown) { $labelsValue = [string[]]$labelList }
$summary = [ordered]@{
    schema = 'lrm.hygiene-summary/1'
    commit = $head
    branch = $Branch
    base   = $mergeBase
    labels = $labelsValue
    passed = $failed.Count -eq 0
    checks = $records
}
$md = [Text.StringBuilder]::new()
[void]$md.AppendLine('### Hygiene')
[void]$md.AppendLine('')
[void]$md.AppendLine("Commit ``$head``, branch ``$Branch``, labels $labelText.")
[void]$md.AppendLine('')
[void]$md.AppendLine('| Check | Result | Errors | Warnings | Measured |')
[void]$md.AppendLine('| --- | --- | ---: | ---: | --- |')
foreach ($rec in $records) {
    $measured = @($rec.metrics.GetEnumerator() | ForEach-Object { "$($_.Key) $($_.Value)" }) -join '; '
    [void]$md.AppendLine("| $($rec.name) | $(if ($rec.passed) { 'pass' } else { '**FAIL**' }) | $($rec.errors) | $($rec.warnings) | $($measured.Replace('|', '\|')) |")
}
[void]$md.AppendLine('')
[void]$md.AppendLine("Result: **$(if ($failed.Count) { "FAILED ($($failed -join ', '))" } else { 'PASSED' })**")

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$utf8 = [Text.UTF8Encoding]::new($false)
[IO.File]::WriteAllText((Join-Path $OutDir 'summary.json'), ($summary | ConvertTo-Json -Depth 8) + "`n", $utf8)
[IO.File]::WriteAllText((Join-Path $OutDir 'summary.md'), $md.ToString(), $utf8)
if ($Ci -and $env:GITHUB_STEP_SUMMARY) { [IO.File]::AppendAllText($env:GITHUB_STEP_SUMMARY, $md.ToString(), $utf8) }

Write-Host ''
Write-Host $md.ToString()
if ($failed.Count) { exit 1 }
exit 0
