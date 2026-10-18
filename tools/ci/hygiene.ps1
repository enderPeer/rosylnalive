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
      - Policy: the branch is judged by docs/ownership.json and tools/ci/hygiene.json as they are at the merge base,
        so a branch cannot loosen the rules it is checked by. Map coverage uses the checked-out map, which must
        describe the checked-out tree. When the merge base predates a file or the map's contract section, the
        checked-out one is used and the summary says so.
      - Labels: -Labels, else (with -Ci on a pull request) read live through `gh api` (so a re-run sees labels added
        after the push), falling back to the event payload. Outside a PR, labels are unknown and the label rules are
        reported, not enforced.
      - Integrators (with -Ci on a pull request): an int/ branch passes only when the PR author, and the exception
        label counts only when whoever last applied it, is listed under 'integrators' in the merge base's map.

    Output: one block per check, then a table. artifacts/hygiene/summary.json (schema lrm.hygiene-summary/1, one
    record per check, named hygiene.<check>) and summary.md; with -Ci the table is also appended to
    GITHUB_STEP_SUMMARY.

    Exit code: 0 when every check passes, 1 when any check has an error, 2 for a usage or git error.

.EXAMPLE
    pwsh tools/ci/hygiene.ps1                              # a branch based on main
    pwsh tools/ci/hygiene.ps1 -Base origin/ws/S1-core      # a stacked branch
    pwsh tools/ci/hygiene.ps1 -Labels contract-change      # as a PR with that label
    pwsh tools/ci/hygiene.ps1 -Only ownership,contract-label
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
    # Checks to run, as a list or comma-separated: ownership, layering, sizes, eol, contract-label.
    [string[]]$Only,
    [switch]$Ci,
    [string]$OutDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'Hygiene.psm1') -Force

$RepoRoot = [IO.Path]::GetFullPath($RepoRoot)
if (-not $OutDir) { $OutDir = Join-Path $RepoRoot 'artifacts/hygiene' }
$allChecks = @('ownership', 'layering', 'sizes', 'eol', 'contract-label')
$checks = @(if ($Only) { $Only | ForEach-Object { $_ -split ',' } | ForEach-Object Trim | Where-Object { $_ } } else { $allChecks })
$unknownChecks = @($checks | Where-Object { $allChecks -cnotcontains $_ })
if ($unknownChecks.Count -or -not $checks.Count) {
    Write-Host "error: hygiene: unknown check(s) '$($unknownChecks -join ', ')' in -Only; use any of $($allChecks -join ', ')" -ForegroundColor Red
    exit 2
}

# ---- context ------------------------------------------------------------------------------------

$labelsKnown = $PSBoundParameters.ContainsKey('Labels')
$labelList = @(Split-Labels $Labels)
$labelSource = if ($labelsKnown) { 'argument' } else { 'unknown' }
$ghEvent = $null
$pr = $null
$verify = $false
$author = ''
try {
    if ($Ci) {
        $eventName = $env:GITHUB_EVENT_NAME
        if ($env:GITHUB_EVENT_PATH -and (Test-Path -LiteralPath $env:GITHUB_EVENT_PATH)) { $ghEvent = Get-Content -Raw -LiteralPath $env:GITHUB_EVENT_PATH | ConvertFrom-Json }
        $pr = if ($ghEvent) { Get-Prop $ghEvent 'pull_request' } else { $null }
        if ($pr) {
            $verify = $true
            $author = [string](Get-Prop (Get-Prop $pr 'user') 'login' '')
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
    # The checked-out map describes the checked-out tree (coverage); the merge base's map judges the branch.
    $map = Read-OwnershipMap (Join-Path $RepoRoot 'docs/ownership.json')
    $isMain = $Branch -ceq 'main'
    $needDiff = ($checks -contains 'contract-label') -or ($checks -contains 'ownership')
    $changes = @()
    $mergeBase = $null
    $policyMap = $map
    $policySource = 'the checked-out map'
    if (-not $isMain -and ($needDiff -or $checks -contains 'sizes')) {
        try { $mergeBase = Get-GitMergeBase $RepoRoot $Base $Head }
        catch { if ($needDiff) { throw } else { Write-Host "warning: no merge base of $Base and $Head ($($_.Exception.Message)); sizes uses the checked-out tools/ci/hygiene.json" -ForegroundColor Yellow } }
    }
    if ($mergeBase) {
        $policySource = "the merge base $($mergeBase.Substring(0, 12))"
        $policyMap = Get-GitOwnershipMap $RepoRoot $mergeBase
    }
    $kind = if ($isMain) { 'main' } elseif ($policyMap) { (Resolve-Branch $policyMap $Branch).Kind } else { 'no trusted map' }
    if ($needDiff -and $mergeBase) {
        $changes = @(Get-GitChanges $RepoRoot $Base $Head)
        Add-GitAppendInfo $RepoRoot $Base $Head $mergeBase $changes $policyMap
    }
    $contractPick = Select-ContractMap $policyMap $map
    $exceptionActor = ''
    $exceptionLabel = if ($contractPick.Map.Contract) { $contractPick.Map.Contract.ExceptionLabel } else { 'ownership-exception' }
}
catch {
    Write-Host "error: hygiene: $($_.Exception.Message)" -ForegroundColor Red
    exit 2
}

if ($verify -and $labelsKnown -and $labelList -ccontains $exceptionLabel) {
    $exceptionActor = Get-GitHubLabelActor $env:GITHUB_REPOSITORY $pr $exceptionLabel $ghEvent
}
$labelText = if ($labelsKnown) { if ($labelList.Count) { $labelList -join ', ' } else { '(none)' } } else { '(unknown)' }
Write-Host "hygiene: repo $RepoRoot"
Write-Host "hygiene: branch $Branch ($kind); diff $(if ($mergeBase -and $needDiff) { "$($mergeBase.Substring(0, 12))..$Head, $($changes.Count) file(s)" } else { 'not needed' }); labels $labelText [$labelSource]"
Write-Host "hygiene: policy from $policySource$(if ($mergeBase -and -not $policyMap) { ' (it has no docs/ownership.json)' })$(if ($verify) { "; PR author $(if ($author) { $author } else { 'unknown' })" })$(if ($exceptionActor) { "; '$exceptionLabel' set by $exceptionActor" })"

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
                Test-Ownership -Map $map -PolicyMap $policyMap -PolicySource $policySource -Tracked $tracked -Branch $Branch -Changes $changes `
                    -Labels $labelList -LabelsKnown $labelsKnown -Verify $verify -Author $author -ExceptionActor $exceptionActor
            }
            'layering' { Test-Layering -RepoRoot $RepoRoot }
            'sizes' {
                $config = if ($mergeBase) { Get-GitHygieneConfig $RepoRoot $mergeBase } else { $null }
                $own = $null -eq $config
                if ($own) { $config = Read-HygieneConfig (Join-Path $RepoRoot 'tools/ci/hygiene.json') }
                $res = Test-Sizes -Config $config -Entries @(Get-GitTreeEntries $RepoRoot 'HEAD')
                $res.Metrics['policy'] = if ($own) { 'checked_out' } else { 'merge_base' }
                if ($own -and $mergeBase) { Add-Finding $res note "$policySource has no tools/ci/hygiene.json; the checked-out one was used" }
                $res
            }
            'eol' { Test-Eol -Entries @(Get-GitEolEntries $RepoRoot) }
            'contract-label' {
                $cm = $contractPick.Map
                $frozen = Test-GitTag $RepoRoot $cm.Contract.FreezeTag
                $added = if ($isMain -or -not $mergeBase) { @() } else { @(Get-GitAddedLines $RepoRoot $Base $Head 'docs/contracts/CHANGELOG.md') }
                $res = Test-ContractLabel -Map $cm -Changes $changes -Labels $labelList -LabelsKnown $labelsKnown -Frozen $frozen -ChangelogAdded $added
                $res.Metrics['policy'] = if ($contractPick.Fallback) { 'checked_out' } else { 'merge_base' }
                if ($isMain) { Add-Finding $res note 'main: the label applies to pull requests only' }
                elseif ($contractPick.Fallback) { Add-Finding $res note "$policySource has no contract section in docs/ownership.json (it predates S0b); the checked-out one was used" }
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
    policy = $policySource
    labels = $labelsValue
    passed = $failed.Count -eq 0
    checks = $records
}
$md = [Text.StringBuilder]::new()
[void]$md.AppendLine('### Hygiene')
[void]$md.AppendLine('')
[void]$md.AppendLine("Commit ``$head``, branch ``$Branch``, labels $labelText, policy from $policySource.")
[void]$md.AppendLine('')
[void]$md.AppendLine('| Check | Result | Errors | Warnings | Measured |')
[void]$md.AppendLine('| --- | --- | ---: | ---: | --- |')
foreach ($rec in $records) {
    [void]$md.AppendLine("| $($rec.name) | $(if ($rec.passed) { 'pass' } else { '**FAIL**' }) | $($rec.errors) | $($rec.warnings) | $(([string]$rec.summary).Replace('|', '\|')) |")
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
