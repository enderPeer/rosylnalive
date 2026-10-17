<#
.SYNOPSIS
    Line-ending check: `git ls-files --eol` against .gitattributes.

.DESCRIPTION
    The index holds LF only (i/crlf and i/mixed fail), every file has a text attribute, and binary content is declared
    binary in .gitattributes. CRLF in the working tree alone is a warning (a stale checkout); *.cmd and *.bat are CRLF
    in the working tree by design.

    Exit code: 0 passed, 1 findings, 2 usage or git error.

.EXAMPLE
    pwsh tools/ci/check-eol.ps1
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
    $r = Test-Eol -Entries @(Get-GitEolEntries $RepoRoot)
}
catch {
    Write-Host "error: check-eol: $($_.Exception.Message)" -ForegroundColor Red
    exit 2
}
exit (Complete-HygieneCheck $r $ResultPath $clock.Elapsed.TotalMilliseconds)
