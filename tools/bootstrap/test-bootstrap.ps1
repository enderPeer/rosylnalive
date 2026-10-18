<#
.SYNOPSIS
    Self-test of bootstrap.ps1's file entries against a loopback HTTP server that counts requests.

.DESCRIPTION
    Runs bootstrap.ps1 with a fixture toolchain.json and a temporary LRM_DATA, in child processes, and checks:
      - an approved entry downloads once, and a second run downloads nothing (the request count stays 1);
      - a pending entry is never downloaded, even with -AllowDownload; its pinned local copy is verified, a corrupted
        copy fails the run, and a missing one is skipped;
      - a pending entry without pins is skipped even when a copy exists;
      - a missing approved file without -AllowDownload fails, and with it downloads again;
      - a download that does not match its pin fails and leaves neither the file nor a .partial behind.
    Nothing leaves the machine: the server listens on 127.0.0.1 only and serves a generated payload.

    Exit code: 0 when every check holds, otherwise 1.

.EXAMPLE
    pwsh tools/bootstrap/test-bootstrap.ps1
#>
[CmdletBinding()]
param([switch]$Keep)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$boot = Join-Path $PSScriptRoot 'bootstrap.ps1'
$pwsh = (Get-Process -Id $PID).Path
$root = Join-Path ([IO.Path]::GetTempPath()) ("lrm-bootstrap-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
$data = Join-Path $root 'data'
New-Item -ItemType Directory -Force -Path $data | Out-Null

$payload = [Text.Encoding]::UTF8.GetBytes(('LRM bootstrap fixture payload' + "`n") * 64)
$sha = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($payload)).ToLowerInvariant()

# A loopback HTTP/1.1 server that answers every GET with the payload and counts the requests.
$state = [hashtable]::Synchronized(@{ Requests = 0; Stop = $false })
$listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
$listener.Start()
$port = $listener.LocalEndpoint.Port
$server = Start-ThreadJob -ArgumentList $listener, $state, $payload -ScriptBlock {
    param($listener, $state, $payload)
    while (-not $state.Stop) {
        $accept = $listener.AcceptTcpClientAsync()
        while (-not $accept.Wait(50)) { if ($state.Stop) { return } }
        $client = $accept.Result
        try {
            $stream = $client.GetStream()
            $reader = [IO.StreamReader]::new($stream, [Text.Encoding]::ASCII, $false, 1024, $true)
            while ($true) { $line = $reader.ReadLine(); if ($null -eq $line -or $line -eq '') { break } }
            $state.Requests++
            $head = [Text.Encoding]::ASCII.GetBytes("HTTP/1.1 200 OK`r`nContent-Type: application/octet-stream`r`nContent-Length: $($payload.Length)`r`nConnection: close`r`n`r`n")
            $stream.Write($head, 0, $head.Length)
            $stream.Write($payload, 0, $payload.Length)
            $stream.Flush()
        }
        finally { $client.Close() }
    }
}

function Entry([string]$Approval, [string]$Install, $Sha256 = $sha) {
    [ordered]@{
        managedBy = 'bootstrap'; kind = 'file'; approval = $Approval; version = 'fixture'
        url = "http://127.0.0.1:$port/fixture.bin"; size = $payload.Length; sha256 = $Sha256; install = $Install
    }
}
$toolchain = [ordered]@{
    schema          = 'lrm.toolchain/1'
    dataRoot        = $data
    approvedFile    = Entry 'approved (fixture)' 'toolchain/fixture/approved.bin'
    pendingLocal    = Entry 'pending' 'models/fixture/local.bin'
    pendingMissing  = Entry 'pending' 'models/fixture/missing.bin'
    pendingUnpinned = Entry 'pending' 'models/fixture/unpinned.bin' $null
    badDownload     = Entry 'approved (fixture)' 'toolchain/fixture/bad.bin' ('0' * 64)
}
$toolchainPath = Join-Path $root 'toolchain.json'
[IO.File]::WriteAllText($toolchainPath, ($toolchain | ConvertTo-Json -Depth 4), [Text.UTF8Encoding]::new($false))
function Put([string]$Rel, [byte[]]$Bytes) {
    $p = Join-Path $data $Rel
    New-Item -ItemType Directory -Force -Path (Split-Path $p) | Out-Null
    [IO.File]::WriteAllBytes($p, $Bytes)
}
Put 'models/fixture/local.bin' $payload
Put 'models/fixture/unpinned.bin' ([Text.Encoding]::UTF8.GetBytes("anything`n"))

$failures = 0
function Invoke-Bootstrap([string]$Name, [string[]]$Arguments, [int]$ExpectCode, [int]$ExpectRequests, [string[]]$Patterns) {
    $out = & $pwsh -NoProfile -NonInteractive -File $boot -ToolchainPath $toolchainPath -DataRoot $data @Arguments 2>&1 | Out-String
    $code = $LASTEXITCODE
    $problems = @()
    if ($code -ne $ExpectCode) { $problems += "exit $code, expected $ExpectCode" }
    if ($state.Requests -ne $ExpectRequests) { $problems += "$($state.Requests) request(s) so far, expected $ExpectRequests" }
    foreach ($p in $Patterns) { if ($out -notmatch $p) { $problems += "output does not match /$p/" } }
    if ($problems) {
        $script:failures++
        Write-Host "  FAIL  $Name`: $($problems -join '; ')" -ForegroundColor Red
        Write-Host ($out.Trim() -replace '(?m)^', '        ')
    }
    else { Write-Host "  ok    $Name (exit $code, $($state.Requests) request(s) so far)" }
}

try {
    Write-Host "bootstrap self-test: loopback server on 127.0.0.1:$port, LRM_DATA $data" -ForegroundColor Cyan
    $entries = @('-Only', 'approvedFile,pendingLocal,pendingMissing,pendingUnpinned')
    Invoke-Bootstrap 'first run downloads the approved file once' (@('-AllowDownload') + $entries) 0 1 @(
        'approvedFile: downloading http://127.0.0.1', 'approvedFile fixture: OK', 'pendingLocal fixture: OK .*local copy verified; download not approved',
        'pendingMissing fixture: SKIPPED, the download awaits user approval; nothing fetched; not present',
        'pendingUnpinned fixture: SKIPPED.*a copy exists at .* but is not verified')
    Invoke-Bootstrap 'second run downloads nothing' (@('-AllowDownload') + $entries) 0 1 @('approvedFile fixture: OK', 'bootstrap: OK')
    Put 'models/fixture/local.bin' ($payload + [byte[]]@(10))
    Invoke-Bootstrap 'a corrupted pending copy fails and is not re-downloaded' (@('-AllowDownload') + $entries) 1 1 @("pendingLocal: .* is $($payload.Length + 1) bytes, the pin says $($payload.Length)")
    Put 'models/fixture/local.bin' $payload
    Remove-Item -LiteralPath (Join-Path $data 'toolchain/fixture/approved.bin')
    Invoke-Bootstrap 'a missing approved file without -AllowDownload fails' $entries 1 1 @('Run bootstrap with -AllowDownload')
    Invoke-Bootstrap 'with -AllowDownload it downloads again' (@('-AllowDownload') + $entries) 0 2 @('approvedFile: downloading http://127.0.0.1')
    Invoke-Bootstrap 'a download that misses its pin fails' @('-AllowDownload', '-Only', 'badDownload') 1 3 @('badDownload: .* has SHA-256 [0-9a-f]{64}, the pin says 0{64}')
    $left = @(Get-ChildItem -LiteralPath (Join-Path $data 'toolchain/fixture') -Filter 'bad.bin*' -ErrorAction SilentlyContinue)
    if ($left) { $failures++; Write-Host "  FAIL  the failed download left $($left.Name -join ', ')" -ForegroundColor Red }
    else { Write-Host '  ok    the failed download left neither the file nor a .partial' }
}
finally {
    $state.Stop = $true
    Wait-Job $server -Timeout 5 | Out-Null
    Remove-Job $server -Force
    $listener.Stop()
    if (-not $Keep) { Remove-Item -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue } else { Write-Host "kept $root" }
}

if ($failures) { Write-Host "bootstrap self-test: FAILED ($failures)" -ForegroundColor Red; exit 1 }
Write-Host 'bootstrap self-test: OK'
exit 0
