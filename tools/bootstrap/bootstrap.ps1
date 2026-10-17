<#
.SYNOPSIS
    Verifies or installs the pinned toolchain under LRM_DATA (DataRootAndToolchain contract).

.DESCRIPTION
    Reads tools/bootstrap/toolchain.json and handles every entry marked managedBy=bootstrap:

      dxc        Verified in place: SHA-256 of each binary, `dxc --version`, Authenticode signer. When it is not
                 installed, it is extracted from the pinned nupkg in LRM_DATA/downloads; the nupkg is downloaded
                 only with -AllowDownload (CI runners, approved under AP-1) and is checked against its pinned size
                 and SHA-512 before use. Without the nupkg and without -AllowDownload the script fails loudly.
      vkxml      Pending user approval: skipped with a message, never downloaded.
      vkheaders  Pending user approval: skipped with a message, never downloaded.

    The exit code is 0 when every approved entry verifies, otherwise 1. Running it again downloads nothing.

.PARAMETER DataRoot
    Overrides LRM_DATA (default: the dataRoot in toolchain.json).

.PARAMETER AllowDownload
    Download a missing approved package from its pinned URL. Use on CI runners only.

.PARAMETER Only
    Restrict to some entries, e.g. -Only dxc.

.EXAMPLE
    pwsh tools/bootstrap/bootstrap.ps1
    pwsh tools/bootstrap/bootstrap.ps1 -AllowDownload      # CI
#>
[CmdletBinding()]
param(
    [string]$DataRoot,
    [switch]$AllowDownload,
    [ValidateSet('dxc', 'vkxml', 'vkheaders')]
    [string[]]$Only
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$toolchainPath = Join-Path $PSScriptRoot 'toolchain.json'
$toolchain = Get-Content -Raw -LiteralPath $toolchainPath | ConvertFrom-Json
if ($toolchain.schema -ne 'lrm.toolchain/1') { throw "toolchain.json: unknown schema '$($toolchain.schema)'" }

if (-not $DataRoot) { $DataRoot = if ($env:LRM_DATA) { $env:LRM_DATA } else { $toolchain.dataRoot } }
$DataRoot = [IO.Path]::GetFullPath($DataRoot)
Write-Host "bootstrap: LRM_DATA = $DataRoot"

function Under([string]$rel) { Join-Path $DataRoot $rel }

function Get-Sha256([string]$path) { (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() }

function Get-Sha512Base64([string]$path) {
    $stream = [IO.File]::OpenRead($path)
    try { [Convert]::ToBase64String([Security.Cryptography.SHA512]::HashData($stream)) } finally { $stream.Dispose() }
}

function Test-Package($e, [string]$path) {
    $len = (Get-Item -LiteralPath $path).Length
    if ($len -ne $e.size) { throw "$path is $len bytes, the pin says $($e.size)" }
    $hash = Get-Sha512Base64 $path
    if ($hash -ne $e.sha512Base64) { throw "$path has SHA-512 $hash, the pin says $($e.sha512Base64)" }
}

function Install-Dxc($e) {
    $install = Under $e.install
    $exe = Under $e.exe
    if (-not (Test-Path -LiteralPath $exe)) {
        if (Test-Path -LiteralPath $install) {
            throw "dxc: $install exists but $($e.exe) is missing; remove that folder by hand and run bootstrap again"
        }
        $nupkg = Under $e.download
        if (-not (Test-Path -LiteralPath $nupkg)) {
            if (-not $AllowDownload) {
                throw ("dxc $($e.version) is not installed at $install and its package is not at $nupkg. " +
                    "Run bootstrap with -AllowDownload (approved for CI runners under AP-1), or put the pinned nupkg there.")
            }
            New-Item -ItemType Directory -Force -Path (Split-Path $nupkg) | Out-Null
            $partial = "$nupkg.partial"
            Write-Host "dxc: downloading $($e.url) ($($e.size) bytes)"
            Invoke-WebRequest -Uri $e.url -OutFile $partial -UseBasicParsing
            Test-Package $e $partial
            Move-Item -LiteralPath $partial -Destination $nupkg -Force
        }
        Test-Package $e $nupkg
        Write-Host "dxc: extracting $nupkg"
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $staging = "$install.staging-$PID"
        $zip = [IO.Compression.ZipFile]::OpenRead($nupkg)
        try {
            foreach ($entry in $zip.Entries) {
                $name = $entry.FullName.Replace('\', '/')
                if ($name.EndsWith('/')) { continue }
                if (-not ($e.extract | Where-Object { $name -like $_ })) { continue }
                $target = [IO.Path]::GetFullPath((Join-Path $staging $name))
                if (-not $target.StartsWith([IO.Path]::GetFullPath($staging))) { throw "dxc: unsafe path in package: $name" }
                New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
                [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $false)
            }
        }
        finally { $zip.Dispose() }
        New-Item -ItemType Directory -Force -Path (Split-Path $install) | Out-Null
        Move-Item -LiteralPath $staging -Destination $install
    }

    foreach ($p in $e.sha256.PSObject.Properties) {
        $file = Join-Path $install $p.Name
        if (-not (Test-Path -LiteralPath $file)) { throw "dxc: $file is missing" }
        $have = Get-Sha256 $file
        if ($have -ne $p.Value) { throw "dxc: $file has SHA-256 $have, the pin says $($p.Value)" }
    }

    $version = (& $exe --version 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) { throw "dxc: '$exe --version' exited $LASTEXITCODE" }
    if ($version -ne $e.versionOutput) { throw "dxc: --version printed '$version', the pin says '$($e.versionOutput)'" }

    if ($IsWindows) {
        $sig = Get-AuthenticodeSignature -LiteralPath $exe
        if ($sig.Status -ne 'Valid' -or $sig.SignerCertificate.Subject -notlike "*$($e.signer)*") {
            throw "dxc: Authenticode status $($sig.Status), signer '$($sig.SignerCertificate.Subject)'; expected a valid '$($e.signer)' signature"
        }
    }
    Write-Host "dxc $($e.version): OK  $exe" -ForegroundColor Green
}

function Skip-Pending([string]$key, $e) {
    $path = Under $e.install
    $where = if (Test-Path -LiteralPath $path) { "a copy exists at $path but is not verified (no pinned hash yet)" } else { "not present" }
    Write-Host "${key} $($e.version): SKIPPED, the download awaits user approval (AP-1); nothing fetched; $where" -ForegroundColor Yellow
}

$keys = @('dxc', 'vkxml', 'vkheaders')
if ($Only) { $keys = @($keys | Where-Object { $Only -contains $_ }) }
$failures = 0
foreach ($key in $keys) {
    $e = $toolchain.$key
    if ($e.managedBy -ne 'bootstrap') { continue }
    try {
        if ($e.approval -eq 'pending') { Skip-Pending $key $e; continue }
        switch ($key) {
            'dxc' { Install-Dxc $e }
            default { throw "${key}: approved, but bootstrap has no installer for it yet" }
        }
    }
    catch {
        $failures++
        Write-Host "bootstrap: ${key}: $($_.Exception.Message)" -ForegroundColor Red
    }
}

if ($failures -gt 0) { Write-Host "bootstrap: FAILED ($failures entr$(if ($failures -eq 1) { 'y' } else { 'ies' }))" -ForegroundColor Red; exit 1 }
Write-Host 'bootstrap: OK'
