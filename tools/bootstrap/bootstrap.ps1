<#
.SYNOPSIS
    Verifies or installs the pinned toolchain under LRM_DATA (DataRootAndToolchain contract).

.DESCRIPTION
    Reads tools/bootstrap/toolchain.json and handles every entry marked managedBy=bootstrap, according to its kind:

      dxc-nupkg  (dxc) Verified in place: SHA-256 of each binary, `dxc --version`, Authenticode signer. When it is
                 not installed, it is extracted from the pinned nupkg in LRM_DATA/downloads. The nupkg is downloaded
                 only with -AllowDownload and is checked against its pinned size and SHA-512 before use.
      file       (vkxml) One file at its install path, checked against its pinned size and SHA-256 on every run. A
                 missing file is downloaded with -AllowDownload to a .partial file, checked, then moved into place.
      zip        (vkheaders) An archive kept in LRM_DATA/downloads and checked against its pinned size and SHA-256.
                 It is extracted into a staging folder (only the `extract` patterns, without the archive's single
                 top folder when stripTopFolder is set), which is then moved into place. The install folder records
                 the archive hash and every file's SHA-256 in .lrm-install.json, and every run verifies the files
                 against that record and against any `files` pins in toolchain.json.

    Approval gates every entry. Only an entry whose `approval` says "approved" is verified or installed; a pending or
    declined entry is skipped with a message and never downloaded. An approved file or zip entry also needs its url,
    size and sha256 pinned. -ComputePins (with -AllowDownload) downloads such an entry once, prints the size and
    SHA-256 to pin, and installs nothing; the integrator checks them against the upstream release and commits them.

    The exit code is 0 when every approved entry verifies, otherwise 1. Running it again downloads nothing.

.PARAMETER DataRoot
    Overrides LRM_DATA (default: the dataRoot in toolchain.json).

.PARAMETER AllowDownload
    Download a missing approved package from its pinned URL. Use on CI runners, or once after an approval.

.PARAMETER Only
    Restrict to some entries, e.g. -Only dxc or -Only vkxml,vkheaders.

.PARAMETER ComputePins
    For approved file and zip entries whose size or sha256 is not pinned yet: download once (needs -AllowDownload),
    print the values to pin, and install nothing.

.EXAMPLE
    pwsh tools/bootstrap/bootstrap.ps1
    pwsh tools/bootstrap/bootstrap.ps1 -AllowDownload                               # CI
    pwsh tools/bootstrap/bootstrap.ps1 -Only vkxml -ComputePins -AllowDownload      # after approval, before pinning
#>
[CmdletBinding()]
param(
    [string]$DataRoot,
    [switch]$AllowDownload,
    [string[]]$Only,
    [switch]$ComputePins
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

# An optional field of a toolchain entry, or $null (StrictMode forbids reading a missing property).
function Get-Field($e, [string]$name) { if ($e.PSObject.Properties[$name]) { $e.$name } else { $null } }

function Get-Sha256([string]$path) { (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() }

function Get-Sha512Base64([string]$path) {
    $stream = [IO.File]::OpenRead($path)
    try { [Convert]::ToBase64String([Security.Cryptography.SHA512]::HashData($stream)) } finally { $stream.Dispose() }
}

function Get-RelativePath([string]$root, [string]$path) { [IO.Path]::GetRelativePath($root, $path).Replace('\', '/') }

function Test-Approved($e) {
    $a = [string](Get-Field $e 'approval')
    ($a -match '\bapproved\b') -and -not ($a -match '\b(pending|declined)\b')
}

# Downloads $e.url to "$dest.partial", runs $verify on it, then moves it to $dest. Needs -AllowDownload.
function Save-Download([string]$key, $e, [string]$dest, [scriptblock]$verify) {
    if (-not $AllowDownload) {
        throw ("$key $($e.version) is not at $dest. Run bootstrap with -AllowDownload (only for an approved entry), " +
            "or put the pinned file there.")
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
    $partial = "$dest.partial"
    Write-Host "${key}: downloading $($e.url) ($($e.size) bytes)"
    try {
        Invoke-WebRequest -Uri $e.url -OutFile $partial -UseBasicParsing
        & $verify $partial
    }
    catch {
        Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
        throw
    }
    Move-Item -LiteralPath $partial -Destination $dest -Force
}

# Extracts the entries of $archive that match $patterns into $staging (a fresh folder). With $stripTop, the archive
# must have one top folder, which is dropped from every path (GitHub source archives).
function Expand-Pinned([string]$key, [string]$archive, [object[]]$patterns, [string]$staging, [bool]$stripTop) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    if (Test-Path -LiteralPath $staging) { Remove-Item -LiteralPath $staging -Recurse -Force }
    $root = [IO.Path]::GetFullPath($staging) + [IO.Path]::DirectorySeparatorChar
    $zip = [IO.Compression.ZipFile]::OpenRead($archive)
    try {
        $prefix = ''
        if ($stripTop) {
            $tops = @($zip.Entries | ForEach-Object { $_.FullName.Replace('\', '/').Split('/')[0] } | Sort-Object -Unique)
            if ($tops.Count -ne 1) { throw "${key}: expected one top folder in $archive, found $($tops.Count)" }
            $prefix = $tops[0] + '/'
        }
        $count = 0
        foreach ($entry in $zip.Entries) {
            $name = $entry.FullName.Replace('\', '/')
            if ($name.EndsWith('/')) { continue }
            if ($prefix) {
                if (-not $name.StartsWith($prefix)) { continue }
                $name = $name.Substring($prefix.Length)
            }
            if (-not ($patterns | Where-Object { $name -like $_ })) { continue }
            $target = [IO.Path]::GetFullPath((Join-Path $staging $name))
            if (-not $target.StartsWith($root)) { throw "${key}: unsafe path in archive: $($entry.FullName)" }
            New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $false)
            $count++
        }
        if ($count -eq 0) { throw "${key}: no entry of $archive matches $($patterns -join ', ')" }
    }
    finally { $zip.Dispose() }
}

function Test-Sha512Pin([string]$key, $e, [string]$path) {
    $len = (Get-Item -LiteralPath $path).Length
    if ($len -ne [long]$e.size) { throw "${key}: $path is $len bytes, the pin says $($e.size)" }
    $hash = Get-Sha512Base64 $path
    if ($hash -ne $e.sha512Base64) { throw "${key}: $path has SHA-512 $hash, the pin says $($e.sha512Base64)" }
}

function Test-Sha256Pin([string]$key, $e, [string]$path) {
    $len = (Get-Item -LiteralPath $path).Length
    if ($len -ne [long]$e.size) { throw "${key}: $path is $len bytes, the pin says $($e.size)" }
    $hash = Get-Sha256 $path
    if ($hash -ne ([string]$e.sha256).ToLowerInvariant()) { throw "${key}: $path has SHA-256 $hash, the pin says $($e.sha256)" }
}

function Assert-Pinned([string]$key, $e) {
    $missing = @(foreach ($k in 'url', 'size', 'sha256') { if (-not (Get-Field $e $k)) { $k } })
    if ($missing) {
        throw ("${key}: approved, but $($missing -join ', ') not pinned in toolchain.json. Run " +
            "bootstrap -Only $key -ComputePins -AllowDownload once, check the printed values against the upstream " +
            "release, and pin them.")
    }
}

function Install-Dxc([string]$key, $e) {
    $install = Under $e.install
    $exe = Under $e.exe
    if (-not (Test-Path -LiteralPath $exe)) {
        if (Test-Path -LiteralPath $install) {
            throw "${key}: $install exists but $($e.exe) is missing; remove that folder by hand and run bootstrap again"
        }
        $nupkg = Under $e.download
        if (-not (Test-Path -LiteralPath $nupkg)) {
            Save-Download $key $e $nupkg { param($p) Test-Sha512Pin $key $e $p }.GetNewClosure()
        }
        Test-Sha512Pin $key $e $nupkg
        Write-Host "${key}: extracting $nupkg"
        $staging = "$install.staging-$PID"
        Expand-Pinned $key $nupkg @($e.extract) $staging $false
        New-Item -ItemType Directory -Force -Path (Split-Path $install) | Out-Null
        Move-Item -LiteralPath $staging -Destination $install
    }

    foreach ($p in $e.sha256.PSObject.Properties) {
        $file = Join-Path $install $p.Name
        if (-not (Test-Path -LiteralPath $file)) { throw "${key}: $file is missing" }
        $have = Get-Sha256 $file
        if ($have -ne $p.Value) { throw "${key}: $file has SHA-256 $have, the pin says $($p.Value)" }
    }

    $version = (& $exe --version 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) { throw "${key}: '$exe --version' exited $LASTEXITCODE" }
    if ($version -ne $e.versionOutput) { throw "${key}: --version printed '$version', the pin says '$($e.versionOutput)'" }

    if ($IsWindows) {
        $sig = Get-AuthenticodeSignature -LiteralPath $exe
        if ($sig.Status -ne 'Valid' -or $sig.SignerCertificate.Subject -notlike "*$($e.signer)*") {
            throw "${key}: Authenticode status $($sig.Status), signer '$($sig.SignerCertificate.Subject)'; expected a valid '$($e.signer)' signature"
        }
    }
    Write-Host "$key $($e.version): OK  $exe" -ForegroundColor Green
}

function Install-File([string]$key, $e) {
    Assert-Pinned $key $e
    $dest = Under $e.install
    if (-not (Test-Path -LiteralPath $dest)) {
        Save-Download $key $e $dest { param($p) Test-Sha256Pin $key $e $p }.GetNewClosure()
    }
    Test-Sha256Pin $key $e $dest
    Write-Host "$key $($e.version): OK  $dest" -ForegroundColor Green
}

function Install-Zip([string]$key, $e) {
    Assert-Pinned $key $e
    $install = Under $e.install
    $stampPath = Join-Path $install '.lrm-install.json'
    if (-not (Test-Path -LiteralPath $install)) {
        $archive = Under $e.download
        if (-not (Test-Path -LiteralPath $archive)) {
            Save-Download $key $e $archive { param($p) Test-Sha256Pin $key $e $p }.GetNewClosure()
        }
        Test-Sha256Pin $key $e $archive
        Write-Host "${key}: extracting $archive"
        $staging = "$install.staging-$PID"
        Expand-Pinned $key $archive @($e.extract) $staging ([bool](Get-Field $e 'stripTopFolder'))
        [string[]]$names = @(Get-ChildItem -LiteralPath $staging -Recurse -File | ForEach-Object { Get-RelativePath $staging $_.FullName })
        [Array]::Sort($names, [StringComparer]::Ordinal)
        $files = [ordered]@{}
        foreach ($n in $names) { $files[$n] = Get-Sha256 (Join-Path $staging $n) }
        $stamp = [ordered]@{
            schema = 'lrm.toolchain-install/1'; entry = $key; version = $e.version; url = $e.url
            archiveSha256 = ([string]$e.sha256).ToLowerInvariant(); files = $files
        }
        [IO.File]::WriteAllText((Join-Path $staging '.lrm-install.json'), ($stamp | ConvertTo-Json -Depth 4) + "`n", [Text.UTF8Encoding]::new($false))
        New-Item -ItemType Directory -Force -Path (Split-Path $install) | Out-Null
        Move-Item -LiteralPath $staging -Destination $install
    }
    elseif (-not (Test-Path -LiteralPath $stampPath)) {
        throw "${key}: $install exists but has no .lrm-install.json; remove that folder by hand and run bootstrap again"
    }

    $stamp = Get-Content -Raw -LiteralPath $stampPath | ConvertFrom-Json
    if ($stamp.archiveSha256 -ne ([string]$e.sha256).ToLowerInvariant() -or $stamp.version -ne $e.version) {
        throw ("${key}: $install came from archive $($stamp.archiveSha256) ($($stamp.version)), the pin says " +
            "$($e.sha256) ($($e.version)); remove that folder by hand and run bootstrap again")
    }
    $recorded = @($stamp.files.PSObject.Properties)
    foreach ($p in $recorded) {
        $file = Join-Path $install $p.Name
        if (-not (Test-Path -LiteralPath $file)) { throw "${key}: $file is missing" }
        $have = Get-Sha256 $file
        if ($have -ne $p.Value) { throw "${key}: $file has SHA-256 $have, but the archive installed $($p.Value)" }
    }
    $extra = @(Get-ChildItem -LiteralPath $install -Recurse -File | ForEach-Object { Get-RelativePath $install $_.FullName } |
            Where-Object { $_ -ne '.lrm-install.json' -and -not $stamp.files.PSObject.Properties[$_] })
    if ($extra) { throw "${key}: $install holds files the archive did not install: $($extra -join ', ')" }
    $pins = Get-Field $e 'files'
    if ($pins) {
        foreach ($p in $pins.PSObject.Properties) {
            $file = Join-Path $install $p.Name
            if (-not (Test-Path -LiteralPath $file)) { throw "${key}: pinned file $file is missing" }
            $have = Get-Sha256 $file
            if ($have -ne $p.Value) { throw "${key}: $file has SHA-256 $have, the pin says $($p.Value)" }
        }
    }
    Write-Host "$key $($e.version): OK  $install ($($recorded.Count) files)" -ForegroundColor Green
}

function Show-Pins([string]$key, $e) {
    $kind = Get-Field $e 'kind'
    if ($kind -notin @('file', 'zip')) { throw "${key}: -ComputePins handles file and zip entries, not '$kind'" }
    if ((Get-Field $e 'size') -and (Get-Field $e 'sha256')) {
        Write-Host "${key}: size and sha256 are already pinned; nothing to compute"
        return
    }
    if (-not (Get-Field $e 'url')) { throw "${key}: no url in toolchain.json" }
    if (-not $AllowDownload) { throw "${key}: -ComputePins downloads the file once; add -AllowDownload" }
    $name = [IO.Path]::GetFileName(([Uri]$e.url).AbsolutePath)
    $dest = Under "downloads/pins/$key-$($e.version)-$name.unpinned"
    New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
    Write-Host "${key}: downloading $($e.url) to compute its pins (nothing is installed)"
    Invoke-WebRequest -Uri $e.url -OutFile $dest -UseBasicParsing
    $len = (Get-Item -LiteralPath $dest).Length
    $pinnedSize = Get-Field $e 'size'
    if ($pinnedSize -and $len -ne [long]$pinnedSize) { throw "${key}: downloaded $len bytes, but the pinned size is $pinnedSize" }
    $hash = Get-Sha256 $dest
    Write-Host "${key}: check these against the upstream release, then pin them in toolchain.json:" -ForegroundColor Yellow
    Write-Host "  `"size`": $len,"
    Write-Host "  `"sha256`": `"$hash`""
    Write-Host "  (downloaded copy: $dest)"
}

function Skip-Unapproved([string]$key, $e) {
    $approval = [string](Get-Field $e 'approval')
    $why = if ($approval -match '\bpending\b') { 'the download awaits user approval' } else { "it is not approved ('$approval')" }
    $install = Get-Field $e 'install'
    $where = if ($install -and (Test-Path -LiteralPath (Under $install))) { "a copy exists at $(Under $install) but is not verified" } else { 'not present' }
    Write-Host "${key} $($e.version): SKIPPED, $why; nothing fetched; $where" -ForegroundColor Yellow
}

$keys = @($toolchain.PSObject.Properties | Where-Object { $_.Value -is [pscustomobject] -and (Get-Field $_.Value 'managedBy') -eq 'bootstrap' } |
        ForEach-Object Name)
if ($Only) {
    # `pwsh -File` passes "a,b" as one string; accept both forms.
    $Only = @($Only | ForEach-Object { $_ -split ',' } | ForEach-Object Trim | Where-Object { $_ })
    $unknown = @($Only | Where-Object { $keys -notcontains $_ })
    if ($unknown) { Write-Host "bootstrap: not a bootstrap entry in toolchain.json: $($unknown -join ', ') (entries: $($keys -join ', '))" -ForegroundColor Red; exit 1 }
    $keys = @($keys | Where-Object { $Only -contains $_ })
}

$failures = 0
foreach ($key in $keys) {
    $e = $toolchain.$key
    try {
        if (-not (Test-Approved $e)) { Skip-Unapproved $key $e; continue }
        if ($ComputePins) { Show-Pins $key $e; continue }
        $kind = Get-Field $e 'kind'
        switch ($kind) {
            'dxc-nupkg' { Install-Dxc $key $e }
            'file' { Install-File $key $e }
            'zip' { Install-Zip $key $e }
            default { throw "${key}: unknown kind '$kind' in toolchain.json (expected dxc-nupkg, file or zip)" }
        }
    }
    catch {
        $failures++
        $message = $_.Exception.Message
        if ($message.StartsWith("${key}: ")) { $message = $message.Substring($key.Length + 2) }
        Write-Host "bootstrap: ${key}: $message" -ForegroundColor Red
    }
}

if ($failures -gt 0) { Write-Host "bootstrap: FAILED ($failures entr$(if ($failures -eq 1) { 'y' } else { 'ies' }))" -ForegroundColor Red; exit 1 }
Write-Host 'bootstrap: OK'
