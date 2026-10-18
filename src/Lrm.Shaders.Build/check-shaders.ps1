<#
.SYNOPSIS
    Shader checks for CI and local runs (ShaderBuildAndSourceConvention contract, owner R4).

.DESCRIPTION
    Run after the Release build; it uses the built Lrm.Shaders.Build and the build's shader outputs. It checks what the
    build itself cannot show:
      1. Diff-clean codegen: Kernels.g.fs and lrm_args.g.hlsli equal a fresh generation from catalog.json.
      2. Reproducibility: a clean compile into an empty folder gives a byte-identical shaders.manifest.json, so both
         ShaderSetHash values reproduce.
      3. Incrementality: a no-change rebuild of Lrm.Shaders skips LrmCompileShadersCore, so dxc never starts.
      4. Pinning: the compile step refuses a missing DXC and a DXC whose version line differs (the Windows Kits dxc
         when it is installed, otherwise another executable).
    Prints a Markdown summary (DXC, ShaderSetHash per target, kernels, missing sources) and appends it to
    $GITHUB_STEP_SUMMARY on CI. Exit code 0 when every check holds, otherwise 1.

.EXAMPLE
    pwsh src/Lrm.Shaders.Build/check-shaders.ps1
#>
[CmdletBinding()]
param(
    [ValidateSet('Release', 'Debug')]
    [string]$Configuration = 'Release',
    [string]$DataRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$toolchainPath = Join-Path $repo 'tools/bootstrap/toolchain.json'
$toolchain = Get-Content -Raw -LiteralPath $toolchainPath | ConvertFrom-Json
if (-not $DataRoot) { $DataRoot = if ($env:LRM_DATA) { $env:LRM_DATA } else { $toolchain.dataRoot } }
$dxc = Join-Path $DataRoot $toolchain.dxc.exe
$versionLine = $toolchain.dxc.versionOutput
$config = $Configuration.ToLowerInvariant()
$tool = Join-Path $repo "artifacts/bin/Lrm.Shaders.Build/$config/Lrm.Shaders.Build.dll"
$built = Join-Path $repo "artifacts/obj/Lrm.Shaders/$config/shaders"
$manifestPath = Join-Path $built 'shaders.manifest.json'

$failures = [System.Collections.Generic.List[string]]::new()
function Check([bool]$ok, [string]$what) {
    if ($ok) { Write-Host "  ok    $what" } else { Write-Host "  FAIL  $what" -ForegroundColor Red; $failures.Add($what) }
}
function Invoke-Tool([string[]]$arguments) {
    $out = & dotnet $tool @arguments 2>&1 | Out-String
    [pscustomobject]@{ Code = $LASTEXITCODE; Output = $out.Trim() }
}

foreach ($p in @($tool, $manifestPath, $dxc)) {
    if (-not (Test-Path -LiteralPath $p)) {
        Write-Host "check-shaders: $p is missing (bootstrap, then build Lrm.slnx -c $Configuration first)" -ForegroundColor Red
        exit 1
    }
}
Write-Host "check-shaders: $tool"

# 1. Diff-clean codegen.
$r = Invoke-Tool @('gen-catalog', '--repo', $repo, '--check')
if ($r.Output) { Write-Host $r.Output }
Check ($r.Code -eq 0) 'generated catalog files are current (gen-catalog --check)'

# 2. A clean compile reproduces the manifest byte for byte.
$fresh = Join-Path ([IO.Path]::GetTempPath()) ("lrm-shaders-" + [guid]::NewGuid().ToString('n'))
try {
    $r = Invoke-Tool @('compile', '--repo', $repo, '--out', $fresh, '--dxc', $dxc, '--dxc-version', $versionLine, '--toolchain', $toolchainPath)
    if ($r.Output) { Write-Host $r.Output }
    Check ($r.Code -eq 0) 'a clean compile into an empty folder succeeds'
    if ($r.Code -eq 0) {
        $a = [IO.File]::ReadAllBytes($manifestPath)
        $b = [IO.File]::ReadAllBytes((Join-Path $fresh 'shaders.manifest.json'))
        Check ([Linq.Enumerable]::SequenceEqual($a, $b)) 'the clean manifest equals the build manifest byte for byte (ShaderSetHash reproduces)'
    }
}
finally {
    Remove-Item -LiteralPath $fresh -Recurse -Force -ErrorAction SilentlyContinue
}

# 3. A no-change rebuild skips the shader target.
$log = & dotnet build (Join-Path $repo 'src/Lrm.Shaders/Lrm.Shaders.fsproj') -c $Configuration --no-restore --nologo -v:d 2>&1 | Out-String
Check ($LASTEXITCODE -eq 0) 'the no-change rebuild succeeds'
Check ($log -match 'Skipping target "LrmCompileShadersCore" because all output files are up-to-date') 'a no-change rebuild skips LrmCompileShadersCore (dxc never starts)'

# 4. Only the pinned DXC.
$scratch = Join-Path ([IO.Path]::GetTempPath()) ("lrm-shaders-neg-" + [guid]::NewGuid().ToString('n'))
try {
    $r = Invoke-Tool @('compile', '--repo', $repo, '--out', $scratch, '--dxc', (Join-Path $scratch 'no-such-dxc.exe'), '--dxc-version', $versionLine)
    Check ($r.Code -ne 0 -and $r.Output -match 'pinned DXC is missing') 'a missing DXC fails'

    $kits = @(Get-ChildItem -Path "${env:ProgramFiles(x86)}\Windows Kits\10\bin\*\x64\dxc.exe" -ErrorAction SilentlyContinue | Select-Object -First 1)
    $other = if ($kits) { $kits[0].FullName } else { (Get-Command dotnet).Source }
    $r = Invoke-Tool @('compile', '--repo', $repo, '--out', $scratch, '--dxc', $other, '--dxc-version', $versionLine)
    Check ($r.Code -ne 0 -and $r.Output -match 'is not the pinned DXC') "a DXC with another version line fails ($other)"
}
finally {
    Remove-Item -LiteralPath $scratch -Recurse -Force -ErrorAction SilentlyContinue
}

# Summary.
$m = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
$md = [System.Text.StringBuilder]::new()
[void]$md.AppendLine('### Shaders (LrmCompileShaders)')
[void]$md.AppendLine('')
[void]$md.AppendLine("DXC ``$($m.dxcVersion)``; catalog ``$($m.catalogHash)``.")
[void]$md.AppendLine('')
[void]$md.AppendLine('| Target | ShaderSetHash | Kernels |')
[void]$md.AppendLine('| --- | --- | ---: |')
foreach ($t in @('dxil', 'spirv')) {
    $e = $m.targets.$t
    [void]$md.AppendLine("| $t | ``$($e.shaderSetHash)`` | $(@($e.kernels).Count) |")
}
[void]$md.AppendLine('')
$missing = @($m.missing)
[void]$md.AppendLine("Missing sources ($($missing.Count)): $(if ($missing) { $missing -join ', ' } else { 'none' }).")
$verdict = if ($failures.Count -gt 0) { "FAILED ($($failures.Count))" } else { 'PASSED' }
[void]$md.AppendLine('')
[void]$md.AppendLine("Checks: **$verdict**")
$text = $md.ToString().Replace("`r`n", "`n")
Write-Host ''
Write-Host $text
if ($env:GITHUB_STEP_SUMMARY) { $text | Out-File -FilePath $env:GITHUB_STEP_SUMMARY -Append -Encoding utf8 }

if ($failures.Count -gt 0) { Write-Host "check-shaders: FAILED ($($failures.Count) check(s))" -ForegroundColor Red; exit 1 }
Write-Host 'check-shaders: OK'
