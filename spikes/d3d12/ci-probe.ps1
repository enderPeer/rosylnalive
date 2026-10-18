<#
.SYNOPSIS
    R5: measures in-box WARP on the machine it runs on (meant for a one-off run on a windows-latest runner) and runs
    the WARP spike's cpu and gpu tiers: D1's selftests and E1's conformance harness on d3d12-warp.

.DESCRIPTION
    Read-only towards the machine: it changes no system setting, installs nothing and downloads nothing (the pinned
    DXC comes from the workflow's bootstrap step, as in build-test). It prints the OS build, the processor count and
    the in-box WARP and D3D12 versions; builds spikes/d3d12/WarpSpike.fsproj and the CLI; runs `warp-info`, then the
    spike's cpu and gpu tiers (LRM_WARP_CPUS defaults to "all" here: a runner is not shared); renders every run log
    the gpu tier wrote with `lrm report`; and writes a Markdown summary to -Results/summary.md and, on GitHub Actions,
    to the step summary. The exit code is the gpu tier's (0 passed, 1 failed, 3 WARP rejected).

.EXAMPLE
    pwsh spikes/d3d12/ci-probe.ps1 -Results artifacts/warp-ci
#>
[CmdletBinding()]
param(
    [string]$Results = 'artifacts/warp-ci',
    [switch]$NoBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$Results = [IO.Path]::GetFullPath((Join-Path $repo $Results))
New-Item -ItemType Directory -Force -Path $Results | Out-Null
if (-not $env:LRM_WARP_CPUS) { $env:LRM_WARP_CPUS = 'all' }
$lines = [Collections.Generic.List[string]]::new()
function Say([string]$m) { Write-Host $m; $lines.Add($m) }

# The machine.
$cv = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion'
$os = [Environment]::OSVersion.Version
Say "# d3d12-warp probe (R5)"
Say ""
Say "| Fact | Value |"
Say "| --- | --- |"
Say "| OS | $($cv.ProductName), $($cv.EditionID) $($cv.DisplayVersion), build $($os.Major).$($os.Minor).$($os.Build).$($cv.UBR) |"
Say "| Logical processors | $([Environment]::ProcessorCount) |"
Say "| LRM_WARP_CPUS | $($env:LRM_WARP_CPUS) |"
foreach ($dll in 'd3d10warp.dll', 'd3d12.dll', 'D3D12Core.dll', 'dxgi.dll') {
    $p = Join-Path $env:SystemRoot "System32/$dll"
    $v = if (Test-Path $p) { (Get-Item $p).VersionInfo.FileVersion } else { 'missing' }
    Say "| $dll | $v |"
}
Say ""

$dll = Join-Path $repo 'artifacts/bin/WarpSpike/release/WarpSpike.dll'
$cli = Join-Path $repo 'artifacts/bin/Lrm.Cli/release/Lrm.Cli.dll'
if (-not $NoBuild) {
    $clock = [Diagnostics.Stopwatch]::StartNew()
    & dotnet build (Join-Path $repo 'spikes/d3d12/WarpSpike.fsproj') -c Release --nologo -v quiet
    if ($LASTEXITCODE -ne 0) { Say "spike build failed"; exit 1 }
    & dotnet build (Join-Path $repo 'apps/Lrm.Cli/Lrm.Cli.fsproj') -c Release --nologo -v quiet
    if ($LASTEXITCODE -ne 0) { Say "CLI build failed"; exit 1 }
    Say "Build: $([math]::Round($clock.Elapsed.TotalSeconds, 1)) s (the spike, its references and the CLI)."
    Say ""
}

Say '```'
& dotnet $dll warp-info 2>&1 | ForEach-Object { Say "$_" }
$info = $LASTEXITCODE
Say '```'
if ($info -ne 0) { Say "WARP is rejected for d3d12-warp (exit $info): no d3d12-warp job on this image."; $code = 3 }
else {
    # LRM_DATA, else the dataRoot of tools/bootstrap/toolchain.json (DataRootAndToolchain).
    $data = if ($env:LRM_DATA) { $env:LRM_DATA } else { (Get-Content -Raw (Join-Path $repo 'tools/bootstrap/toolchain.json') | ConvertFrom-Json).dataRoot }
    $runs = Join-Path $data 'runs'
    $before = @(if (Test-Path $runs) { Get-ChildItem $runs -Directory | ForEach-Object Name })
    $code = 0
    foreach ($tier in 'cpu', 'gpu') {
        $env:LRM_TEST_TIER = $tier
        $clock = [Diagnostics.Stopwatch]::StartNew()
        & dotnet $dll --sequenced --no-spinner --results (Join-Path $Results $tier) 2>&1 | Tee-Object -FilePath (Join-Path $Results "$tier.log") | Out-Host
        $c = $LASTEXITCODE
        Say "- $tier tier: exit $c in $([math]::Round($clock.Elapsed.TotalSeconds, 1)) s"
        if ($c -ne 0) { $code = 1 }
    }
    Remove-Item Env:LRM_TEST_TIER

    $new = @(if (Test-Path $runs) { Get-ChildItem $runs -Directory | Where-Object { $before -notcontains $_.Name } | Sort-Object Name })
    foreach ($r in $new) {
        & dotnet $cli report $r.FullName --out (Join-Path $Results "reports/$($r.Name)") | Out-Host
        Say "- report of run $($r.Name): reports/$($r.Name)/kernels.md"
    }

    $summary = Join-Path $Results 'gpu/WarpSpike.gpu.summary.json'
    if (Test-Path $summary) {
        $s = Get-Content -Raw $summary | ConvertFrom-Json
        foreach ($g in $s.groups) {
            Say "- gpu group $($g.name): $($g.passed) passed, $($g.failed) failed, $($g.errored) errored, $($g.ignored) ignored, $([math]::Round($g.durationMs / 1000, 1)) s"
        }
    }
    $log = Join-Path $Results 'gpu.log'
    if (Test-Path $log) {
        Say ""
        Say '```'
        # Expecto colours its own lines; drop the escape sequences before matching the spike's lines.
        Get-Content $log | ForEach-Object { [regex]::Replace($_, '\x1b\[[0-9;?]*[A-Za-z]', '') } |
            Where-Object { $_ -match '^(fp32 denormal probe|throughput:|d3d12-warp:|EnumWarpAdapter|  feature level|D1.s backend|versus )' } |
            ForEach-Object { Say $_ }
        Say '```'
    }
}

$md = Join-Path $Results 'summary.md'
$lines | Set-Content -Path $md -Encoding utf8
if ($env:GITHUB_STEP_SUMMARY) { $lines | Out-File -FilePath $env:GITHUB_STEP_SUMMARY -Append -Encoding utf8 }
exit $code
