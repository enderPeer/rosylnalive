<#
.SYNOPSIS
    R5: measures in-box WARP on the machine it runs on (meant for a one-off run on a windows-latest runner) and runs
    the WARP spike's cpu and gpu tiers: D1's selftests and E1's conformance harness on d3d12-warp.

.DESCRIPTION
    Read-only towards the machine: it changes no system setting, installs nothing and downloads nothing (the pinned
    DXC comes from the workflow's bootstrap step, as in build-test). It prints the OS, the processor count and the
    in-box WARP and D3D12 versions (and whether D3D12SDKLayers.dll, the debug layer, is present); builds
    spikes/d3d12/WarpSpike.fsproj and the CLI; runs `warp-info`, then the spike's cpu and gpu tiers (LRM_WARP_CPUS
    defaults to "all" here: a runner is not shared); renders every run log the gpu tier wrote with `lrm report`; and
    writes a Markdown summary to -Results/summary.md and, on GitHub Actions, to the step summary. The summary is
    written on every path, failures included.

    Exit codes:
      0  WARP accepted and both tiers passed;
      1  a build, `warp-info` or a tier failed, or the script itself failed: the probe could not measure, so this is
         no verdict on WARP;
      3  `warp-info` ran and rejected WARP for d3d12-warp (its own exit 3): the image's WARP is not viable.

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

# `warp-info`'s exit code for a rejected verdict (Main.info); any other non-zero code is a failure of the probe.
$Rejected = 3
$script:code = 1

function Write-Machine {
    $os = [Environment]::OSVersion.Version
    $cv = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion'
    # The registry's ProductName says "Windows 10" on Windows 11; Win32_OperatingSystem's caption names the product.
    $caption = try { (Get-CimInstance -ClassName Win32_OperatingSystem).Caption.Trim() } catch { "unknown ($($_.Exception.Message))" }
    $release = if ($cv.PSObject.Properties['DisplayVersion']) { $cv.DisplayVersion } else { '' }
    Say "| OS | $caption, $release, build $($os.Major).$($os.Minor).$($os.Build).$($cv.UBR) |"
    Say "| Logical processors | $([Environment]::ProcessorCount) |"
    Say "| LRM_WARP_CPUS | $($env:LRM_WARP_CPUS) |"
    foreach ($dll in 'd3d10warp.dll', 'd3d12.dll', 'D3D12Core.dll', 'dxgi.dll', 'D3D12SDKLayers.dll') {
        $p = Join-Path $env:SystemRoot "System32/$dll"
        $v = if (Test-Path $p) { (Get-Item $p).VersionInfo.FileVersion } else { 'missing' }
        Say "| $dll | $v |"
    }
}

function Invoke-Probe {
    $dll = Join-Path $repo 'artifacts/bin/WarpSpike/release/WarpSpike.dll'
    $cli = Join-Path $repo 'artifacts/bin/Lrm.Cli/release/Lrm.Cli.dll'
    if (-not $NoBuild) {
        $clock = [Diagnostics.Stopwatch]::StartNew()
        & dotnet build (Join-Path $repo 'spikes/d3d12/WarpSpike.fsproj') -c Release --nologo -v quiet | Out-Host
        if ($LASTEXITCODE -ne 0) { Say "Probe failed: the spike build failed (exit $LASTEXITCODE)."; return }
        & dotnet build (Join-Path $repo 'apps/Lrm.Cli/Lrm.Cli.fsproj') -c Release --nologo -v quiet | Out-Host
        if ($LASTEXITCODE -ne 0) { Say "Probe failed: the CLI build failed (exit $LASTEXITCODE)."; return }
        Say "Build: $([math]::Round($clock.Elapsed.TotalSeconds, 1)) s (the spike, its references and the CLI)."
        Say ""
    }

    Say '```'
    & dotnet $dll warp-info 2>&1 | ForEach-Object { Say "$_" }
    $info = $LASTEXITCODE
    Say '```'
    if ($info -eq $Rejected) {
        Say "WARP is rejected for d3d12-warp (warp-info exit $info): no d3d12-warp job on this image."
        $script:code = $Rejected
        return
    }
    if ($info -ne 0) {
        Say "Probe failed: warp-info exited $info (a crash, not a verdict). Fix the probe and run it again; this says nothing about WARP."
        return
    }

    # LRM_DATA, else the dataRoot of tools/bootstrap/toolchain.json (DataRootAndToolchain).
    $data = if ($env:LRM_DATA) { $env:LRM_DATA } else { (Get-Content -Raw (Join-Path $repo 'tools/bootstrap/toolchain.json') | ConvertFrom-Json).dataRoot }
    $runs = Join-Path $data 'runs'
    $before = @(if (Test-Path $runs) { Get-ChildItem $runs -Directory | ForEach-Object Name })
    $failed = $false
    foreach ($tier in 'cpu', 'gpu') {
        $env:LRM_TEST_TIER = $tier
        $clock = [Diagnostics.Stopwatch]::StartNew()
        & dotnet $dll --sequenced --no-spinner --results (Join-Path $Results $tier) 2>&1 | Tee-Object -FilePath (Join-Path $Results "$tier.log") | Out-Host
        $c = $LASTEXITCODE
        Say "- $tier tier: exit $c in $([math]::Round($clock.Elapsed.TotalSeconds, 1)) s"
        if ($c -ne 0) { $failed = $true }
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
        # Expecto colours its own lines; drop the escape sequences before matching the spike's lines.
        $plain = @(Get-Content $log | ForEach-Object { [regex]::Replace($_, '\x1b\[[0-9;?]*[A-Za-z]', '') })
        # Whether D1's backend ran WARP under the debug layer (D3D12Backend.DebugLayer, the adapter test's device).
        $debug = $plain | Where-Object { $_ -match "^D1.s backend on WARP: .*debug layer (.+)$" } | Select-Object -First 1
        $debugText = if ($debug -and $debug -match 'debug layer (.+)$') { $Matches[1] } else { 'not recorded (the adapter test did not run)' }
        Say "- debug layer on WARP's first device (D3D12Backend.DebugLayer): $debugText"
        Say ""
        Say '```'
        $plain | Where-Object { $_ -match '^(fp32 denormal probe|throughput:|d3d12-warp:|EnumWarpAdapter|  feature level|D1.s backend|versus )' } |
            ForEach-Object { Say $_ }
        Say '```'
    }
    $script:code = if ($failed) { 1 } else { 0 }
}

try {
    Say "# d3d12-warp probe (R5)"
    Say ""
    Say "| Fact | Value |"
    Say "| --- | --- |"
    Write-Machine
    Say ""
    Invoke-Probe
}
catch {
    Say ""
    Say "Probe failed: $($_.Exception.Message) ($($_.InvocationInfo.PositionMessage -replace '\s+', ' '))"
    $script:code = 1
}
finally {
    Say ""
    Say "Exit code: $($script:code) (0 accepted and passed; 1 the probe or a tier failed, no verdict on WARP; 3 WARP rejected)."
    $md = Join-Path $Results 'summary.md'
    $lines | Set-Content -Path $md -Encoding utf8
    if ($env:GITHUB_STEP_SUMMARY) { $lines | Out-File -FilePath $env:GITHUB_STEP_SUMMARY -Append -Encoding utf8 }
}
exit $script:code
