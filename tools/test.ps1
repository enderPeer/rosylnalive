<#
.SYNOPSIS
    Runs one test tier across every test exe in docs/project-graph.json (TestTiers contract).

.DESCRIPTION
    Builds the solution (unless -NoBuild), then runs each test exe as `dotnet <dll>` with LRM_TEST_TIER set, one
    after the other. Each exe holds the tier's mutexes itself (Lrm.Testing), so parallel agents queue on the GPU
    instead of colliding. Per-exe JSON summaries and JUnit files go to -Results; a Markdown summary for PR bodies is
    printed at the end and written to <Results>/summary.md.

    The run fails when any exe fails, when the tier ran no tests at all (a missing tier), or when a required backend
    ran no tests. Required backends come from LRM_REQUIRED_BACKENDS (for example "d3d12,vulkan"):
      - gpu tier: always enforced.
      - model tier: enforced with -RequireBackends (GPU model gates). Without it, for example M2b's CPU-only
        gate1-cpu run, a required backend that ran no model tests is listed in the summary as a note, not a failure.
    The bench tier is informational: it never fails the run.

.EXAMPLE
    pwsh tools/test.ps1 -Tier cpu
    pwsh tools/test.ps1 -Tier gpu -Project Lrm.Gpu.Vulkan.Tests
    pwsh tools/test.ps1 -Tier cpu -NoBuild -FailOnFocusedTests    # what CI runs
    pwsh tools/test.ps1 -Tier model -RequireBackends              # a GPU model gate
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet('cpu', 'gpu', 'model', 'bench')]
    [string]$Tier,
    [ValidateSet('Release', 'Debug')]
    [string]$Configuration = 'Release',
    [switch]$NoBuild,
    # Run only these test exes (project names from docs/project-graph.json).
    [string[]]$Project,
    [string]$Results,
    # Fail when a test is focused (ftestCase, ftestList); CI sets this.
    [switch]$FailOnFocusedTests,
    # Model tier: fail when a backend in LRM_REQUIRED_BACKENDS ran no model tests (always on for the gpu tier).
    [switch]$RequireBackends,
    # Extra arguments for every Expecto exe, e.g. -ExpectoArgs '--filter','cpu.selftest.locks'
    [string[]]$ExpectoArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $Results) { $Results = Join-Path $repo "artifacts/test-results/$Tier" }
$Results = [IO.Path]::GetFullPath($Results)
$isBench = $Tier -eq 'bench'

function Write-Fail([string]$m) { Write-Host "test.ps1: $m" -ForegroundColor Red }

$required = @(if ($env:LRM_REQUIRED_BACKENDS) { $env:LRM_REQUIRED_BACKENDS -split '[,; ]+' | Where-Object { $_ } | ForEach-Object { $_.ToLowerInvariant() } })
$unknownBackends = @($required | Where-Object { @('d3d12', 'vulkan') -notcontains $_ })
if ($unknownBackends) { Write-Fail "LRM_REQUIRED_BACKENDS: unknown backend(s) $($unknownBackends -join ', '); expected d3d12, vulkan"; exit 1 }
if ($RequireBackends -and -not $required) { Write-Fail '-RequireBackends needs LRM_REQUIRED_BACKENDS (for example d3d12,vulkan)'; exit 1 }
if ($RequireBackends -and $Tier -notin @('gpu', 'model')) { Write-Fail "-RequireBackends applies to the gpu and model tiers, not $Tier"; exit 1 }

if (-not $NoBuild) {
    Write-Host "test.ps1: building Lrm.slnx ($Configuration)"
    & dotnet build (Join-Path $repo 'Lrm.slnx') -c $Configuration --nologo -v quiet
    if ($LASTEXITCODE -ne 0) { Write-Fail "build failed"; exit 1 }
}

$graph = Get-Content -Raw (Join-Path $repo 'docs/project-graph.json') | ConvertFrom-Json
$tests = @($graph.projects | Where-Object kind -eq 'test' | Sort-Object name)
if ($Project) {
    # `pwsh -File` passes "a,b" as one string; accept both forms.
    $Project = @($Project | ForEach-Object { $_ -split ',' } | ForEach-Object Trim | Where-Object { $_ })
    $unknown = @($Project | Where-Object { $tests.name -notcontains $_ })
    if ($unknown) { Write-Fail "not a test project in docs/project-graph.json: $($unknown -join ', ')"; exit 1 }
    $tests = @($tests | Where-Object { $Project -contains $_.name })
}

New-Item -ItemType Directory -Force -Path $Results | Out-Null
$configDir = $Configuration.ToLowerInvariant()
$extra = @($ExpectoArgs)
if ($FailOnFocusedTests) { $extra += '--fail-on-focused-tests' }
if ($env:CI -eq 'true') { $extra += '--no-spinner' }

$env:LRM_TEST_TIER = $Tier
$env:LRM_REPO_ROOT = $repo
# DataRootAndToolchain: the default data root has one source, toolchain.json's dataRoot.
if (-not $env:LRM_DATA) { $env:LRM_DATA = (Get-Content -Raw (Join-Path $repo 'tools/bootstrap/toolchain.json') | ConvertFrom-Json).dataRoot }

$rows = @()
$failed = @()
$started = Get-Date
foreach ($t in $tests) {
    $dll = Join-Path $repo "artifacts/bin/$($t.name)/$configDir/$($t.name).dll"
    if (-not (Test-Path -LiteralPath $dll)) { Write-Fail "$($t.name): $dll is missing (build first)"; $failed += $t.name; continue }
    $summaryPath = Join-Path $Results "$($t.name).$Tier.summary.json"
    Remove-Item -LiteralPath $summaryPath -ErrorAction SilentlyContinue
    Get-ChildItem -LiteralPath $Results -Filter "$($t.name).*.junit.xml" -ErrorAction SilentlyContinue | Remove-Item

    $clock = [Diagnostics.Stopwatch]::StartNew()
    & dotnet $dll --results $Results @extra
    $code = $LASTEXITCODE
    $clock.Stop()

    $s = if (Test-Path -LiteralPath $summaryPath) { Get-Content -Raw $summaryPath | ConvertFrom-Json } else { $null }
    if (-not $s) { Write-Fail "$($t.name): exited $code without writing a summary"; $failed += $t.name; continue }
    $groups = @($s.groups)
    $sum = { param($k) ($groups | ForEach-Object { $_.$k } | Measure-Object -Sum).Sum }
    $rows += [pscustomobject]@{
        Exe      = $t.name
        Groups   = (@($groups | ForEach-Object name) -join ' ')
        Passed   = [int](& $sum 'passed')
        Failed   = [int](& $sum 'failed')
        Errored  = [int](& $sum 'errored')
        Ignored  = [int](& $sum 'ignored')
        Exit     = $code
        Seconds  = [math]::Round($clock.Elapsed.TotalSeconds, 1)
        Backends = @($groups | Where-Object { $_.backend -and ($_.passed + $_.failed + $_.errored) -gt 0 } | ForEach-Object backend)
        Devices  = @($s.devices)
    }
    if ($code -ne 0) { $failed += $t.name }
}

# ---- checks ---------------------------------------------------------------------------------------

$problems = @()
$notes = @()
if ($failed) { $problems += "failed: $($failed -join ', ')" }
$executed = ($rows | ForEach-Object { $_.Passed + $_.Failed + $_.Errored } | Measure-Object -Sum).Sum
if (-not $executed) { $problems += "missing tier: no $Tier tests ran in any of $($tests.Count) test exe(s)" }
if ($required -and $Tier -in @('gpu', 'model')) {
    $ran = @($rows | ForEach-Object { $_.Backends })
    $enforced = $Tier -eq 'gpu' -or $RequireBackends
    foreach ($b in $required) {
        if ($ran -contains $b) { continue }
        if ($enforced) { $problems += "required backend '$b' (LRM_REQUIRED_BACKENDS) ran no $Tier tests" }
        else { $notes += "note: required backend '$b' (LRM_REQUIRED_BACKENDS) ran no $Tier tests; not enforced without -RequireBackends" }
    }
}

# ---- summary --------------------------------------------------------------------------------------

$head = (& git -C $repo rev-parse --short HEAD 2>$null)
$dirty = if (& git -C $repo status --porcelain 2>$null) { ' (uncommitted changes)' } else { '' }
$devices = @($rows | ForEach-Object { $_.Devices } | Sort-Object backend, device, driver, shaderSetHash -Unique)
$md = [System.Text.StringBuilder]::new()
[void]$md.AppendLine("### tools/test.ps1 -Tier $Tier")
[void]$md.AppendLine('')
[void]$md.AppendLine("Commit ``$head``$dirty, $Configuration, $($started.ToUniversalTime().ToString('yyyy-MM-dd HH:mm')) UTC, machine ``$([Environment]::MachineName)``.")
if ($devices) {
    foreach ($d in $devices) { [void]$md.AppendLine("- Device: $($d.backend) $($d.device), driver $($d.driver), ShaderSetHash $(if ($d.shaderSetHash) { $d.shaderSetHash } else { 'n/a' })") }
}
[void]$md.AppendLine('')
[void]$md.AppendLine('| Exe | Groups | Passed | Failed | Errored | Ignored | Exit | Seconds |')
[void]$md.AppendLine('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |')
foreach ($r in $rows) {
    $g = if ($r.Groups) { $r.Groups } else { '-' }
    [void]$md.AppendLine("| $($r.Exe) | $g | $($r.Passed) | $($r.Failed) | $($r.Errored) | $($r.Ignored) | $($r.Exit) | $($r.Seconds) |")
}
$tot = { param($k) ($rows | ForEach-Object { $_.$k } | Measure-Object -Sum).Sum }
[void]$md.AppendLine("| **total** | | **$(& $tot 'Passed')** | **$(& $tot 'Failed')** | **$(& $tot 'Errored')** | **$(& $tot 'Ignored')** | | |")
[void]$md.AppendLine('')
$verdict = if ($problems -and -not $isBench) { 'FAILED' } elseif ($problems) { 'informational (bench never gates)' } else { 'PASSED' }
[void]$md.AppendLine("Result: **$verdict**")
foreach ($p in $problems) { [void]$md.AppendLine("- $p") }
foreach ($n in $notes) { [void]$md.AppendLine("- $n") }

$text = $md.ToString().Replace("`r`n", "`n")
[IO.File]::WriteAllText((Join-Path $Results 'summary.md'), $text, [Text.UTF8Encoding]::new($false))
Write-Host ''
Write-Host $text

if ($problems -and -not $isBench) {
    $problems | ForEach-Object { Write-Fail $_ }
    exit 1
}
exit 0
