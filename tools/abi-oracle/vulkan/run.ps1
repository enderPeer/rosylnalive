<#
.SYNOPSIS
    The Vulkan ABI oracle (VulkanInteropApi contract): checks the runtime layout of Lrm.Gpu.Vulkan.Interop with cl.exe
    against Vulkan-Headers 1.4.365, and cross-checks R1's spike layouts.

.DESCRIPTION
    Runs `Lrm.Gpu.Vulkan.Interop.Tests abi-oracle`, which
      1. dumps the runtime layout of every bound item (struct and union sizes, alignments, offsets, field sizes and
         kinds, enum values, handle sizes, constants, sTypes, structextends, aliases and every delegate signature) as
         vk_abi.g.c (C11) and vk_sig.g.cpp (C++17);
      2. fails when the committed copies in this folder differ (-Update rewrites them instead);
      3. compiles them with cl.exe (VS 2022 Build Tools through vswhere and vcvars64.bat) against the pinned
         Vulkan-Headers (the vkheaders entry of tools/bootstrap/toolchain.json, its vulkan_core.h checked against the
         install record), links oracle_main.c and runs it.
    Then, when spikes/vk-heap/src/Layout.fs exists, `dotnet fsi spike-crosscheck.fsx` compares R1's hand-written C
    declarations with the binding.
    Exit codes: 0 passed; 1 failed or stale; 2 no Visual Studio C++ toolchain or no Vulkan-Headers.
    The cpu tier runs the same checks (Lrm.Gpu.Vulkan.Interop.Tests), plus mutants that must fail.

.EXAMPLE
    pwsh tools/abi-oracle/vulkan/run.ps1             # build, check the committed sources, compile and run
    pwsh tools/abi-oracle/vulkan/run.ps1 -Update     # after changing the bindings: rewrite the committed sources
    pwsh tools/abi-oracle/vulkan/run.ps1 -NoBuild    # what CI runs after its Release build
#>
[CmdletBinding()]
param(
    [switch]$Update,
    [switch]$NoBuild,
    [switch]$NoCompile,
    [ValidateSet('Release', 'Debug')]
    [string]$Configuration = 'Release'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$project = Join-Path $repo 'tests/Lrm.Gpu.Vulkan.Interop.Tests'

if (-not $NoBuild) {
    & dotnet build $project -c $Configuration --nologo -v quiet
    if ($LASTEXITCODE -ne 0) { Write-Host 'abi-oracle: build failed' -ForegroundColor Red; exit 1 }
}

$bin = Join-Path $repo "artifacts/bin/Lrm.Gpu.Vulkan.Interop.Tests/$($Configuration.ToLowerInvariant())"
$dll = Join-Path $bin 'Lrm.Gpu.Vulkan.Interop.Tests.dll'
if (-not (Test-Path -LiteralPath $dll)) { Write-Host "abi-oracle: $dll is missing (build first)" -ForegroundColor Red; exit 1 }

$arguments = @('abi-oracle')
if ($Update) { $arguments += '--update' }
if ($NoCompile) { $arguments += '--no-compile' }

$env:LRM_REPO_ROOT = $repo
& dotnet $dll @arguments
$code = $LASTEXITCODE
if ($code -ne 0 -or $NoCompile) { exit $code }

$layout = Join-Path $repo 'spikes/vk-heap/src/Layout.fs'
if (Test-Path -LiteralPath $layout) {
    & dotnet fsi --quiet --nowarn:3886 (Join-Path $PSScriptRoot 'spike-crosscheck.fsx') (Join-Path $bin 'Lrm.Gpu.Vulkan.Interop.dll')
    if ($LASTEXITCODE -ne 0) { exit 1 }
} else {
    Write-Host 'abi-oracle: spikes/vk-heap/src/Layout.fs is gone; the spike cross-check is skipped'
}
exit 0
