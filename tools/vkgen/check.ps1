<#
.SYNOPSIS
    Codegen diff for the Vulkan interop: regenerates in memory and fails when the committed files are stale.

.DESCRIPTION
    Runs tools/vkgen (Lrm.Tools.VkGen) with --check against the vk.xml pinned in tools/bootstrap/toolchain.json
    (the vkxml entry: version, size and SHA-256; bootstrap installs it under LRM_DATA). Exit codes: 0 current;
    1 stale (regenerate with `dotnet run --project tools/vkgen/Lrm.Tools.VkGen -c Release`, then refresh the ABI
    oracle with `pwsh tools/abi-oracle/vulkan/run.ps1 -Update`); 2 generator error; 4 vk.xml is missing or differs
    from the pin (run `pwsh tools/bootstrap/bootstrap.ps1`).

.EXAMPLE
    pwsh tools/vkgen/check.ps1
    pwsh tools/vkgen/check.ps1 -NoBuild    # CI, after the Release build
#>
[CmdletBinding()]
param(
    [switch]$NoBuild,
    [ValidateSet('Release', 'Debug')]
    [string]$Configuration = 'Release'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$project = Join-Path $repo 'tools/vkgen/Lrm.Tools.VkGen'

if (-not $NoBuild) {
    & dotnet build $project -c $Configuration --nologo -v quiet
    if ($LASTEXITCODE -ne 0) { Write-Host 'vkgen: build failed' -ForegroundColor Red; exit 2 }
}

$dll = Join-Path $repo "artifacts/bin/Lrm.Tools.VkGen/$($Configuration.ToLowerInvariant())/Lrm.Tools.VkGen.dll"
if (-not (Test-Path -LiteralPath $dll)) { Write-Host "vkgen: $dll is missing (build first)" -ForegroundColor Red; exit 2 }

& dotnet $dll --check --repo $repo
exit $LASTEXITCODE
