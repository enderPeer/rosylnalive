<#
.SYNOPSIS
    Runs LRM's typed argument blocks on the GPU through R1's descriptor-heap spike: the on-device evidence of
    ShaderBuildAndSourceConvention v1.1, "The argument block type" (owner R4).

.DESCRIPTION
    R1's spike (spikes/vk-heap, VulkanHeapRecipe) reads its arguments through `struct { uint4 d[8]; }`. LRM's kernels
    read them through the generated per-kernel blocks `lrm_args_<kernel>` (KernelAbiAndCatalog). This script runs
    LRM's own blocks through the spike, so the recipe's push data and root constants are shown to reach them:
      1. It exports the tree of -SpikeRef with `git archive` into the work folder. Nothing in the checkout changes.
      2. It replaces the spike's scale_f32.hlsl and args_echo.hlsl with LRM's selftests
         (src/Lrm.Shaders/hlsl/selftest), and puts lrm.hlsli and lrm_args.g.hlsli (src/Lrm.Shaders/include) beside
         them. Their typed blocks are 20 and 128 bytes, and the spike's host code writes the same argument words.
         table_echo and bf16_unpack stay the spike's own: their arguments and offsets differ from LRM's selftests.
      3. It builds the spike and runs its gpu tier. The tier takes the per-backend mutexes through Lrm.Testing,
         poisons every output, runs every case twice on D3D12 and on 9 Vulkan configurations, compares each gating
         case word for word with the CPU, and runs negative controls with the dispatches removed.
    The spike writes a run log under LRM_DATA/runs (gpu.dispatch intervals and gpu.spike.* checks, in the RunLog
    layout) and its results page under <WorkDir>/results. The GPU part takes about 5 seconds. A local check only:
    CI has no GPU, and spikes/** is outside the solution and CI (ProjectGraph contract).

.PARAMETER SpikeRef
    The commit whose spikes/vk-heap is used. Default: HEAD when it contains the spike (after R1 merges), otherwise
    origin/ws/R1-vk-heap-spike.

.PARAMETER WorkDir
    The work folder. Default: artifacts/typed-args-spike/<commit> (gitignored). It is recreated on every run.

.EXAMPLE
    pwsh src/Lrm.Shaders.Build/typed-args-spike.ps1
#>
[CmdletBinding()]
param(
    [string]$SpikeRef,
    [string]$WorkDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$spikeProject = 'spikes/vk-heap/VkHeapSpike.fsproj'

if (-not $SpikeRef) {
    & git -C $repo cat-file -e "HEAD:$spikeProject" 2>$null
    $SpikeRef = if ($LASTEXITCODE -eq 0) { 'HEAD' } else { 'origin/ws/R1-vk-heap-spike' }
}

$sha = (& git -C $repo rev-parse --verify "$SpikeRef^{commit}").Trim()
if ($LASTEXITCODE -ne 0) { throw "typed-args-spike: cannot resolve $SpikeRef" }
& git -C $repo cat-file -e "${sha}:$spikeProject"
if ($LASTEXITCODE -ne 0) { throw "typed-args-spike: $SpikeRef ($sha) has no $spikeProject" }

if (-not $WorkDir) { $WorkDir = Join-Path $repo "artifacts/typed-args-spike/$($sha.Substring(0, 12))" }
$WorkDir = [System.IO.Path]::GetFullPath($WorkDir)
if (Test-Path $WorkDir) { Remove-Item -Recurse -Force $WorkDir }
$tree = Join-Path $WorkDir 'tree'
New-Item -ItemType Directory -Force -Path $tree | Out-Null

Write-Host "typed-args-spike: spike from $SpikeRef ($sha) into $tree"
$tar = Join-Path $WorkDir 'tree.tar'
& git -C $repo archive --format=tar -o $tar $sha
if ($LASTEXITCODE -ne 0) { throw 'typed-args-spike: git archive failed' }
& tar -xf $tar -C $tree
if ($LASTEXITCODE -ne 0) { throw 'typed-args-spike: tar failed' }
Remove-Item -Force $tar

# LRM's sources, from this checkout, over the spike's own.
$hlsl = Join-Path $tree 'spikes/vk-heap/hlsl'
$swapped = @(
    @{ From = 'src/Lrm.Shaders/include/lrm.hlsli'; To = 'lrm.hlsli' }
    @{ From = 'src/Lrm.Shaders/include/lrm_args.g.hlsli'; To = 'lrm_args.g.hlsli' }
    @{ From = 'src/Lrm.Shaders/hlsl/selftest/scale_f32.hlsl'; To = 'scale_f32.hlsl' }
    @{ From = 'src/Lrm.Shaders/hlsl/selftest/args_echo.hlsl'; To = 'args_echo.hlsl' }
)
foreach ($s in $swapped) {
    $src = Join-Path $repo $s.From
    Copy-Item -LiteralPath $src -Destination (Join-Path $hlsl $s.To) -Force
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $src).Hash.ToLowerInvariant()
    Write-Host ("typed-args-spike: {0} <- {1} (sha256 {2})" -f $s.To, $s.From, $hash)
}

Push-Location $tree
try {
    & dotnet build $spikeProject -c Release --nologo
    if ($LASTEXITCODE -ne 0) { throw 'typed-args-spike: the spike does not build' }

    $results = Join-Path $WorkDir 'results'
    $env:LRM_TEST_TIER = 'gpu'
    & dotnet (Join-Path $tree 'artifacts/bin/VkHeapSpike/release/VkHeapSpike.dll') --sequenced --results $results
    $code = $LASTEXITCODE
}
finally {
    Remove-Item Env:LRM_TEST_TIER -ErrorAction SilentlyContinue
    Pop-Location
}

Write-Host "typed-args-spike: gpu tier exit $code; results in $results"
exit $code
