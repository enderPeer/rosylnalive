<#
.SYNOPSIS
    Self-test of the hygiene checks: every fixture fails as designed, and every positive control passes.

.DESCRIPTION
    Runs, in order:
      1. vectors     the known answers of docs/ownership.json (fixtures/ownership/vectors.json): owners, branch
                     resolution, glob matching, rejected globs and contract files. The contract text shows the same
                     table.
      2. cases       fixtures/<check>/cases.json for ownership, sizes, eol and contract-label, through the pure
                     functions of Hygiene.psm1.
      3. layering    fixtures/layering/cases.json: each case edits a temporary copy of the graph inputs and runs
                     gen-project-graph.ps1 -Check there.
      4. end-to-end  temporary git repositories with fixture branches, checked by the real check-*.ps1 scripts in
                     child processes, so git parsing and exit codes are covered too.

    A 'fail' fixture passes the self-test only when it produces errors and each of its 'match' patterns matches one;
    a 'pass' fixture only when it produces no error. The hygiene CI job runs this before the checks themselves.

    Exit code: 0 when every fixture behaves as designed, otherwise 1.

.EXAMPLE
    pwsh tools/ci/test-hygiene.ps1
#>
[CmdletBinding()]
param(
    [string]$OutDir,
    # Keep the temporary fixture repositories for inspection.
    [switch]$Keep
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'Hygiene.psm1') -Force

$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$fixtures = Join-Path $PSScriptRoot 'fixtures'
if (-not $OutDir) { $OutDir = Join-Path $repo 'artifacts/hygiene' }
$pwsh = (Get-Process -Id $PID).Path
$tmpRoot = Join-Path ([IO.Path]::GetTempPath()) ("lrm-hygiene-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Force -Path $tmpRoot | Out-Null

$results = [Collections.Generic.List[object]]::new()
$produced = [Collections.Generic.List[object]]::new()
function Add-Outcome([string]$Group, [string]$Name, [string]$Expect, [string[]]$Problems, [string]$Detail) {
    $ok = -not $Problems
    $results.Add([pscustomobject]@{ Group = $Group; Name = $Name; Expect = $Expect; Ok = $ok; Problems = @($Problems); Detail = $Detail })
    if ($ok) { Write-Host "  ok    $Group/$Name ($Detail)" }
    else { Write-Host "  FAIL  $Group/$Name ($Detail): $($Problems -join '; ')" -ForegroundColor Red }
}

# Compares a check result with a case's expectation; returns the problems.
function Compare-Expectation($Case, $Result) {
    $errors = @($Result.Findings | Where-Object Severity -EQ 'error' | ForEach-Object Message)
    $warnings = @($Result.Findings | Where-Object Severity -EQ 'warning' | ForEach-Object Message)
    $problems = @()
    if ($Case.expect -ceq 'fail') {
        if (-not $errors.Count) { $problems += 'expected errors, got none' }
        foreach ($m in @(Get-Prop $Case 'match' @())) { if (-not @($errors | Where-Object { $_ -cmatch $m }).Count) { $problems += "no error matches /$m/ (errors: $($errors -join ' | '))" } }
        $n = Get-Prop $Case 'errors'
        if ($null -ne $n -and $errors.Count -ne [int]$n) { $problems += "expected $n error(s), got $($errors.Count): $($errors -join ' | ')" }
    }
    elseif ($Case.expect -ceq 'pass') {
        if ($errors.Count) { $problems += "expected no errors, got: $($errors -join ' | ')" }
    }
    else { $problems += "unknown expect '$($Case.expect)'" }
    foreach ($w in @(Get-Prop $Case 'warn' @())) { if (-not @($warnings | Where-Object { $_ -cmatch $w }).Count) { $problems += "no warning matches /$w/" } }
    , $problems
}

function Get-Detail($Case, $Result) {
    $e = Get-FindingCount $Result 'error'
    if ($Case.expect -ceq 'fail') { "fails as designed, $e error(s)" } else { "passes, $(Get-FindingCount $Result 'warning') warning(s)" }
}

function Read-Cases([string]$Check) { @((Get-Content -Raw -LiteralPath (Join-Path $fixtures "$Check/cases.json") | ConvertFrom-Json).cases) }

$map = Read-OwnershipMap (Join-Path $repo 'docs/ownership.json')
$config = Read-HygieneConfig (Join-Path $PSScriptRoot 'hygiene.json')

try {
    # ---- 1. vectors -----------------------------------------------------------------------------
    Write-Host 'vectors (docs/ownership.json known answers)' -ForegroundColor Cyan
    $v = Get-Content -Raw -LiteralPath (Join-Path $fixtures 'ownership/vectors.json') | ConvertFrom-Json
    $p = @(); $n = 0
    foreach ($o in $v.owners) { $n++; $got = @(Get-PathOwners $map $o.path); if ($got.Count -ne 1 -or $got[0] -cne $o.family) { $p += "$($o.path): expected $($o.family), got [$($got -join ', ')]" } }
    Add-Outcome 'vectors' 'owners' 'pass' $p "$n paths"
    $p = @(); $n = 0
    foreach ($b in $v.branches) {
        $n++
        $got = Resolve-Branch $map $b.branch
        $fam = [string](Get-Prop $b 'family' '')
        if ($got.Kind -cne $b.kind -or [string]$got.Family -cne $fam) { $p += "$($b.branch): expected $($b.kind)/$fam, got $($got.Kind)/$($got.Family)" }
    }
    Add-Outcome 'vectors' 'branches' 'pass' $p "$n branches"
    $p = @(); $n = 0
    foreach ($g in $v.globs) { $n++; $got = Test-Glob $g.path $g.glob; if ($got -ne [bool]$g.match) { $p += "'$($g.glob)' on '$($g.path)': expected $($g.match), got $got" } }
    Add-Outcome 'vectors' 'globs' 'pass' $p "$n globs"
    $p = @(); $n = 0
    foreach ($g in $v.badGlobs) { $n++; try { [void](ConvertTo-GlobRegex $g); $p += "'$g' was accepted" } catch { } }
    Add-Outcome 'vectors' 'rejected-globs' 'pass' $p "$n globs rejected"
    $p = @(); $n = 0
    foreach ($c in $v.contractFiles) { $n++; $got = Test-ContractFile $map $c.path; if ($got -ne [bool]$c.contract) { $p += "$($c.path): expected contract=$($c.contract), got $got" } }
    Add-Outcome 'vectors' 'contract-files' 'pass' $p "$n paths"

    # ---- 2. cases through the pure functions ----------------------------------------------------
    Write-Host 'cases (pure functions)' -ForegroundColor Cyan
    foreach ($c in Read-Cases 'ownership') {
        $m = if (Get-Prop $c 'map') { Read-OwnershipMap (Join-Path $fixtures "ownership/$($c.map)") } else { $map }
        $known = $null -ne $c.PSObject.Properties['labels']
        $r = Test-Ownership -Map $m -Tracked @(Get-Prop $c 'tracked' @()) -Branch ([string](Get-Prop $c 'branch' '')) `
            -Changes @(Get-Prop $c 'changes' @() | ForEach-Object { ConvertTo-Change $_ }) -Labels @(Get-Prop $c 'labels' @()) -LabelsKnown $known
        $produced.Add($r)
        Add-Outcome 'ownership' $c.name $c.expect (Compare-Expectation $c $r) (Get-Detail $c $r)
    }
    foreach ($c in Read-Cases 'sizes') {
        $entries = foreach ($e in $c.entries) {
            $count = [int](Get-Prop $e 'repeat' 0)
            if ($count -le 0) { [pscustomobject]@{ Path = $e.path; Size = [long]$e.size; Type = (Get-Prop $e 'type' 'blob') }; continue }
            $ext = [IO.Path]::GetExtension($e.path)
            $stem = $e.path.Substring(0, $e.path.Length - $ext.Length)
            for ($k = 0; $k -lt $count; $k++) { [pscustomobject]@{ Path = "$stem-$k$ext"; Size = [long]$e.size; Type = 'blob' } }
        }
        $r = Test-Sizes -Config $config -Entries @($entries)
        $produced.Add($r)
        Add-Outcome 'sizes' $c.name $c.expect (Compare-Expectation $c $r) (Get-Detail $c $r)
    }
    foreach ($c in Read-Cases 'eol') {
        $r = Test-Eol -Entries @(ConvertFrom-EolListing @($c.listing))
        $produced.Add($r)
        Add-Outcome 'eol' $c.name $c.expect (Compare-Expectation $c $r) (Get-Detail $c $r)
    }
    foreach ($c in Read-Cases 'contract-label') {
        $known = $null -ne $c.PSObject.Properties['labels']
        $r = Test-ContractLabel -Map $map -Changes @($c.changes | ForEach-Object { ConvertTo-Change $_ }) -Labels @(Get-Prop $c 'labels' @()) `
            -LabelsKnown $known -Frozen ([bool]$c.frozen) -ChangelogAdded @(Get-Prop $c 'changelogAdded' @())
        $produced.Add($r)
        Add-Outcome 'contract-label' $c.name $c.expect (Compare-Expectation $c $r) (Get-Detail $c $r)
    }

    # ---- 3. layering on temporary copies of the graph inputs ------------------------------------
    Write-Host 'layering (gen-project-graph.ps1 -Check on edited copies)' -ForegroundColor Cyan
    $graphJson = Get-Content -Raw -LiteralPath (Join-Path $repo 'docs/project-graph.json')
    $inputs = @('docs/project-graph.json', 'Directory.Packages.props', 'build/ProjectKinds.g.props', 'build/ProjectGraph.g.props', 'Lrm.slnx')
    $inputs += @(($graphJson | ConvertFrom-Json).projects | ForEach-Object path)
    $inputs += @(Get-ChildItem -LiteralPath $repo -Recurse -Filter *.lrm.targets -File | ForEach-Object { [IO.Path]::GetRelativePath($repo, $_.FullName).Replace('\', '/') } |
            Where-Object { -not ($_.StartsWith('artifacts/') -or $_.StartsWith('spikes/')) })
    function New-GraphCopy([string]$Name) {
        $root = Join-Path $tmpRoot "layering-$Name"
        foreach ($rel in $inputs) {
            $dst = Join-Path $root $rel
            New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
            Copy-Item -LiteralPath (Join-Path $repo $rel) -Destination $dst
        }
        $root
    }
    $layeringRoots = @{}
    foreach ($c in Read-Cases 'layering') {
        $root = New-GraphCopy $c.name
        $layeringRoots[$c.name] = $root
        $edits = @(Get-Prop $c 'edits' @())
        if ($edits.Count) {
            $g = $graphJson | ConvertFrom-Json
            foreach ($e in $edits) {
                $proj = if ($e.op -cne 'addProject') { @($g.projects | Where-Object name -CEQ $e.project)[0] } else { $null }
                switch ($e.op) {
                    'addRef' { $proj.refs = @($proj.refs) + $e.ref }
                    'addBuildRef' { $proj | Add-Member -Force -NotePropertyName buildRefs -NotePropertyValue (@(Get-Prop $proj 'buildRefs' @()) + $e.ref) }
                    'addPackage' { $proj | Add-Member -Force -NotePropertyName packages -NotePropertyValue (@(Get-Prop $proj 'packages' @()) + $e.package) }
                    'addProject' { $g.projects = @($g.projects) + $e.project }
                    default { throw "layering/$($c.name): unknown edit '$($e.op)'" }
                }
            }
            [IO.File]::WriteAllText((Join-Path $root 'docs/project-graph.json'), ($g | ConvertTo-Json -Depth 20), [Text.UTF8Encoding]::new($false))
        }
        $files = Get-Prop $c 'files'
        if ($files) {
            foreach ($f in $files.PSObject.Properties) {
                $dst = Join-Path $root $f.Name
                New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
                [IO.File]::WriteAllText($dst, [string]$f.Value, [Text.UTF8Encoding]::new($false))
            }
        }
        $r = Test-Layering -RepoRoot $root
        $produced.Add($r)
        Add-Outcome 'layering' $c.name $c.expect (Compare-Expectation $c $r) (Get-Detail $c $r)
    }

    # Every record name and metric key the checks produced follows the TelemetryFacade naming rule.
    $p = @()
    foreach ($r in $produced) {
        $name = Get-RecordName $r
        if (-not (Test-TelemetryName $name)) { $p += "record name '$name'" }
        foreach ($k in $r.Metrics.Keys) { if (-not (Test-TelemetryName $k)) { $p += "$name metric '$k'" } }
    }
    Add-Outcome 'vectors' 'telemetry-names' 'pass' @($p | Sort-Object -Unique) "$($produced.Count) results"

    # ---- 4. end to end: fixture git repositories and the real scripts ---------------------------
    Write-Host 'end-to-end (fixture repositories, check-*.ps1 in child processes)' -ForegroundColor Cyan
    function Git([string]$Root, [string[]]$Arguments) { [void](Invoke-Git $Root $Arguments) }
    function New-FixtureRepo([string]$Name, [hashtable]$Files) {
        $root = Join-Path $tmpRoot $Name
        New-Item -ItemType Directory -Force -Path $root | Out-Null
        Git $root @('init', '-q', '-b', 'main')
        Git $root @('config', 'core.autocrlf', 'false')
        Git $root @('config', 'user.name', 'lrm-fixture')
        Git $root @('config', 'user.email', 'fixture@lrm.invalid')
        foreach ($k in $Files.Keys) { Set-FixtureFile $root $k $Files[$k] }
        Git $root @('add', '-A')
        Git $root @('commit', '-q', '-m', 'fixture base')
        $root
    }
    function Set-FixtureFile([string]$Root, [string]$Rel, $Content) {
        $dst = Join-Path $Root $Rel
        New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
        if ($Content -is [byte[]]) { [IO.File]::WriteAllBytes($dst, $Content) } else { [IO.File]::WriteAllText($dst, [string]$Content, [Text.UTF8Encoding]::new($false)) }
    }
    function Commit-Branch([string]$Root, [string]$Branch, [scriptblock]$Change) {
        Git $Root @('checkout', '-q', '-B', $Branch, 'main')
        & $Change
        Git $Root @('add', '-A')
        Git $Root @('commit', '-q', '-m', "fixture $Branch")
    }
    # Runs a check script in a child process; expectCode 0 or 1, and a pattern its output must contain.
    function Test-Script([string]$Name, [string]$Script, [string[]]$Arguments, [int]$ExpectCode, [string]$Pattern) {
        $out = & $pwsh -NoProfile -NonInteractive -File (Join-Path $PSScriptRoot $Script) @Arguments 2>&1 | Out-String
        $code = $LASTEXITCODE
        $p = @()
        if ($code -ne $ExpectCode) { $p += "exit $code, expected $ExpectCode; output: $($out.Trim() -replace '\s+', ' ')" }
        if ($Pattern -and $out -cnotmatch $Pattern) { $p += "output does not match /$Pattern/: $($out.Trim() -replace '\s+', ' ')" }
        $detail = if ($ExpectCode -eq 1) { "exit 1 as designed" } else { 'exit 0' }
        Add-Outcome 'e2e' $Name $(if ($ExpectCode -eq 1) { 'fail' } else { 'pass' }) $p $detail
    }

    $ownershipJson = Get-Content -Raw -LiteralPath (Join-Path $repo 'docs/ownership.json')
    $changelog = "# Contract changelog`n`n## 2026-10-18`n`n- **LrmCore v1** (``lrm-core.md``, S1): first version.`n"

    # Ownership: the fixture branch of the acceptance criterion.
    $o = New-FixtureRepo 'e2e-ownership' @{
        'docs/ownership.json' = $ownershipJson; 'src/Lrm.Core/Hash.fs' = "module Hash`n"
        'docs/contracts/CHANGELOG.md' = $changelog; 'goldens/README.md' = "# Goldens`n"
    }
    Commit-Branch $o 'ws/O1-fixture-bad' { Set-FixtureFile $o 'src/Lrm.Core/Hash.fs' "module Hash`n// O1 was here`n" }
    Commit-Branch $o 'ws/S1-fixture-good' { Set-FixtureFile $o 'src/Lrm.Core/Hash.fs' "module Hash`n// S1 owns this`n" }
    Commit-Branch $o 'ws/O1-fixture-append' { Set-FixtureFile $o 'docs/contracts/CHANGELOG.md' ($changelog + "- **GoldenSet v1** (``golden-set.md``, O1): first version.`n") }
    Commit-Branch $o 'ws/O1-fixture-rewrite' { Set-FixtureFile $o 'docs/contracts/CHANGELOG.md' $changelog.Replace('first version', 'rewritten') }
    Commit-Branch $o 'ws/O1-fixture-move' { Git $o @('mv', 'goldens/README.md', 'src/Lrm.Core/README.md') }
    Test-Script 'ownership-fixture-branch' 'check-ownership.ps1' @('-RepoRoot', $o, '-Branch', 'ws/O1-fixture-bad', '-Base', 'main', '-Head', 'ws/O1-fixture-bad', '-Labels', '') 1 'src/Lrm\.Core/Hash\.fs is owned by family S1, not by family O1'
    Test-Script 'ownership-own-path' 'check-ownership.ps1' @('-RepoRoot', $o, '-Branch', 'ws/S1-fixture-good', '-Base', 'main', '-Head', 'ws/S1-fixture-good', '-Labels', '') 0 'ownership: OK'
    Test-Script 'ownership-changelog-append' 'check-ownership.ps1' @('-RepoRoot', $o, '-Branch', 'ws/O1-fixture-append', '-Base', 'main', '-Head', 'ws/O1-fixture-append') 0 'shared append-only path; this change only adds lines'
    Test-Script 'ownership-changelog-rewrite' 'check-ownership.ps1' @('-RepoRoot', $o, '-Branch', 'ws/O1-fixture-rewrite', '-Base', 'main', '-Head', 'ws/O1-fixture-rewrite') 1 'is shared append-only'
    Test-Script 'ownership-move-into-other-family' 'check-ownership.ps1' @('-RepoRoot', $o, '-Branch', 'ws/O1-fixture-move', '-Base', 'main', '-Head', 'ws/O1-fixture-move') 1 'src/Lrm\.Core/README\.md is owned by family S1, not by family O1 \(branch ws/O1-fixture-move, status A\)'
    Test-Script 'ownership-current-branch' 'check-ownership.ps1' @('-RepoRoot', $o, '-Base', 'main') 1 'branch ws/O1-fixture-move \(family O1\)'

    # Layering: the script itself on an unmodified copy and on the hand-written reference.
    Test-Script 'layering-unmodified' 'check-layering.ps1' @('-RepoRoot', $layeringRoots['unmodified-copy']) 0 'layering: OK'
    Test-Script 'layering-bad-project-reference' 'check-layering.ps1' @('-RepoRoot', $layeringRoots['handwritten-project-reference']) 1 '<ProjectReference> is not allowed'

    # Sizes: a real 1.5 MiB blob, outside and inside goldens/.
    $big = [byte[]]::new(1572864)
    [Random]::new(7).NextBytes($big)
    $s = New-FixtureRepo 'e2e-sizes' @{ 'README.md' = "# fixture`n"; '.gitattributes' = "* text=auto eol=lf`n*.dat binary`n*.safetensors binary`n" }
    Commit-Branch $s 'ws/T1-fixture-big' { Set-FixtureFile $s 'docs/reports/blob.dat' $big }
    Commit-Branch $s 'ws/O1-fixture-golden' { Set-FixtureFile $s 'goldens/tiny/g9/blob.safetensors' $big }
    $hyg = Join-Path $PSScriptRoot 'hygiene.json'
    Test-Script 'sizes-large-blob' 'check-sizes.ps1' @('-RepoRoot', $s, '-Rev', 'ws/T1-fixture-big', '-Config', $hyg) 1 'docs/reports/blob\.dat is 1\.50 MiB \(1572864 bytes\), over the 1\.00 MiB limit'
    Test-Script 'sizes-golden-blob' 'check-sizes.ps1' @('-RepoRoot', $s, '-Rev', 'ws/O1-fixture-golden', '-Config', $hyg) 0 'sizes: OK'

    # EOL: a CRLF blob forced into the index past the filters, as a broken client would commit it.
    $e = New-FixtureRepo 'e2e-eol' @{ '.gitattributes' = (Get-Content -Raw -LiteralPath (Join-Path $repo '.gitattributes')); 'src/A.fs' = "module A`n" }
    Test-Script 'eol-clean' 'check-eol.ps1' @('-RepoRoot', $e) 0 'eol: OK'
    Set-FixtureFile $e 'tools/crlf.ps1' "Write-Host 1`r`nWrite-Host 2`r`n"
    $blob = (Invoke-Git $e @('hash-object', '-w', '--no-filters', 'tools/crlf.ps1')).Out.Trim()
    Git $e @('update-index', '--add', '--cacheinfo', "100644,$blob,tools/crlf.ps1")
    Test-Script 'eol-crlf-in-index' 'check-eol.ps1' @('-RepoRoot', $e) 1 'tools/crlf\.ps1 has crlf line endings in the index'

    # Contract label: before and after the freeze tag, with and without the label and the CHANGELOG entry.
    $c = New-FixtureRepo 'e2e-contract' @{ 'docs/ownership.json' = $ownershipJson; 'docs/contracts/lrm-core.md' = "# LrmCore`n"; 'docs/contracts/CHANGELOG.md' = $changelog }
    Commit-Branch $c 'ws/S1-fix-dtype' { Set-FixtureFile $c 'docs/contracts/lrm-core.md' "# LrmCore`n`nDType 9 = U16.`n" }
    $args1 = @('-RepoRoot', $c, '-Base', 'main', '-Head', 'ws/S1-fix-dtype')
    Test-Script 'contract-before-freeze' 'check-contract-label.ps1' ($args1 + @('-Labels', '')) 0 'contracts-v1 not tagged yet'
    Git $c @('tag', 'contracts-v1', 'main')
    Test-Script 'contract-frozen-no-label' 'check-contract-label.ps1' ($args1 + @('-Labels', '')) 1 "without the 'contract-change' label: docs/contracts/lrm-core\.md"
    Test-Script 'contract-label-no-changelog' 'check-contract-label.ps1' ($args1 + @('-Labels', 'contract-change')) 1 'needs an entry in docs/contracts/CHANGELOG\.md'
    Git $c @('checkout', '-q', 'ws/S1-fix-dtype')
    Set-FixtureFile $c 'docs/contracts/CHANGELOG.md' ($changelog + "- **LrmCore v2** (``lrm-core.md``, S1): DType 9 = U16.`n")
    Git $c @('commit', '-q', '-am', 'changelog entry')
    Test-Script 'contract-complete' 'check-contract-label.ps1' ($args1 + @('-Labels', 'contract-change')) 0 'contract-label: OK'
}
finally {
    if (-not $Keep) { Remove-Item -LiteralPath $tmpRoot -Recurse -Force -ErrorAction SilentlyContinue }
    else { Write-Host "fixture repositories kept in $tmpRoot" }
}

# ---- summary ------------------------------------------------------------------------------------

$bad = @($results | Where-Object { -not $_.Ok })
$failFixtures = @($results | Where-Object Expect -CEQ 'fail')
$md = [Text.StringBuilder]::new()
[void]$md.AppendLine('### Hygiene self-test')
[void]$md.AppendLine('')
[void]$md.AppendLine('| Group | Fixtures | Designed to fail | Behaved as designed |')
[void]$md.AppendLine('| --- | ---: | ---: | ---: |')
foreach ($g in ($results | Group-Object Group)) {
    $f = @($g.Group | Where-Object Expect -CEQ 'fail').Count
    $ok = @($g.Group | Where-Object Ok).Count
    [void]$md.AppendLine("| $($g.Name) | $($g.Count) | $f | $ok |")
}
[void]$md.AppendLine('')
$verdict = if ($bad.Count) { "FAILED: $($bad.Count) of $($results.Count) did not behave as designed" } else { 'PASSED' }
[void]$md.AppendLine("Result: **$verdict**. $($failFixtures.Count) fixtures are designed to fail; $($results.Count - $failFixtures.Count) are positive controls and vector sets.")
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
[IO.File]::WriteAllText((Join-Path $OutDir 'selftest.md'), $md.ToString(), [Text.UTF8Encoding]::new($false))
if ($env:GITHUB_STEP_SUMMARY) { [IO.File]::AppendAllText($env:GITHUB_STEP_SUMMARY, $md.ToString() + "`n", [Text.UTF8Encoding]::new($false)) }
Write-Host ''
Write-Host $md.ToString()
if ($bad.Count) { exit 1 }
exit 0
