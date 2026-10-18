<#
.SYNOPSIS
    Generates build/ProjectGraph.g.props, build/ProjectKinds.g.props and Lrm.slnx from docs/project-graph.json.

.DESCRIPTION
    docs/project-graph.json is the only source of ProjectReferences and PackageReferences (ProjectGraph contract).
    This script validates the graph (schema, files on disk, layering rules), checks that every fsproj holds file
    items only and that every <Name>.lrm.targets sits next to its registered project, and writes:
      - build/ProjectKinds.g.props  project identity, kind and group membership, imported by Directory.Build.props
      - build/ProjectGraph.g.props  ProjectReference and PackageReference items, imported by Directory.Build.targets
      - Lrm.slnx                    every project in the graph, nothing else (spikes/** is never listed)
    Output is byte-deterministic: UTF-8 without BOM, LF line endings, sorted where order carries no meaning.

.PARAMETER Check
    Do not write anything. Exit 1 if any committed output differs from a fresh generation (the CI codegen diff).

.EXAMPLE
    pwsh tools/ci/gen-project-graph.ps1
    pwsh tools/ci/gen-project-graph.ps1 -Check
#>
[CmdletBinding()]
param(
    [switch]$Check,
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$graphPath = Join-Path $RepoRoot 'docs/project-graph.json'
$packagesPath = Join-Path $RepoRoot 'Directory.Packages.props'
$errors = [System.Collections.Generic.List[string]]::new()

function Fail([string]$message) { $errors.Add($message) }

$graph = Get-Content -Raw -LiteralPath $graphPath | ConvertFrom-Json
if ($graph.schema -ne 'lrm.project-graph/1') { throw "docs/project-graph.json: unknown schema '$($graph.schema)'" }

$kindNames = @($graph.kinds.PSObject.Properties.Name)
$groups = @{}
foreach ($g in $graph.groups.PSObject.Properties) { $groups[$g.Name] = @($g.Value) }

# ---- projects: shape and files on disk ------------------------------------------------------------

$byName = [ordered]@{}
$byPath = @{}
foreach ($p in $graph.projects) {
    $refs = @(if ($p.PSObject.Properties['refs']) { $p.refs } else { @() })
    $buildRefs = @(if ($p.PSObject.Properties['buildRefs']) { $p.buildRefs } else { @() })
    $packages = @(if ($p.PSObject.Properties['packages']) { $p.packages } else { @() })
    $proj = [pscustomobject]@{
        Name = [string]$p.name; Path = [string]$p.path; Kind = [string]$p.kind; Owner = [string]$p.owner
        Refs = $refs; BuildRefs = $buildRefs; Packages = $packages
    }
    if ($byName.Contains($proj.Name)) { Fail "duplicate project name '$($proj.Name)'"; continue }
    if ($byPath.ContainsKey($proj.Path)) { Fail "duplicate project path '$($proj.Path)'"; continue }
    $byName[$proj.Name] = $proj
    $byPath[$proj.Path] = $proj

    if ($kindNames -notcontains $proj.Kind) { Fail "$($proj.Name): unknown kind '$($proj.Kind)'" }
    if ([IO.Path]::GetFileName($proj.Path) -ne "$($proj.Name).fsproj") { Fail "$($proj.Name): file name must be $($proj.Name).fsproj" }
    if ($proj.Path -match '\\') { Fail "$($proj.Name): use forward slashes in path" }
    $allowedRoots = switch ($proj.Kind) {
        'lib' { @('src/', 'tests/') } 'app' { @('apps/') } 'test' { @('tests/') } 'tool' { @('src/', 'tools/') } default { @() }
    }
    if (-not ($allowedRoots | Where-Object { $proj.Path.StartsWith($_) })) {
        Fail "$($proj.Name): a $($proj.Kind) project must live under $($allowedRoots -join ' or ')"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot $proj.Path))) { Fail "$($proj.Name): $($proj.Path) does not exist" }
}

# Every fsproj in the repo (outside spikes/ and artifacts/) must be registered.
$onDisk = Get-ChildItem -LiteralPath $RepoRoot -Recurse -Filter *.fsproj -File |
    ForEach-Object { [IO.Path]::GetRelativePath($RepoRoot, $_.FullName).Replace('\', '/') } |
    Where-Object { -not ($_.StartsWith('spikes/') -or $_.StartsWith('artifacts/') -or $_ -match '(^|/)(bin|obj)/') }
foreach ($f in $onDisk) { if (-not $byPath.ContainsKey($f)) { Fail "$f is not registered in docs/project-graph.json" } }

# An fsproj holds file items only: <Project Sdk="Microsoft.NET.Sdk"> with ItemGroups of Compile, EmbeddedResource,
# None and Content items. Build logic goes in <Name>.lrm.targets next to it, which Directory.Build.targets imports.
$fileItems = @('Compile', 'EmbeddedResource', 'None', 'Content')
$skipNodes = @([Xml.XmlNodeType]::Comment, [Xml.XmlNodeType]::Whitespace, [Xml.XmlNodeType]::SignificantWhitespace)
foreach ($proj in $byName.Values) {
    $full = Join-Path $RepoRoot $proj.Path
    if (-not (Test-Path -LiteralPath $full)) { continue }
    $hint = "an fsproj holds only ItemGroups of $($fileItems -join ', ') items; build logic goes in $($proj.Name).lrm.targets"
    try { [xml]$x = Get-Content -Raw -LiteralPath $full }
    catch { Fail "$($proj.Path): not well-formed XML: $($_.Exception.Message)"; continue }
    $root = $x.DocumentElement
    if ($root.Name -ne 'Project' -or $root.Attributes.Count -ne 1 -or $root.GetAttribute('Sdk') -ne 'Microsoft.NET.Sdk') {
        Fail "$($proj.Path): the root element must be <Project Sdk=`"Microsoft.NET.Sdk`"> with no other attributes"
    }
    foreach ($n in $root.ChildNodes) {
        if ($skipNodes -contains $n.NodeType) { continue }
        if ($n.NodeType -ne [Xml.XmlNodeType]::Element -or $n.Name -ne 'ItemGroup') { Fail "$($proj.Path): <$($n.Name)> is not allowed; $hint"; continue }
        foreach ($a in $n.Attributes) { if (@('Condition', 'Label') -notcontains $a.Name) { Fail "$($proj.Path): ItemGroup attribute '$($a.Name)' is not allowed" } }
        foreach ($i in $n.ChildNodes) {
            if ($skipNodes -contains $i.NodeType) { continue }
            if ($i.NodeType -ne [Xml.XmlNodeType]::Element -or $fileItems -notcontains $i.Name) { Fail "$($proj.Path): <$($i.Name)> is not allowed; $hint" }
        }
    }
}

# A <Name>.lrm.targets file is imported only next to its own registered <Name>.fsproj; anything else is a stray.
$extensions = Get-ChildItem -LiteralPath $RepoRoot -Recurse -Filter *.lrm.targets -File |
    ForEach-Object { [IO.Path]::GetRelativePath($RepoRoot, $_.FullName).Replace('\', '/') } |
    Where-Object { -not ($_.StartsWith('spikes/') -or $_.StartsWith('artifacts/') -or $_ -match '(^|/)(bin|obj)/') }
foreach ($t in $extensions) {
    $fsproj = $t.Substring(0, $t.Length - '.lrm.targets'.Length) + '.fsproj'
    if (-not $byPath.ContainsKey($fsproj)) { Fail "$t has no registered project $fsproj next to it, so nothing imports it" }
}

# ---- references and packages --------------------------------------------------------------------

[xml]$packagesXml = Get-Content -Raw -LiteralPath $packagesPath
$pinned = @($packagesXml.Project.ItemGroup.PackageVersion | Where-Object { $_ } | ForEach-Object { $_.Include })
foreach ($pkg in $graph.packages) { if ($pinned -notcontains $pkg) { Fail "package '$pkg' has no PackageVersion in Directory.Packages.props" } }

foreach ($proj in $byName.Values) {
    foreach ($r in @($proj.Refs) + @($proj.BuildRefs)) {
        if (-not $byName.Contains($r)) { Fail "$($proj.Name): reference to unknown project '$r'" }
        elseif ($r -eq $proj.Name) { Fail "$($proj.Name): references itself" }
    }
    foreach ($list in @(@{ n = 'refs'; v = $proj.Refs }, @{ n = 'buildRefs'; v = $proj.BuildRefs }, @{ n = 'packages'; v = $proj.Packages })) {
        $dupes = @($list.v | Group-Object | Where-Object Count -gt 1 | ForEach-Object Name)
        if ($dupes) { Fail "$($proj.Name): duplicate $($list.n) entries: $($dupes -join ', ')" }
    }
    foreach ($pkg in $proj.Packages) { if (@($graph.packages) -notcontains $pkg) { Fail "$($proj.Name): package '$pkg' is not in the top-level packages list" } }
}

# ---- layering rules -----------------------------------------------------------------------------

function Expand-Patterns([object[]]$patterns) {
    $out = @()
    foreach ($pat in $patterns) {
        if ($pat.StartsWith('@')) {
            $key = $pat.Substring(1)
            if (-not $groups.ContainsKey($key)) { Fail "rules: unknown group '$pat'"; continue }
            $out += Expand-Patterns $groups[$key]
        }
        else { $out += $pat }
    }
    , $out
}
function Test-Match([string]$name, [object[]]$patterns) {
    foreach ($pat in $patterns) { if ($name -like $pat) { return $true } }
    $false
}

# Transitive closure over refs and buildRefs, with the path that reaches each project (for messages).
$closure = @{}
function Get-Closure([string]$start) {
    if ($closure.ContainsKey($start)) { return $closure[$start] }
    $seen = [ordered]@{}
    $queue = [System.Collections.Generic.Queue[object]]::new()
    $queue.Enqueue(@($start))
    while ($queue.Count -gt 0) {
        $chain = $queue.Dequeue()
        $cur = $chain[-1]
        if (-not $byName.Contains($cur)) { continue }
        foreach ($next in @($byName[$cur].Refs) + @($byName[$cur].BuildRefs)) {
            if (-not $seen.Contains($next) -and $next -ne $start) {
                $seen[$next] = ($chain + $next) -join ' -> '
                $queue.Enqueue($chain + $next)
            }
        }
    }
    $closure[$start] = $seen
    $seen
}

# Cycle check first; the other rules assume a DAG.
$state = @{}
function Visit([string]$n, [string[]]$stack) {
    if ($state[$n] -eq 'done') { return }
    if ($state[$n] -eq 'active') { Fail "cycle: $(($stack + $n) -join ' -> ')"; return }
    $state[$n] = 'active'
    foreach ($next in @($byName[$n].Refs) + @($byName[$n].BuildRefs)) { if ($byName.Contains($next)) { Visit $next ($stack + $n) } }
    $state[$n] = 'done'
}
foreach ($n in $byName.Keys) { Visit $n @() }

foreach ($rule in $graph.rules) {
    $id = $rule.id
    $has = { param($k) $null -ne $rule.PSObject.Properties[$k] }
    if ((& $has 'forbid')) {
        $forbid = Expand-Patterns @($rule.forbid)
        $except = if (& $has 'except') { Expand-Patterns @($rule.except) } else { @() }
        $from = @($byName.Values | Where-Object {
                $inNames = (& $has 'from') -and (Test-Match $_.Name (Expand-Patterns @($rule.from)))
                $inKinds = (& $has 'fromKinds') -and (@($rule.fromKinds) -contains $_.Kind)
                ($inNames -or $inKinds) -and -not (Test-Match $_.Name $except)
            })
        foreach ($p in $from) {
            $reach = Get-Closure $p.Name
            foreach ($q in $reach.Keys) {
                if (Test-Match $q $forbid) { Fail "rule ${id}: $($reach[$q]) ($($rule.why))" }
            }
        }
    }
    if ((& $has 'allowBuildRefs')) {
        $allowed = @($rule.allowBuildRefs)
        foreach ($p in $byName.Values) {
            foreach ($b in $p.BuildRefs) { if ($allowed -notcontains "$($p.Name) -> $b") { Fail "rule ${id}: $($p.Name) -> $b is not an allowed build-only edge" } }
        }
    }
    if ((& $has 'forbidRefKinds')) {
        foreach ($p in $byName.Values) {
            foreach ($r in $p.Refs) {
                if ($byName.Contains($r) -and (@($rule.forbidRefKinds) -contains $byName[$r].Kind)) {
                    Fail "rule ${id}: $($p.Name) -> $r references a $($byName[$r].Kind) ($($rule.why))"
                }
            }
        }
    }
    if ((& $has 'forbidPaths')) {
        foreach ($p in $byName.Values) {
            foreach ($prefix in $rule.forbidPaths) { if ($p.Path.StartsWith($prefix)) { Fail "rule ${id}: $($p.Path) ($($rule.why))" } }
        }
    }
}

if ($errors.Count -gt 0) {
    $errors | ForEach-Object { Write-Host "error: $_" -ForegroundColor Red }
    Write-Host "docs/project-graph.json: $($errors.Count) error(s); nothing generated." -ForegroundColor Red
    exit 1
}

# ---- generation ---------------------------------------------------------------------------------

$header = '  <!-- GENERATED by tools/ci/gen-project-graph.ps1 from docs/project-graph.json. Do not edit: change the JSON and regenerate. -->'
function Escape([string]$s) { [System.Security.SecurityElement]::Escape($s) }

# Group membership, for build settings keyed on groups in Directory.Build.props (for example Tailcalls=false for GPU
# code). LrmGroups lists the groups whose patterns match the project's name; LrmReaches the groups that its references
# reach, transitively. Both are ';'-wrapped and sorted, so $(LrmGroups.Contains(';gpu;')) tests one group.
$groupNames = @($groups.Keys | Sort-Object)
function Get-GroupsOf([string]$name) {
    @($groupNames | Where-Object { Test-Match $name (Expand-Patterns @("@$_")) })
}
function Format-GroupList([string[]]$list) {
    $items = @($list | Where-Object { $_ })
    if ($items.Count -eq 0) { return '' }
    ';' + ($items -join ';') + ';'
}

$kinds = [System.Text.StringBuilder]::new()
[void]$kinds.Append("<Project>`n$header`n")
foreach ($p in ($byName.Values | Sort-Object Path)) {
    $member = Format-GroupList (Get-GroupsOf $p.Name)
    $reached = Format-GroupList @((Get-Closure $p.Name).Keys | ForEach-Object { Get-GroupsOf $_ } | Sort-Object -Unique)
    [void]$kinds.Append("  <PropertyGroup Condition=`"'`$(LrmProjectPath)' == '$(Escape $p.Path)'`">`n")
    [void]$kinds.Append("    <LrmGraphName>$(Escape $p.Name)</LrmGraphName>`n")
    [void]$kinds.Append("    <LrmProjectKind>$(Escape $p.Kind)</LrmProjectKind>`n")
    [void]$kinds.Append("    <LrmOwner>$(Escape $p.Owner)</LrmOwner>`n")
    if ($member) { [void]$kinds.Append("    <LrmGroups>$(Escape $member)</LrmGroups>`n") }
    if ($reached) { [void]$kinds.Append("    <LrmReaches>$(Escape $reached)</LrmReaches>`n") }
    [void]$kinds.Append("  </PropertyGroup>`n")
}
[void]$kinds.Append("</Project>`n")

$refsOut = [System.Text.StringBuilder]::new()
[void]$refsOut.Append("<Project>`n$header`n")
foreach ($p in ($byName.Values | Sort-Object Path)) {
    if (@($p.Refs).Count + @($p.BuildRefs).Count + @($p.Packages).Count -eq 0) { continue }
    [void]$refsOut.Append("  <ItemGroup Condition=`"'`$(LrmGraphName)' == '$(Escape $p.Name)'`">`n")
    foreach ($r in $p.Refs) {
        [void]$refsOut.Append("    <ProjectReference Include=`"`$(LrmRepoRoot)$(Escape $byName[$r].Path)`" LrmGraph=`"true`" />`n")
    }
    foreach ($r in $p.BuildRefs) {
        [void]$refsOut.Append("    <ProjectReference Include=`"`$(LrmRepoRoot)$(Escape $byName[$r].Path)`" LrmGraph=`"true`" LrmBuildOnly=`"true`" ReferenceOutputAssembly=`"false`" Private=`"false`" />`n")
    }
    foreach ($pkg in $p.Packages) {
        [void]$refsOut.Append("    <PackageReference Include=`"$(Escape $pkg)`" LrmGraph=`"true`" />`n")
    }
    [void]$refsOut.Append("  </ItemGroup>`n")
}
[void]$refsOut.Append("</Project>`n")

$sln = [System.Text.StringBuilder]::new()
[void]$sln.Append("<!-- GENERATED by tools/ci/gen-project-graph.ps1 from docs/project-graph.json. Do not edit. -->`n<Solution>`n")
$folders = $byName.Values | Group-Object { $_.Path.Split('/')[0] } | Sort-Object Name
foreach ($f in $folders) {
    [void]$sln.Append("  <Folder Name=`"/$(Escape $f.Name)/`">`n")
    foreach ($p in ($f.Group | Sort-Object Path)) { [void]$sln.Append("    <Project Path=`"$(Escape $p.Path)`" />`n") }
    [void]$sln.Append("  </Folder>`n")
}
[void]$sln.Append("</Solution>`n")

$outputs = [ordered]@{
    'build/ProjectKinds.g.props' = $kinds.ToString()
    'build/ProjectGraph.g.props' = $refsOut.ToString()
    'Lrm.slnx'                   = $sln.ToString()
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
$stale = @()
foreach ($rel in $outputs.Keys) {
    $full = Join-Path $RepoRoot $rel
    $want = $outputs[$rel]
    $have = if (Test-Path -LiteralPath $full) { [IO.File]::ReadAllText($full).Replace("`r`n", "`n") } else { $null }
    if ($Check) {
        if ($have -ne $want) { $stale += $rel }
    }
    elseif ($have -ne $want) {
        New-Item -ItemType Directory -Force -Path (Split-Path $full) | Out-Null
        [IO.File]::WriteAllText($full, $want, $utf8)
        Write-Host "wrote $rel"
    }
}

if ($Check) {
    if ($stale.Count -gt 0) {
        $stale | ForEach-Object { Write-Host "error: $_ is out of date; run pwsh tools/ci/gen-project-graph.ps1 and commit the result" -ForegroundColor Red }
        exit 1
    }
    Write-Host "project graph: $($byName.Count) projects, all rules hold, generated files are up to date"
}
else {
    Write-Host "project graph: $($byName.Count) projects, all rules hold"
}
