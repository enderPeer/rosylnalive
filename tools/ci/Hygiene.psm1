<#
    Shared logic of the hygiene checks (OwnershipAndContractChange contract; S0b).

    The five checks are pure functions over plain inputs, so the fixtures in tools/ci/fixtures can drive them without
    git. The check-*.ps1 scripts gather the same inputs from git and call these functions; hygiene.ps1 runs all five
    with the CI context; test-hygiene.ps1 proves that every fixture fails as designed.

      Test-Ownership       docs/ownership.json against the tracked tree and the branch's changes
      Test-Layering        tools/ci/gen-project-graph.ps1 -Check (the ProjectGraph rules and generated files)
      Test-Sizes           per-file limits, budgets and forbidden files (tools/ci/hygiene.json)
      Test-Eol             `git ls-files --eol` against .gitattributes (LF only in the index)
      Test-ContractLabel   the contract-change label and its CHANGELOG entry

    Paths are repo-relative with forward slashes and compared ordinally (case-sensitive), as git stores them.
#>
Set-StrictMode -Version Latest

$script:Ordinal = [StringComparer]::Ordinal
$script:GlobCache = [Collections.Generic.Dictionary[string, regex]]::new([StringComparer]::Ordinal)
$script:Changelog = 'docs/contracts/CHANGELOG.md'

# ---- results ------------------------------------------------------------------------------------

# Record names and metric keys follow the TelemetryFacade naming rule (dot-separated segments of a-z, 0-9 and _, each
# starting with a letter; at most 128 characters), and metrics are numbers, booleans or short strings, so T1's reports
# and dashboard can ingest the hygiene summary as Check records with attributes. test-hygiene.ps1 enforces the rule.
$script:TelemetryName = [regex]::new('^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$')
function Test-TelemetryName([string]$Name) { $Name.Length -ge 1 -and $Name.Length -le 128 -and $script:TelemetryName.IsMatch($Name) }

function New-HygieneResult([string]$Check) {
    [pscustomobject]@{
        Check    = $Check
        Findings = [Collections.Generic.List[object]]::new()
        Metrics  = [ordered]@{}
        Summary  = ''
    }
}

# hygiene.<check> with '-' as '_' (hygiene.contract_label).
function Get-RecordName($Result) { 'hygiene.' + $Result.Check.Replace('-', '_') }

function Add-Finding($Result, [ValidateSet('error', 'warning', 'note')][string]$Severity, [string]$Message) {
    $Result.Findings.Add([pscustomobject]@{ Severity = $Severity; Message = $Message })
}

function Get-FindingCount($Result, [string]$Severity) { @($Result.Findings | Where-Object Severity -EQ $Severity).Count }

# One line for people: the result's Summary, else its metrics.
function Get-ResultText($Result) {
    if ($Result.Summary) { return $Result.Summary }
    @($Result.Metrics.GetEnumerator() | ForEach-Object { "$($_.Key) $($_.Value)" }) -join '; '
}

function Write-HygieneResult($Result) {
    foreach ($f in $Result.Findings) {
        $color = switch ($f.Severity) { 'error' { 'Red' } 'warning' { 'Yellow' } default { 'Gray' } }
        Write-Host "$($f.Severity): $($f.Message)" -ForegroundColor $color
    }
    $errors = Get-FindingCount $Result 'error'
    $warnings = Get-FindingCount $Result 'warning'
    $text = Get-ResultText $Result
    if ($errors -gt 0) { Write-Host "$($Result.Check): FAILED ($errors error(s), $warnings warning(s)): $text" -ForegroundColor Red }
    else { Write-Host "$($Result.Check): OK ($warnings warning(s)): $text" -ForegroundColor Green }
}

# The record a check contributes to artifacts/hygiene/summary.json (schema lrm.hygiene-summary/1): a Check record in
# the TelemetryFacade sense (name, passed) with the metrics as its attributes.
function ConvertTo-HygieneRecord($Result, [double]$DurationMs = 0) {
    [ordered]@{
        name       = Get-RecordName $Result
        passed     = (Get-FindingCount $Result 'error') -eq 0
        errors     = Get-FindingCount $Result 'error'
        warnings   = Get-FindingCount $Result 'warning'
        durationMs = [math]::Round($DurationMs, 1)
        summary    = Get-ResultText $Result
        metrics    = $Result.Metrics
        findings   = @($Result.Findings | ForEach-Object { [ordered]@{ severity = $_.Severity; message = $_.Message } })
    }
}

# Shared ending of the check-*.ps1 scripts: print, optionally write the record, return the exit code.
function Complete-HygieneCheck($Result, [string]$ResultPath, [double]$DurationMs = 0) {
    Write-HygieneResult $Result
    if ($ResultPath) {
        $dir = Split-Path -Parent $ResultPath
        if ($dir) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
        [IO.File]::WriteAllText($ResultPath, ((ConvertTo-HygieneRecord $Result $DurationMs) | ConvertTo-Json -Depth 6) + "`n", [Text.UTF8Encoding]::new($false))
    }
    if ((Get-FindingCount $Result 'error') -gt 0) { 1 } else { 0 }
}

# An optional property of a JSON object, or $Default (StrictMode forbids reading a missing property).
function Get-Prop($Object, [string]$Name, $Default = $null) {
    if ($null -ne $Object -and $Object.PSObject.Properties[$Name]) { $Object.$Name } else { $Default }
}

# ---- globs --------------------------------------------------------------------------------------

<#
    Glob syntax of docs/ownership.json and tools/ci/hygiene.json: '**/' matches zero or more whole segments, a '**'
    elsewhere matches anything, '*' matches within one segment and '?' one character other than '/'. Everything else is
    literal; brackets, braces and backslashes are rejected so that a glob never silently means something else.
#>
function ConvertTo-GlobRegex([string]$Glob) {
    if (-not $Glob) { throw 'empty glob' }
    if ($Glob -match '[\[\]{}\\]') { throw "glob '$Glob': only *, ** and ? are special; brackets, braces and backslashes are not supported" }
    $sb = [Text.StringBuilder]::new('^')
    $i = 0
    while ($i -lt $Glob.Length) {
        $c = $Glob[$i]
        if ($c -eq '*') {
            if ($i + 1 -lt $Glob.Length -and $Glob[$i + 1] -eq '*') {
                if ($i + 2 -lt $Glob.Length -and $Glob[$i + 2] -eq '/') { [void]$sb.Append('(?:[^/]+/)*'); $i += 3; continue }
                [void]$sb.Append('.*'); $i += 2; continue
            }
            [void]$sb.Append('[^/]*'); $i++; continue
        }
        if ($c -eq '?') { [void]$sb.Append('[^/]'); $i++; continue }
        [void]$sb.Append([regex]::Escape([string]$c)); $i++
    }
    [void]$sb.Append('$')
    $sb.ToString()
}

function Test-Glob([string]$Path, [string]$Glob) {
    $re = $null
    if (-not $script:GlobCache.TryGetValue($Glob, [ref]$re)) {
        $re = [regex]::new((ConvertTo-GlobRegex $Glob), [Text.RegularExpressions.RegexOptions]::CultureInvariant)
        $script:GlobCache[$Glob] = $re
    }
    $re.IsMatch($Path)
}

# Include globs, minus '!' excludes, as one predicate.
function Split-Globs([object[]]$Globs) {
    $inc = [Collections.Generic.List[string]]::new()
    $exc = [Collections.Generic.List[string]]::new()
    foreach ($g in $Globs) {
        $s = [string]$g
        if ($s.StartsWith('!')) { $exc.Add($s.Substring(1)) } else { $inc.Add($s) }
    }
    foreach ($g in @($inc) + @($exc)) { [void](ConvertTo-GlobRegex $g) }
    [pscustomobject]@{ Include = @($inc); Exclude = @($exc) }
}

function Test-GlobSet([string]$Path, $Set) {
    $hit = $false
    foreach ($g in $Set.Include) { if (Test-Glob $Path $g) { $hit = $true; break } }
    if (-not $hit) { return $false }
    foreach ($g in $Set.Exclude) { if (Test-Glob $Path $g) { return $false } }
    $true
}

# ---- docs/ownership.json ------------------------------------------------------------------------

function Read-OwnershipMap([string]$Path) {
    $j = Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
    if ((Get-Prop $j 'schema') -cne 'lrm.ownership/1') { throw "${Path}: unknown schema '$(Get-Prop $j 'schema')'" }
    $families = [Collections.Generic.Dictionary[string, object]]::new($script:Ordinal)
    $slices = [Collections.Generic.Dictionary[string, string]]::new($script:Ordinal)
    foreach ($p in $j.families.PSObject.Properties) {
        $set = Split-Globs @($p.Value.paths)
        $families[$p.Name] = [pscustomobject]@{ Name = $p.Name; Slices = @($p.Value.slices); Paths = $set }
        foreach ($s in @($p.Value.slices)) {
            if ($slices.ContainsKey($s)) { throw "${Path}: slice '$s' is listed under both $($slices[$s]) and $($p.Name)" }
            $slices[$s] = $p.Name
        }
    }
    $shared = @(foreach ($s in @(Get-Prop $j 'shared' @())) {
            $mode = [string](Get-Prop $s 'mode')
            if ($mode -cne 'append') { throw "${Path}: shared path '$($s.path)' has mode '$mode'; only 'append' is defined" }
            [void](ConvertTo-GlobRegex $s.path)
            [pscustomobject]@{ Path = [string]$s.path; Mode = $mode; Why = [string](Get-Prop $s 'why') }
        })
    $c = Get-Prop $j 'contract'
    if (-not $c) { throw "${Path}: no 'contract' section (label, exceptionLabel, freezeTag, files)" }
    $contract = [pscustomobject]@{
        Label          = [string]$c.label
        ExceptionLabel = [string]$c.exceptionLabel
        FreezeTag      = [string]$c.freezeTag
        Files          = Split-Globs @($c.files)
    }
    [pscustomobject]@{ Families = $families; Slices = $slices; Shared = $shared; Contract = $contract }
}

# The families that own a path. A healthy map gives exactly one.
function Get-PathOwners($Map, [string]$Path) {
    @(foreach ($f in $Map.Families.Values) { if (Test-GlobSet $Path $f.Paths) { $f.Name } })
}

function Get-SharedRule($Map, [string]$Path) {
    foreach ($s in $Map.Shared) { if (Test-Glob $Path $s.Path) { return $s } }
    $null
}

<#
    Branch names (OwnershipAndContractChange): ws/<id>-<slug>, ws/<id>-fix-<slug> and int/<slug>. <id> is a slice of
    docs/ownership.json, matched case-sensitively; <slug> is lowercase letters and digits joined by '-' (int/ also
    allows '.'). main is accepted and checked for map coverage only.
#>
function Resolve-Branch($Map, [string]$Branch) {
    if ($Branch -ceq 'main') { return [pscustomobject]@{ Kind = 'main'; Id = $null; Family = $null; Fix = $false } }
    if ($Branch -cmatch '^int/[a-z0-9]+(?:[-.][a-z0-9]+)*$') {
        return [pscustomobject]@{ Kind = 'integrator'; Id = $null; Family = $null; Fix = $false }
    }
    if ($Branch -cmatch '^ws/([A-Za-z0-9]+)-(fix-)?[a-z0-9]+(?:-[a-z0-9]+)*$') {
        $id = $Matches[1]
        $fix = [bool]$Matches[2]
        $family = $null
        if ($Map.Slices.TryGetValue($id, [ref]$family)) {
            return [pscustomobject]@{ Kind = 'workstream'; Id = $id; Family = $family; Fix = $fix }
        }
        return [pscustomobject]@{ Kind = 'unknown-id'; Id = $id; Family = $null; Fix = $fix }
    }
    [pscustomobject]@{ Kind = 'invalid'; Id = $null; Family = $null; Fix = $false }
}

# A change: Status (A, M, D, T), Path, Added and Deleted line counts ($null for a binary diff).
function New-Change([string]$Status, [string]$Path, $Added = 0, $Deleted = 0) {
    [pscustomobject]@{ Status = $Status; Path = $Path; Added = $Added; Deleted = $Deleted }
}

function ConvertTo-Change($Object) {
    New-Change ([string](Get-Prop $Object 'status' 'M')) ([string]$Object.path) (Get-Prop $Object 'added' 1) (Get-Prop $Object 'deleted' 0)
}

<#
    Ownership: (1) every tracked file is owned by exactly one family; (2) on a ws/ branch, every changed path is owned
    by the branch's family, or is a shared append-only path that the change only appends to. The exception label lets
    the integrator accept a reviewed out-of-path edit: the paths are then listed as warnings instead of errors.
#>
function Test-Ownership {
    param(
        [Parameter(Mandatory)] $Map,
        [string[]]$Tracked = @(),
        [string]$Branch,
        [object[]]$Changes = @(),
        [string[]]$Labels = @(),
        [bool]$LabelsKnown = $false
    )
    $r = New-HygieneResult 'ownership'

    $unowned = 0
    $multi = 0
    foreach ($p in $Tracked) {
        $owners = @(Get-PathOwners $Map $p)
        if ($owners.Count -eq 0) { $unowned++; Add-Finding $r error "$p is owned by no family in docs/ownership.json" }
        elseif ($owners.Count -gt 1) { $multi++; Add-Finding $r error "$p is owned by several families ($($owners -join ', ')); globs must not overlap" }
    }
    $r.Metrics['tracked'] = $Tracked.Count
    $r.Metrics['unowned'] = $unowned
    $r.Metrics['multi_owned'] = $multi
    $coverage = if ($unowned -or $multi) { "$($Tracked.Count) files, $unowned unowned, $multi owned twice" } else { "$($Tracked.Count) files, each owned by one family" }
    $r.Summary = $coverage

    if (-not $Branch) {
        Add-Finding $r note 'no branch given: only map coverage was checked'
        return $r
    }
    $b = Resolve-Branch $Map $Branch
    $r.Metrics['branch'] = $Branch
    $r.Metrics['changed'] = $Changes.Count
    $r.Summary = "$coverage; branch $Branch ($($b.Kind))"
    switch ($b.Kind) {
        'invalid' {
            Add-Finding $r error "branch '$Branch' does not follow ws/<id>-<slug>, ws/<id>-fix-<slug> or int/<slug>"
            return $r
        }
        'unknown-id' {
            Add-Finding $r error "branch '$Branch' names workstream '$($b.Id)', which is no slice in docs/ownership.json"
            return $r
        }
        'main' {
            Add-Finding $r note 'main: no branch diff is checked; changes reach main only through reviewed PRs'
            return $r
        }
        'integrator' {
            Add-Finding $r note "int/ branch: the integrator may touch any path ($($Changes.Count) changed)"
            return $r
        }
    }

    $family = $b.Family
    $r.Metrics['family'] = $family
    $hasException = $LabelsKnown -and ($Labels -ccontains $Map.Contract.ExceptionLabel)
    $outside = 0
    foreach ($c in $Changes) {
        $owners = @(Get-PathOwners $Map $c.Path)
        if ($owners -ccontains $family) { continue }
        $shared = Get-SharedRule $Map $c.Path
        if ($shared) {
            if ($c.Status -cin @('A', 'M') -and $null -ne $c.Deleted -and [int]$c.Deleted -eq 0) {
                Add-Finding $r note "$($c.Path): shared append-only path; this change only adds lines"
                continue
            }
            $what = if ($c.Status -ceq 'D') { 'deletes the file' } elseif ($null -eq $c.Deleted) { 'is a binary change' } else { "deletes or rewrites $($c.Deleted) line(s)" }
            Add-Finding $r error "$($c.Path) is shared append-only ($($shared.Why)); a family $family branch may only add lines, but this change $what"
            $outside++
            continue
        }
        $owner = if ($owners.Count) { "family $($owners -join ', ')" } else { 'no family' }
        $outside++
        if ($hasException) {
            Add-Finding $r warning "$($c.Path) is owned by $owner, not $family; accepted by the '$($Map.Contract.ExceptionLabel)' label"
        }
        else {
            Add-Finding $r error "$($c.Path) is owned by $owner, not by family $family (branch $Branch, status $($c.Status))"
        }
    }
    $r.Metrics['outside'] = $outside
    $r.Summary = "$coverage; branch $Branch (family $family): $($Changes.Count) changed, $outside outside the family"
    $r
}

# ---- layering -----------------------------------------------------------------------------------

<#
    Layering is the ProjectGraph contract's own validator, tools/ci/gen-project-graph.ps1 -Check, run against a repo
    root. It checks the rules in docs/project-graph.json, the fsproj shape (file items only, so no hand-written
    ProjectReference), registration of every fsproj and that the generated files are up to date.
#>
function Test-Layering {
    param([Parameter(Mandatory)][string]$RepoRoot, [string]$Generator = (Join-Path $PSScriptRoot 'gen-project-graph.ps1'))
    $r = New-HygieneResult 'layering'
    $global:LASTEXITCODE = 0
    $lines = @()
    try {
        $lines = @(& { & $Generator -Check -RepoRoot $RepoRoot } *>&1 | ForEach-Object { "$_" })
        $code = $global:LASTEXITCODE
    }
    catch {
        $lines += "error: $($_.Exception.Message)"
        $code = 1
    }
    foreach ($l in $lines) {
        if ($l -match '^error: (.*)$') { Add-Finding $r error $Matches[1] }
        elseif ($l -match '^project graph: (\d+) projects') { $r.Metrics['projects'] = [int]$Matches[1] }
    }
    if ($code -ne 0 -and (Get-FindingCount $r 'error') -eq 0) {
        Add-Finding $r error "gen-project-graph.ps1 -Check exited $code`: $($lines -join ' | ')"
    }
    $r.Metrics['rule_errors'] = Get-FindingCount $r 'error'
    $r.Summary = if ($r.Metrics.Contains('projects')) { "$($r.Metrics['projects']) projects, every rule holds, generated files up to date" } else { "$(Get-FindingCount $r 'error') error(s) in the project graph" }
    $r
}

# ---- sizes and forbidden files ------------------------------------------------------------------

function Read-HygieneConfig([string]$Path) {
    $j = Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
    if ((Get-Prop $j 'schema') -cne 'lrm.hygiene/1') { throw "${Path}: unknown schema '$(Get-Prop $j 'schema')'" }
    $s = $j.sizes
    [pscustomobject]@{
        DefaultMaxFileBytes = [long]$s.defaultMaxFileBytes
        FileLimits          = @(foreach ($f in @($s.fileLimits)) { [void](ConvertTo-GlobRegex $f.glob); [pscustomobject]@{ Glob = [string]$f.glob; Max = [long]$f.maxFileBytes; Why = [string]$f.why } })
        Budgets             = @(foreach ($b in @($s.budgets)) {
                [void](ConvertTo-GlobRegex $b.glob)
                $name = [string](Get-Prop $b 'name')
                if ($name -cnotmatch '^[a-z][a-z0-9_]*$') { throw "${Path}: budget '$($b.glob)' needs a name of a-z, 0-9 and _ (a telemetry key segment)" }
                [pscustomobject]@{ Name = $name; Glob = [string]$b.glob; Max = [long]$b.maxTotalBytes; Why = [string]$b.why }
            })
        Forbidden           = @(foreach ($f in @($s.forbidden)) {
                [pscustomobject]@{ Glob = [string]$f.glob; Set = (Split-Globs (@([string]$f.glob) + @(Get-Prop $f 'except' @() | ForEach-Object { "!$_" }))); Why = [string]$f.why }
            })
    }
}

# Byte counts for messages, formatted with the invariant culture (hygiene output must not depend on the host).
function Format-Bytes([long]$n) {
    $inv = [Globalization.CultureInfo]::InvariantCulture
    if ($n -ge 1MB) { [string]::Format($inv, '{0:0.00} MiB', $n / 1MB) }
    elseif ($n -ge 1KB) { [string]::Format($inv, '{0:0.0} KiB', $n / 1KB) }
    else { "$n B" }
}

<#
    Sizes: every tracked blob is at most its limit (the first fileLimits glob that matches, else the default); every
    budget's matching blobs together stay under its total; no forbidden file is tracked; no submodule exists.
    Entries: Path, Size (bytes, as committed) and Type (blob, commit).
#>
function Test-Sizes {
    param([Parameter(Mandatory)] $Config, [object[]]$Entries = @())
    $r = New-HygieneResult 'sizes'
    $total = [long]0
    $largest = $null
    $totals = @{}
    foreach ($b in $Config.Budgets) { $totals[$b.Glob] = [long]0 }
    foreach ($e in $Entries) {
        $path = [string]$e.Path
        if ([string](Get-Prop $e 'Type' 'blob') -ceq 'commit') { Add-Finding $r error "$path is a submodule; LRM has none (everything large lives under LRM_DATA)"; continue }
        $size = [long]$e.Size
        $total += $size
        if ($null -eq $largest -or $size -gt $largest.Size) { $largest = [pscustomobject]@{ Path = $path; Size = $size } }
        foreach ($f in $Config.Forbidden) {
            if (Test-GlobSet $path $f.Set) { Add-Finding $r error "$path must not be committed: $($f.Why) (rule $($f.Glob))"; break }
        }
        $limit = $Config.DefaultMaxFileBytes
        $why = 'the default per-file limit'
        foreach ($f in $Config.FileLimits) { if (Test-Glob $path $f.Glob) { $limit = $f.Max; $why = $f.Why; break } }
        if ($size -gt $limit) { Add-Finding $r error "$path is $(Format-Bytes $size) ($size bytes), over the $(Format-Bytes $limit) limit: $why" }
        foreach ($b in $Config.Budgets) { if (Test-Glob $path $b.Glob) { $totals[$b.Glob] += $size } }
    }
    $r.Metrics['files'] = $Entries.Count
    $r.Metrics['total_bytes'] = $total
    if ($largest) { $r.Metrics['largest_path'] = $largest.Path; $r.Metrics['largest_bytes'] = $largest.Size }
    $budgetText = @()
    foreach ($b in $Config.Budgets) {
        $t = $totals[$b.Glob]
        $pct = if ($b.Max -gt 0) { [string]::Format([Globalization.CultureInfo]::InvariantCulture, '{0:0.0}', 100.0 * $t / $b.Max) } else { '0' }
        $r.Metrics["budget.$($b.Name).bytes"] = $t
        $r.Metrics["budget.$($b.Name).max_bytes"] = $b.Max
        $budgetText += "$($b.Glob) $(Format-Bytes $t) of $(Format-Bytes $b.Max) ($pct%)"
        if ($t -gt $b.Max) { Add-Finding $r error "files under $($b.Glob) total $(Format-Bytes $t) ($t bytes), over the $(Format-Bytes $b.Max) budget: $($b.Why)" }
    }
    $largestText = if ($largest) { "; largest $($largest.Path) ($(Format-Bytes $largest.Size))" } else { '' }
    $r.Summary = "$($Entries.Count) files, $(Format-Bytes $total)$largestText; budgets: $($budgetText -join ', ')"
    $r
}

# ---- line endings -------------------------------------------------------------------------------

<#
    EOL: entries of `git ls-files --eol` (Index, Worktree, Attr, Path). The index holds LF only: i/crlf and i/mixed
    fail. Every file has a text attribute from .gitattributes. Binary content (i/-text) must be declared binary.
    CRLF in the working tree alone (a stale checkout) is a warning, except where the attributes ask for CRLF.
#>
function Test-Eol {
    param([object[]]$Entries = @())
    $r = New-HygieneResult 'eol'
    $counts = [ordered]@{ lf = 0; none = 0; binary = 0; crlf_checkout = 0 }
    foreach ($e in $Entries) {
        $i = [string]$e.Index; $w = [string]$e.Worktree; $a = ([string]$e.Attr).Trim(); $p = [string]$e.Path
        $binaryAttr = $a -cmatch '(^|\s)-text(\s|$)'
        $crlfAttr = $a -cmatch '(^|\s)eol=crlf(\s|$)'
        if ($i -cin @('crlf', 'mixed')) {
            Add-Finding $r error "$p has $i line endings in the index (attr '$a'); the repository stores LF only. Fix: git add --renormalize $p"
        }
        if (-not $a) { Add-Finding $r error "$p has no text attribute; .gitattributes must cover every file (* text=auto eol=lf)" }
        if ($i -ceq '-text') {
            $counts.binary++
            if (-not $binaryAttr) { Add-Finding $r error "$p holds binary content but .gitattributes does not mark it binary; add a rule such as '*.$([IO.Path]::GetExtension($p).TrimStart('.')) binary'" }
        }
        elseif ($i -ceq 'lf') { $counts.lf++ }
        elseif ($i -ceq 'none' -or $i -ceq '') { $counts.none++ }
        if ($crlfAttr) { $counts.crlf_checkout++ }
        elseif ($w -cin @('crlf', 'mixed') -and -not $binaryAttr) {
            Add-Finding $r warning "$p has $w line endings in the working tree only (the index is LF); check it out again or run git add --renormalize $p"
        }
    }
    $r.Metrics['files'] = $Entries.Count
    foreach ($k in $counts.Keys) { $r.Metrics[$k] = $counts[$k] }
    $r.Summary = "$($Entries.Count) files: $($counts.lf) LF, $($counts.none) without line endings, $($counts.binary) binary, $($counts.crlf_checkout) CRLF by attribute"
    $r
}

function ConvertFrom-EolListing([string[]]$Lines) {
    foreach ($l in $Lines) {
        if (-not $l) { continue }
        $tab = $l.IndexOf("`t")
        if ($tab -lt 0) { throw "not a git ls-files --eol line: '$l'" }
        $info = $l.Substring(0, $tab)
        if ($info -notmatch '^i/(\S*)\s+w/(\S*)\s+attr/(.*)$') { throw "not a git ls-files --eol line: '$l'" }
        [pscustomobject]@{ Index = $Matches[1]; Worktree = $Matches[2]; Attr = $Matches[3].Trim(); Path = $l.Substring($tab + 1) }
    }
}

# ---- contract-change label ----------------------------------------------------------------------

function Test-ContractFile($Map, [string]$Path) { Test-GlobSet $Path $Map.Contract.Files }

<#
    Contract label. After the freeze tag (contracts-v1) exists, a PR that changes a contract file needs the
    contract-change label. A labelled PR must change a contract file, must add lines to docs/contracts/CHANGELOG.md,
    and every contract text it touches (docs/contracts/<name>.md) must be named in those added lines.
#>
function Test-ContractLabel {
    param(
        [Parameter(Mandatory)] $Map,
        [object[]]$Changes = @(),
        [string[]]$Labels = @(),
        [bool]$LabelsKnown = $false,
        [bool]$Frozen = $false,
        [string[]]$ChangelogAdded = @()
    )
    $r = New-HygieneResult 'contract-label'
    $label = $Map.Contract.Label
    $tag = $Map.Contract.FreezeTag
    $touched = @($Changes | Where-Object { Test-ContractFile $Map $_.Path })
    $texts = @($touched | Where-Object { $_.Path -cmatch '^docs/contracts/[^/]+\.md$' })
    $r.Metrics['frozen'] = $Frozen
    $r.Metrics['contract_files_changed'] = $touched.Count
    $r.Metrics['labels_known'] = $LabelsKnown
    $state = if ($Frozen) { "$tag exists" } else { "$tag not tagged yet" }
    $r.Summary = "$state; $($touched.Count) contract file(s) changed; labels $(if ($LabelsKnown) { 'known' } else { 'unknown' })"
    if (-not $LabelsKnown) {
        if ($touched.Count -and $Frozen) {
            Add-Finding $r warning "PR labels are unknown (not a pull request), so the '$label' label was not checked for: $(@($touched.Path) -join ', ')"
        }
        elseif ($touched.Count) { Add-Finding $r note "$($touched.Count) contract file(s) changed; $tag does not exist yet, so no label is needed" }
        return $r
    }
    $hasLabel = $Labels -ccontains $label
    $r.Metrics['labelled'] = $hasLabel
    $r.Summary = "$state; $($touched.Count) contract file(s) changed; '$label' label $(if ($hasLabel) { 'set' } else { 'not set' })"
    if ($Frozen -and $touched.Count -and -not $hasLabel) {
        Add-Finding $r error "contract files changed after $tag without the '$label' label: $(@($touched.Path) -join ', ')"
    }
    if (-not $Frozen -and $touched.Count) { Add-Finding $r note "$($touched.Count) contract file(s) changed; $tag does not exist yet, so no label is needed" }
    if ($hasLabel) {
        if (-not $touched.Count) { Add-Finding $r error "the '$label' label is set, but no contract file changed; remove the label" }
        $log = @($Changes | Where-Object { $_.Path -ceq $script:Changelog -and $_.Status -cne 'D' -and ($null -eq $_.Added -or [int]$_.Added -gt 0) })
        if (-not $log.Count -or -not $ChangelogAdded.Count) {
            Add-Finding $r error "a '$label' PR needs an entry in $($script:Changelog), but the PR adds no line to it"
        }
        else {
            $added = $ChangelogAdded -join "`n"
            foreach ($t in $texts) {
                $name = [IO.Path]::GetFileName($t.Path)
                if (-not $added.Contains($name)) { Add-Finding $r error "$($script:Changelog): no added line names $name, which this PR changes" }
            }
        }
    }
    $r
}

# ---- git inputs ---------------------------------------------------------------------------------

function Invoke-Git {
    param([Parameter(Mandatory)][string]$RepoRoot, [Parameter(Mandatory)][string[]]$Arguments, [switch]$AllowFailure)
    $psi = [Diagnostics.ProcessStartInfo]::new('git')
    foreach ($a in @('-C', $RepoRoot) + $Arguments) { $psi.ArgumentList.Add($a) }
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.StandardOutputEncoding = [Text.UTF8Encoding]::new($false)
    $psi.StandardErrorEncoding = [Text.UTF8Encoding]::new($false)
    $p = [Diagnostics.Process]::Start($psi)
    $out = $p.StandardOutput.ReadToEndAsync()
    $err = $p.StandardError.ReadToEnd()
    $p.WaitForExit()
    $res = [pscustomobject]@{ ExitCode = $p.ExitCode; Out = $out.Result; Err = $err.Trim() }
    if ($res.ExitCode -ne 0 -and -not $AllowFailure) { throw "git $($Arguments -join ' ') exited $($res.ExitCode): $($res.Err)" }
    $res
}

function Split-Nul([string]$Text) { @($Text.Split([char]0) | Where-Object { $_ -ne '' }) }

# Blobs of a commit's tree: Path, Size, Type.
function Get-GitTreeEntries([string]$RepoRoot, [string]$Rev = 'HEAD') {
    foreach ($rec in Split-Nul (Invoke-Git $RepoRoot @('ls-tree', '-r', '-l', '-z', $Rev)).Out) {
        $tab = $rec.IndexOf("`t")
        $meta = $rec.Substring(0, $tab) -split '\s+'
        $size = if ($meta[3] -eq '-') { 0 } else { [long]$meta[3] }
        [pscustomobject]@{ Path = $rec.Substring($tab + 1); Size = $size; Type = $meta[1] }
    }
}

function Get-GitEolEntries([string]$RepoRoot) {
    ConvertFrom-EolListing (Split-Nul (Invoke-Git $RepoRoot @('ls-files', '--eol', '-z')).Out)
}

function Get-GitMergeBase([string]$RepoRoot, [string]$Base, [string]$Head) {
    (Invoke-Git $RepoRoot @('merge-base', $Base, $Head)).Out.Trim()
}

# Changes from the merge base of Base and Head to Head. Renames count as a deletion plus an addition.
function Get-GitChanges([string]$RepoRoot, [string]$Base, [string]$Head) {
    $range = "$Base...$Head"
    $status = [Collections.Generic.Dictionary[string, string]]::new($script:Ordinal)
    $tokens = @(Split-Nul (Invoke-Git $RepoRoot @('diff', '--no-renames', '-z', '--name-status', $range)).Out)
    for ($k = 0; $k + 1 -lt $tokens.Count; $k += 2) { $status[$tokens[$k + 1]] = $tokens[$k].Substring(0, 1) }
    $counts = [Collections.Generic.Dictionary[string, object]]::new($script:Ordinal)
    foreach ($rec in Split-Nul (Invoke-Git $RepoRoot @('diff', '--no-renames', '-z', '--numstat', $range)).Out) {
        $parts = $rec.Split("`t", 3)
        $counts[$parts[2]] = if ($parts[0] -eq '-') { , @($null, $null) } else { , @([int]$parts[0], [int]$parts[1]) }
    }
    foreach ($path in $status.Keys) {
        $n = $null
        $added = 0; $deleted = 0
        if ($counts.TryGetValue($path, [ref]$n)) { $added = $n[0]; $deleted = $n[1] }
        New-Change $status[$path] $path $added $deleted
    }
}

function Get-GitAddedLines([string]$RepoRoot, [string]$Base, [string]$Head, [string]$Path) {
    $diff = (Invoke-Git $RepoRoot @('diff', '--no-renames', '-U0', "$Base...$Head", '--', $Path)).Out
    @($diff -split "`n" | Where-Object { $_.StartsWith('+') -and -not $_.StartsWith('+++') } | ForEach-Object { $_.Substring(1).TrimEnd("`r") })
}

function Test-GitTag([string]$RepoRoot, [string]$Tag) {
    (Invoke-Git $RepoRoot @('rev-parse', '-q', '--verify', "refs/tags/$Tag") -AllowFailure).ExitCode -eq 0
}

function Get-GitBranch([string]$RepoRoot) {
    $b = (Invoke-Git $RepoRoot @('rev-parse', '--abbrev-ref', 'HEAD')).Out.Trim()
    if ($b -eq 'HEAD') { $null } else { $b }
}

# Labels as a list of names from a comma-separated or repeated -Labels argument.
function Split-Labels([string[]]$Labels) {
    @($Labels | ForEach-Object { "$_" -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

Export-ModuleMember -Function *
