namespace Lrm.Gpu.Core

open System
open System.Text
open System.Text.RegularExpressions

/// The kernel source convention (ShaderBuildAndSourceConvention contract), checked by the shader build before DXC
/// runs. One kernel per file at its catalog path: the first include is "lrm.hlsli", the others are helper headers
/// beside the kernel; arguments come only from LRM_ARGS(<kernel>) and resources only from LRM_BUF, lrm_buf and
/// lrm_tensor; LRM_NUMTHREADS(<kernel>) gives numthreads; one entry point, void main(...). Helper headers (.hlsli
/// under hlsl/) follow the same code rules and include nothing, so a kernel's include closure is its own source, the
/// helpers it names and include/. There is no conditional compilation, so both targets compile one body. DXC
/// enforces the rest (groupshared at most 32 KiB, -WX).
[<RequireQualifiedAccess>]
module SourceConvention =
    /// Removes // and /* */ comments, keeping line breaks, so the checks see only code.
    let stripComments (text: string) =
        let sb = StringBuilder(text.Length)
        let mutable i = 0

        while i < text.Length do
            if i + 1 < text.Length && text[i] = '/' && text[i + 1] = '/' then
                while i < text.Length && text[i] <> '\n' do
                    i <- i + 1
            elif i + 1 < text.Length && text[i] = '/' && text[i + 1] = '*' then
                i <- i + 2

                while i < text.Length
                      && not (i + 1 < text.Length && text[i] = '*' && text[i + 1] = '/') do
                    if text[i] = '\n' then
                        sb.Append '\n' |> ignore

                    i <- i + 1

                i <- i + 2
            else
                sb.Append text[i] |> ignore
                i <- i + 1

        sb.ToString()

    /// Identifiers a kernel or helper must not use, with the reason.
    let forbidden =
        [
            "ResourceDescriptorHeap", "use LRM_BUF or lrm_buf"
            "SamplerDescriptorHeap", "kernels use no samplers"
            "NonUniformResourceIndex", "every LRM_BUF index is dynamically uniform (it comes from the table)"
            "register", "bindings come from lrm.hlsli"
            "cbuffer", "arguments come from LRM_ARGS"
            "tbuffer", "arguments come from LRM_ARGS"
            "ConstantBuffer", "arguments come from LRM_ARGS"
            "ByteAddressBuffer", "use RWByteAddressBuffer through LRM_BUF"
            "StructuredBuffer", "typed views are not allowed"
            "RWStructuredBuffer", "typed views are not allowed"
            "AppendStructuredBuffer", "typed views are not allowed"
            "ConsumeStructuredBuffer", "typed views are not allowed"
            "Buffer", "typed views are not allowed"
            "RWBuffer", "typed views are not allowed"
            "SamplerState", "kernels use no samplers"
            "IncrementCounter", "counters are not allowed"
            "DecrementCounter", "counters are not allowed"
            "half", "16-bit types are not allowed"
            "min16float", "16-bit types are not allowed"
            "min10float", "16-bit types are not allowed"
            "min16int", "16-bit types are not allowed"
            "min12int", "16-bit types are not allowed"
            "min16uint", "16-bit types are not allowed"
            "float16_t", "16-bit types are not allowed"
            "int16_t", "16-bit types are not allowed"
            "uint16_t", "16-bit types are not allowed"
            "SV_DispatchThreadID", "use SV_GroupID with LRM_GROUP_LINEAR, so folded grids stay correct"
            "numthreads", "use LRM_NUMTHREADS(<kernel>)"
            "vk", "SPIR-V-only intrinsics are not allowed: both targets compile one body"
        ]

    /// Identifier prefixes a kernel or helper must not use, with the reason.
    let forbiddenPrefixes =
        [
            "Wave", "wave ops are not allowed"
            "Quad", "quad ops are not allowed"
            "Interlocked", "atomics are not allowed"
            "Texture", "typed views are not allowed"
            "RWTexture", "typed views are not allowed"
            "__", "compiler macros such as __spirv__ are not allowed: both targets compile one body"
        ]

    let private identifier =
        Regex(@"[A-Za-z_][A-Za-z0-9_]*", RegexOptions.CultureInvariant)

    let private includeLine =
        Regex("^#\\s*include\\s+\"([^\"]+)\"$", RegexOptions.CultureInvariant)

    let private directiveName = Regex(@"^([A-Za-z_]*)", RegexOptions.CultureInvariant)

    let private conditionals = set [ "if"; "ifdef"; "ifndef"; "elif"; "else"; "endif" ]

    /// The code lines (comments removed, trimmed) and the preprocessor directives among them, as (name, line).
    let private directives (code: string) =
        code.Split('\n')
        |> Array.map _.Trim()
        |> Array.filter (fun t -> t.StartsWith "#")
        |> Array.map (fun t -> directiveName.Match(t.Substring(1).TrimStart()).Groups[1].Value, t)
        |> List.ofArray

    /// The checks that kernels and helpers share: directives other than #include, attributes and identifiers.
    let private codeProblems (code: string) (p: ResizeArray<string>) =
        for name, line in directives code do
            match name with
            | "include" -> ()
            | "define" ->
                if Regex.IsMatch(line, @"^#\s*define\s+(LRM_|lrm_)", RegexOptions.CultureInvariant) then
                    p.Add "macros named LRM_* or lrm_* belong to lrm.hlsli"
            | "pragma" -> p.Add "#pragma is not allowed (flags come from the build)"
            | "undef" -> p.Add "#undef is not allowed"
            | c when conditionals.Contains c ->
                p.Add(sprintf "#%s is not allowed: no conditional compilation, both targets compile one body" c)
            | _ -> p.Add(sprintf "unsupported directive: %s (only #include and #define)" line)

        if code.Contains "[[vk::" then
            p.Add "[[vk::...]] attributes belong to lrm.hlsli"

        let ids = identifier.Matches code |> Seq.map _.Value |> Set.ofSeq

        for word, why in forbidden do
            if ids.Contains word then
                p.Add(sprintf "'%s' is not allowed: %s" word why)

        for prefix, why in forbiddenPrefixes do
            for id in ids do
                if id.StartsWith(prefix, StringComparison.Ordinal) && id.Length > prefix.Length then
                    p.Add(sprintf "'%s' is not allowed: %s" id why)

    let private count (code: string) (pattern: string) =
        Regex.Matches(code, pattern, RegexOptions.CultureInvariant).Count

    /// True for a helper header name a kernel may include: a plain file name ending in .hlsli that does not start
    /// with "lrm" (so it can never shadow lrm.hlsli or lrm_args.g.hlsli in DXC's include search).
    let isHelperName (name: string) =
        not (String.IsNullOrEmpty name)
        && not (name.Contains "/" || name.Contains "\\")
        && name.EndsWith(".hlsli", StringComparison.Ordinal)
        && name.Length > ".hlsli".Length
        && not (name.StartsWith("lrm", StringComparison.OrdinalIgnoreCase))

    /// The quoted includes of a source in order, comments removed; malformed include lines are left out (the lint
    /// reports them).
    let includes (text: string) : string list =
        directives (stripComments text)
        |> List.choose (fun (name, line) ->
            if name = "include" then
                let m = includeLine.Match line
                if m.Success then Some m.Groups[1].Value else None
            else
                None)

    /// The helper headers a kernel includes: every include after "lrm.hlsli", as written (names beside the kernel).
    let helpers (text: string) : string list =
        includes text |> List.filter ((<>) "lrm.hlsli") |> List.distinct

    /// The convention violations of one kernel source; empty when it is clean.
    let lint (k: KernelSpec) (text: string) : string list =
        let code = stripComments text
        let p = ResizeArray<string>()

        for name, line in directives code do
            if name = "include" && not (includeLine.Match line).Success then
                p.Add(sprintf "unsupported include: %s" line)

        let incs = includes text

        for name in incs do
            if name <> "lrm.hlsli" && not (isHelperName name) then
                p.Add(
                    sprintf
                        "include \"%s\": only lrm.hlsli and .hlsli helpers beside the kernel (a plain name, not starting with lrm)"
                        name
                )

        match incs with
        | "lrm.hlsli" :: rest when not (List.contains "lrm.hlsli" rest) -> ()
        | "lrm.hlsli" :: _ -> p.Add "\"lrm.hlsli\" is included more than once"
        | _ -> p.Add "the first include must be \"lrm.hlsli\""

        codeProblems code p
        let name = Regex.Escape k.Name

        if
            count code @"\bLRM_ARGS\s*\(" <> 1
            || count code (@"\bLRM_ARGS\s*\(\s*" + name + @"\s*\)") <> 1
        then
            p.Add(sprintf "the source must use LRM_ARGS(%s) exactly once, and no other LRM_ARGS" k.Name)

        if
            count code @"\bLRM_NUMTHREADS\s*\(" <> 1
            || count code (@"\bLRM_NUMTHREADS\s*\(\s*" + name + @"\s*\)") <> 1
        then
            p.Add(sprintf "the source must use LRM_NUMTHREADS(%s) exactly once, and no other LRM_NUMTHREADS" k.Name)

        if count code @"\bvoid\s+main\s*\(" <> 1 then
            p.Add "the source must define exactly one entry point, void main(...)"

        List.ofSeq p |> List.distinct

    /// The convention violations of one helper header (`fileName` is its name, for example "rows.hlsli"); empty when
    /// it is clean. A helper follows the kernels' code rules, includes nothing (the kernel includes lrm.hlsli first and
    /// then its helpers), and declares no arguments, numthreads or entry point.
    let lintHelper (fileName: string) (text: string) : string list =
        let code = stripComments text
        let p = ResizeArray<string>()

        if not (isHelperName fileName) then
            p.Add(sprintf "helper '%s': a helper is a .hlsli file whose name does not start with lrm" fileName)

        for name, _ in directives code do
            if name = "include" then
                p.Add "#include is not allowed in a helper: the kernel includes lrm.hlsli and each helper itself"

        codeProblems code p

        if count code @"\bLRM_ARGS\s*\(" > 0 then
            p.Add "LRM_ARGS belongs to the kernel source"

        if count code @"\bLRM_NUMTHREADS\s*\(" > 0 then
            p.Add "LRM_NUMTHREADS belongs to the kernel source"

        if count code @"\bmain\s*\(" > 0 then
            p.Add "the entry point main belongs to the kernel source"

        List.ofSeq p |> List.distinct
