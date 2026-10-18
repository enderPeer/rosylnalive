namespace Lrm.Gpu.Core

open System
open System.Text
open System.Text.RegularExpressions

/// The kernel source convention (ShaderBuildAndSourceConvention contract), checked by the shader build before DXC
/// runs. One kernel per file at its catalog path: the first include is "lrm.hlsli"; arguments come only from
/// LRM_ARGS(<kernel>) and resources only from LRM_BUF, lrm_buf and lrm_tensor; LRM_NUMTHREADS(<kernel>) gives
/// numthreads; one entry point, void main(...). DXC enforces the rest (groupshared at most 32 KiB, -WX).
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

    /// Identifiers a kernel must not use, with the reason.
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
        ]

    /// Identifier prefixes a kernel must not use, with the reason.
    let forbiddenPrefixes =
        [
            "Wave", "wave ops are not allowed"
            "Quad", "quad ops are not allowed"
            "Interlocked", "atomics are not allowed"
            "Texture", "typed views are not allowed"
            "RWTexture", "typed views are not allowed"
        ]

    let private identifier =
        Regex(@"[A-Za-z_][A-Za-z0-9_]*", RegexOptions.CultureInvariant)

    let private includeLine =
        Regex("^#\\s*include\\s+\"([^\"]+)\"$", RegexOptions.CultureInvariant)

    /// The convention violations of one kernel source; empty when it is clean.
    let lint (k: KernelSpec) (text: string) : string list =
        let code = stripComments text
        let p = ResizeArray<string>()
        let lines = code.Split('\n') |> Array.map _.Trim()
        let includes = ResizeArray<string>()

        for t in lines do
            if t.StartsWith "#" then
                let directive = t.Substring(1).TrimStart()

                if directive.StartsWith "include" then
                    let m = includeLine.Match t

                    if not m.Success then
                        p.Add(sprintf "unsupported include: %s" t)
                    else
                        let name = m.Groups[1].Value
                        includes.Add name

                        if
                            name <> "lrm.hlsli"
                            && (name.Contains "/" || name.Contains "\\" || not (name.EndsWith ".hlsli"))
                        then
                            p.Add(sprintf "include \"%s\": only lrm.hlsli and .hlsli files beside the kernel" name)
                elif directive.StartsWith "pragma" then
                    p.Add "#pragma is not allowed (flags come from the build)"
                elif directive.StartsWith "undef" then
                    p.Add "#undef is not allowed"
                elif Regex.IsMatch(directive, @"^define\s+(LRM_|lrm_)") then
                    p.Add "macros named LRM_* or lrm_* belong to lrm.hlsli"

        if includes.Count = 0 || includes[0] <> "lrm.hlsli" then
            p.Add "the first include must be \"lrm.hlsli\""

        if code.Contains "[[vk::" then
            p.Add "[[vk::...]] attributes belong to lrm.hlsli"

        let count (pattern: string) =
            Regex.Matches(code, pattern, RegexOptions.CultureInvariant).Count

        let name = Regex.Escape k.Name

        if
            count @"\bLRM_ARGS\s*\(" <> 1
            || count (@"\bLRM_ARGS\s*\(\s*" + name + @"\s*\)") <> 1
        then
            p.Add(sprintf "the source must use LRM_ARGS(%s) exactly once, and no other LRM_ARGS" k.Name)

        if
            count @"\bLRM_NUMTHREADS\s*\(" <> 1
            || count (@"\bLRM_NUMTHREADS\s*\(\s*" + name + @"\s*\)") <> 1
        then
            p.Add(sprintf "the source must use LRM_NUMTHREADS(%s) exactly once, and no other LRM_NUMTHREADS" k.Name)

        if count @"\bvoid\s+main\s*\(" <> 1 then
            p.Add "the source must define exactly one entry point, void main(...)"

        let ids = identifier.Matches code |> Seq.map _.Value |> Set.ofSeq

        for word, why in forbidden do
            if ids.Contains word then
                p.Add(sprintf "'%s' is not allowed: %s" word why)

        for prefix, why in forbiddenPrefixes do
            for id in ids do
                if id.StartsWith(prefix, StringComparison.Ordinal) && id.Length > prefix.Length then
                    p.Add(sprintf "'%s' is not allowed: %s" id why)

        List.ofSeq p |> List.distinct
