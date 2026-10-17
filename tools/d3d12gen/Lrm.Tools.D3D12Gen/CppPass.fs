/// Reads the C++ declarations of the same headers: `MIDL_INTERFACE("iid") Name : public Base`. They give each
/// interface's direct base (interfaces without methods, such as ID3D12Pageable, never appear in the C vtables'
/// DECLSPEC_XFGVIRT markers) and a second copy of the IID, which must match DEFINE_GUID.
module Lrm.Tools.D3D12Gen.CppPass

open System
open System.Collections.Generic
open Lrm.Tools.D3D12Gen.Lexer

type CppInterface =
    {
        Name: string
        Base: string
        Iid: Guid
    }

let scan (tokens: Token[]) : IReadOnlyDictionary<string, CppInterface> =
    let out = Dictionary<string, CppInterface>(StringComparer.Ordinal)
    let mutable i = 0

    while i < tokens.Length do
        if tokens[i].Text = "MIDL_INTERFACE" && i + 8 < tokens.Length then
            // MIDL_INTERFACE ( "guid" ) Name : public Base
            let t k = tokens[i + k]

            if
                (t 1).Text = "("
                && (t 2).Kind = Str
                && (t 3).Text = ")"
                && (t 5).Text = ":"
                && (t 6).Text = "public"
            then
                let iid = Guid((t 2).Text.Trim('"'))
                let name = (t 4).Text
                let baseName = (t 7).Text

                if out.ContainsKey name then
                    failwithf "%s:%d: MIDL_INTERFACE %s appears twice" (t 4).File (t 4).Line name

                out[name] <-
                    {
                        Name = name
                        Base = baseName
                        Iid = iid
                    }

            i <- i + 8
        else
            i <- i + 1

    out
