/// Tokens of the preprocessed header text: identifiers, numbers, strings and punctuation. Comments are dropped.
module Lrm.Tools.D3D12Gen.Lexer

open System
open Lrm.Tools.D3D12Gen.Preprocessor

type TokenKind =
    | Ident
    | Number
    | Str
    | Punct

type Token =
    {
        Kind: TokenKind
        Text: string
        File: string
        Line: int
    }

let private puncts = [ "<<"; ">>"; "->"; "::"; "..." ]

/// Tokenizes the surviving lines. Block comments may span lines.
let tokenize (lines: Line list) : Token[] =
    let out = ResizeArray<Token>()
    let mutable inComment = false

    for l in lines do
        let s = l.Text
        let mutable i = 0

        let add kind (text: string) =
            out.Add
                {
                    Kind = kind
                    Text = text
                    File = l.File
                    Line = l.Number
                }

        while i < s.Length do
            if inComment then
                match s.IndexOf("*/", i) with
                | -1 -> i <- s.Length
                | j ->
                    inComment <- false
                    i <- j + 2
            else
                let c = s[i]

                if Char.IsWhiteSpace c then
                    i <- i + 1
                elif c = '/' && i + 1 < s.Length && s[i + 1] = '/' then
                    i <- s.Length
                elif c = '/' && i + 1 < s.Length && s[i + 1] = '*' then
                    inComment <- true
                    i <- i + 2
                elif Char.IsLetter c || c = '_' then
                    let start = i

                    while i < s.Length && (Char.IsLetterOrDigit s[i] || s[i] = '_') do
                        i <- i + 1

                    add Ident (s.Substring(start, i - start))
                elif Char.IsDigit c then
                    let start = i

                    while i < s.Length && (Char.IsLetterOrDigit s[i] || s[i] = '.') do
                        i <- i + 1

                    add Number (s.Substring(start, i - start))
                elif c = '"' || c = '\'' then
                    let start = i
                    i <- i + 1

                    while i < s.Length && s[i] <> c do
                        if s[i] = '\\' then
                            i <- i + 1

                        i <- i + 1

                    i <- i + 1
                    add Str (s.Substring(start, min (i - start) (s.Length - start)))
                else
                    match
                        puncts
                        |> List.tryFind (fun p -> String.CompareOrdinal(s, i, p, 0, p.Length) = 0)
                    with
                    | Some p ->
                        add Punct p
                        i <- i + p.Length
                    | None ->
                        add Punct (string c)
                        i <- i + 1

    out.ToArray()
