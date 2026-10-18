namespace Lrm.Tokenizer

open System.Globalization
open System.Text

/// The class of a Unicode scalar value in the Qwen Split regex (TokenizerApi contract, "Pre-tokenizer"). The regex
/// reads only \p{L}, \p{N} and \s, which never overlap, so every scalar is in exactly one class.
type CharClass =
    /// \p{L}: Lu, Ll, Lt, Lm or Lo.
    | Letter = 0uy
    /// \p{N}: Nd, Nl or No.
    | Number = 1uy
    /// \s: White_Space (U+0009..U+000D, U+0020, U+0085, U+00A0, U+1680, U+2000..U+200A, U+2028, U+2029, U+202F,
    /// U+205F, U+3000).
    | Space = 2uy
    /// Everything else, marks and unassigned scalars included.
    | Other = 3uy

/// Character classes from the runtime's Unicode tables (System.Text.Rune). On .NET 10 they are Unicode 16.0 and equal
/// the oracle's Oniguruma tables on every scalar value; the cpu tier checks all 1,112,064 against the oracle's probe
/// (tests/Lrm.Tokenizer.Tests/oracle/unicode-oracle.json, ADR M1-0001).
[<RequireQualifiedAccess>]
module CharClass =
    /// The class of scalar value `v` from the runtime's tables.
    let ofScalar (v: int) : CharClass =
        let r = Rune v

        match Rune.GetUnicodeCategory r with
        | UnicodeCategory.UppercaseLetter
        | UnicodeCategory.LowercaseLetter
        | UnicodeCategory.TitlecaseLetter
        | UnicodeCategory.ModifierLetter
        | UnicodeCategory.OtherLetter -> CharClass.Letter
        | UnicodeCategory.DecimalDigitNumber
        | UnicodeCategory.LetterNumber
        | UnicodeCategory.OtherNumber -> CharClass.Number
        | _ ->
            if Rune.IsWhiteSpace r then
                CharClass.Space
            else
                CharClass.Other

    /// The BMP part of the table, built once; surrogate code units are Other (they never reach the regex).
    let private bmp: CharClass[] =
        Array.init 0x10000 (fun c ->
            if c >= 0xD800 && c <= 0xDFFF then
                CharClass.Other
            else
                ofScalar c)

    /// The class of scalar value `v`.
    let get (v: int) : CharClass =
        if v < 0x10000 then bmp[v] else ofScalar v

    /// The letter a scalar matches in the case-insensitive contraction group (?i:'s|'t|'re|'ve|'m|'ll|'d), or '\000'.
    /// This is simple case folding (CaseFolding.txt, status C and S) restricted to the group's letters: U+017F LATIN
    /// SMALL LETTER LONG S folds to s, and every other letter of the group matches only its own two ASCII cases. The
    /// cpu tier checks this against the oracle for every letter of Unicode.
    let contractionLetter (v: int) : char =
        match v with
        | 0x53
        | 0x73
        | 0x17F -> 's'
        | 0x54
        | 0x74 -> 't'
        | 0x52
        | 0x72 -> 'r'
        | 0x45
        | 0x65 -> 'e'
        | 0x56
        | 0x76 -> 'v'
        | 0x4D
        | 0x6D -> 'm'
        | 0x4C
        | 0x6C -> 'l'
        | 0x44
        | 0x64 -> 'd'
        | _ -> '\000'
