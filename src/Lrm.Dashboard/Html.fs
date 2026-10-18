namespace Lrm.Dashboard

open System
open System.IO
open System.Text

/// HTML text helpers. Every string that comes from a run log goes through `text` or `attr`, so a log can never inject
/// markup into the dashboard, and characters HTML cannot hold never reach the file.
[<RequireQualifiedAccess>]
module Html =
    /// Appends `s` escaped for HTML text (`quotes` also escapes double and single quotes, for attribute values).
    /// Control characters other than tab and LF are dropped (CR is a line break and becomes nothing), noncharacters
    /// U+FFFE and U+FFFF and unpaired surrogates become U+FFFD; a surrogate pair is kept as it is.
    let appendEscaped (sb: StringBuilder) (quotes: bool) (s: string) =
        if not (isNull s) then
            let mutable i = 0

            while i < s.Length do
                let c = s[i]

                match c with
                | '&' -> sb.Append "&amp;" |> ignore
                | '<' -> sb.Append "&lt;" |> ignore
                | '>' -> sb.Append "&gt;" |> ignore
                | '"' when quotes -> sb.Append "&quot;" |> ignore
                | '\'' when quotes -> sb.Append "&#39;" |> ignore
                | '\t'
                | '\n' -> sb.Append c |> ignore
                | _ when c < ' ' || (c >= '\u007f' && c <= '\u009f') -> ()
                | '￾'
                | '￿' -> sb.Append '�' |> ignore
                | _ when Char.IsHighSurrogate c ->
                    if i + 1 < s.Length && Char.IsLowSurrogate s[i + 1] then
                        sb.Append(c).Append(s[i + 1]) |> ignore
                        i <- i + 1
                    else
                        sb.Append '�' |> ignore
                | _ when Char.IsLowSurrogate c -> sb.Append '�' |> ignore
                | _ -> sb.Append c |> ignore

                i <- i + 1

    /// `s` escaped for HTML text content.
    let text (s: string) =
        let sb = StringBuilder()
        appendEscaped sb false s
        sb.ToString()

    /// `s` escaped for a double-quoted attribute value.
    let attr (s: string) =
        let sb = StringBuilder()
        appendEscaped sb true s
        sb.ToString()

    /// A fragment id made from text: lowercase ASCII letters and digits, every other run of characters one '-'.
    let slug (s: string) =
        let sb = StringBuilder()
        let mutable dash = false

        for c in (if isNull s then "" else s) do
            let c = Char.ToLowerInvariant c

            if (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') then
                if dash && sb.Length > 0 then
                    sb.Append '-' |> ignore

                dash <- false
                sb.Append c |> ignore
            else
                dash <- true

        if sb.Length = 0 then "section" else sb.ToString()

    /// The href of `target` (a file or folder path) from a page in `fromDir`, when the target lies inside `fromDir`: a
    /// relative path with forward slashes, each segment percent-encoded, so the href never depends on how the platform
    /// spells separators. None for a target outside `fromDir`: its only href would be an absolute file: URI, which
    /// would put a local path into the page and make its bytes depend on where the folders are (ADR T2-0001, 7).
    let href (fromDir: string) (target: string) =
        let fromDir = Path.GetFullPath fromDir
        let target = Path.GetFullPath target
        let rel = Path.GetRelativePath(fromDir, target)

        if
            rel = target
            || Path.IsPathRooted rel
            || rel = ".."
            || rel.StartsWith("..\\", StringComparison.Ordinal)
            || rel.StartsWith("../", StringComparison.Ordinal)
        then
            None
        else
            Some(
                rel.Replace('\\', '/').Split('/')
                |> Array.map Uri.EscapeDataString
                |> String.concat "/"
            )

    /// Whether an href stays on the machine, the only hrefs the page holds (ADR T2-0001, decisions 1 and 10): a fragment,
    /// a local file: URI (file:/// and then not another slash, which would name a network host), or a relative path that
    /// names no scheme and does not start with a slash. Other than a fragment, a target with a backslash, a space or a
    /// control character is refused, because browsers turn a backslash into a slash, drop spaces and controls at the
    /// ends and drop tabs and line ends inside: two backslashes before a host, or a slash, a tab and a slash, would
    /// become "//host", a network path that a file: page opens as an SMB request.
    let isLocalHref (h: string) =
        not (String.IsNullOrEmpty h)
        && (h.StartsWith '#'
            || h |> Seq.forall (fun c -> c > ' ' && c <> '\\' && c <> '\u007f')
               && ((h.StartsWith("file:///", StringComparison.Ordinal)
                    && h.Length > 8
                    && h[8] <> '/'
                    && h[8] <> '%')
                   || (not (h.Contains ':') && not (h.StartsWith '/'))))

    /// The href of `target` from a page in `fromDir` as a relative path, even when it climbs out of `fromDir`.
    let relativeHref (fromDir: string) (target: string) =
        Path.GetRelativePath(Path.GetFullPath fromDir, Path.GetFullPath target).Replace('\\', '/').Split('/')
        |> Array.map Uri.EscapeDataString
        |> String.concat "/"
