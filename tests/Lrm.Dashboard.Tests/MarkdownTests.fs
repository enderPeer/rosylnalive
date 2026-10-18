module Lrm.Dashboard.Tests.MarkdownTests

open System.IO
open Expecto
open Lrm.Dashboard

let private figure (alt: string) (src: string) =
    if src = "charts/x-c.svg" then
        Some("<figure>" + Html.text alt + "</figure>")
    else
        None

let private options =
    {
        HeadingShift = 1
        IdPrefix = "v"
        Figure = figure
    }

/// Vector MD1 (ADR T2-0001): every block and inline form the report templates produce.
let vectorInput =
    String.concat
        "\n"
        [
            "# Title *x*"
            ""
            "> **Synthetic.** a `b|c` <tag>"
            ""
            "**PASS.** rel_l2 of L3.q_proj and _em_ and a\\*b"
            "next line"
            ""
            "- one"
            "- two [link](#x) [net](https://example.com/a)"
            ""
            "| A | B | C |"
            "| --- | ---: | :---: |"
            "| x\\|y | 1 | `c` |"
            "| short |"
            ""
            "![Chart [1]](charts/x-c.svg)"
            ""
            "![missing](charts/none.svg)"
            ""
            "## Title *x*"
        ]

let vectorOutput =
    String.concat
        "\n"
        [
            "<h2 id=\"v-title-x\">Title <em>x</em></h2>"
            "<blockquote>"
            "<p><strong>Synthetic.</strong> a <code>b|c</code> &lt;tag&gt;</p>"
            "</blockquote>"
            "<p><strong>PASS.</strong> rel_l2 of L3.q_proj and <em>em</em> and a*b"
            "next line</p>"
            "<ul>"
            "<li>one</li>"
            "<li>two <a href=\"#x\">link</a> net (<code>https://example.com/a</code>)</li>"
            "</ul>"
            "<div class=\"d-scroll\"><table class=\"d-table\">"
            "<thead><tr><th>A</th><th class=\"d-r\">B</th><th class=\"d-c\">C</th></tr></thead>"
            "<tbody>"
            "<tr><td>x|y</td><td class=\"d-r\">1</td><td class=\"d-c\"><code>c</code></td></tr>"
            "<tr><td>short</td><td class=\"d-r\"></td><td class=\"d-c\"></td></tr>"
            "</tbody></table></div>"
            "<figure>Chart [1]</figure>"
            "<p class=\"d-muted\">missing</p>"
            "<h3 id=\"v-title-x-2\">Title <em>x</em></h3>"
            ""
        ]

let tests =
    testList
        "cpu.dashboard.markdown"
        [
            testCase "vector MD1 renders to exactly the expected HTML"
            <| fun () ->
                let html, headings = Markdown.render options vectorInput
                Expect.equal html vectorOutput "MD1"

                Expect.equal
                    (headings |> List.map (fun h -> h.Level, h.Id))
                    [ 2, "v-title-x"; 3, "v-title-x-2" ]
                    "headings, shifted, with unique ids"

            testCase "CRLF input renders like LF input"
            <| fun () ->
                Expect.equal (fst (Markdown.render options (vectorInput.Replace("\n", "\r\n")))) vectorOutput "CRLF"

            testCase "no markup from the source reaches the page"
            <| fun () ->
                let md =
                    "<script>alert(1)</script>\n\n| <b> | x |\n| --- | --- |\n| <i>\"'& | `<x>` |"

                let html, _ = Markdown.render options md
                Expect.isFalse (html.Contains "<script") "script escaped"
                Expect.isFalse (html.Contains "<b>") "b escaped"
                Expect.isFalse (html.Contains "<i>") "i escaped"
                Expect.stringContains html "&lt;script&gt;alert(1)&lt;/script&gt;" "escaped text"
                Expect.stringContains html "<code>&lt;x&gt;</code>" "escaped code"

            testCase "underscores inside words never start emphasis"
            <| fun () ->
                let html =
                    Markdown.inlineHtml (fun _ _ -> None) "rel_l2 max_abs L3.q_proj snake_case_name"

                Expect.equal html "rel_l2 max_abs L3.q_proj snake_case_name" "no emphasis"

            testCase "unclosed delimiters and brackets stay text"
            <| fun () ->
                let html = Markdown.inlineHtml (fun _ _ -> None) "a ** b `c [d](e f * g"
                Expect.equal html "a ** b `c [d](e f * g" "literal"

            testCase "links never point at the network"
            <| fun () ->
                let html =
                    Markdown.inlineHtml
                        (fun _ _ -> None)
                        "[a](http://x.test) [b](https://x.test) [c](report/parity.md) [d](file:///C:/r.md) [e](#k)"

                Expect.equal
                    html
                    ("a (<code>http://x.test</code>) b (<code>https://x.test</code>) <a href=\"report/parity.md\">c</a> "
                     + "<a href=\"file:///C:/r.md\">d</a> <a href=\"#k\">e</a>")
                    "only relative, file: and fragment links are links"

            testCase "network paths, and what a browser turns into one, stay text"
            <| fun () ->
                // From a file: page, //host and \\host open file://host/..., an SMB request that leaves the machine.
                for target in
                    [
                        "//evil.example/share/x"
                        "\\\\evil.example\\share\\x"
                        "\\/evil.example/x"
                        "/\\evil.example/x"
                        "/\t/evil.example/x"
                        " //evil.example/x"
                        "/x/report.md"
                        "file://evil.example/share/x"
                        "file:////evil.example/share/x"
                        "file:///%2F%2Fevil.example/x"
                        "javascript:alert(1)"
                        "data:text/html,x"
                        "a\\b.md"
                    ] do
                    let html = Markdown.inlineHtml (fun _ _ -> None) ("[click](" + target + ")")
                    Expect.isFalse (html.Contains "<a ") ("no link for " + target)
                    Expect.stringStarts html "click (<code>" ("shown as text: " + target)
                    Expect.isFalse (Html.isLocalHref target) ("not local: " + target)

                for target in [ "#run-x"; "#kind-a b"; "report/parity.md"; "../c/d.md"; "file:///C:/r.md" ] do
                    Expect.isTrue (Html.isLocalHref target) ("local: " + target)

            testCase "Html.text drops control characters and replaces what HTML cannot hold"
            <| fun () ->
                let s =
                    "a\u0001b\rc\td\ne\u007f\u0085f\uFFFEg" + string '\uD800' + "h" + "\uD83D\uDE00"

                Expect.equal (Html.text s) ("abc\td\nef\uFFFDg\uFFFDh" + "\uD83D\uDE00") "cleaned"
                Expect.equal (Html.attr "\"'<>&") "&quot;&#39;&lt;&gt;&amp;" "attribute"
                Expect.equal (Html.text "\"'") "\"'" "quotes stay in text"

            testCase "slugs are lowercase ASCII words joined by dashes"
            <| fun () ->
                Expect.equal (Html.slug "Per pair: cpu32-vs-torch32") "per-pair-cpu32-vs-torch32" "slug"
                Expect.equal (Html.slug "  fp32 denormals ") "fp32-denormals" "trimmed"
                Expect.equal (Html.slug "∑") "section" "no letters"

            testCase "hrefs are relative inside the page's folder, and there is none outside it"
            <| fun () ->
                let root = Path.Combine(Path.GetTempPath(), "lrm-dashboard-href")
                let inside = Path.Combine(root, "runs", "r 1", "report", "parity.md")
                let outside = Path.Combine(Path.GetTempPath(), "elsewhere", "x.md")
                Expect.equal (Html.href root inside) (Some "runs/r%201/report/parity.md") "relative"
                Expect.isNone (Html.href root outside) "no absolute file: URI"
                Expect.isNone (Html.href root (Path.Combine(root, "..", "x.md"))) "nor one climbing out"

                Expect.equal
                    (Html.relativeHref (Path.Combine(root, "a", "b")) (Path.Combine(root, "c", "d.md")))
                    "../../c/d.md"
                    "relative, climbing out"
        ]
