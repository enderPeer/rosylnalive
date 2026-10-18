/// The dashboard's fixed CSS and JavaScript (ADR T2-0001). Both are inline in the page, so the file needs nothing but
/// itself: no network, no fonts, no CDN. The chart theme is T1's own (Theme.style), included once for every chart,
/// plus rules that let the theme toggle override the viewer's colour-scheme preference.
module Lrm.Dashboard.Assets

open System
open System.Text
open Lrm.Reports

/// On a phone the runs tables become cards, and each cell is preceded by its column's name.
let private rowLabels (table: string) (labels: string list) =
    labels
    |> List.mapi (fun i l ->
        table
        + " td:nth-child("
        + Lrm.Core.Invariant.i32 (i + 1)
        + ")::before{content:\""
        + l
        + " \"}")
    |> String.concat ""

/// The page CSS. Colours are tokens on :root, redefined for dark mode under prefers-color-scheme (unless the toggle
/// chose light) and under [data-theme=dark]. Class names start with "d-" so they never meet the chart classes.
let pageCss =
    let light =
        "--bg:#f4f3f0;--surface:#fcfcfb;--surface2:#f0efec;--border:#e6e5e0;--ink:#0b0b0b;--ink2:#52514e;"
        + "--muted:#6a6964;--link:#1c5cab;color-scheme:light"

    // --muted is at least 4.5:1 against every surface it sits on (4.78 on --surface2 light, 4.80 dark).
    let dark =
        "--bg:#111110;--surface:#1a1a19;--surface2:#242423;--border:#2c2c2a;--ink:#f0efec;--ink2:#c3c2b7;"
        + "--muted:#918f89;--link:#86b6ef;color-scheme:dark"

    String.concat
        "\n"
        [
            ":root{"
            + light
            + ";--good:#0ca30c;--warn:#fab219;--serious:#ec835a;--crit:#d03b3b}"
            "@media (prefers-color-scheme:dark){:root:not([data-theme=light]){"
            + dark
            + "}}"
            ":root[data-theme=dark]{" + dark + "}"
            "*,*::before,*::after{box-sizing:border-box}"
            "[hidden]{display:none!important}"
            "html{-webkit-text-size-adjust:100%;text-size-adjust:100%}"
            "body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;overflow-wrap:anywhere}"
            "a{color:var(--link)}"
            "code{font:.88em ui-monospace,SFMono-Regular,Consolas,monospace;background:var(--surface2);padding:0 .25em;border-radius:4px}"
            "pre{overflow-x:auto;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px 12px}"
            "pre code{background:none;padding:0}"
            "a:focus-visible,button:focus-visible,select:focus-visible,input:focus-visible,figure:focus-visible,h2:focus-visible{outline:2px solid var(--link);outline-offset:2px}"
            "h2[tabindex]:focus:not(:focus-visible){outline:none}"
            ".d-skip{position:absolute;left:16px;top:-48px;z-index:3;background:var(--surface);color:var(--link);border:1px solid var(--border);border-radius:6px;padding:4px 10px}"
            ".d-skip:focus{top:8px}"
            ".d-top{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--border)}"
            ".d-bar,.d-nav,main,footer{max-width:1120px;margin:0 auto;padding-left:16px;padding-right:16px}"
            ".d-bar{display:flex;flex-wrap:wrap;align-items:center;gap:6px 16px;padding-top:10px;padding-bottom:6px}"
            ".d-brand{font-weight:650;font-size:17px;line-height:1.3;margin:0}"
            ".d-brand a{color:var(--ink);text-decoration:none}"
            ".d-sum{color:var(--muted);font-size:13px}"
            ".d-live{font-size:13px;color:var(--ink2);display:inline-flex;align-items:center;gap:6px}"
            ".d-live::before{content:\"\";width:8px;height:8px;border-radius:50%;background:var(--good)}"
            ".d-live[data-paused]::before{background:var(--warn)}"
            ".d-live[data-paused]::after{content:\"(waiting: a field has focus, a chart is zoomed or text is selected)\";color:var(--muted)}"
            "#d-theme{margin-left:auto;font:inherit;font-size:13px;color:var(--ink);background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:3px 10px;cursor:pointer}"
            ".d-nav{position:relative;display:flex;flex-wrap:wrap;gap:6px;padding-top:2px;padding-bottom:10px}"
            // A pill never gets wider than the page: a long kind name ends in an ellipsis (its title has it all).
            ".d-nav a{display:inline-flex;max-width:100%;min-width:0;font-size:13px;color:var(--ink2);text-decoration:none;border:1px solid var(--border);background:var(--surface);border-radius:999px;padding:1px 10px;white-space:nowrap}"
            ".d-nav a span{min-width:0;overflow:hidden;text-overflow:ellipsis}"
            ".d-nav a b{flex:none;font-weight:600;color:var(--muted);margin-left:5px}"
            ".d-nav a[aria-current]{background:var(--ink);color:var(--bg);border-color:var(--ink)}"
            ".d-nav a[aria-current] b{color:inherit}"
            "main{padding-top:12px;padding-bottom:40px}"
            "footer{color:var(--muted);font-size:12px;padding-bottom:24px}"
            "h2{font-size:22px;line-height:1.3;margin:6px 0 4px;font-weight:650}"
            "h3{font-size:18px;line-height:1.35;margin:30px 0 8px;font-weight:650}"
            "h4,h5,h6{font-size:15px;margin:22px 0 6px;font-weight:650}"
            "p,ul,ol,blockquote{max-width:780px}"
            "blockquote{margin:12px 0;padding:8px 14px;background:var(--surface);border:1px solid var(--border);border-left:4px solid var(--warn);border-radius:0 8px 8px 0}"
            "blockquote p{margin:4px 0}"
            ".d-muted{color:var(--muted)}"
            ".d-crumbs{font-size:13px;color:var(--muted);margin:0}"
            ".d-crumbs a{color:var(--muted)}"
            ".d-tags{display:flex;flex-wrap:wrap;gap:6px 8px;align-items:center;margin:6px 0 0;font-size:13px}"
            ".d-tag{display:inline-block;max-width:100%;overflow:hidden;text-overflow:ellipsis;vertical-align:bottom;font-size:12px;border:1px solid var(--border);background:var(--surface);border-radius:4px;padding:0 6px;color:var(--ink2);white-space:nowrap}"
            ".d-small{font-size:12px;margin:-6px 0 10px}"
            ".d-st{display:inline-flex;align-items:center;gap:6px;white-space:nowrap}"
            ".d-st::before{content:\"\";width:8px;height:8px;border-radius:50%;background:var(--muted);flex:none}"
            ".d-st-ok::before{background:var(--good)}.d-st-failed::before{background:var(--crit)}"
            ".d-st-running::before{background:var(--warn)}.d-st-aborted::before{background:var(--serious)}"
            ".d-tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(128px,1fr));gap:8px;margin:14px 0}"
            ".d-tile{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:8px 12px;min-width:0}"
            ".d-tile span{display:block;font-size:12px;color:var(--muted)}"
            ".d-tile b{display:block;font-size:19px;font-weight:600;line-height:1.3}"
            ".d-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 16px;margin:12px 0}"
            ".d-card>:first-child{margin-top:0}.d-card>:last-child{margin-bottom:0}"
            ".d-links{font-size:14px}"
            ".d-hint{display:none;font-size:12px;color:var(--muted);margin:8px 0 0}"
            ".d-toc{display:flex;flex-wrap:wrap;gap:2px 14px;font-size:13px;margin:10px 0 0;padding:8px 0;border-top:1px solid var(--border);border-bottom:1px solid var(--border)}"
            ".d-scroll{overflow-x:auto;max-width:100%;margin:8px 0 16px}"
            ".d-table{border-collapse:collapse;font-size:13.5px;font-variant-numeric:tabular-nums;background:var(--surface);overflow-wrap:normal}"
            ".d-table th,.d-table td{padding:5px 10px;border-bottom:1px solid var(--border);text-align:left;vertical-align:top}"
            ".d-table th{background:var(--surface2);font-weight:600;color:var(--ink2);white-space:nowrap}"
            ".d-table .d-r{text-align:right}.d-table .d-c{text-align:center}"
            ".d-table tr.d-changed td:first-child{font-weight:600}"
            ".d-table td.d-nw{white-space:nowrap}"
            // Targets scroll clear of the sticky header; the script sets --d-hdr to its height.
            ".d-view,.d-view [id]{scroll-margin-top:var(--d-hdr,112px)}"
            ".d-table td.d-wrap{min-width:16em;overflow-wrap:anywhere}"
            ".d-filters{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:flex-end;margin:12px 0 4px}"
            ".d-filters label{display:flex;flex-direction:column;gap:2px;font-size:12px;color:var(--muted);min-width:0}"
            "select,input{font:inherit;font-size:14px;color:var(--ink);background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:3px 8px;max-width:100%}"
            "input[type=search]{width:16em}"
            "figure.d-chart{margin:12px 0;overflow-x:auto;cursor:zoom-in;max-width:100%}"
            "figure.d-chart svg{display:block;max-width:100%;height:auto}"
            "figure.d-chart.d-zoom{cursor:zoom-out}figure.d-chart.d-zoom svg{max-width:none}"
            ".d-pair-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}"
            ".d-slot{min-width:0}"
            ".d-slot figure.d-chart{margin:4px 0}"
            ".d-slot-label{font-size:12px;color:var(--muted)}"
            ".d-pairs{margin:6px 0 18px;padding-left:20px}.d-pairs li{margin:3px 0}"
            // Views: without JavaScript the targeted view shows and the runs index (the last view) hides behind it;
            // with JavaScript the script marks the current view. :has rules stand alone, so a browser without :has
            // drops only them.
            ".d-view{display:none}"
            "#runs{display:block}"
            ".d-view:target{display:block}"
            ".d-view:target~#runs{display:none}"
            ".d-view:has(:target){display:block}"
            ".d-view:has(:target)~#runs{display:none}"
            "main.d-js .d-view{display:none!important}"
            "main.d-js .d-view.d-on{display:block!important}"
            "@media (max-width:900px){.d-pair-grid{grid-template-columns:1fr}}"
            // On a phone the header scrolls away with the page and the views are one row of pills that scrolls
            // sideways inside itself, so the view's verdict and first chart come before the fold.
            "@media (max-height:500px){.d-top{position:static}}"
            "@media (max-width:640px){"
            + ".d-top{position:static}"
            + ".d-bar{padding-top:8px;padding-bottom:4px;gap:4px 12px}"
            + ".d-nav{flex-wrap:nowrap;overflow-x:auto;scrollbar-width:thin;padding-bottom:8px}"
            + ".d-nav a{flex:none}"
            + "h2{font-size:19px}h3{font-size:17px}.d-hint{display:block}"
            + ".d-tiles{grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;margin:10px 0}"
            + ".d-tile{padding:5px 8px}.d-tile span{font-size:11.5px}.d-tile b{font-size:15px}"
            + ".d-rows thead{display:none}"
            + ".d-rows,.d-rows tbody,.d-rows tr,.d-rows td{display:block;width:100%}"
            + ".d-rows{background:none}"
            + ".d-rows tr{background:var(--surface);border:1px solid var(--border);border-radius:8px;margin:0 0 8px;padding:6px 10px}"
            + ".d-rows td{border:0;padding:1px 0;text-align:left!important;overflow-wrap:anywhere}"
            + ".d-rows td::before{color:var(--muted);font-size:12px}"
            + rowLabels ".d-rows" [ "Started"; "Run"; "Kind"; "Status"; "Duration"; "Nodes"; "Class"; "Title" ]
            + rowLabels ".d-nokind" [ "Started"; "Run"; "Status"; "Duration"; "Nodes"; "Class"; "Title" ]
            + "}"
            "@media print{.d-top{position:static}#d-theme,.d-filters{display:none}}"
        ]

/// The chart theme: T1's Theme.style once for the whole page (every chart SVG embeds the same rules), then the
/// light and the dark rules again under [data-theme=light] and [data-theme=dark], so the toggle wins over the
/// viewer's prefers-color-scheme in both directions.
let chartCss =
    let style = Theme.style
    let marker = "@media (prefers-color-scheme:dark){"
    let at = style.IndexOf(marker, StringComparison.Ordinal)

    if at < 0 || not (style.EndsWith('}')) then
        invalidOp "Theme.style no longer has the expected light and dark blocks"

    let lightRules = style.Substring(0, at)

    let darkRules =
        style.Substring(at + marker.Length, style.Length - at - marker.Length - 1)

    let scoped (prefix: string) (rules: string) =
        let sb = StringBuilder()

        for rule in rules.Split('}', StringSplitOptions.RemoveEmptyEntries) do
            let brace = rule.IndexOf '{'

            if brace > 0 && rule[0] = '.' then
                let selectors =
                    rule.Substring(0, brace).Split(',')
                    |> Array.map (fun s -> prefix + " " + s.Trim())
                    |> String.concat ","

                sb.Append(selectors).Append(rule.Substring brace).Append('}') |> ignore

        sb.ToString()

    style
    + "\n"
    + scoped ":root[data-theme=light] .d-chart" lightRules
    + "\n"
    + scoped ":root[data-theme=dark] .d-chart" darkRules

/// Runs before the first paint: applies the theme the viewer chose last time, if any.
let headScript =
    "try{var m=localStorage.getItem('lrm.dashboard.theme');"
    + "if(m==='light'||m==='dark')document.documentElement.setAttribute('data-theme',m)}catch(e){}"

/// The page behaviour: view switching (focus moves to the new view's heading), the skip link, the theme toggle, the
/// runs filter, compare chart slots, chart zoom and, in watch mode, the reload. The page works without it: views then
/// switch through :target, and a watch page reloads through the meta refresh in <noscript>.
///
/// Watch mode reloads with location.reload() every data-watch seconds, which keeps the view the reader navigated to
/// (a meta refresh reloads the URL the page was opened with, and Chromium drops the fragment). It waits while the tab
/// is hidden, a form field has focus, a chart is zoomed or text is selected, and keeps the scroll position and the
/// runs filter across the reload (sessionStorage, guarded).
let bodyScript =
    String.concat
        "\n"
        [
            "(function(){'use strict';"
            "var d=document,root=d.documentElement,K='lrm.dashboard.theme',modes=['auto','light','dark'];"
            "function load(k){try{return localStorage.getItem(k)}catch(e){return null}}"
            "function save(k,v){try{if(v===null)localStorage.removeItem(k);else localStorage.setItem(k,v)}catch(e){}}"
            "var W='lrm.dashboard.watch:',watching=root.hasAttribute('data-watch');"
            "function sget(k){try{return sessionStorage.getItem(W+k)}catch(e){return null}}"
            "function sset(k,v){try{sessionStorage.setItem(W+k,v)}catch(e){}}"
            "var mode=load(K);if(modes.indexOf(mode)<0)mode='auto';"
            "var tb=d.getElementById('d-theme');"
            "function apply(){if(mode==='auto')root.removeAttribute('data-theme');else root.setAttribute('data-theme',mode);"
            "if(tb){tb.textContent='Theme: '+mode;tb.setAttribute('aria-label','Colour theme: '+mode+'. Click to change.')}}"
            "apply();if(tb)tb.addEventListener('click',function(){mode=modes[(modes.indexOf(mode)+1)%3];save(K,mode==='auto'?null:mode);apply()});"
            "var main=d.querySelector('main');if(main)main.classList.add('d-js');"
            "function target(){var h='';try{h=decodeURIComponent(location.hash.slice(1))}catch(e){}return h&&d.getElementById(h)}"
            "function isView(el){return !!(el&&el.classList&&el.classList.contains('d-view'))}"
            "function current(){var el=target(),v=el&&el.closest?el.closest('.d-view'):null;return v||d.getElementById('runs')}"
            // A compare slot gets a clone of the run view's figure, with every id and id reference suffixed.
            "function fill(slot){if(slot.getAttribute('data-done'))return;var from=d.getElementById(slot.getAttribute('data-src'));if(!from)return;"
            "var n=slot.getAttribute('data-n'),c=from.cloneNode(true);c.removeAttribute('id');c.classList.remove('d-zoom');c.setAttribute('aria-pressed','false');"
            "[].forEach.call(c.querySelectorAll('[id]'),function(e){e.id=e.id+'-'+n});"
            "function ren(el,at){var l=el&&el.getAttribute(at);if(l)el.setAttribute(at,l.split(' ').map(function(x){return x+'-'+n}).join(' '))}"
            "ren(c.querySelector('svg'),'aria-labelledby');ren(c,'aria-describedby');"
            "slot.textContent='';slot.appendChild(c);slot.setAttribute('data-done','1')}"
            "var nav=d.querySelector('.d-nav');"
            "function show(){var v=current();[].forEach.call(d.querySelectorAll('.d-view.d-on'),function(x){if(x!==v)x.classList.remove('d-on')});"
            "if(v){v.classList.add('d-on');[].forEach.call(v.querySelectorAll('.d-slot[data-src]'),fill)}"
            "var k=v&&v.getAttribute('data-nav');"
            "[].forEach.call(d.querySelectorAll('.d-nav a'),function(a){if(a.getAttribute('href')==='#'+k){a.setAttribute('aria-current','page');"
            // On a phone the pills are one row that scrolls sideways: bring the current one into it.
            "if(nav&&(a.offsetLeft<nav.scrollLeft||a.offsetLeft+a.offsetWidth>nav.scrollLeft+nav.clientWidth))nav.scrollLeft=Math.max(0,a.offsetLeft-16)}"
            "else a.removeAttribute('aria-current')})}"
            // Targets scroll clear of the header while it is sticky (not on a phone, where it scrolls away).
            "var hdr=d.querySelector('.d-top');"
            "function measure(){if(hdr)root.style.setProperty('--d-hdr',((getComputedStyle(hdr).position==='sticky'?hdr.offsetHeight:0)+8)+'px')}"
            "measure();window.addEventListener('resize',measure);"
            // A view starts at the top of the page; a target inside a view scrolls into place below the header.
            "function place(){var el=target();if(!el||isView(el))window.scrollTo(0,0);else el.scrollIntoView()}"
            "function heading(v){return v&&v.querySelector('h2[tabindex]')}"
            // A switch to another view moves focus to its heading; a target inside a view keeps the browser's focus.
            "function focusView(){var el=target();if(el&&!isView(el))return;var h=heading(current());if(h)h.focus({preventScroll:true})}"
            "window.addEventListener('hashchange',function(){show();place();focusView()});show();"
            "var skip=d.querySelector('.d-skip');"
            "if(skip){skip.hidden=false;skip.addEventListener('click',function(e){e.preventDefault();var h=heading(current());if(h)h.focus()})}"
            "var body=d.getElementById('d-runs-body'),fk,fs,fq;"
            "if(body){var rows=[].slice.call(body.rows),fc=d.getElementById('d-f-count');fk=d.getElementById('d-f-kind');fs=d.getElementById('d-f-status');fq=d.getElementById('d-f-q');"
            "d.getElementById('d-filters').hidden=false;"
            "if(watching){var saved=sget('filter');if(saved){try{var f=JSON.parse(saved);fk.value=f.k||'';fs.value=f.s||'';fq.value=f.q||''}catch(e){}}}"
            "var filter=function(){var k=fk.value,s=fs.value,q=fq.value.trim().toLowerCase(),n=0;"
            "rows.forEach(function(r){var ok=(!k||r.getAttribute('data-kind')===k)&&(!s||r.getAttribute('data-status')===s)&&(!q||r.textContent.toLowerCase().indexOf(q)>=0);r.hidden=!ok;if(ok)n++});"
            "fc.textContent=n+' of '+rows.length+' runs'};"
            "fk.addEventListener('change',filter);fs.addEventListener('change',filter);fq.addEventListener('input',filter);filter()}"
            "function zoom(f){var on=f.classList.toggle('d-zoom');f.setAttribute('aria-pressed',on?'true':'false')}"
            "d.addEventListener('click',function(e){var f=e.target&&e.target.closest?e.target.closest('figure.d-chart'):null;if(f&&!(e.target.closest('a')))zoom(f)});"
            "d.addEventListener('keydown',function(e){if(e.key!=='Enter'&&e.key!==' ')return;var f=e.target;if(f&&f.matches&&f.matches('figure.d-chart')){e.preventDefault();zoom(f)}});"
            "function restore(){var y=watching?sget('scroll:'+location.hash):null;if(y!==null)window.scrollTo(0,+y);else place()}"
            "if(d.readyState==='complete')restore();else window.addEventListener('load',function(){setTimeout(restore,0)});"
            "if(watching){var every=Math.max(1,+root.getAttribute('data-watch')||3)*1000,due=Date.now()+every,live=d.querySelector('.d-live');"
            "window.addEventListener('pagehide',function(){sset('scroll:'+location.hash,String(window.scrollY));if(body)sset('filter',JSON.stringify({k:fk.value,s:fs.value,q:fq.value}))});"
            "function busy(){var a=d.activeElement,t=a&&a.tagName;"
            "return t==='INPUT'||t==='SELECT'||t==='TEXTAREA'||!!d.querySelector('figure.d-chart.d-zoom')||(window.getSelection?String(window.getSelection()):'')!==''}"
            "setInterval(function(){if(d.hidden||Date.now()<due)return;var b=busy();"
            "if(live){if(b)live.setAttribute('data-paused','');else live.removeAttribute('data-paused')}"
            "if(!b)location.reload()},500)}"
            "})();"
        ]
