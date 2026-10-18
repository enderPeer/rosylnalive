/// Process start-up shared by every LRM executable (ProjectGraph and LrmCore contracts: every exe calls
/// Process.init first).
module Lrm.Core.Process

open System
open System.Globalization
open System.Text

/// "e" followed by U+0301 COMBINING ACUTE ACCENT; NFC composes it to U+00E9.
let private canaryInput = "e\u0301"

let private canaryExpected = "\u00e9"

/// True when the runtime is in globalization-invariant mode. This follows the runtime's own precedence, verified on
/// .NET 10: DOTNET_SYSTEM_GLOBALIZATION_INVARIANT decides when it is "1", "true", "0" or "false" (any case), even
/// against the runtimeconfig; otherwise the System.Globalization.Invariant switch decides; otherwise false.
let globalizationInvariant () =
    let fromEnvironment =
        match Environment.GetEnvironmentVariable "DOTNET_SYSTEM_GLOBALIZATION_INVARIANT" with
        | null -> None
        | v when v = "1" || v.Equals("true", StringComparison.OrdinalIgnoreCase) -> Some true
        | v when v = "0" || v.Equals("false", StringComparison.OrdinalIgnoreCase) -> Some false
        | _ -> None

    match fromEnvironment with
    | Some value -> value
    | None ->
        match AppContext.TryGetSwitch "System.Globalization.Invariant" with
        | true, value -> value
        | _ -> false

/// The NFC canary: true when string normalization composes "e" + U+0301 to U+00E9. Globalization-invariant mode turns
/// normalization into a silent no-op, so this is false there.
let nfcCanary () =
    try
        canaryInput.Normalize NormalizationForm.FormC = canaryExpected
    with _ ->
        false

/// The start-up checks, without side effects: not globalization-invariant, the NFC canary holds, and the process is
/// 64-bit little-endian.
let check () : Result<unit, string> =
    if globalizationInvariant () then
        Error
            "the runtime is in globalization-invariant mode (InvariantGlobalization or DOTNET_SYSTEM_GLOBALIZATION_INVARIANT), which silently disables Unicode normalization. LRM needs ICU; turn invariant mode off."
    elif not (nfcCanary ()) then
        Error
            "the NFC canary failed: \"e\\u0301\".Normalize(FormC) did not give \"\\u00e9\". Unicode normalization is unavailable (ICU missing?)."
    elif not BitConverter.IsLittleEndian then
        Error "LRM needs a little-endian host"
    elif not Environment.Is64BitProcess then
        Error "LRM needs a 64-bit process (x64)"
    else
        Ok()

/// Sets the invariant culture for the whole process, after the start-up checks. Fails loudly with
/// InvalidOperationException when a check fails. Safe to call more than once.
let init () =
    match check () with
    | Error problem -> raise (InvalidOperationException("LRM Process.init: " + problem))
    | Ok() ->
        CultureInfo.DefaultThreadCurrentCulture <- CultureInfo.InvariantCulture
        CultureInfo.DefaultThreadCurrentUICulture <- CultureInfo.InvariantCulture
        CultureInfo.CurrentCulture <- CultureInfo.InvariantCulture
        CultureInfo.CurrentUICulture <- CultureInfo.InvariantCulture
