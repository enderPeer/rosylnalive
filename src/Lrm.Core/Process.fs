/// Process start-up shared by every LRM executable (ProjectGraph contract: every exe calls Process.init first).
/// S0a seeds this minimal version so the exes it creates can call it; S1 owns the file and completes it per the
/// LrmCore contract.
module Lrm.Core.Process

open System
open System.Globalization
open System.Text

/// Sets the invariant culture for the whole process and fails loudly in globalization-invariant mode, which
/// silently turns Unicode normalization into a no-op.
let init () =
    // NFC canary: "e" + COMBINING ACUTE ACCENT must compose to U+00E9.
    if "e\u0301".Normalize NormalizationForm.FormC <> "\u00e9" then
        failwith "Unicode normalization is unavailable (globalization-invariant mode?). LRM refuses to run without ICU."

    CultureInfo.DefaultThreadCurrentCulture <- CultureInfo.InvariantCulture
    CultureInfo.DefaultThreadCurrentUICulture <- CultureInfo.InvariantCulture
    CultureInfo.CurrentCulture <- CultureInfo.InvariantCulture
    CultureInfo.CurrentUICulture <- CultureInfo.InvariantCulture
