module Lrm.Core.Tests.ProcessTests

open System
open System.Globalization
open System.Text
open Expecto
open Lrm.Core
open Lrm.Core.Tests.Common

/// The child-process probes behind "--probe <name>" (see Main.fs).
let runProbe (name: string) : int =
    match name with
    | "init" ->
        Process.init ()
        printfn "init ok; culture=%s" CultureInfo.CurrentCulture.Name
        0
    | "nfc" ->
        printfn "nfc=%b invariant=%b" (Process.nfcCanary ()) (Process.globalizationInvariant ())
        0
    | other ->
        eprintfn "unknown probe '%s'" other
        2

/// Text produced by many Lrm.Core functions. It must not depend on the current culture.
let fingerprint () : string list =
    let r = DetRng(42UL, DetRng.streamOf "fingerprint")

    [
        for x in [ 1.5f; -0.0f; 0.1f; 1e-5f; 3.4028235e38f; Single.NaN; Single.NegativeInfinity ] do
            Invariant.f32 x
        for x in [ 1.5; -0.0; 0.1; 1.0 / 3.0; 1e300; 5e-324; Double.PositiveInfinity ] do
            Invariant.f64 x
        Invariant.i64 -1234567L
        Invariant.u64 UInt64.MaxValue
        Invariant.format $"{1.5} {2.25f} {-7}"
        Invariant.f64 (Invariant.parseF64 "1.5e3")
        Invariant.f32 (Invariant.parseF32 "-2.5")
        (Hash.sha256Utf8 "abc").Hex
        (Canonical.hash "lrm.test/1" CanonicalTests.katWrite).Hex
        (ExecutionClass.id ExecutionClassTests.cpuKat).Value.Hex
        ExecutionClass.describe ExecutionClassTests.gpuKat
        for _ in 1..8 do
            Invariant.u32 (r.NextU32())
        Invariant.f64 (r.NextF64())
        Invariant.f32 (r.NextF32())
        for x in [ 1.0f; 3.14159265f; -65504.0f; 1e-8f ] do
            sprintf "%04x %04x" (Bf16.narrowRne x) (F16.narrowRne x)
        for t in DType.all do
            DType.name t + "/" + DType.safetensorsName t
        string (DType.tryParse "I32")
        (RunId.create (DateTimeOffset(2026, 10, 17, 23, 15, 0, TimeSpan.Zero)) 0xabcdefu).Value
        string (SafeName.isValid "title")
        string (SafeName.isValid "TITLE")
        string (TelemetryNames.isValid "model.layer")
        string (TelemetryNames.isValid "MODEL.LAYER")
        string (TelemetryNames.isReserved "gpu.dispatch")
        string (ExecutionBackend.tryParse "D3D12")
    ]

let tests =
    testList
        "process"
        [
            testCase "NFC canary: Unicode normalization works in this process"
            <| fun () ->
                Expect.isTrue (Process.nfcCanary ()) "canary"
                Expect.isFalse (Process.globalizationInvariant ()) "not invariant"
                Expect.equal ("e\u0301".Normalize NormalizationForm.FormC) "\u00e9" "e + U+0301"
                Expect.equal ("A\u030a".Normalize NormalizationForm.FormC) "\u00c5" "A + U+030A"
                Expect.equal ("\u1100\u1161\u11a8".Normalize NormalizationForm.FormC) "\uac01" "Hangul jamo"
                Expect.equal ("\u00e9".Normalize NormalizationForm.FormD) "e\u0301" "NFD"
                Expect.equal (Process.check ()) (Ok()) "check"

            testCase "init sets the invariant culture"
            <| fun () ->
                withCulture "de-DE" (fun () ->
                    Process.init ()
                    Expect.equal CultureInfo.CurrentCulture.Name "" "invariant")

                Expect.equal CultureInfo.DefaultThreadCurrentCulture CultureInfo.InvariantCulture "default"

            testCase "init succeeds in a normal child process"
            <| fun () ->
                let r = runSelf [ "--probe"; "init" ] [] Own
                Expect.equal r.ExitCode 0 (sprintf "exit code; stderr: %s" r.StdErr)
                Expect.stringContains r.StdOut "init ok" "stdout"

            testCase "init fails loudly in globalization-invariant mode (runtimeconfig switch)"
            <| fun () ->
                let r = runSelf [ "--probe"; "init" ] [] (InvariantSwitch true)
                Expect.notEqual r.ExitCode 0 "init must fail"
                Expect.stringContains r.StdErr "LRM Process.init" "message names Process.init"
                Expect.stringContains r.StdErr "globalization-invariant mode" "message names the cause"
                Expect.isFalse (r.StdOut.Contains "init ok") "init did not finish"

            testCase "init fails loudly in globalization-invariant mode (environment variable)"
            <| fun () ->
                // The variable wins even over the exe's own runtimeconfig (Invariant=false), so a machine-wide
                // DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1 is caught too.
                for config in [ Own; NoInvariantSwitch ] do
                    let r =
                        runSelf [ "--probe"; "init" ] [ "DOTNET_SYSTEM_GLOBALIZATION_INVARIANT", "1" ] config

                    Expect.notEqual r.ExitCode 0 (sprintf "init must fail (%A)" config)
                    Expect.stringContains r.StdErr "globalization-invariant mode" (sprintf "message (%A)" config)

            testCase "the NFC canary itself detects invariant mode"
            <| fun () ->
                // The canary reads no switch: normalization really is a silent no-op in invariant mode.
                let r = runSelf [ "--probe"; "nfc" ] [] (InvariantSwitch true)
                Expect.equal r.ExitCode 0 (sprintf "probe; stderr: %s" r.StdErr)
                Expect.stringContains r.StdOut "nfc=false invariant=true" "invariant child"

            testCase "invariant-mode detection follows the runtime's precedence"
            <| fun () ->
                // Verified on .NET 10: the variable decides when it is 1, true, 0 or false, even against the
                // runtimeconfig switch; any other value leaves the switch in charge. In every case the NFC canary
                // agrees with globalizationInvariant.
                let var value =
                    [ "DOTNET_SYSTEM_GLOBALIZATION_INVARIANT", value ]

                let cases =
                    [
                        var "1", Own, "nfc=false invariant=true"
                        var "TRUE", Own, "nfc=false invariant=true"
                        var "yes", Own, "nfc=true invariant=false"
                        var "0", InvariantSwitch true, "nfc=true invariant=false"
                        var "false", InvariantSwitch true, "nfc=true invariant=false"
                        var "yes", InvariantSwitch true, "nfc=false invariant=true"
                        [], NoInvariantSwitch, "nfc=true invariant=false"
                    ]

                for env, config, expected in cases do
                    let r = runSelf [ "--probe"; "nfc" ] env config
                    Expect.equal r.ExitCode 0 (sprintf "probe %A %A; stderr: %s" env config r.StdErr)
                    Expect.stringContains r.StdOut expected (sprintf "%A %A" env config)

            testCase "results are identical under de-DE and tr-TR"
            <| fun () ->
                let reference = withCulture "" fingerprint

                withCulture "de-DE" (fun () ->
                    Expect.equal ((1.5).ToString()) "1,5" "de-DE really is active (needs ICU)"
                    Expect.equal (fingerprint ()) reference "de-DE")

                withCulture "tr-TR" (fun () ->
                    Expect.equal ("I".ToLower()) "\u0131" "tr-TR really is active (dotless i)"
                    Expect.equal (fingerprint ()) reference "tr-TR")
        ]
