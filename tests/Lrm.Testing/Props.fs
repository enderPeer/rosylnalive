/// FsCheck with a fixed seed (TestTiers contract), overridable through LRM_FSCHECK_SEED.
module Lrm.Testing.Props

open System
open System.Globalization
open Expecto

/// Default replay seed and gamma. The gamma must be odd (SplitMix64 stream).
let defaultSeed = (0x4C524D2D46534B31UL, 0x9E3779B97F4A7C15UL)

let private parseU64 (s: string) =
    let s = s.Trim()

    if s.StartsWith("0x", StringComparison.OrdinalIgnoreCase) then
        UInt64.TryParse(s.Substring 2, NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture)
    else
        UInt64.TryParse(s, NumberStyles.None, CultureInfo.InvariantCulture)

/// Parses "seed" or "seed,gamma" (decimal or 0x-hex). Without a gamma the default gamma is used.
let parseSeed (text: string) : Result<uint64 * uint64, string> =
    match text.Split(',') |> Array.map parseU64 with
    | [| true, seed |] -> Ok(seed, snd defaultSeed)
    | [| true, seed; true, gamma |] when gamma % 2UL = 1UL -> Ok(seed, gamma)
    | [| true, _; true, _ |] -> Error "LRM_FSCHECK_SEED: the gamma must be odd"
    | _ -> Error(sprintf "LRM_FSCHECK_SEED: expected \"seed\" or \"seed,gamma\" (decimal or 0x hex), got '%s'" text)

/// The seed in effect for this process.
let seed =
    lazy
        (match Env.tryGet "LRM_FSCHECK_SEED" with
         | None -> defaultSeed
         | Some text ->
             match parseSeed text with
             | Ok s -> s
             | Error e -> failwith e)

/// "0x...,0x..." for logs, in the form LRM_FSCHECK_SEED accepts.
let describe (seed: uint64, gamma: uint64) = sprintf "0x%016x,0x%016x" seed gamma

/// FsCheck configuration with the fixed replay seed.
let config =
    lazy
        (let s, g = seed.Force()

         { FsCheckConfig.defaultConfig with
             replay = Some(s, g, None)
         })

/// A property test with the fixed seed. Use this instead of Expecto's testProperty.
let testProperty (name: string) (property: 'a) =
    testPropertyWithConfig (config.Force()) name property
