/// How many logical processors WARP may use. WARP rasterizes on the CPU with a worker per logical processor, so on the
/// 32-thread development machine an unbounded run takes every core from the other agents. The spike pins its own
/// process (never the system) to the first N logical processors before it creates a device: LRM_WARP_CPUS = N, or
/// "all". The default, 4, is the vCPU count of a GitHub-hosted windows-latest runner for a public repository, so the
/// timings below model the CI job the ADR proposes.
module Lrm.Spikes.D3D12.Affinity

open System
open System.Diagnostics
open Lrm.Core

/// The default processor count: a windows-latest runner of a public repository has 4 vCPUs.
[<Literal>]
let DefaultCpus = 4

/// LRM_WARP_CPUS parsed: Ok None for "all", Ok (Some n) for 1 <= n <= 64, Error for anything else.
let parse (text: string option) : Result<int option, string> =
    match text with
    | None -> Ok(Some DefaultCpus)
    | Some t when String.Equals(t.Trim(), "all", StringComparison.OrdinalIgnoreCase) -> Ok None
    | Some t ->
        match Invariant.tryParseI32 (t.Trim()) with
        | Some n when n >= 1 && n <= 64 -> Ok(Some n)
        | _ -> Error(sprintf "LRM_WARP_CPUS must be 'all' or a count from 1 to 64, not '%s'" t)

/// The first `n` set bits of `current` (bit i is logical processor i): the processors the process keeps.
let mask (n: int) (current: int64) : int64 =
    let mutable kept = 0L
    let mutable left = max 1 n
    let mutable bit = 0

    while left > 0 && bit < 64 do
        if current &&& (1L <<< bit) <> 0L then
            kept <- kept ||| (1L <<< bit)
            left <- left - 1

        bit <- bit + 1

    kept

/// What the process runs on after `apply`.
type Applied =
    {
        /// LRM_WARP_CPUS as given, or the default.
        Requested: string
        /// Logical processors the process may use.
        Cpus: int
        /// Logical processors of the machine.
        MachineCpus: int
        /// The affinity mask, as hex.
        Mask: string
    }

let private bitCount (m: int64) =
    Numerics.BitOperations.PopCount(uint64 m)

/// Pins this process to the processors LRM_WARP_CPUS names (only this process: no system setting changes). Call it
/// before the first D3D12 device: WARP's workers inherit the process affinity.
let apply () : Applied =
    let requested = Environment.GetEnvironmentVariable "LRM_WARP_CPUS"
    let p = Process.GetCurrentProcess()
    let machine = Environment.ProcessorCount

    match parse (Option.ofObj requested) with
    | Error e -> invalidOp e
    | Ok None ->
        let m = int64 p.ProcessorAffinity

        {
            Requested = "all"
            Cpus = bitCount m
            MachineCpus = machine
            Mask = sprintf "%x" m
        }
    | Ok(Some n) ->
        let m = mask n (int64 p.ProcessorAffinity)
        p.ProcessorAffinity <- nativeint m

        {
            Requested = (if isNull requested then sprintf "%d (default)" DefaultCpus else requested)
            Cpus = bitCount m
            MachineCpus = machine
            Mask = sprintf "%x" m
        }
