namespace Lrm.Serving

open System
open System.Text.Json
open System.Text.Json.Nodes
open Lrm.Core
open Lrm.Cluster

/// The outcome of one pipeline run as the coordinator saw it: C1's RunOutcome plus the workload's tokens and logits
/// digest. It holds only content and counters, never times, so it travels between processes as one JSON line.
type ServingOutcome =
    {
        /// RunStatus.name: completed, unverified, diverged or aborted.
        Status: string
        Job: ContentId
        RunDigest: ContentId
        ReplicaDigest: ContentId
        /// ChainDigest.logits over every head output.
        LogitsDigest: ContentId
        Tokens: int list list
        /// LaneEnd.name per lane.
        Ends: string list
        Units: int
        Epoch: uint64
        Reissues: int
        Duplicates: int
        Replays: int
        Rejected: int
        CrossChecked: int
        Unconfirmed: int
        Divergences: int
    }

[<RequireQualifiedAccess>]
module ServingOutcome =
    /// The line prefix of an outcome a coordinator process prints.
    [<Literal>]
    let Prefix = "LRM-COORD-OUTCOME "

    /// The outcome of a finished run and its workload.
    let ofRun (o: RunOutcome) (r: ServingResult) =
        {
            Status = RunStatus.name o.Status
            Job = o.Job
            RunDigest = o.RunDigest
            ReplicaDigest = o.ReplicaDigest
            LogitsDigest = r.LogitsDigest
            Tokens = r.Tokens
            Ends = r.Ends |> List.map LaneEnd.name
            Units = o.Units
            Epoch = o.Epoch
            Reissues = o.Reissues
            Duplicates = o.Duplicates
            Replays = o.Replays
            Rejected = o.Rejected
            CrossChecked = o.CrossChecked
            Unconfirmed = o.Unconfirmed
            Divergences = o.Divergences.Length
        }

    /// One line of JSON.
    let toJson (o: ServingOutcome) : string =
        let j = JsonObject()
        j["status"] <- JsonValue.Create o.Status
        j["job"] <- JsonValue.Create o.Job.Hex
        j["runDigest"] <- JsonValue.Create o.RunDigest.Hex
        j["replicaDigest"] <- JsonValue.Create o.ReplicaDigest.Hex
        j["logitsDigest"] <- JsonValue.Create o.LogitsDigest.Hex
        let tokens = JsonArray()

        for lane in o.Tokens do
            let a = JsonArray()

            for t in lane do
                a.Add(JsonValue.Create t)

            tokens.Add a

        j["tokens"] <- tokens
        let ends = JsonArray()

        for e in o.Ends do
            ends.Add(JsonValue.Create e)

        j["ends"] <- ends
        j["units"] <- JsonValue.Create o.Units
        j["epoch"] <- JsonValue.Create(Invariant.u64 o.Epoch)
        j["reissues"] <- JsonValue.Create o.Reissues
        j["duplicates"] <- JsonValue.Create o.Duplicates
        j["replays"] <- JsonValue.Create o.Replays
        j["rejected"] <- JsonValue.Create o.Rejected
        j["crossChecked"] <- JsonValue.Create o.CrossChecked
        j["unconfirmed"] <- JsonValue.Create o.Unconfirmed
        j["divergences"] <- JsonValue.Create o.Divergences
        j.ToJsonString()

    /// Parses toJson's text; None for anything else.
    let tryParse (text: string) : ServingOutcome option =
        try
            use doc = JsonDocument.Parse text
            let root = doc.RootElement
            let str (key: string) = root.GetProperty(key).GetString()
            let int (key: string) = root.GetProperty(key).GetInt32()

            Some
                {
                    Status = str "status"
                    Job = ContentId.parse (str "job")
                    RunDigest = ContentId.parse (str "runDigest")
                    ReplicaDigest = ContentId.parse (str "replicaDigest")
                    LogitsDigest = ContentId.parse (str "logitsDigest")
                    Tokens =
                        [
                            for lane in root.GetProperty("tokens").EnumerateArray() ->
                                [ for t in lane.EnumerateArray() -> t.GetInt32() ]
                        ]
                    Ends = [ for e in root.GetProperty("ends").EnumerateArray() -> e.GetString() ]
                    Units = int "units"
                    Epoch = Invariant.parseU64 (str "epoch")
                    Reissues = int "reissues"
                    Duplicates = int "duplicates"
                    Replays = int "replays"
                    Rejected = int "rejected"
                    CrossChecked = int "crossChecked"
                    Unconfirmed = int "unconfirmed"
                    Divergences = int "divergences"
                }
        with _ ->
            None

    /// Parses a coordinator's outcome line (with its prefix).
    let tryParseLine (line: string) =
        if isNull line || not (line.StartsWith(Prefix, StringComparison.Ordinal)) then
            None
        else
            tryParse (line.Substring Prefix.Length)

    /// How the outcome differs from the oracles, one line per difference; empty when the run is bit-identical: status
    /// completed with no divergence, the plan-order run's run and replica digests, and the single-process run's logits
    /// digest and greedy ids.
    let differences (reference: QwenReference) (o: ServingOutcome) : string list =
        [
            if o.Status <> "completed" then
                yield sprintf "status %s, not completed" o.Status
            if o.Divergences > 0 then
                yield sprintf "%d divergence(s)" o.Divergences
            if o.Job <> reference.JobId then
                yield sprintf "job %s, not %s" o.Job.Short reference.JobId.Short
            if o.RunDigest <> reference.Pipeline.RunDigest then
                yield
                    sprintf
                        "run digest %s, not the plan-order run's %s"
                        o.RunDigest.Short
                        reference.Pipeline.RunDigest.Short
            if o.ReplicaDigest <> reference.Pipeline.ReplicaDigest then
                yield
                    sprintf
                        "replica digest %s, not the plan-order run's %s"
                        o.ReplicaDigest.Short
                        reference.Pipeline.ReplicaDigest.Short
            if o.LogitsDigest <> reference.Single.LogitsDigest then
                yield
                    sprintf
                        "logits digest %s, not the single-process run's %s"
                        o.LogitsDigest.Short
                        reference.Single.LogitsDigest.Short
            if o.Tokens <> reference.Single.Tokens then
                yield sprintf "greedy ids %A, not the single-process run's %A" o.Tokens reference.Single.Tokens
        ]

    /// One summary line.
    let describe (o: ServingOutcome) =
        sprintf
            "%s, %d units, epoch %d, %d reissues, %d duplicates, %d replays, %d cross-checked, run digest %s, logits digest %s"
            o.Status
            o.Units
            o.Epoch
            o.Reissues
            o.Duplicates
            o.Replays
            o.CrossChecked
            o.RunDigest.Short
            o.LogitsDigest.Short
