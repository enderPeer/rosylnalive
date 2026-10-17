/// Helpers and known-answer vectors shared by the Lrm.Cluster tests.
module Lrm.Cluster.Tests.Common

open System
open Lrm.Core
open Lrm.Cluster

/// Lowercase hex.
let hex (bytes: byte[]) = Convert.ToHexStringLower bytes

/// Bytes from hex that may contain spaces for readability.
let unhex (text: string) =
    Convert.FromHexString(text.Replace(" ", ""))

/// The vectors of docs/contracts/cluster.md, computed independently by vectors/cluster_vectors.py (Python hashlib).
module Vectors =
    /// ExecutionClass.id of FakeJob.executionClass.
    let fakeClass = "86d7e37b8867c60d8b79289e75bd715066544d2036790ebac258eb909107ee04"

    /// F1: Lease from the coordinator, epoch 7, seq 42, DurationMs 1000, LeaseEcho 41.
    let f1Payload = "e80300002900000000000000"

    let f1Frame =
        String.concat
            ""
            [
                "4c524d31" // magic "LRM1"
                "0100" // version 1
                "0400" // kind 4 (Lease)
                "00000000" // flags
                "00000000" // reserved
                "0700000000000000" // epoch 7
                "0000000000000000" // sender 0 (coordinator)
                "2a00000000000000" // seq 42
                "0c000000" // payload length 12
                "00000000" // reserved
                "0000000000000000000000000000000000000000000000000000000000000000" // unit id: none
                f1Payload
                "ca9a4fb535a0703a6464a32b4a939f809c9251f439361d91d3babd83e962f74a" // trailer
            ]

    /// The job id of U1 and U2: the bytes 01 02 ... 20.
    let unitJob = ContentId.OfBytes(Array.init 32 (fun i -> byte (i + 1)))

    let u1Payload =
        "090000006677642e737461676501000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20"
        + "86d7e37b8867c60d8b79289e75bd715066544d2036790ebac258eb909107ee04010000000200000000000000"
        + "0000000000000000000000000000000000000000000000000000000000000000"
        + "43dc986338811e8f6ba04bbba1545fdb098e2e138a1392cb276961ebfe901a2c0400000004000000"

    let u1Id = "b40822f7972990ed40b8c5d94d2bacb84df0586d900b5c2b81fc7a296bea44b3"

    let u2Payload =
        "090000006677642e737461676501000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20"
        + "86d7e37b8867c60d8b79289e75bd715066544d2036790ebac258eb909107ee04010000000200000001000000"
        + "b40822f7972990ed40b8c5d94d2bacb84df0586d900b5c2b81fc7a296bea44b3"
        + "ad6b5a670a9ed1a436317420b9e3b8bca558b3ac96adeed05537b66050c342c10400000004000000"

    let u2Id = "27e7e84f456d780f4e22a0504941bf186cf0746a3110e35b00701bf9759dbcc9"

    /// F2: Assign of U1 with input "abc", epoch 3, seq 9, the Retransmit flag.
    let f2Frame =
        "4c524d31010006000100000000000000030000000000000000000000000000000900000000000000aa00000000000000"
        + u1Id
        + u1Payload
        + "03000000616263"
        + "d1d88a5a72ac401416cf63a4352c18a6403e16d27bb7851868a929009077b5e9"

    /// The fake job { Seed 1; Lanes 2; Steps 3; Stages 3; PromptBytes 16; HiddenBytes 64 }.
    let smallJob =
        {
            Seed = 1UL
            Lanes = 2
            Steps = 3
            Stages = 3
            PromptBytes = 16
            HiddenBytes = 64
        }

    let smallJobId = "0d93494f91cde7827594a012c9ff0461e53a2c95a37d8f0fb96c5152fe568a77"
    let smallRun = "2f358424718b1c4c3b4bb1efb4cc4d7dcde1bb54acff4695e8f4b7f77960c14c"

    let smallReplicas =
        "d55283da804392ea3875f0686c394ff211e87d70f2234b611a0134f0cd1dbf2a"

    let smallTokens =
        [
            [ 3360429926u; 1564617581u; 3831387546u ]
            [ 3456249224u; 414529344u; 200954126u ]
        ]

    /// FakeJob.standard.
    let standardJobId =
        "77b04d909bce35e504e0249e2fa172f5d50ff8d6421f476e09cc3370395d578f"

    let standardRun = "507fb323a4f9e5efda3e7a0ce78bd9ff1c1ed07d6a95b8520850357bf90eec16"

    let standardReplicas =
        "b0a5410211781844069c973e94a169274ba43db3f50ba3b44a6976b5068f344f"

/// U1 of the vectors: fwd.stage v1, lane 1, stage 2, step 0, input "abc", Args u32 4.
let u1 =
    WorkUnit.fwdStage
        Vectors.unitJob
        FakeJob.classId
        1u
        2u
        0u
        UnitId.none
        (BlobId.ofBytes "abc"B)
        (Canonical.encode (fun w -> w.U32 4u))

/// U2 of the vectors: step 1 of U1's chain, input empty.
let u2 =
    { u1 with
        Step = 1u
        Prev = WorkUnit.id u1
        Input = BlobId.ofBytes [||]
    }

/// The reference of a fake job, computed once per job.
let reference =
    let cache = Collections.Concurrent.ConcurrentDictionary<FakeJob, FakeReference>()
    fun (job: FakeJob) -> cache.GetOrAdd(job, FakeReference.run)

/// A one-line description of a simulated run, for failure messages.
let describe (r: SimResult) =
    match r.Outcome with
    | None ->
        sprintf
            "seed %d did not finish: %A (%d events, %d ms virtual)"
            r.Seed
            r.Failure
            r.Stats.Events
            (r.Stats.VirtualNs / Ns.PerMs)
    | Some o ->
        sprintf
            "seed %d: %A, digest %s, %d units, epoch %d, %d reissues, %d replays, %d divergences, %d ms virtual"
            r.Seed
            o.Status
            o.RunDigest.Short
            o.Units
            o.Epoch
            o.Reissues
            o.Replays
            o.Divergences.Length
            (r.Stats.VirtualNs / Ns.PerMs)

/// The run digest of a finished simulation; fails the test otherwise.
let digestOf (r: SimResult) =
    match r.Outcome with
    | Some o -> o.RunDigest
    | None -> failwith (describe r)
