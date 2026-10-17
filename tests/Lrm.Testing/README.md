# Lrm.Testing

The shared test runner behind every test exe (TestTiers contract). Owner: S0.

## Writing tests

Each test exe declares its root lists per tier and hands them to `Runner.main`:

```fsharp
module Lrm.Gpu.Vulkan.Tests.Main

open Expecto
open Lrm.Testing

let tiers: TestTiers =
    { TestTiers.empty with
        Cpu = [ testCase "arg block packs to 128 bytes" (fun () -> ()) ]
        Gpu = [ Backend.Vulkan, [ Determinism.testTwice "scale_f32 round trip" (fun () -> [||]) ] ]
        Model = [ None, []; Some Backend.Vulkan, [] ]
        Bench = [] }

[<EntryPoint>]
let main argv = Runner.main argv tiers
```

- **cpu**: pure CPU tests. CI runs this tier with `--fail-on-focused-tests`.
- **gpu**: grouped by backend. The runner holds that backend's mutex while the group runs. Run GPU work twice and compare bytes (`Determinism.runTwice` or `Determinism.testTwice`); time with the faster run. Record the device with `Devices.record` so summaries show device, driver and ShaderSetHash.
- **model**: full-model tests. `None` groups hold `Local\LrmModelCpu`; GPU groups hold their backend mutex, then `Local\LrmModelCpu`.
- **bench**: local timing, informational only; it holds every mutex.
- **Property tests:** use `Props.testProperty`, which replays a fixed FsCheck seed. Override it with `LRM_FSCHECK_SEED=seed[,gamma]` (decimal or `0x` hex; the gamma must be odd).
- **Child processes** started by a test get `Env.workerEnvironment` (`LRM_TEST_WORKER=1`); workers never take a mutex.

## Running

`pwsh tools/test.ps1 -Tier <tier>` builds, runs every test exe and prints a Markdown summary for the PR body. A single exe can also run directly: `LRM_TEST_TIER=cpu dotnet artifacts/bin/<exe>/release/<exe>.dll [expecto args]`. A missing or unknown tier fails. `LRM_BACKENDS=d3d12` restricts a run to some backends.

**Required backends.** `LRM_REQUIRED_BACKENDS` (for example `d3d12,vulkan`) names the backends that must run tests:

- **gpu tier:** always enforced. A gpu run fails when a listed backend ran no gpu tests.
- **model tier:** enforced only with `-RequireBackends`, which GPU model gates use (X1b, I1, I2, I3). Without the switch, a listed backend that ran no model tests appears as a note in the summary and does not fail the run. This keeps CPU-only model runs such as M2b's gate1-cpu valid while `LRM_REQUIRED_BACKENDS` is set.
- `-RequireBackends` without `LRM_REQUIRED_BACKENDS`, or on the cpu or bench tier, is a usage error. An unknown backend name in `LRM_REQUIRED_BACKENDS` also fails the run.

## Mutexes and the queue

The mutexes are `Local\LrmGpu.d3d12`, `Local\LrmGpu.vulkan` and `Local\LrmModelCpu`, always taken in that order. A waiting process keeps a ticket in `LRM_DATA/locks` and prints its queue position and the holder PID every minute; the holder writes a `.holder` file. The wait limit is 120 minutes (`LRM_LOCK_WAIT_MINUTES` overrides it). An abandoned mutex (the holder died) counts as acquired.

Integrator priority: when `LRM_DATA/locks/verify.flag` contains a repo root (the verify worktree), runs from that tree go first, and other waiters do not take a mutex while a verify run is queued. Delete the flag when verification ends.
