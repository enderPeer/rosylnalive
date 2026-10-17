# TestTiers contract

Owner: S0 (S0a). Status: v1. Consumers: every test exe. The how-to guide is `tests/Lrm.Testing/README.md`.

## Test exes and tiers

- **Test exes.** Every `test` project in `docs/project-graph.json` is an Expecto exe whose entry point is `Runner.main argv tiers`. `tiers` is a `TestTiers` record that declares all four root lists, empty ones included:
  - `Cpu: Test list`: pure CPU tests. CI runs only this tier, with `--fail-on-focused-tests`.
  - `Gpu: (Backend * Test list) list`: GPU tests grouped by backend (`d3d12`, `vulkan`).
  - `Model: (Backend option * Test list) list`: full-model tests. `None` is CPU-only; `Some b` runs on backend `b`.
  - `Bench: Test list`: local timing. It is informational and never gates.
- **Selection.** `LRM_TEST_TIER` (or `--tier`) selects the tier. A missing or unknown tier exits 2. `LRM_BACKENDS` (for example `d3d12`) can restrict the GPU backends.

## Groups and mutexes

The runner splits a tier into groups and runs each group as one Expecto run under its mutexes. Groups without tests are dropped.

| Group | Mutexes held |
| --- | --- |
| `cpu` | none |
| `gpu.<backend>` | `Local\LrmGpu.<backend>` |
| `model.cpu` | `Local\LrmModelCpu` |
| `model.<backend>` | `Local\LrmGpu.<backend>`, then `Local\LrmModelCpu` |
| `bench` | all three |

- **Order.** Mutexes are always taken in the order `Local\LrmGpu.d3d12`, `Local\LrmGpu.vulkan`, `Local\LrmModelCpu`.
- **Holder.** An abandoned mutex (its holder died) counts as acquired. The holder writes a `.holder` file in `LRM_DATA/locks`.
- **Waiting.** A waiting process keeps a ticket in `LRM_DATA/locks` and prints its queue position and the holder PID every minute.
- **Wait limit.** The limit is 120 minutes; `LRM_LOCK_WAIT_MINUTES` overrides it. Reaching it exits 3.
- **Workers.** Child processes started with `Env.workerEnvironment` (`LRM_TEST_WORKER=1`) never take a mutex.
- **Verify priority.** When `LRM_DATA/locks/verify.flag` names a repo root (the integrator's verify worktree), runs from that tree go first. Other waiters do not take a mutex while a priority ticket is queued.

## Determinism and properties

- **GPU work runs twice.** `Determinism.runTwice` and `Determinism.testTwice` require byte-identical results and report the faster time.
- **Device record.** GPU tests record their device with `Devices.record backend device driver shaderSetHash`.
- **Property tests.** `Props.testProperty` replays a fixed FsCheck seed, `0x4c524d2d46534b31,0x9e3779b97f4a7c15`. `LRM_FSCHECK_SEED=seed[,gamma]` overrides it; values are decimal or `0x` hex, and the gamma must be odd. The seed in effect appears in every summary.

## Results

- **Files.** With `--results <dir>`, each exe writes `<exe>.<tier>.summary.json` and one `<exe>.<group>.junit.xml` per group. The summary uses schema `lrm.test-summary/1`:
  - top level: `exe`, `tier`, `fscheckSeed` and `exitCode`;
  - `groups`, one per group: `name`, `backend`, `locks`, `passed`, `failed`, `errored`, `ignored`, `durationMs`, `lockWaitMs`, `exitCode` and `error`;
  - `devices`: `backend`, `device`, `driver` and `shaderSetHash`.
- **Exit codes.** 0 means passed, 1 test failures, 2 a usage error or missing tier, and 3 the mutex wait limit.

## tools/test.ps1

`pwsh tools/test.ps1 -Tier <tier>` builds the solution (unless `-NoBuild`), then runs every test exe (or those named with `-Project`) as `dotnet <dll>`. It sets `LRM_TEST_TIER`, `LRM_REPO_ROOT` and `LRM_DATA`, the last one defaulting to the `dataRoot` in `toolchain.json`. It writes `<results>/summary.md` for PR bodies. The run fails in these cases:

- **An exe fails** or writes no summary.
- **Missing tier:** the tier ran no tests in any exe.
- **Required backends.** `LRM_REQUIRED_BACKENDS` (for example `d3d12,vulkan`) names backends that must run tests:
  - gpu tier: always enforced;
  - model tier: enforced only with `-RequireBackends`, which GPU model gates (X1b, I1, I2, I3) use. Without the switch, a listed backend that ran no model tests is reported as a note in the summary. CPU-only model runs, such as M2b's gate1-cpu, therefore stay valid while the variable is set;
  - `-RequireBackends` without `LRM_REQUIRED_BACKENDS`, or on the cpu or bench tier, is a usage error, and so is an unknown backend name.
- **Bench** never fails the run.
