# matmul_bf16w: matmul_bf16w throughput on d3d12


**PASS.** Every row alone equals its rows in the batched calls on all 6 shapes, and every dispatch is under the 100 ms budget.

- Throughput points: 62, the fastest 15.10 TFLOP/s (lm_head, m = 256)
- lm_head decode (m = 1): 718.85 µs, 433.72 GB/s of weight and activation traffic
- Slowest dispatch: 21.30 ms (lm_head, m = 1024), 21.3% of the 100 ms budget
- fp32 denormals: preserve

## Throughput

Each point is the faster of 5 back-to-back dispatches after one warm-up, from the backend's `gpu.dispatch` timers. FLOPs are the FLOP registry's 2mnk; traffic is its minimum (x and y in fp32, w in bf16, each element once).

![Throughput by m](charts/matmul-throughput.svg)

![Memory traffic by m](charts/matmul-bandwidth.svg)

![Dispatch time by m](charts/matmul-time.svg)

| Shape | m | n | k | Time | FLOP/s | Traffic | Of budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q_proj | 1 | 2,048 | 1,024 | 38.91 µs | 107.79 GFLOP/s | 108.11 GB/s | 0.0% |
| q_proj | 4 | 2,048 | 1,024 | 38.91 µs | 431.16 GFLOP/s | 109.05 GB/s | 0.0% |
| q_proj | 16 | 2,048 | 1,024 | 41.98 µs | 1.60 TFLOP/s | 104.59 GB/s | 0.0% |
| q_proj | 37 | 2,048 | 1,024 | 64.51 µs | 2.41 TFLOP/s | 72.06 GB/s | 0.1% |
| q_proj | 64 | 2,048 | 1,024 | 75.78 µs | 3.54 TFLOP/s | 65.73 GB/s | 0.1% |
| q_proj | 128 | 2,048 | 1,024 | 76.80 µs | 6.99 TFLOP/s | 75.09 GB/s | 0.1% |
| q_proj | 256 | 2,048 | 1,024 | 96.26 µs | 11.16 TFLOP/s | 76.26 GB/s | 0.1% |
| q_proj | 512 | 2,048 | 1,024 | 184.32 µs | 11.65 TFLOP/s | 56.89 GB/s | 0.2% |
| q_proj | 1,024 | 2,048 | 1,024 | 345.09 µs | 12.45 TFLOP/s | 48.62 GB/s | 0.3% |
| q_proj | 2,048 | 2,048 | 1,024 | 645.12 µs | 13.32 TFLOP/s | 45.51 GB/s | 0.6% |
| q_proj | 4,000 | 2,048 | 1,024 | 1.24 ms | 13.51 TFLOP/s | 42.95 GB/s | 1.2% |
| kv_proj | 1 | 1,024 | 1,024 | 38.91 µs | 53.89 GFLOP/s | 54.11 GB/s | 0.0% |
| kv_proj | 4 | 1,024 | 1,024 | 38.91 µs | 215.58 GFLOP/s | 54.74 GB/s | 0.0% |
| kv_proj | 16 | 1,024 | 1,024 | 43.01 µs | 780.19 GFLOP/s | 51.81 GB/s | 0.0% |
| kv_proj | 37 | 1,024 | 1,024 | 64.51 µs | 1.20 TFLOP/s | 37.21 GB/s | 0.1% |
| kv_proj | 64 | 1,024 | 1,024 | 75.78 µs | 1.77 TFLOP/s | 34.59 GB/s | 0.1% |
| kv_proj | 128 | 1,024 | 1,024 | 75.78 µs | 3.54 TFLOP/s | 41.51 GB/s | 0.1% |
| kv_proj | 256 | 1,024 | 1,024 | 75.78 µs | 7.08 TFLOP/s | 55.35 GB/s | 0.1% |
| kv_proj | 512 | 1,024 | 1,024 | 96.26 µs | 11.16 TFLOP/s | 65.36 GB/s | 0.1% |
| kv_proj | 1,024 | 1,024 | 1,024 | 185.34 µs | 11.59 TFLOP/s | 56.57 GB/s | 0.2% |
| kv_proj | 2,048 | 1,024 | 1,024 | 345.09 µs | 12.45 TFLOP/s | 54.69 GB/s | 0.3% |
| kv_proj | 4,000 | 1,024 | 1,024 | 641.02 µs | 13.09 TFLOP/s | 54.39 GB/s | 0.6% |
| o_proj | 1 | 1,024 | 2,048 | 76.80 µs | 54.61 GFLOP/s | 54.77 GB/s | 0.1% |
| o_proj | 4 | 1,024 | 2,048 | 75.78 µs | 221.41 GFLOP/s | 56.00 GB/s | 0.1% |
| o_proj | 16 | 1,024 | 2,048 | 83.97 µs | 799.22 GFLOP/s | 52.29 GB/s | 0.1% |
| o_proj | 37 | 1,024 | 2,048 | 126.98 µs | 1.22 TFLOP/s | 36.61 GB/s | 0.1% |
| o_proj | 64 | 1,024 | 2,048 | 149.50 µs | 1.80 TFLOP/s | 33.32 GB/s | 0.1% |
| o_proj | 128 | 1,024 | 2,048 | 149.50 µs | 3.59 TFLOP/s | 38.58 GB/s | 0.1% |
| o_proj | 256 | 1,024 | 2,048 | 150.53 µs | 7.13 TFLOP/s | 48.76 GB/s | 0.2% |
| o_proj | 512 | 1,024 | 2,048 | 188.42 µs | 11.40 TFLOP/s | 55.65 GB/s | 0.2% |
| o_proj | 1,024 | 1,024 | 2,048 | 363.52 µs | 11.81 TFLOP/s | 46.15 GB/s | 0.4% |
| o_proj | 2,048 | 1,024 | 2,048 | 681.98 µs | 12.60 TFLOP/s | 43.05 GB/s | 0.7% |
| o_proj | 4,000 | 1,024 | 2,048 | 1.27 ms | 13.19 TFLOP/s | 41.95 GB/s | 1.3% |
| gate_up_proj | 1 | 3,072 | 1,024 | 38.91 µs | 161.68 GFLOP/s | 162.11 GB/s | 0.0% |
| gate_up_proj | 4 | 3,072 | 1,024 | 38.91 µs | 646.74 GFLOP/s | 163.37 GB/s | 0.0% |
| gate_up_proj | 16 | 3,072 | 1,024 | 41.98 µs | 2.40 TFLOP/s | 156.10 GB/s | 0.0% |
| gate_up_proj | 37 | 3,072 | 1,024 | 64.51 µs | 3.61 TFLOP/s | 106.92 GB/s | 0.1% |
| gate_up_proj | 64 | 3,072 | 1,024 | 76.80 µs | 5.24 TFLOP/s | 95.57 GB/s | 0.1% |
| gate_up_proj | 128 | 3,072 | 1,024 | 96.26 µs | 8.37 TFLOP/s | 87.15 GB/s | 0.1% |
| gate_up_proj | 256 | 3,072 | 1,024 | 178.18 µs | 9.04 TFLOP/s | 58.85 GB/s | 0.2% |
| gate_up_proj | 512 | 3,072 | 1,024 | 273.41 µs | 11.78 TFLOP/s | 53.69 GB/s | 0.3% |
| gate_up_proj | 1,024 | 3,072 | 1,024 | 521.22 µs | 12.36 TFLOP/s | 44.26 GB/s | 0.5% |
| gate_up_proj | 2,048 | 3,072 | 1,024 | 973.82 µs | 13.23 TFLOP/s | 40.92 GB/s | 1.0% |
| gate_up_proj | 4,000 | 3,072 | 1,024 | 1.85 ms | 13.62 TFLOP/s | 38.88 GB/s | 1.8% |
| down_proj | 1 | 1,024 | 3,072 | 114.69 µs | 54.86 GFLOP/s | 55.00 GB/s | 0.1% |
| down_proj | 4 | 1,024 | 3,072 | 113.66 µs | 221.41 GFLOP/s | 55.93 GB/s | 0.1% |
| down_proj | 16 | 1,024 | 3,072 | 124.93 µs | 805.77 GFLOP/s | 52.46 GB/s | 0.1% |
| down_proj | 37 | 1,024 | 3,072 | 190.46 µs | 1.22 TFLOP/s | 36.22 GB/s | 0.2% |
| down_proj | 64 | 1,024 | 3,072 | 224.26 µs | 1.80 TFLOP/s | 32.73 GB/s | 0.2% |
| down_proj | 128 | 1,024 | 3,072 | 225.28 µs | 3.57 TFLOP/s | 37.24 GB/s | 0.2% |
| down_proj | 256 | 1,024 | 3,072 | 224.26 µs | 7.18 TFLOP/s | 46.76 GB/s | 0.2% |
| down_proj | 512 | 1,024 | 3,072 | 279.55 µs | 11.52 TFLOP/s | 52.51 GB/s | 0.3% |
| down_proj | 1,024 | 1,024 | 3,072 | 543.74 µs | 11.85 TFLOP/s | 42.43 GB/s | 0.5% |
| down_proj | 2,048 | 1,024 | 3,072 | 1.02 ms | 12.60 TFLOP/s | 38.95 GB/s | 1.0% |
| down_proj | 4,000 | 1,024 | 3,072 | 1.92 ms | 13.09 TFLOP/s | 37.37 GB/s | 1.9% |
| lm_head | 1 | 151,936 | 1,024 | 718.85 µs | 432.87 GFLOP/s | 433.72 GB/s | 0.7% |
| lm_head | 4 | 151,936 | 1,024 | 728.06 µs | 1.71 TFLOP/s | 430.75 GB/s | 0.7% |
| lm_head | 16 | 151,936 | 1,024 | 821.25 µs | 6.06 TFLOP/s | 390.81 GB/s | 0.8% |
| lm_head | 64 | 151,936 | 1,024 | 1.48 ms | 13.48 TFLOP/s | 237.08 GB/s | 1.5% |
| lm_head | 256 | 151,936 | 1,024 | 5.28 ms | 15.10 TFLOP/s | 88.67 GB/s | 5.3% |
| lm_head | 512 | 151,936 | 1,024 | 10.59 ms | 15.04 TFLOP/s | 58.95 GB/s | 10.6% |
| lm_head | 1,024 | 151,936 | 1,024 | 21.30 ms | 14.96 TFLOP/s | 44.02 GB/s | 21.3% |

## Batch invariance and CpuRef

Per shape, one m = 512 call, one m = 37 call and single-row calls (m = 1) over the same activation rows and weight. The rows alone must equal the rows of both batched calls bit for bit (BatchInvariance). The CpuRef column compares the single rows with CpuRef F32 (Rows.dotBf16); it is informational, because the conformance bound is KernelBounds.

| Shape | n | k | Calls (m) | Rows | Rows alone vs batched | vs CpuRef F32 |
| --- | ---: | ---: | --- | ---: | --- | --- |
| q_proj | 2048 | 1024 | 1,37,512 | 9 | ✓ identical | ✓ bit-exact |
| kv_proj | 1024 | 1024 | 1,37,512 | 9 | ✓ identical | ✓ bit-exact |
| o_proj | 1024 | 2048 | 1,37,512 | 9 | ✓ identical | ✓ bit-exact |
| gate_up_proj | 3072 | 1024 | 1,37,512 | 9 | ✓ identical | ✓ bit-exact |
| down_proj | 1024 | 3072 | 1,37,512 | 9 | ✓ identical | ✓ bit-exact |
| lm_head | 151936 | 1024 | 1,37,512 | 9 | ✓ identical | ✓ bit-exact |

## Arithmetic probes

Two dot products whose result tells the arithmetic apart: whether a `mad` step is fused (one rounding, as CpuRef's FusedMultiplyAdd), and whether the steps run in increasing p. Informational: the bound is KernelBounds.

| Probe | Result | CpuRef F32 | Other choice | What it shows |
| --- | --- | --- | --- | --- |
| fused | `30800000` | `30800000` | `00000000` | mad is one fused multiply-add per step (2^-30), not a rounded product and a sum (0): as CpuRef F32 |
| order | `00000000` | `00000000` | `3f800000` | the steps run in increasing p (+0), not in decreasing p (1): as CpuRef F32 |

## Run

| Field | Value |
| --- | --- |
| Run | `20261018-132542-33830713` |
| Title | matmul_bf16w throughput on d3d12 |
| Kind | kernels |
| Status | ok |
| Started | 2026-10-18T13:25:42.403Z |
| Ended | 2026-10-18T13:25:43.972Z |
| Execution class | d3d12 pci:10de:2704, 32.0.16.1692, win-x64 shaders 2d513d812671 |
| Class id | `4b1614e079f0` |
| Driver | 32.0.16.1692 |
| Commit | `558815bd640c4dee35eea5bfc49ee9d65ab113ac` |
| Nodes | 1 |
| Records | 1,026 |
| Synthetic | no |

| Parameter | Value |
| --- | --- |
| backend | d3d12 |
| device | NVIDIA GeForce RTX 4080 |
| reps | 5 |
| source | tests/Lrm.Kernels.Conformance/Cases/matmul/MatmulBench.fs (K2) |

### Log problems

None.
