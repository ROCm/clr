# Graph frontier checks and microbenchmarks

These standalone HIP programs cover the ordering and ownership affected by this
runtime change. They need Linux x86-64, one allocated MI355X/gfx950 GPU, Python 3,
and ROCm 7.2.4 `hipcc`. They require no model, framework extension, dataset, Slurm,
or internal checkout. Run each process with exactly one visible GPU.

## Correctness checks

Pass a directory containing the candidate `libamdhip64.so` and its matching
`libhsa-runtime64.so`; see the [runtime build guide](../../docs/graph_frontier_runtime.md).
The runner compiles the fixtures, preloads both exact libraries in each child,
verifies their loader paths and SHA-256 values, checks numerical results, and
audits private-generation retirement. Each child has a 120-second timeout.
The output directory must not already exist.

```sh
HIP_VISIBLE_DEVICES=0 python3 hipamd/tests/graph_frontier/run.py \
  --runtime-dir /path/to/candidate/lib --output /tmp/frontier-checks
python3 hipamd/tests/graph_frontier/run.py --audit /tmp/frontier-checks
```

The 13 cases exercise:

| Fixture | Behavior checked |
| --- | --- |
| `graph_signal_generations.cpp` | In-flight parameter updates, generation reuse, destruction before synchronization, and resource caps of 2/8/32 |
| `graph_frontier_boundaries.cpp` | Upload ordering, public events, callbacks, long graph chains, concurrent observation, and resource exhaustion |
| `graph_frontier_idle.cpp` | Destruction initiates retirement without a later HIP operation |
| `graph_frontier_host_fed.cpp` | Capacity fallback makes progress while work awaits a later host action |
| `graph_boundary_alias.cpp` | Physical-queue aliases preserve dependencies across graph boundaries |
| `graph_lane_lifetime.cpp` | Node disable/update, stream switching, single-queue fallback, and partial-publication fault cleanup |
| `../native_event_wait/native_pool_pressure.cpp` | Busy native instruction storage falls back without blocking host submission |

One boundary case sets obsolete V10 enable variables to zero and still requires
frontier trace evidence. This verifies the single enabled policy. The single-hardware-queue
case requires ordinary submission. Tracing and fault injection are
used only by these untimed checks.

## Microbenchmarks

```sh
HIP_VISIBLE_DEVICES=0 python3 hipamd/tests/graph_frontier/run.py \
  --runtime-dir /path/to/stock/lib --output /tmp/stock-micros --benchmarks-only
HIP_VISIBLE_DEVICES=0 python3 hipamd/tests/graph_frontier/run.py \
  --runtime-dir /path/to/candidate/lib --output /tmp/candidate-micros --benchmarks-only
```

Repeat in fresh processes, alternating which runtime goes first. Keep GPU
visibility and hardware queue count identical. The runner disables tracing and
records CSV timings separately from its library receipts and correctness checks.
`--audit` rechecks saved output without a GPU.

`graph_microbenchmark.cpp` tests one or two independent kernel chains, groups of
1 or 50 kernels, and three bounded spin lengths. It checks the exact increment
count after every trial. Its `gpu_us`, `host_us`, and `submit_us` columns are
normalized per unit of work; grouped graph launches contain 50 units. It is a
synthetic scheduling probe, separate from the historical Hassan GEMM benchmark.

`minimal_wait.cpp` compares a 2,048-increment graph alone, with a pending external
waiter, and with an already-completed waiter. The producer event must still report `hipErrorNotReady`
when the external waiter is submitted. It checks producer output on every replay.

The suite exposes existing limitations: tiny two-root grouped graphs can be
slower than stock. It does not establish full-model performance or validate
shared-ring saturation with arbitrary host-controlled dependencies.
