# HIP graph runtime on MI355X

This branch provides one enabled runtime configuration. Loading its HIP/HSA
package activates the graph scheduling policy and eligible gfx950 native event
prewaits; no V10 enable flags are required. Applications keep the ordinary HIP
and PyTorch graph, stream, and event APIs.

Eligible captured kernel graphs carry completion dependencies between physical
GPU queues and defer the host-visible join until an event, synchronization, or
ordinary stream operation needs it. Qualified queue placement, lane retirement,
dependency fusion, and covered-tail handling are part of this policy.

The runtime retains ordinary submission for unsupported hardware, mixed or
multi-device graphs, profiling, unavailable native-wait storage, and private
signal-pool exhaustion. Eligibility and memory-ordering checks remain mandatory.
The generic HIP scheduler/debug controls still exist for compatibility; the
segment scheduler defaults to mode 2, matching the measured V10-on profile.
Validation targets Linux x86-64 and MI355X/gfx950. The graph placement and
retirement policy also applies on other HIP devices, but has not been validated
on those targets.

## Controls

The V10 mode switches have been removed, including `GPU_NATIVE_EVENT_WAIT`,
`GPU_GRAPH_NODE_COUNT_PLACEMENT`, and the graph spare, fusion, retirement,
local-signal, and frontier toggles. Setting those old variables to zero does not
disable this runtime. Tracing, bounded fault injection, and the private signal
resource cap remain available for diagnostics. Leave them at their defaults for
timing measurements.

## Packaging and rollback

Keep HIP and HSA together in an isolated package and verify the mapped library
paths and hashes in every worker. PyTorch can load bundled copies, so an intended
preload alone is insufficient evidence. Do not replace libraries in a running
process.

Rollback means restarting workers with stock HIP/HSA libraries, including any
container-local PyTorch library links. There is no feature-off runtime profile.
The package remains experimental; enabling it is a choice made by loading the
package.

## Build and reproduce

Build this CLR checkout for HIP with the ROCm 7.2.4 toolchain and HIP headers at
`bc9af25177f96c0fea93198b89cf4c3cf08f3ea3`:

```sh
cmake -S /path/to/clr -B /path/to/build-clr \
  -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIP_PLATFORM=amd \
  -D__HIP_ENABLE_PCH=OFF -DHIP_COMMON_DIR=/path/to/HIP \
  -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build /path/to/build-clr --parallel 8
```

Pair the resulting HIP library with the HSA library built by the
[companion ROCr PR](https://github.com/MarloweAI/rocm-systems/pull/1), based on
ROCm 7.2.4 source revision `97f5574fe2fdc7bef44fb01545347912ee9f1779`.
That change is required to reproduce GPU-local signal placement. A supported
public signal-placement interface remains an AMD design-review topic.

The [graph frontier suite](../tests/graph_frontier/README.md) provides
13 bounded correctness cases and two standalone microbenchmarks. The
[native event tests](../tests/native_event_wait/README.md) also include
a plain PyTorch counterpart. These are the focused reproductions shipped here.
The broader validation figures below describe the recorded evaluation and are
not the case count of this portable suite.

## Historical workload evidence

Earlier GLM TP4 and microbenchmark measurements remain available in the
[immutable V10 report](https://github.com/MarloweAI/clr/blob/875d0cb00c70a6ddc0f13e84028795b94d73ea59/tools/marlowe-runtime/benchmarks/hassan_four_arm/V10_RESULTS.md)
and its [binary profile](https://github.com/MarloweAI/clr/blob/875d0cb00c70a6ddc0f13e84028795b94d73ea59/tools/marlowe-runtime/benchmarks/hassan_four_arm/V10_EVALUATION.md).
Those measurements apply to the identified historical binaries.

## Recorded validation

On MI355X with ROCm 7.2.4, the default configuration passes 73 bounded
HIP/PyTorch correctness checks. A 329-cell microbenchmark comparison across four
rotated process rounds has a +0.015% median change versus frozen V10-on, with
no regression above 2%. Hassan's two-stream cases improve 28.5–28.6% versus
stock. The runtime still has stock-relative regressions: four broad-screen cells
at 2.5–3.4%, and larger losses on tiny two-root graphs replayed in groups of 50.

The measured HIP SHA-256 is
`c10f29781e9e4007a102bb93f767870a2d1506fd3032a8f634ac881b5a10c681`;
the matched HSA SHA-256 is
`2899f94063127a3c6d0bbba0e5c5c54cab3256499f2f0183f6a53631b5fdad12`.
The bundled portable suite also passes on a fresh build of this source: all
13 correctness cases and 864 timing rows across two alternating stock/candidate
rounds. That build has HIP SHA-256
`a2e735d5bcae61e22198e040b3aa03704f02dc0fd06136270349392349e4f505`
and the same matched HSA. These checks do not include new full-model runs.

## AMD review scope

This is a runtime prototype for technical review. The private ROCr attributes
and native packet encoding need an AMD-supported interface. Validation covers
Linux x86-64 and MI355X; other devices and operating systems remain unqualified.
Shared physical queues saturated behind host-controlled waits need additional
forward-progress validation because frontier submission holds multiple queue
locks. The bounded host-fed capacity test does not cover every such sequence.
