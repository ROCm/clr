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

## Measurements

[Historical V10 results](benchmarks/hassan_four_arm/V10_RESULTS.md) cover the
identified frozen binaries, including the earlier GLM TP4 study. Their old
on/off and central-diagnostic controls describe those historical binaries only;
see the [archived evaluation profile](benchmarks/hassan_four_arm/V10_EVALUATION.md).
New builds have distinct library hashes and require their own validation.

## Current-build validation

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
These checks do not include new full-model runs.
