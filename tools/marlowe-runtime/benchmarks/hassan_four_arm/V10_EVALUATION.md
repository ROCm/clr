# Frozen V10 evaluation profile

This directory is an unchanged copy of Hassan's four-arm reproducer at commit
`b8a96bb4d842f628142e417ebb6753d4492bcaff`. The reproducer's bundled runtime
recipe describes its historical experiment. It was not used to build or select
the V10 runtime evaluated with this CLR branch.

The frozen candidate was built from this branch's runtime sources and the ROCr
library used by the matched graph-frontier experiment. Its measured library
identities are:

- `libamdhip64.so`: `857a3d8d714f057cb04bc5025434200eaa5a3c6a31ad8224d94e59d13cd241a2`
- `libhsa-runtime64.so`: `2899f94063127a3c6d0bbba0e5c5c54cab3256499f2f0183f6a53631b5fdad12`

The candidate-on profile uses the following nonzero controls; every diagnostic
trace and fault-injection control remains zero:

```text
AMD_DIRECT_DISPATCH=1
GPU_MAX_HW_QUEUES=4
GPU_NATIVE_EVENT_WAIT=1
GPU_GRAPH_NODE_COUNT_PLACEMENT=1
GPU_GRAPH_DIAGNOSTIC_LANE_RETIRE=1
GPU_GRAPH_DIAGNOSTIC_COVERED_TAIL=1
GPU_GRAPH_DIAGNOSTIC_FUSE_DEPS=1
GPU_GRAPH_DIAGNOSTIC_FUSE_FINAL=1
GPU_GRAPH_DIAGNOSTIC_SPARE=1
GPU_GRAPH_DIAGNOSTIC_QUALIFIED_SPARE=1
DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=2
GPU_GRAPH_DIAGNOSTIC_KERNEL_RETIRE=1
GPU_GRAPH_DIAGNOSTIC_LOCAL_SIGNALS=1
GPU_GRAPH_DIAGNOSTIC_FRONTIER=1
GPU_GRAPH_DIAGNOSTIC_FRONTIER_DISTRIBUTED=1
GPU_STREAMOPS_CP_WAIT=0
GPU_NATIVE_EVENT_TRACE=0
GPU_GRAPH_DIAGNOSTIC_FRONTIER_TIMESTAMPS=0
```

Feature-off uses the same candidate libraries with native wait, node-count
placement, lane retirement, covered-tail fusion, qualified spare, kernel
retirement, local signals, frontier, and distributed-frontier controls set to
zero. It sets `DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=1`. Feature-off is therefore
not the stock ROCm runtime. The central diagnostic profile is also not
feature-off: it keeps the candidate-on profile and changes only
`GPU_GRAPH_DIAGNOSTIC_FRONTIER_DISTRIBUTED=0`.

## Deployment and rollback

Keep V10 in an isolated package containing both exact libraries and load the
package before importing PyTorch. Record the mapped library paths, SHA-256
values, and full control profile in every worker. Do not replace system ROCm
libraries in place.

Qualification is still in progress. Deployment requires completed and repeated
microbenchmark, HiSparse C1/C4 prefetch-off/on, and standard GLM C1/C4/C16
stock/off/on comparisons. Until those gates pass, the package is experimental.

Rollback is process-level: stop candidate workers and restart them without the
package preload and without candidate controls. Verify the restarted workers map
the stock image's HIP and HSA libraries. Do not change libraries in a live
process, and do not describe candidate feature-off as stock rollback.
