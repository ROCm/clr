# Pending event interference on gfx950

These standalone reproducers submit the same graph of2,048 increments in three
cases: no waiter, a pending event waited by an otherwise empty side stream, and
an already-completed event waited by that stream. Every measured replay checks
producer output. The Python version uses ordinary PyTorch streams, events and
CUDAGraph; it has no custom extension. PyTorch and C++ kernel implementations
differ, so compare each program against its own no-wait control.

Build the HIP version with `hipcc -O3 -std=c++17 --offload-arch=gfx950
minimal_wait.cpp -o minimal_wait`. Run on one allocated GPU and preserve GPU
visibility. The Python version requires a compatible ROCm PyTorch installation.

Select a matching HIP and HSA runtime using your container's loader configuration.
Verify that exactly one of each is mapped: PyTorch may otherwise load a bundled
HIP runtime despite LD_LIBRARY_PATH. For the tested ROCm7.2.4 image, use both
`LD_PRELOAD=libhsa-runtime64.so:libamdhip64.so` and an LD_LIBRARY_PATH with the
candidate runtime directory first. Inspect the Python stderr library receipt.

Run the identical executable/script in separate processes with
`GPU_NATIVE_EVENT_WAIT=0` and `GPU_NATIVE_EVENT_WAIT=1`, alternating order across
repeats. Leave GPU_MAX_HW_QUEUES at its default. Do not enable trace logging in
clean timing runs. Keep the CPU query proving that the producer is still pending
before submitting its external waiter; completed-event cases do not exercise
the problem. A reduced difference is a mitigation result, not zero-cost waiting
or a prediction of full-model throughput.

This release backport is default-off and gfx950-only. It retains the original
AQL dependency packet after a native prewait. IRQ-backed signal layout and PM4
encoding follow ROCr internals and require an AMD-owned supported interface for
broader deployment. Do not treat this test alone as production qualification.

`native_pool_pressure.cpp` is a bounded enqueue-progress regression test. It holds
one producer event pending while submitting 2,300 waits and consumer kernels on
another stream, below the default AQL queue capacity. A two-second watchdog keeps
a failing runtime recoverable. Exit 7 means the watchdog fired or the producer
completed before all submissions; correct output alone is insufficient. The test
uses one visible GPU and prints the actual HIP/HSA mappings. Compile with:

```
hipcc -O2 -std=c++17 -pthread --offload-arch=gfx950 native_pool_pressure.cpp -o native_pool_pressure
GPU_NATIVE_EVENT_WAIT=1 <release>/run ./native_pool_pressure
```

The original RC3 runtime needs watchdog release at the second native instruction
pool rotation; its disabled mode completes submission while the event is pending.
The optional native path must fall back to AQL when an instruction chunk is busy.
