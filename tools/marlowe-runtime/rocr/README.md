# Matching ROCr signal storage

The measured runtime uses GPU-local, CPU-mapped storage for private graph
completion signals. `signal-locality.patch` is the exact two-file ROCr change
used by the tested HSA binary. Its upstream pin and SHA-256 are in
[source-lock.json](source-lock.json). The patch applies inside
`projects/rocr-runtime` of `ROCm/rocm-systems` at
`97f5574fe2fdc7bef44fb01545347912ee9f1779`.

```sh
git -C /path/to/rocm-systems checkout 97f5574fe2fdc7bef44fb01545347912ee9f1779
git -C /path/to/rocm-systems apply --check \
  --directory=projects/rocr-runtime /path/to/clr/tools/marlowe-runtime/rocr/signal-locality.patch
git -C /path/to/rocm-systems apply \
  --directory=projects/rocr-runtime /path/to/clr/tools/marlowe-runtime/rocr/signal-locality.patch
```

Build that subtree with its upstream CMake instructions and development
dependencies (`libelf`, `libdrm`, `pkg-config`, `rocm-core`, and `rocm-llvm-dev`).
The tested build used ROCm 7.2.4 clang/clang++, Release, and
`-DROCM_PATCH_VERSION=70204`. Copy the resulting `libhsa-runtime64.so` into the
same isolated directory as the candidate HIP library; do not install it over a
system runtime. The native test runner verifies the libraries actually loaded.

The private attributes use bits 62 and 63 of `hsa_amd_signal_create`. They create
real `BusyWaitSignal` objects over dedicated 4 KiB backing storage and avoid a
CPU locked read/modify/write during destruction. The caller must retire all GPU
readers before release and must not register CPU waiters or asynchronous host
handlers on these private signals. Destroying one with a retained host waiter or
asynchronous owner aborts the process. Ordinary public signals are unchanged.

This is an experimental implementation dependency, not a supported public HSA
extension. AMD review is needed to choose a supported signal-placement and
lifetime interface. A stock ROCr build does not reproduce the measured placement
policy. The pinned patch retains its matched-host diagnostic allocation branch;
CLR uses only its GPU-local allocation.
