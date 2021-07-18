## Prerequisites

-   Install mesa-common-dev
-   Either build or install [COMGR](https://github.com/RadeonOpenCompute/ROCm-CompilerSupport), [CLANG](https://github.com/RadeonOpenCompute/llvm-project) and [Device Library](https://github.com/RadeonOpenCompute/ROCm-Device-Libs)

## Getting the source code

```bash
git clone -b rocm-4.4.x https://github.com/ROCm-Developer-Tools/hipamd.git
git clone -b rocm-4.4.x https://github.com/ROCm-Developer-Tools/hip.git
git clone -b rocm-4.4.x https://github.com/ROCm-Developer-Tools/ROCclr.git
git clone -b rocm-4.4.x https://github.com/RadeonOpenCompute/ROCm-OpenCL-Runtime.git
```

## Set the environment variables

```bash
export HIPAMD_DIR="$(readlink -f hipamd)"
export HIP_DIR="$(readlink -f hip)"
export ROCclr_DIR="$(readlink -f ROCclr)"
export OPENCL_DIR="$(readlink -f ROCm-OpenCL-Runtime)"
```

## Build AMDHIP
Commands to build amdhip are as following,

```bash
cd "$HIPAMD_DIR"
mkdir -p build; cd build
cmake -DHIP_COMMON_DIR=$HIP_DIR -DAMD_OPENCL_PATH=$OPENCL_DIR -DROCCLR_PATH=$ROCCLR_DIR -DCMAKE_PREFIX_PATH="/opt/rocm/" -DCMAKE_INSTALL_PREFIX=$PWD/install ..
make -j$(nproc)
sudo make install
```

Note,
HIP_COMMON_DIR looks for hip common source codes.
By default, release version of AMDHIP is built.

