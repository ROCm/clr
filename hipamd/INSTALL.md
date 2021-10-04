## Prerequisites

-   Install mesa-common-dev
-   Either build or install [COMGR](https://github.com/RadeonOpenCompute/ROCm-CompilerSupport), [CLANG](https://github.com/RadeonOpenCompute/llvm-project) and [Device Library](https://github.com/RadeonOpenCompute/ROCm-Device-Libs)

## Getting the source code

```bash
git clone -b develop https://github.com/ROCm-Developer-Tools/hipamd.git
git clone -b develop https://github.com/ROCm-Developer-Tools/hip.git
git clone -b develop https://github.com/ROCm-Developer-Tools/ROCclr.git
git clone -b develop https://github.com/RadeonOpenCompute/ROCm-OpenCL-Runtime.git
```

## Set the environment variables

```bash
export HIPAMD_DIR="$(readlink -f hipamd)"
export HIP_DIR="$(readlink -f hip)"
export ROCclr_DIR="$(readlink -f ROCclr)"
export OPENCL_DIR="$(readlink -f ROCm-OpenCL-Runtime)"
```

## Build HIPAMD
Commands to build hipamd are as following,

```bash
cd "$HIPAMD_DIR"
mkdir -p build; cd build
cmake -DHIP_COMMON_DIR=$HIP_DIR -DAMD_OPENCL_PATH=$OPENCL_DIR -DROCCLR_PATH=$ROCCLR_DIR -DCMAKE_PREFIX_PATH="<ROCM_PATH>/" -DCMAKE_INSTALL_PREFIX=$PWD/install ..
make -j$(nproc)
sudo make install
```

Note,
HIP_COMMON_DIR looks for hip common ([HIP](https://github.com/ROCm-Developer-Tools/HIP/)) source codes.
By default, release version of hipamd is built.

