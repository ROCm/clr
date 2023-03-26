## Prerequisites

-   Install mesa-common-dev
-   Either build or install [COMGR](https://github.com/RadeonOpenCompute/ROCm-CompilerSupport), [CLANG](https://github.com/RadeonOpenCompute/llvm-project) and [Device Library](https://github.com/RadeonOpenCompute/ROCm-Device-Libs)

## Branch of repository

Before get HIP source code, set the expected branch of repository at the variable HIP_BRANCH.
For example, for ROCm5.0 release branch, set
```
export HIP_BRANCH=rocm-5.0.x
```

ROCm5.1 release branch, set
```
export HIP_BRANCH=rocm-5.1.x
```
Similiar format for future branches.

## Getting the source code

```bash
git clone -b $HIP_BRANCH https://github.com/ROCm-Developer-Tools/hipamd.git
git clone -b $HIP_BRANCH https://github.com/ROCm-Developer-Tools/hip.git
git clone -b $HIP_BRANCH https://github.com/ROCm-Developer-Tools/ROCclr.git
git clone -b $HIP_BRANCH https://github.com/RadeonOpenCompute/ROCm-OpenCL-Runtime.git
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
cmake -DHIP_COMMON_DIR=$HIP_DIR -DAMD_OPENCL_PATH=$OPENCL_DIR -DROCCLR_PATH=$ROCCLR_DIR -DCMAKE_PREFIX_PATH="<ROCM_PATH>/" ..
make -j$(nproc)
sudo make install
```

Please note, HIP_COMMON_DIR looks for hip common ([HIP](https://github.com/ROCm-Developer-Tools/HIP/)) source codes.
By default, release version of hipamd is built. hip will be installed to the default path <ROCM_PATH>/hip

Developer can use cmake option CMAKE_INSTALL_PREFIX to define the path where hip is expected to be installed, commands to build are as following,
```bash
cd "$HIPAMD_DIR"
mkdir -p build; cd build
cmake -DHIP_COMMON_DIR=$HIP_DIR -DAMD_OPENCL_PATH=$OPENCL_DIR -DROCCLR_PATH=$ROCCLR_DIR -DCMAKE_PREFIX_PATH="<ROCM_PATH>/" -DCMAKE_INSTALL_PREFIX=$PWD/install ..
make -j$(nproc)
sudo make install
```

After installation, make sure HIP_PATH is pointed to the path where hip is installed.

