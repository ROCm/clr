# ROCclr - Radeon Open Compute Common Language Runtime
ROCclr is a virtual device interface that compute runtimes interact with to different backends such as ROCr or PAL
This abstraction allows runtimes to work on Windows as well as on Linux without much effort.

# DISCLAIMER

The information contained herein is for informational purposes only, and is subject to change without notice. In addition, any stated support is planned and is also subject to change. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information. Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described herein. No license, including implied or arising by estoppel, to any intellectual property rights is granted by this document. Terms and limitations applicable to the purchase or use of AMD’s products are as set forth in a signed agreement between the parties or in AMD's Standard Terms and Conditions of Sale.

© 2020 Advanced Micro Devices, Inc. All Rights Reserved.

## Repository branches
The repository maintains several branches. The branches that are of importance are:

- Main branch: This is the stable branch. It is up to date with the latest release branch, for example, if the latest ROCM release is rocm-4.1.x, main branch will be the repository based on this release.
- Release branches. These are branches corresponding to each ROCM release, listed with release tags, such as rocm-4.0.x, rocm-4.1.x, etc.

## Building

### Prerequisites

-   Install mesa-common-dev
-   Either build or install [COMGR](https://github.com/RadeonOpenCompute/ROCm-CompilerSupport), [CLANG](https://github.com/RadeonOpenCompute/llvm-project) and [Device Library](https://github.com/RadeonOpenCompute/ROCm-Device-Libs)

### Getting the source code

```bash
git clone -b main https://github.com/ROCm-Developer-Tools/ROCclr.git
git clone -b main https://github.com/RadeonOpenCompute/ROCm-OpenCL-Runtime.git
```

### Set the environment variables

```bash
export ROCclr_DIR="$(readlink -f ROCclr)"
export OPENCL_DIR="$(readlink -f ROCm-OpenCL-Runtime)"
```

### Build ROCclr
Here is command to build ROCclr:

```bash
cd "$ROCclr_DIR"
mkdir -p build; cd build
cmake -DOPENCL_DIR="$OPENCL_DIR" -DCMAKE_INSTALL_PREFIX=/opt/rocm/rocclr ..
make -j$(nproc)
sudo make install
```

### Optional steps to build HIP runtime
Enter the directory where git cloned the ROCClr and OpenCL. Run the following commands:

```bash
git clone -b main https://github.com/ROCm-Developer-Tools/HIP.git
export HIP_DIR="$(readlink -f HIP)"
cd "$HIP_DIR"
mkdir -p build; cd build
cmake -DCMAKE_PREFIX_PATH="$ROCclr_DIR/build;/opt/rocm/" ..
make -j$(nproc)
```

### Release build
For release build, add "-DCMAKE_BUILD_TYPE=Release" to the cmake command line.
