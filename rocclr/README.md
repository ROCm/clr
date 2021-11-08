# ROCclr - Radeon Open Compute Common Language Runtime
ROCclr is a virtual device interface that compute runtimes interact with to different backends such as ROCr or PAL
This abstraction allows runtimes to work on Windows as well as on Linux without much effort.

# DISCLAIMER

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard versionchanges, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated.AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

© 2021 Advanced Micro Devices, Inc. All Rights Reserved.

## Repository branches
The repository maintains several branches. The branches that are of importance are:

- Main branch: This is the stable branch. It is up to date with the latest release branch, for example, if the latest ROCM release is rocm-4.1.x, main branch will be the repository based on this release.
- Develop branch: This is the default branch, on which the new features are still under development and visible. While this maybe of interest to many, it should be noted that this branch and the features under development might not be stable.
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
