# OpenCL™ Compatible Runtime

-   OpenCL 2.0 compatible language runtime
-   Supports offline and in-process/in-memory compilation

## DISCLAIMER

The information contained herein is for informational purposes only, and is subject to change without notice. In addition, any stated support is planned and is also subject to change. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information. Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described herein. No license, including implied or arising by estoppel, to any intellectual property rights is granted by this document. Terms and limitations applicable to the purchase or use of AMD’s products are as set forth in a signed agreement between the parties or in AMD's Standard Terms and Conditions of Sale.

© 2020 Advanced Micro Devices, Inc. All Rights Reserved.

## Getting the source code
Download the git projects using the following commands:

```bash
git clone -b main https://github.com/RadeonOpenCompute/ROCm-OpenCL-Runtime.git
```

## Repository branches

The repository maintains several branches. The branches that are of importance are:

- Main branch: This is the stable branch. It is up to date with the latest release branch, for example, if the latest ROCM release is rocm-4.1, main branch will be the repository based on this release.
- Release branches: These are branches corresponding to each ROCM release, listed with release tags, such as rocm-4.0, rocm-4.1, etc.

## Setup OpenCL
Copy the amdocl64.icd file to /etc/OpenCL/vendors

```bash
sudo cp api/opencl/config/amdocl64.icd /etc/OpenCL/vendors/
```

## Building
Follow these steps:

-   Build ROCclr first. Follow the steps in the following link to build ROCclr
   [ROCclr Readme](https://github.com/ROCm-Developer-Tools/ROCclr)
   In this step, $OPENCL_DIR and $ROCclr_DIR are defined.

-   Building OpenCL
Run these commands:

```bash
cd "$OPENCL_DIR"
mkdir -p build; cd build
cmake -DUSE_COMGR_LIBRARY=ON -DCMAKE_PREFIX_PATH="$ROCclr_DIR/build;/opt/rocm/" ..
make -j$(nproc)
```

Note: For release build, add "-DCMAKE_BUILD_TYPE=Release" to the cmake command line.

---
OpenCL™ is registered Trademark of Apple
