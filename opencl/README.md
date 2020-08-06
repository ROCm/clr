# OpenCL™ Compatible Runtime

-   OpenCL 2.0 compatible language runtime
-   Supports offline and in-process/in-memory compilation

## Getting the source code
Download the git projects using the following commands:

```bash
git clone -b master-next https://github.com/RadeonOpenCompute/ROCm-OpenCL-Runtime.git
```

## Repository branches
The repository maintains several branches. The branches that are of importance are:

-    master-next: This is the default branch.

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
