# ROCm OpenCL™ Runtime

Please build/install ROCclr first.

  Please view build steps here:
  https://github.com/ROCm-Developer-Tools/ROCclr

## Building OpenCL

cd $OPENCL_DIR

mkdir -p build; cd build

cmake -DCMAKE_PREFIX_PATH=/path/to/rocclr/build/or/install ..

make


