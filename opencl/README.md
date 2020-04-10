# ROCm OpenCL™ Runtime 

Please build/install ROCclr first.

  Please view build steps here:
  https://github.com/ROCm-Developer-Tools/ROCclr

## Building OpenCL

cd $OPENCL_DIR

mkdir -p build; cd build

cmake -DVDI_DIR="$VDI_DIR" -DLIBVDI_STATIC_DIR="$VDI_DIR/build" ..

make


