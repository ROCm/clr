#include "spirvUtils.h"

const unsigned SPRVMagicNumber = 0x07230203;

bool isSPIRVMagic(const void* Image, size_t Length) {
  if (Image == nullptr || Length < sizeof(unsigned))
    return false;
  auto Magic = static_cast<const unsigned*>(Image);
  return *Magic == SPRVMagicNumber;
}

// ToDo: replace this with SPIR-V validator when it is available.
bool
validateSPIRV(const void *Image, size_t Length) {
  return isSPIRVMagic(Image, Length);
}


