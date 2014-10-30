#ifndef AMDILUTILS_H_
#define AMDILUTILS_H_

#include <string>

namespace amdilUtils {
// Change all private uav length in a kernel
void changePrivateUAVLength(std::string& kernel, unsigned length);

bool isKernelMemoryBound(const std::string& kernel);

}
#endif /* AMDILUTILS_H_ */
