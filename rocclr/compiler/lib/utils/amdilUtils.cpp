#include "amdilUtils.hpp"
#include <regex>
#include <string>
#include <sstream>

// Change all private uav length in a kernel
void amdilUtils::changePrivateUAVLength(std::string& kernel, unsigned length) {
  std::regex pattern("dcl_typeless_uav_id\\(([[:digit:]]+)\\)_stride"
      "\\(([[:digit:]]+)\\)_length\\([[:digit:]]+\\)_access\\(private\\)");
  std::stringstream ss;
  ss << "dcl_typeless_uav_id($1)_stride($2)_length(" << length <<
      ")_access(private)";
  kernel = std::regex_replace(kernel, pattern, ss.str());
}
