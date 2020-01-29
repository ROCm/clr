//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef COUNTERS_HPP_
#define COUNTERS_HPP_

#include "top.hpp"

namespace amd {

/*! \addtogroup Runtime
 *  @{
 *
 *  \addtogroup Devicecounter
 *  @{
 */

/*! \class Counter
 *
 *  \brief The container class for the performance counters
 */
class Counter : public RuntimeObject {
 public:
  //! RTTI internal implementation
  virtual ObjectType objectType() const { return ObjectTypeCounter; }
};

/*@}*/
/*@}*/ } // namespace amd

#endif  // COUNTERS_HPP_
