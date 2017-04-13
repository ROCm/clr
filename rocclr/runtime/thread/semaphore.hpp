//
// Copyright (c) 2008,2010 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef SEMAPHORE_HPP_
#define SEMAPHORE_HPP_

#include "top.hpp"
#include "utils/util.hpp"

#include <atomic>
#if defined(__linux__)
#include <semaphore.h>
#endif /*linux*/


namespace amd {

/*! \addtogroup Threads
 *  @{
 *
 *  \addtogroup Synchronization
 *  @{
 */

class Thread;

//! \brief Counting semaphore
class Semaphore : public HeapObject {
 private:
  std::atomic_int state_;  //!< This semaphore's value.

#ifdef _WIN32
  void* handle_;  //!< The semaphore object's handle.
  char padding_[64 - sizeof(void*) - sizeof(std::atomic_int)];
#else  // !_WIN32
  sem_t sem_;  //!< The semaphore object's identifier.
  char padding_[64 - sizeof(sem_t) - sizeof(std::atomic_int)];
#endif /*!_WIN32*/

 public:
  Semaphore();
  ~Semaphore();

  //! \brief Decrement this semaphore
  void wait();

  //! \brief Increment this semaphore
  void post();

  //! \brief Reset this semaphore.
  void reset() { state_.store(0, std::memory_order_release); }
};

/*! @}
 *  @}
 */

}  // namespace amd

#endif /*SEMAPHORE_HPP_*/
