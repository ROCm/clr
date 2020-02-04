/* Copyright (c) 2008-present Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

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
