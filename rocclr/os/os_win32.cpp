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

#if defined(_WIN32) || defined(__CYGWIN__)

#include "os/os.hpp"
#include "thread/thread.hpp"

#include <windows.h>
#include <process.h>
#include <tchar.h>
#include <time.h>
#include <intrin.h>

#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>
#include <algorithm>

#ifndef WINAPI
#define WINAPI
#endif


BOOL(WINAPI* pfnGetNumaNodeProcessorMaskEx)(USHORT, PGROUP_AFFINITY) = NULL;

namespace amd {

static size_t allocationGranularity_;

static LONG WINAPI divExceptionFilter(struct _EXCEPTION_POINTERS* ep);

#ifdef _WIN64
PVOID divExceptionHandler = NULL;
#endif  // _WIN64

static double PerformanceFrequency;

typedef BOOL(WINAPI* SetThreadGroupAffinity_fn)(__in HANDLE, __in CONST GROUP_AFFINITY*,
                                                __out_opt PGROUP_AFFINITY);
static SetThreadGroupAffinity_fn pfnSetThreadGroupAffinity = NULL;

#pragma section(".CRT$XCU", long, read)
__declspec(allocate(".CRT$XCU")) bool (*__init)(void) = Os::init;

bool Os::init() {
  static bool initialized_ = false;

  // We could use InitOnceExecuteOnce here:
  if (initialized_) {
    return true;
  }
  initialized_ = true;

  SYSTEM_INFO si;
  ::GetSystemInfo(&si);
  pageSize_ = si.dwPageSize;
  allocationGranularity_ = (size_t)si.dwAllocationGranularity;
  processorCount_ = si.dwNumberOfProcessors;

  LARGE_INTEGER frequency;
  QueryPerformanceFrequency(&frequency);
  PerformanceFrequency = (double)frequency.QuadPart;

  HMODULE handle = ::LoadLibrary("kernel32.dll");
  if (handle != NULL) {
    pfnSetThreadGroupAffinity =
        (SetThreadGroupAffinity_fn)::GetProcAddress(handle, "SetThreadGroupAffinity");
    pfnGetNumaNodeProcessorMaskEx = (BOOL(WINAPI*)(USHORT, PGROUP_AFFINITY))::GetProcAddress(
        handle, "GetNumaNodeProcessorMaskEx");
  }

  return Thread::init();
}

#pragma section(".CRT$XTU", long, read)
__declspec(allocate(".CRT$XTU")) void (*__exit)(void) = Os::tearDown;

void Os::tearDown() { Thread::tearDown(); }

void* Os::loadLibrary_(const char* filename) {
  if (filename != NULL) {
    HMODULE hModule = ::LoadLibrary(filename);
    return hModule;
  }
  return NULL;
}

void Os::unloadLibrary(void* handle) { ::FreeLibrary((HMODULE)handle); }

void* Os::getSymbol(void* handle, const char* name) {
  return ::GetProcAddress((HMODULE)handle, name);
}

static inline int memProtToOsProt(Os::MemProt prot) {
  switch (prot) {
    case Os::MEM_PROT_NONE:
      return PAGE_NOACCESS;
    case Os::MEM_PROT_READ:
      return PAGE_READONLY;
    case Os::MEM_PROT_RW:
      return PAGE_READWRITE;
    case Os::MEM_PROT_RWX:
      return PAGE_EXECUTE_READWRITE;
    default:
      break;
  }
  ShouldNotReachHere();
  return -1;
}

address Os::reserveMemory(address start, size_t size, size_t alignment, MemProt prot) {
  size = alignUp(size, pageSize());
  alignment = std::max(allocationGranularity_, alignUp(alignment, allocationGranularity_));
  assert(isPowerOfTwo(alignment) && "not a power of 2");

  size_t requested = size + alignment - allocationGranularity_;
  address mem, aligned;
  do {
    mem = (address)VirtualAlloc(start, requested, MEM_RESERVE, memProtToOsProt(prot));

    // check for out of memory.
    if (mem == NULL) return NULL;

    aligned = alignUp(mem, alignment);

    // check for already aligned memory.
    if (aligned == mem && size == requested) {
      return mem;
    }

    // try to reserve the aligned address.
    if (VirtualFree(mem, 0, MEM_RELEASE) == 0) {
      assert(!"VirtualFree failed");
    }

    mem = (address)VirtualAlloc(aligned, size, MEM_RESERVE, memProtToOsProt(prot));
    assert((mem == NULL || mem == aligned) && "VirtualAlloc failed");

  } while (mem != aligned);

  return mem;
}

bool Os::releaseMemory(void* addr, size_t size) { return VirtualFree(addr, 0, MEM_RELEASE) != 0; }

bool Os::commitMemory(void* addr, size_t size, MemProt prot) {
  return VirtualAlloc(addr, size, MEM_COMMIT, memProtToOsProt(prot)) != NULL;
}

bool Os::uncommitMemory(void* addr, size_t size) {
  return VirtualFree(addr, size, MEM_DECOMMIT) != 0;
}

bool Os::protectMemory(void* addr, size_t size, MemProt prot) {
  DWORD OldProtect;
  return VirtualProtect(addr, size, memProtToOsProt(prot), &OldProtect) != 0;
}


uint64_t Os::hostTotalPhysicalMemory() {
  static uint64_t totalPhys = 0;

  if (totalPhys != 0) {
    return totalPhys;
  }

  MEMORYSTATUSEX mstatus;
  mstatus.dwLength = sizeof(mstatus);

  ::GlobalMemoryStatusEx(&mstatus);

  totalPhys = mstatus.ullTotalPhys;
  return totalPhys;
}

void* Os::alignedMalloc(size_t size, size_t alignment) {
  return ::_aligned_malloc(size, alignment);
}

void Os::alignedFree(void* mem) { ::_aligned_free(mem); }


void Os::currentStackInfo(address* base, size_t* size) {
  MEMORY_BASIC_INFORMATION mbInfo;

  address currentStackPage = (address)alignDown((intptr_t)currentStackPtr(), pageSize());

  ::VirtualQuery(currentStackPage, &mbInfo, sizeof(mbInfo));

  address stackBottom = (address)mbInfo.AllocationBase;
  size_t stackSize = 0;

  do {
    stackSize += mbInfo.RegionSize;
    ::VirtualQuery(stackBottom + stackSize, &mbInfo, sizeof(mbInfo));
  } while (stackBottom == (address)mbInfo.AllocationBase);

  *base = stackBottom + stackSize;
  *size = stackSize;

  assert(currentStackPtr() >= *base - *size && currentStackPtr() < *base && "just checking");
}

#define MS_VC_EXCEPTION 0x406D1388
#pragma pack(push, 8)
struct THREADNAME_INFO {
  DWORD dwType;      // Must be 0x1000.
  LPCSTR szName;     // Pointer to name (in user addr space).
  DWORD dwThreadID;  // Thread ID (-1=caller thread).
  DWORD dwFlags;     // Reserved for future use, must be zero.
};
#pragma pack(pop)

static void SetThreadName(DWORD threadId, const char* name) {
  if (name == NULL || *name == '\0') {
    return;
  }

  THREADNAME_INFO info;
  info.dwType = 0x1000;
  info.szName = name;
  info.dwThreadID = threadId;
  info.dwFlags = 0;

  __try {
    ::RaiseException(0x406D1388, 0, sizeof(info) / sizeof(ULONG_PTR), (ULONG_PTR*)&info);
  } __except (EXCEPTION_EXECUTE_HANDLER) {
  }
}

void Os::setCurrentThreadName(const char* name) { SetThreadName(GetCurrentThreadId(), name); }

static LONG WINAPI divExceptionFilter(struct _EXCEPTION_POINTERS* ep) {
  DWORD code = ep->ExceptionRecord->ExceptionCode;

  if ((code == EXCEPTION_INT_DIVIDE_BY_ZERO || code == EXCEPTION_INT_OVERFLOW) &&
      Thread::current()->isWorkerThread()) {
    address insn = (address)ep->ContextRecord->LP64_SWITCH(Eip, Rip);

    if (Os::skipIDIV(insn)) {
      ep->ContextRecord->LP64_SWITCH(Eip, Rip) = (uintptr_t)insn;
      return EXCEPTION_CONTINUE_EXECUTION;
    }
  }
  return EXCEPTION_CONTINUE_SEARCH;
}

bool Os::installSigfpeHandler() {
#ifdef _WIN64
  divExceptionHandler = AddVectoredExceptionHandler(1, divExceptionFilter);
#endif  // _WIN64
  return true;
}

void Os::uninstallSigfpeHandler() {
#ifdef _WIN64
  if (divExceptionHandler != NULL) {
    RemoveVectoredExceptionHandler(divExceptionHandler);
    divExceptionHandler = NULL;
  }
#endif  // _WIN64
}

void* Thread::entry(Thread* thread) {
  void* ret = NULL;
#if !defined(_WIN64)
  __try {
    ret = thread->main();
  } __except (divExceptionFilter(GetExceptionInformation())) {
    // nothing to do here.
  }
#else   // _WIN64
  ret = thread->main();
#endif  // _WIN64

// The current thread exits, thus clear the pointer
#if defined(USE_DECLSPEC_THREAD)
  details::thread_ = NULL;
#else   // !USE_DECLSPEC_THREAD
  TlsSetValue(details::threadIndex_, NULL);
#endif  // !USE_DECLSPEC_THREAD
  return ret;
}

bool Os::isThreadAlive(const Thread& thread) {
  HANDLE handle = (HANDLE)(thread.handle());

  DWORD exitCode = 0;
  if (GetExitCodeThread(handle, &exitCode)) {
    return exitCode == STILL_ACTIVE;
  } else {
    // Could not get thread's exitcode
    return false;
  }
}

const void* Os::createOsThread(Thread* thread) {
  HANDLE handle = ::CreateThread(NULL, thread->stackSize_, (LPTHREAD_START_ROUTINE)Thread::entry,
                                 thread, 0, NULL);
  if (handle == NULL) {
    thread->setState(Thread::FAILED);
  }
  return reinterpret_cast<const void*>(handle);
}

void Os::setThreadAffinity(const void* handle, const Os::ThreadAffinityMask& mask) {
  if (pfnSetThreadGroupAffinity != NULL) {
    GROUP_AFFINITY group = {0};
    for (WORD i = 0; i < sizeof(mask.mask_) / sizeof(KAFFINITY); ++i) {
      group.Mask = mask.mask_[i];
      group.Group = i;
      if (group.Mask != 0) {
        pfnSetThreadGroupAffinity((HANDLE)handle, &group, NULL);
      }
    }
  } else {  // pfnSetThreadGroupAffinity == NULL
    DWORD_PTR threadAffinityMask = (DWORD_PTR)mask.mask_[0];
    if (threadAffinityMask != 0) {
      ::SetThreadAffinityMask((HANDLE)handle, threadAffinityMask);
    }
  }
}

void Os::yield() { ::SwitchToThread(); }

uint64_t Os::timeNanos() {
  LARGE_INTEGER current;
  QueryPerformanceCounter(&current);
  return (uint64_t)((double)current.QuadPart / PerformanceFrequency * 1e9);
}

uint64_t Os::timerResolutionNanos() { return (uint64_t)(1e9 / PerformanceFrequency); }


const char* Os::libraryExtension() { return ".DLL"; }

const char* Os::libraryPrefix() { return NULL; }

const char* Os::objectExtension() { return ".OBJ"; }

char Os::fileSeparator() { return '\\'; }

char Os::pathSeparator() { return ';'; }

bool Os::pathExists(const std::string& path) {
  return GetFileAttributes(path.c_str()) != INVALID_FILE_ATTRIBUTES;
}

bool Os::createPath(const std::string& path) {
  size_t pos = 0;
  while (true) {
    pos = path.find(fileSeparator(), pos);
    const std::string currPath = path.substr(0, pos);
    if (!currPath.empty() && !pathExists(currPath)) {
      if (!CreateDirectory(currPath.c_str(), NULL)) return false;
    }
    if (pos == std::string::npos) break;
    ++pos;
  }
  return true;
}

bool Os::removePath(const std::string& path) {
  size_t pos = std::string::npos;
  bool removed = false;
  while (true) {
    const std::string currPath = path.substr(0, pos);
    if (!currPath.empty()) {
      if (!RemoveDirectory(currPath.c_str())) return removed;
      removed = true;
    }
    if (pos == 0) break;
    pos = path.rfind(fileSeparator(), pos == std::string::npos ? pos : pos - 1);
    if (pos == std::string::npos) break;
  }
  return true;
}

int Os::printf(const char* fmt, ...) {
  va_list ap;
  DWORD dwBytesWritten;

  va_start(ap, fmt);
  int len = ::_vsnprintf(NULL, 0, fmt, ap);
  va_end(ap);
  if (len <= 0) return len;

  va_start(ap, fmt);
  char* str = static_cast<char*>(alloca(len + 1));
  len = ::_vsnprintf(str, len + 1, fmt, ap);
  va_end(ap);
  if (len <= 0) return len;

  ::WriteFile(::GetStdHandle(STD_OUTPUT_HANDLE), str, len, &dwBytesWritten, NULL);

  return len;
}

int Os::systemCall(const std::string& command) {
#if 1
  char* cmd = new char[command.size() + 1];
  fastMemcpy(cmd, command.c_str(), command.size());
  cmd[command.size()] = 0;

  STARTUPINFO si = {0};
  si.cb = sizeof(si);
  PROCESS_INFORMATION pi;

  if (::CreateProcess(NULL, cmd, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi) == 0) {
    delete[] cmd;
    return -1;  // failed
  };

  // Wait until child process exits.
  ::WaitForSingleObject(pi.hProcess, INFINITE);

  DWORD ExitCode = 0;
  ::GetExitCodeProcess(pi.hProcess, &ExitCode);

  // Close process and thread handles.
  ::CloseHandle(pi.hProcess);
  ::CloseHandle(pi.hThread);

  delete[] cmd;
  return (int)ExitCode;
#else
  std::stringstream str;
  str << "\"" << command << "\"";
  return ::system(str.str().c_str());
#endif
}

std::string Os::getEnvironment(const std::string& name) {
  char dstBuf[MAX_PATH];
  size_t dstSize;

  if (::getenv_s(&dstSize, dstBuf, MAX_PATH, name.c_str())) {
    return std::string("");
  }
  return std::string(dstBuf);
}

std::string Os::getTempPath() {
  char tempPath[MAX_PATH];
  uint ret = GetTempPath(MAX_PATH, tempPath);
  if (ret == 0 || (ret == 1 && tempPath[0] == '?')) {
    return std::string(".");
  }

  // If the app was started from an UNC path instead of a DOS path,
  // the temp env var won't be set correctly and will point to windows
  // system directory instead (usually c:/windows/temp), which will be
  // blocked. So we check if the temp path returned by GetTempPath is
  // under windows directory, use . instead
  std::string tempPathStr(tempPath);
  char winPath[MAX_PATH];
  if (GetWindowsDirectory(winPath, MAX_PATH) > 0) {
    // Need to check if tempPath is C:\Windows or C:\Windows\ //
    if (tempPath[strlen(tempPath) - 1] == '\\') {
      tempPath[strlen(tempPath) - 1] = '\0';
      ret--;
    }
    if (_memicmp(tempPath, winPath, ret) == 0) {
      return std::string(".");
    }
  }
  return tempPathStr;
}


std::string Os::getTempFileName() {
  static std::atomic_size_t counter(0);

  std::string tempPath = getTempPath();
  std::stringstream tempFileName;

  tempFileName << tempPath << "\\OCL" << ::_getpid() << 'T' << counter++;
  return tempFileName.str();
}

int Os::unlink(const std::string& path) { return ::_unlink(path.c_str()); }

void Os::cpuid(int regs[4], int info) { return __cpuid(regs, info); }

uint64_t Os::xgetbv(uint32_t ecx) { return (uint64_t)_xgetbv(ecx); }

// Various "fast" memcpy implementation (currently win32 only due to compiler limitations)

// (dgladdin - "recent" below means MMX and later)

// Very optimized memcpy() routine for all AMD Athlon and Duron family.
// This code uses any of FOUR different basic copy methods, depending
// on the transfer size.
// NOTE:  Since this code uses MOVNTQ (also known as "Non-Temporal MOV" or
// "Streaming Store"), and also uses the software prefetchnta instructions,
// be sure youre running on Athlon/Duron or other recent CPU before calling!

#define TINY_BLOCK_COPY 64  // upper limit for movsd type copy
// The smallest copy uses the X86 "movsd" instruction, in an optimized
// form which is an "unrolled loop".

#define IN_CACHE_COPY 64 * 1024  // upper limit for movq/movq copy w/SW prefetch
// Next is a copy that uses the MMX registers to copy 8 bytes at a time,
// also using the "unrolled loop" optimization.   This code uses
// the software prefetch instruction to get the data into the cache.

#define UNCACHED_COPY 197 * 1024  // upper limit for movq/movntq w/SW prefetch
// For larger blocks, which will spill beyond the cache, its faster to
// use the Streaming Store instruction MOVNTQ.   This write instruction
// bypasses the cache and writes straight to main memory.  This code also
// uses the software prefetch instruction to pre-read the data.
// USE 64 * 1024 FOR THIS VALUE IF YOURE ALWAYS FILLING A "CLEAN CACHE"

#define BLOCK_PREFETCH_COPY infinity  // no limit for movq/movntq w/block prefetch
#define CACHEBLOCK 80h                // number of 64-byte blocks (cache lines) for block prefetch
// For the largest size blocks, a special technique called Block Prefetch
// can be used to accelerate the read operations.   Block Prefetch reads
// one address per cache line, for a series of cache lines, in a short loop.
// This is faster than using software prefetch.  The technique is great for
// getting maximum read bandwidth, especially in DDR memory systems.

// Inline assembly syntax for use with Visual C++

void* Os::fastMemcpy(void* dest, const void* src, size_t n) {
#if !defined(_WIN64)

  __asm {

    mov     ecx, [n]        ; number of bytes to copy
    mov     edi, [dest]     ; destination
    mov     esi, [src]      ; source
    mov     ebx, ecx        ; keep a copy of count

    cld
    cmp     ecx, TINY_BLOCK_COPY
    jb      $memcpy_ic_3    ; tiny? skip mmx copy

    cmp     ecx, 32*1024        ; dont align between 32k-64k because
    jbe     $memcpy_do_align    ;  it appears to be slower
    cmp     ecx, 64*1024
    jbe     $memcpy_align_done
$memcpy_do_align:
    mov     ecx, 8          ; a trick thats faster than rep movsb...
    sub     ecx, edi        ; align destination to qword
    and     ecx, 111b       ; get the low bits
    sub     ebx, ecx        ; update copy count
    neg     ecx             ; set up to jump into the array
    add     ecx, offset $memcpy_align_done
    jmp     ecx             ; jump to array of movsbs

align 4
    movsb
    movsb
    movsb
    movsb
    movsb
    movsb
    movsb
    movsb

$memcpy_align_done:         ; destination is dword aligned
    mov     ecx, ebx        ; number of bytes left to copy
    shr     ecx, 6          ; get 64-byte block count
    jz      $memcpy_ic_2    ; finish the last few bytes

    cmp     ecx, IN_CACHE_COPY/64    ; too big 4 cache? use uncached copy
    jae     $memcpy_uc_test

        // This is small block copy that uses the MMX registers to copy 8 bytes
        // at a time.  It uses the "unrolled loop" optimization, and also uses
        // the software prefetch instruction to get the data into the cache.
align 16
$memcpy_ic_1:            ; 64-byte block copies, in-cache copy

    prefetchnta [esi + (200*64/34+192)]        ; start reading ahead

    movq    mm0, [esi+0]    ; read 64 bits
    movq    mm1, [esi+8]
    movq    [edi+0], mm0    ; write 64 bits
    movq    [edi+8], mm1    ; note:  the normal movq writes the
    movq    mm2, [esi+16]   ; data to cache; a cache line will be
    movq    mm3, [esi+24]   ; allocated as needed, to store the data
    movq    [edi+16], mm2
    movq    [edi+24], mm3
    movq    mm0, [esi+32]
    movq    mm1, [esi+40]
    movq    [edi+32], mm0
    movq    [edi+40], mm1
    movq    mm2, [esi+48]
    movq    mm3, [esi+56]
    movq    [edi+48], mm2
    movq    [edi+56], mm3

    add        esi, 64      ; update source pointer
    add        edi, 64      ; update destination pointer
    dec        ecx          ; count down
    jnz        $memcpy_ic_1 ; last 64-byte block?

$memcpy_ic_2:
    mov        ecx, ebx     ; has valid low 6 bits of the byte count
$memcpy_ic_3:
    shr        ecx, 2       ; dword count
    and        ecx, 1111b   ; only look at the "remainder" bits
    neg        ecx          ; set up to jump into the array
    add        ecx, offset $memcpy_last_few
    jmp        ecx          ; jump to array of movsds

$memcpy_uc_test:
    cmp        ecx, UNCACHED_COPY/64    ; big enough? use block prefetch copy
    jae        $memcpy_bp_1

$memcpy_64_test:
    or        ecx, ecx      ; tail end of block prefetch will jump here
    jz        $memcpy_ic_2  ; no more 64-byte blocks left

        // For larger blocks, which will spill beyond the cache, its faster to
        // use the Streaming Store instruction MOVNTQ.   This write instruction
        // bypasses the cache and writes straight to main memory.  This code also
        // uses the software prefetch instruction to pre-read the data.
align 16
$memcpy_uc_1:               ; 64-byte blocks, uncached copy

    prefetchnta [esi + (200*64/34+192)]        ; start reading ahead

    movq    mm0,[esi+0]     ; read 64 bits
    add     edi,64          ; update destination pointer
    movq    mm1,[esi+8]
    add     esi,64          ; update source pointer
    movq    mm2,[esi-48]
    movntq  [edi-64], mm0   ; write 64 bits, bypassing the cache
    movq    mm0,[esi-40]    ; note: movntq also prevents the CPU
    movntq  [edi-56], mm1   ; from READING the destination address
    movq    mm1,[esi-32]    ; into the cache, only to be over-written
    movntq  [edi-48], mm2   ; so that also helps performance
    movq    mm2,[esi-24]
    movntq  [edi-40], mm0
    movq    mm0,[esi-16]
    movntq  [edi-32], mm1
    movq    mm1,[esi-8]
    movntq  [edi-24], mm2
    movntq  [edi-16], mm0
    dec     ecx
    movntq  [edi-8], mm1
    jnz     $memcpy_uc_1    ; last 64-byte block?

    jmp     $memcpy_ic_2    ; almost done

    // For the largest size blocks, a special technique called Block Prefetch
    // can be used to accelerate the read operations.   Block Prefetch reads
    // one address per cache line, for a series of cache lines, in a short loop.
    // This is faster than using software prefetch, in this case.
    // The technique is great for getting maximum read bandwidth,
    // especially in DDR memory systems.
$memcpy_bp_1:               ; large blocks, block prefetch copy

    cmp     ecx, CACHEBLOCK ; big enough to run another prefetch loop?
    jl      $memcpy_64_test ; no, back to regular uncached copy

    mov     eax, CACHEBLOCK / 2  ; block prefetch loop, unrolled 2X
    add     esi, CACHEBLOCK * 64 ; move to the top of the block
align 16
$memcpy_bp_2:
    mov     edx, [esi-64]   ; grab one address per cache line
    mov     edx, [esi-128]  ; grab one address per cache line
    sub     esi, 128        ; go reverse order
    dec     eax             ; count down the cache lines
    jnz     $memcpy_bp_2    ; keep grabbing more lines into cache

    mov     eax, CACHEBLOCK ; now that its in cache, do the copy
align 16
$memcpy_bp_3:
    movq    mm0, [esi   ]   ; read 64 bits
    movq    mm1, [esi+ 8]
    movq    mm2, [esi+16]
    movq    mm3, [esi+24]
    movq    mm4, [esi+32]
    movq    mm5, [esi+40]
    movq    mm6, [esi+48]
    movq    mm7, [esi+56]
    add     esi, 64         ; update source pointer
    movntq  [edi   ], mm0   ; write 64 bits, bypassing cache
    movntq  [edi+ 8], mm1   ; note: movntq also prevents the CPU
    movntq  [edi+16], mm2   ; from READING the destination address
    movntq  [edi+24], mm3   ; into the cache, only to be over-written,
    movntq  [edi+32], mm4   ; so that also helps performance
    movntq  [edi+40], mm5
    movntq  [edi+48], mm6
    movntq  [edi+56], mm7
    add     edi, 64         ; update dest pointer

    dec     eax             ; count down

    jnz     $memcpy_bp_3    ; keep copying
    sub     ecx, CACHEBLOCK ; update the 64-byte block count
    jmp     $memcpy_bp_1    ; keep processing chunks

    // The smallest copy uses the X86 "movsd" instruction, in an optimized
    // form which is an "unrolled loop".   Then it handles the last few bytes.
align 4
    movsd
    movsd            ; perform last 1-15 dword copies
    movsd
    movsd
    movsd
    movsd
    movsd
    movsd
    movsd
    movsd            ; perform last 1-7 dword copies
    movsd
    movsd
    movsd
    movsd
    movsd
    movsd

$memcpy_last_few:           ; dword aligned from before movsds
    mov     ecx, ebx        ; has valid low 2 bits of the byte count
    and     ecx, 11b        ; the last few cows must come home
    jz      $memcpy_final   ; no more, lets leave
    rep     movsb           ; the last 1, 2, or 3 bytes

$memcpy_final:
    emms                    ; clean up the MMX state
    sfence                  ; flush the write buffer
    mov     eax, [dest]     ; ret value = destination pointer

  }
#else  // !defined(_WIN64))

  return memcpy(dest, src, n);

#endif
}

uint64_t Os::offsetToEpochNanos() {
  static uint64_t offset = 0;

  if (offset != 0) {
    return offset;
  }

  FILETIME ft;
  GetSystemTimeAsFileTime(&ft);

  LARGE_INTEGER li;
  li.LowPart = ft.dwLowDateTime;
  li.HighPart = ft.dwHighDateTime;

  uint64_t now = (li.QuadPart - 116444736000000000ull) * 100;
  offset = now - timeNanos();

  return offset;
}

#ifdef _WIN64

address Os::currentStackPtr() { return (address)_AddressOfReturnAddress() + sizeof(void*); }

#else  // !_WIN64

#pragma warning(disable : 4731)

void __stdcall Os::setCurrentStackPtr(address newSp) {
  newSp -= sizeof(void*);
  *(void**)newSp = *(void**)_AddressOfReturnAddress();
  __asm {
        mov esp,newSp
        mov ebp,[ebp]
        ret
  }
}

#endif  // !_WIN64

size_t Os::getPhysicalMemSize() {
  MEMORYSTATUSEX statex;

  statex.dwLength = sizeof(statex);

  if (GlobalMemoryStatusEx(&statex) == 0) {
    return 0;
  }

  return (size_t)statex.ullTotalPhys;
}

void Os::getAppPathAndFileName(std::string& appName, std::string& appPathAndName) {
  char* buff = new char[FILE_PATH_MAX_LENGTH];

  if (GetModuleFileNameA(NULL, buff, FILE_PATH_MAX_LENGTH) != 0) {
    // Get filename without path and extension.
    appPathAndName = buff;
    appName = strrchr(buff, '\\') ? strrchr(buff, '\\') + 1 : buff;
  }
  else {
    appPathAndName = "";
    appName = "";
  }

  delete[] buff;
  return;
}

bool Os::MemoryUnmapFile(const void* mmap_ptr, size_t mmap_size) {
  if(!UnmapViewOfFile(mmap_ptr)) {
    return false;
  }

  return true;
}

bool Os::MemoryMapFile(const char* fname, const void** mmap_ptr, size_t* mmap_size) {
  if ((mmap_ptr == nullptr) || (mmap_size == nullptr)) {
    return false;
  }

  HANDLE map_handle = INVALID_HANDLE_VALUE;
  HANDLE file_handle = INVALID_HANDLE_VALUE;

  file_handle = CreateFileA(fname, GENERIC_READ, 0, NULL, OPEN_EXISTING,
                            FILE_ATTRIBUTE_READONLY, NULL);
  if (file_handle == INVALID_HANDLE_VALUE) {
    return false;
  }

  map_handle = CreateFileMappingA(file_handle, NULL, PAGE_READONLY, 0, 0, NULL);
  if (map_handle == INVALID_HANDLE_VALUE) {
    CloseHandle(file_handle);
    return false;
  }

  *mmap_size = GetFileSize(file_handle, NULL);
  *mmap_ptr = MapViewOfFile(map_handle, FILE_MAP_READ, 0,0,0);

  CloseHandle(file_handle);
  CloseHandle(map_handle);

  if (*mmap_ptr == nullptr) {
    return false;
  }

  return true;
}

}  // namespace amd

#endif  // _WIN32 || __CYGWIN__
