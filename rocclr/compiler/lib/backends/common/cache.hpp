#ifndef AMD_KERNEL_CACHE_H_
#define AMD_KERNEL_CACHE_H_

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <cstdio>
#ifdef __linux__
#include <unistd.h>
#include <fcntl.h>
#include <pwd.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <alloca.h>
#else
#include <windows.h>
#include <shlobj.h>
#include <Lmcons.h>
#include <aclapi.h>
#include <filesystem>
#endif
#include "os/os.hpp"
#include "utils/options.hpp"
#include "../../../sc/Interface/SCLib_Ver.h"

#if _WIN32
#define CloseFile CloseHandle
#define FileHandle HANDLE
#else
#define CloseFile close
#define FileHandle int
#endif

typedef struct _KernelCacheData {
  char *data;
  size_t dataSize;
} KernelCacheData;

// Specialize std::hash
struct HashType {
  std::string data;
  std::string buildOption;
};
namespace std {
template<>
struct hash<HashType> {
public:
  size_t operator()(const HashType &v) const
  {
    size_t h1 = std::hash<std::string>()(v.data);
    size_t h2 = std::hash<std::string>()(v.buildOption);
    return h1 ^ ( h2 << 1 );
  }
};
}

typedef struct _KernelCacheFileHeader {
  char AMD[4]; // 'AMD\0'
  size_t buildOptSize;
  size_t dstSize;
} KernelCacheFileHeader;

/* Kernel Cache File Contents (listed in order) */
// BUild options in text format
// Src data size
// Src data
// Dest data

class KernelCache {
private:
  // TODO: the default cache size (512MB) might be changed later
  static const unsigned int DEFAULT_CACHE_SIZE = 512 * 1024 * 1024;

  unsigned int version;
  unsigned int cacheSize;
  std::string rootPath;
  std::string indexName;
  std::string errorMsg;

  // Set the root path for the cache
  bool setRootPath(const std::string &chipName);

  // Wipe the cache folder structure
  bool wipeCacheFolders();

  // Setup cache tree structure
  bool setUpCacheFolders();

  // Get the cache version and size from the index file
  bool getCacheInfo();

  // Set the cache version and size in the index file
  bool setCacheInfo(unsigned int newVersion, unsigned int newSize);

  // Compute hash value for chunks of data
  unsigned int computeHash(const KernelCacheData *data, const std::string &buildOpts);

  // Computes hash and file name from given data
  void makeFileName(const KernelCacheData *data, const std::string &buildOpts, std::string &pathToFile);

  // Finds path to a file from a given hash value
  void getFilePathFromHash(unsigned int hash, std::string &pathToFile);

#if _WIN32
  // Get Sid of account
  std::unique_ptr<SID> getSid(TCHAR *userName);
#endif

  // Return detailed error message as string
  std::string getLastErrorMsg();

  // Read contents in cacheFile
  bool readFile(FileHandle cacheFile, void *buffer, ssize_t size);

  // Write data to a file
  bool writeFile(FileHandle cacheFile, const void *buffer, ssize_t sizeToWriten);
  bool writeFile(const std::string &fileName, const void *data, size_t size, bool appendable);

  // Set file to only owner accessible
  bool setAccessPermission(const std::string &fileName, bool isFile = false);

  // Set up cache file structure
  bool cacheInit(unsigned int compilerVersion, const std::string &chipName);

  // Get cache entry corresponding to srcData, if it exists
  bool getCacheEntry_helper(const KernelCacheData *srcData, const std::string &buildOpts,
                            std::string &dstData);

  // Control kernel cache test
  bool internalKCacheTestSwitch();

  // Verify whether the file includes the right kernel cache file header
  bool verifyKernelCacheFileHeader(KernelCacheFileHeader &H, const std::string &buildOpts);

  // Remove partially written file
  void removePartiallyWrittenFile(const std::string &fileName);

  // Log error message and close the file
  void logErrorCloseFile(const std::string &errorMsg, const FileHandle file);
public:
  KernelCache() : version(0), cacheSize(0) { rootPath.clear(); indexName.clear(); errorMsg.clear(); }

  // Make cache entry corresponding to srcData, dstData, buildOpts and kernelName
  bool makeCacheEntry(bool isCacheReady, const KernelCacheData *srcData,
                      const std::string &buildOpts, const std::string &dstData);

  // Wrapper function for getCacheEntry
  bool getCacheEntry(bool &isCacheReady, const std::string &deviceName, amd::option::Options *opts,
                     const KernelCacheData *srcData, const std::string &buildOpts,
                     std::string &dstData, const std::string &msg);

  // Log caching error messages for debugging the cache and/or detecting collisions
  void appendLogToFile(std::string extraMsg = "");
};
#endif // AMD_KERNEL_CACHE_H_
