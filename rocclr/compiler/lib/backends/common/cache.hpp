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
#include <sys/types.h>
#else
#include <windows.h>
#include <shlobj.h>
#include <Lmcons.h>
#include <aclapi.h>
#endif
#include "os/os.hpp"

typedef struct _KernelCacheData {
    char *data;
    unsigned int dataSize;
} KernelCacheData;

class KernelCache {
private:
    static const char ESCAPE = 0x7f;
    // TODO: the default cache size (512MB) might be changed later
    static const unsigned int DEFAULT_CACHE_SIZE = 512 * 1024 * 1024;
    unsigned int version;
    unsigned int cacheSize;
    std::string rootPath;
    std::string indexName;
    std::string errorMsg;

    // Set the root path for the cache
    bool setRootPath(const std::string &chipName);

    // Check if file exists
    bool fileExists(const std::string &fileName);

    // Wipe the cache folder structure
    bool wipeCacheFolders();

    // Setup cache tree structure
    bool setUpCacheFolders();

    // Get the cache version and size from the index file
    bool getCacheInfo();

    // Set the cache version and size in the index file
    bool setCacheInfo(unsigned int newVersion, unsigned int newSize);

    // Read contents of a file
    bool readFile(const std::string &fileName, char **contents, size_t &fileSize);

    // Write data to a file
    bool writeFile(const std::string &fileName, const char *data, size_t fileSize);

    // Compute hash value for chunks of data
    unsigned int computeHash(const KernelCacheData *data, const unsigned int numData, const std::string &buildOpts, const std::string &kernelName);

    // Compares two sets of data
    inline bool compareData(const char *data0, const char *data1, const unsigned int size) {
        return (memcmp(data0, data1, size) == 0);
    }

    // Computes hash and file name from given data
    void makeFileName(const KernelCacheData *data, const unsigned int numData, const std::string &buildOpts, const std::string &kernelName, std::string &pathToFile);

    // Finds path to a file from a given hash value
    void getFilePathFromHash(const unsigned int hash, std::string &pathToFile);

    // Finds the cache entry for a chunk of data
    bool findCacheEntry(const KernelCacheData *data, const unsigned int numData, const std::string &buildOpts, const std::string &kernelName, std::string &pathToFile);

    // Creates a cache file in the cache heirarchy
    bool makeCacheEntry(const KernelCacheData *srcData, const unsigned int srcNum, const std::string &buildOpts, const char *dstData, unsigned int dstSize, bool fromModule);

    // Builds a file for storage into the cache
    bool buildFile(const KernelCacheData *srcData, const unsigned int srcNum, const std::string &buildOpts, const std::string &kernelName, const char *dstData, const unsigned int dstSize, const unsigned int dstHash, char **fileData, unsigned int &dataSize);

    // Parses data from a file
    bool parseFile(const char *fileData, const unsigned int dataSize, KernelCacheData **srcData, unsigned int &srcNum, std::string &buildOpts, std::string &kernelName, char **dstData, unsigned int &dstSize, unsigned int &dstHash);

#if _WIN32
    // Get Sid of account
    bool getSid(TCHAR *userName, PSID &sid);
#endif

    // Set file to only owner accessible
    bool setAccessPermission(const std::string fileName, bool isFile = false);
public:
    KernelCache() : version(0) {rootPath.clear(); indexName.clear(); errorMsg.clear(); }

    bool cacheInit(unsigned int compilerVersion, const std::string &chipName);

    // Get cache entry corresponding to srcData, if it exists
    bool getCacheEntry(const KernelCacheData *srcData, const unsigned int srcNum, const std::string &buildOpts, const std::string &kernelName, char **dstData, unsigned int &dstSize, unsigned int &dstHash);

    // Make cache entry corresponding to srcData, dstData, buildOpts and kernelName
    bool makeCacheEntry(const KernelCacheData *srcData, const unsigned int srcNum, const std::string &buildOpts, const std::string &kernelName, const char *dstData, const unsigned int dstSize);

    std::string ErrorMsg() { return errorMsg; }

    // Control kernel cache test
    bool internalKCacheTestSwitch(bool &canUseCache);

    // Log caching error messages for debugging the cache and/or detecting collisions
    void saveLogToFile(std::string extraMsg = " ") {
        if (amd::Os::pathExists(rootPath)) {
            std::string fileName = rootPath + amd::Os::fileSeparator() + "cacheError.log";
            errorMsg += extraMsg;
            writeFile(fileName, errorMsg.c_str(), errorMsg.length());
        }
    }

};

#endif // AMD_KERNEL_CACHE_H_
