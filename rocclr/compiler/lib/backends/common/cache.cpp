#include <sys/stat.h>
#include "cache.hpp"

// Check if a file exists
//
// In:
// fileName - Path to file
//
// Out:
// none
//
// Returns:
// true if the file exists
// false otherwise
//
bool KernelCache::fileExists(const std::string &fileName)
{
    struct stat info;
    return 0 == stat(fileName.c_str(), &info);
}

// Wipe the cache folder structure
//
// In:
// none
//
// Out:
// none
//
// Returns:
// true if folder wipe is ok
// false otherwise
//
bool KernelCache::wipeCacheFolders()
{
    for (int i = 0; i < 16; ++i) {
        std::string dir = rootPath;
        std::stringstream ss;
        ss << amd::Os::fileSeparator() << std::hex << i;
        dir += ss.str();
        if (amd::Os::pathExists(dir)) {
            if (false == amd::Os::removePath(dir)) {
                errorMsg = "Error deleting directory in cache";
                return false;
            }
        }
    }
    return true;
}

// Setup cache tree structure
//
// In:
// none
//
// Out:
// none
//
// Returns:
// true if folders setup is ok
// false otherwise
//
bool KernelCache::setUpCacheFolders()
{
    // Directory structure is distributed as 16 * 16 in order to keep the file count per directory low
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            std::string dir = rootPath;
            std::stringstream ss;
            ss << amd::Os::fileSeparator() << std::hex << i << amd::Os::fileSeparator() << j;
            dir += ss.str();
            if (false == amd::Os::createPath(dir)) {
                errorMsg = "Error creating directory in cache";
                return false;
            }
            // Set folder to only owner accessible
            if (!setAccessPermission(rootPath)) {
                return false;
            }
        }
    }
    return true;
}

#if _WIN32
// Get Sid of account
// Caller will need to free the SID buffer (returned in sid)
//
// In:
// userName - accont name
//
// Out:
// sid - Sid of account
//
// Return:
// true if SID is obtained, false otherwise
//
bool KernelCache::getSid(TCHAR *username, PSID &sid)
{
    if (username == NULL) {
        errorMsg = "Invalid user name in getSid mehtod";
        return false;
    }
    // If a buffer is too small, the count parameter will be set to the size needed.
    const DWORD INITIAL_SIZE = 32;
    SID_NAME_USE sidNameUse;
    DWORD cbSid = 0, cchDomainName = 0;
    DWORD dwSidBufferSize = INITIAL_SIZE, dwDomainBufferSize = INITIAL_SIZE;
    // Create buffers for the SID and the domain name
    sid = (PSID) new BYTE[dwSidBufferSize];
    if (sid == NULL) {
        errorMsg = "Failed to allocate space for SID";
        return false;
    }
    memset(sid, 0, dwSidBufferSize);
    TCHAR *wszDomainName = new TCHAR[dwDomainBufferSize];
    if (wszDomainName == NULL) {
        delete[] sid;
        errorMsg = "Failed to allocate space for domain name";
        return false;
    }
    memset(wszDomainName, 0, dwDomainBufferSize * sizeof(TCHAR));
    // Obtain the SID for the account name passed
    while (true) {
        // Set the count variables to the buffer sizes and retrieve the SID
        cbSid = dwSidBufferSize;
        cchDomainName = dwDomainBufferSize;
        if (LookupAccountName(NULL, username, sid, &cbSid, wszDomainName, &cchDomainName, &sidNameUse)) {
            if (IsValidSid(sid) == FALSE) {
                delete[] sid;
                delete[] wszDomainName;
                errorMsg = "The SID for the account is invalid";
                return false;
            }
            break;
        }
        DWORD dwErrorCode = GetLastError();
        if (dwErrorCode == ERROR_INSUFFICIENT_BUFFER) {
            if (cbSid > dwSidBufferSize) {
                // Reallocate memory for the SID buffer
                delete[] sid;
                sid = (PSID) new BYTE[cbSid];
                if (sid == NULL) {
                    delete[] wszDomainName;
                    errorMsg = "Failed to allocate space for SID";
                    return false;
                }
                memset(sid, 0, cbSid);
                dwSidBufferSize = cbSid;
            }
            if (cchDomainName > dwDomainBufferSize) {
                // Reallocate memory for the domain name buffer
                delete[] wszDomainName;
                wszDomainName = new TCHAR[cchDomainName];
                if (wszDomainName == NULL) {
                    delete[] sid;
                    errorMsg = "Failed to allocate space for domain name";
                }
                memset(wszDomainName, 0, cchDomainName * sizeof(TCHAR));
                dwDomainBufferSize = cchDomainName;
            }
        } else {
            delete[] sid;
            delete[] wszDomainName;
            errorMsg = "Failed to get user security identifier for the account, GetLastError returned"
                       + dwErrorCode;
            return false;
        }
    }
    delete[] wszDomainName;
    return true;
}
#endif

// Set file to only owner accessible
//
// In:
// fileName - Path to file
// isFile - True if fileName is a file, false if it is a path; false by default
//
// Out:
// none
//
// Returns:
// true if access permission is under control
// false otherwise
//
bool KernelCache::setAccessPermission(const std::string fileName, bool isFile)
{
#if _WIN32
    TCHAR username[UNLEN + 1];
    DWORD username_len = UNLEN + 1;
    if (!GetUserName(username, &username_len)) {
        errorMsg = "Failed to get user name for the account";
        return false;
    }
    PSID sid = NULL;
    if (!getSid(username, sid)) {
        return false;
    }
    if (SetNamedSecurityInfo((LPTSTR)(fileName.c_str()), SE_FILE_OBJECT, OWNER_SECURITY_INFORMATION,
                             sid, NULL, NULL, NULL) != ERROR_SUCCESS ) {
        delete[] sid;
        errorMsg = "Failed to set user access permission";
        return false;
    }
    delete[] sid;
#else
    int ret = -1;
    if (isFile) {
        ret = chmod(fileName.c_str(), S_IRUSR | S_IWUSR);
    } else {
        ret = chmod(fileName.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    }
    if (ret < 0) {
        errorMsg = "Failed to set user access permission";
        return false;
    }
#endif
    return true;
}

// Open a file and read its contents
// Caller will need to free the memory created by this function (returned in contents)
//
// In:
// fileName - Path to file
// fileSize - File size
//
// Out:
// conttents - Pointer to file contents
//
// Returns:
// true if the file is read successfully
// false otherwise
//
bool KernelCache::readFile(const std::string &fileName, char **contents, size_t &fileSize)
{
    *contents = NULL;
#if _WIN32
    HANDLE hFile = CreateFile(fileName.c_str(), GENERIC_READ, FILE_SHARE_READ,
                              NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        errorMsg = "Error opening file for reading";
        return false;
    }
    LARGE_INTEGER size;
    if (!GetFileSizeEx(hFile, &size)) {
        errorMsg = "Last error code is " + std::to_string(GetLastError());
        CloseHandle(hFile);
        return false;
    }
    char *memblock = new char [size.QuadPart];
    if (memblock == NULL) {
        errorMsg = "Out of memory";
        CloseHandle(hFile);
        return false;
    }
    DWORD dwBytesRead = 0;
    if(FALSE == ReadFile(hFile, memblock, size.QuadPart, &dwBytesRead, NULL)) {
        errorMsg = "Unable to read cache file";
        CloseHandle(hFile);
        return false;
    }
    if (dwBytesRead != size.QuadPart) {
        errorMsg = "Error reading cache file";
        CloseHandle(hFile);
        return false;
    }
    CloseHandle(hFile);
    *contents = memblock;
    fileSize = size.QuadPart;
#else
    FILE *cacheFile = fopen(fileName.c_str(), "rb");
    if (cacheFile == NULL) {
        errorMsg = "Error opening file for reading";
        return false;
    }
    // Read lock for cache file
    int fd = fileno(cacheFile);
    if (fd == -1) {
        errorMsg = "Error getting file descriptor";
        fclose(cacheFile);
        return false;
    }
    struct flock fl = {F_RDLCK, SEEK_SET, 0, 0};
    if (fcntl(fd, F_SETLK, &fl) == -1) {
        errorMsg = "Error setting file read lock";
        fclose(cacheFile);
        return false;
    }
    // Read the file
    fseek(cacheFile, 0, SEEK_END);
    size_t size = ftell(cacheFile);
    rewind(cacheFile);
    char *memblock = new char [size];
    if (memblock == NULL) {
        errorMsg = "Out of memory";
        fclose(cacheFile);
        return false;
    }
    size_t result = fread(memblock, sizeof(char), size, cacheFile);
    // Unlock the file
    fl.l_type = F_UNLCK;
    if (fcntl(fd, F_SETLK, &fl) == -1) {
        errorMsg = "Error unlock file read lock";
        fclose(cacheFile);
        return false;
    }
    if (result != size) {
        errorMsg = "Error reading cache file";
        fclose(cacheFile);
        delete[] memblock;
        return false;
    }
    fclose(cacheFile);
    *contents = memblock;
    fileSize = size;
#endif
    return true;
}

// Open a file and write its contents
//
// In:
// fileName - Path to file
// data - Pointer to file contents
// size - Data size
//
// Out:
// none
//
// Returns:
// true if the file is written to file successfully
// false otherwise
//
bool KernelCache::writeFile(const std::string &fileName, const char *data, size_t size)
{
#if _WIN32
    HANDLE hFile = CreateFile(fileName.c_str(), GENERIC_WRITE | WRITE_OWNER | READ_CONTROL,
                              0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        errorMsg = "Error opening file for writing";
        return false;
    }
    // Write file exclusively
    DWORD dwBytesWritten = 0;
    if (FALSE == WriteFile(hFile, data, size, &dwBytesWritten, NULL)) {
        errorMsg = "Unable to write to file";
        CloseHandle(hFile);
        return false;
    }
    if (dwBytesWritten != size) {
        errorMsg = "Error writing cache file";
        CloseHandle(hFile);
        return false;
    }
    CloseHandle(hFile);
#else
    FILE *cacheFile = fopen(fileName.c_str(), "wb");
    if (cacheFile == NULL) {
        errorMsg = "Error opening file for writing";
        return false;
    }
    // Exclusive write lock for cache file
    int fd = fileno(cacheFile);
    if (fd == -1) {
        errorMsg = "Error getting file descriptor";
        fclose(cacheFile);
        return false;
    }
    struct flock fl = {F_WRLCK, SEEK_SET, 0, 0};
    if (fcntl(fd, F_SETLK, &fl) == -1) {
        errorMsg = "Error setting file write lock";
	std::cout << errorMsg << std::endl;
        fclose(cacheFile);
        return false;
    }
    // Write data to file
    fwrite(data, sizeof(char), size, cacheFile);
    // Unlock the file
    fl.l_type = F_UNLCK;
    if (fcntl(fd, F_SETLK, &fl) == -1) {
        errorMsg = "Error unlock file write lock";
        fclose(cacheFile);
        return false;
    }
    fclose(cacheFile);
#endif
    // Set file to only owner accessible
    if (!setAccessPermission(fileName, true)) {
        return false;
    }
    return true;
}

// Set the cache's root path
//
// In:
// chipName - Chip name
//
// Out:
// none
//
// Returns:
// true if root path of cache is set successfully
// false otherwise
//
bool KernelCache::setRootPath(const std::string &chipName)
{
    bool ok = false;
    rootPath.clear();
#if _WIN32
    // Set root path to <USER>\AppData\Local\AMD\CLCache
    TCHAR userLocalAppDir[_MAX_PATH];
    // Get path for user specific and non-roaming data
    if (SUCCEEDED(SHGetFolderPath(NULL, CSIDL_LOCAL_APPDATA, NULL, SHGFP_TYPE_CURRENT, userLocalAppDir))) {
        rootPath = userLocalAppDir;
    } else {
        errorMsg = "User's local app dir is not found";
        return false;
    }
    // Ok, we have <USER>\AppData\Local, let's check for the rest and create if needed
    rootPath += "\\AMD";
    if (!amd::Os::pathExists(rootPath)) {
        ok = amd::Os::createPath(rootPath);
        if (!ok) {
            errorMsg = "Failed to create AMD directory";
            return false;
        }
    }
    rootPath += "\\CLCache";
    if (!amd::Os::pathExists(rootPath)) {
        ok = amd::Os::createPath(rootPath);
        if (!ok) {
            errorMsg = "Failed to create CLCache directory";
            return false;
        }
    }
    // Set folder to only owner accessible
    if (!setAccessPermission(rootPath)) {
        return false;
    }
    rootPath += "\\" + chipName;
    if (!amd::Os::pathExists(rootPath)) {
        ok = amd::Os::createPath(rootPath);
        if (!ok) {
            errorMsg = "Failed to create " + chipName + " directory";
            return false;
        }
    }
    // Set folder to only owner accessible
    if (!setAccessPermission(rootPath)) {
        return false;
    }
#else
    const char *homedir = getpwuid(getuid())->pw_dir;
    if (homedir == NULL) {
        errorMsg = "Failed to get HOME directory";
        return false;
    }
    rootPath = homedir;
    // Verify the path exists
    if (!amd::Os::pathExists(rootPath)) {
        errorMsg = "User's home directory is not created";
        return false;
    }
    // Ok, we have <HOME>, let's check for the rest and create if needed
    rootPath += "/.AMD";
    if (!amd::Os::pathExists(rootPath)) {
        ok = amd::Os::createPath(rootPath);
        if (!ok) {
            errorMsg = "Failed to create AMD directory";
            return false;
        }
    }
    rootPath += "/CLCache";
    if (!amd::Os::pathExists(rootPath)) {
        ok = amd::Os::createPath(rootPath);
        if (!ok) {
            errorMsg = "Failed to create CLCache directory";
            return false;
        }
    }
    // Set folder to only owner accessible
    if (!setAccessPermission(rootPath)) {
        return false;
    }
    rootPath += "/" + chipName;
    if (!amd::Os::pathExists(rootPath)) {
        ok = amd::Os::createPath(rootPath);
        if (!ok) {
            errorMsg = "Failed to create " + chipName + " directory";
            return false;
        }
    }
    // Set folder to only owner accessible
    if (!setAccessPermission(rootPath)) {
        return false;
    }
#endif
    return true;
}

// Set the cache version and size
//
// In:
// newVersion - New version for the cache
// newSize - New size for the cache
//
// Out:
// none
//
// Returns:
// true if successful
// false otherwise
//
bool KernelCache::setCacheInfo(unsigned int newVersion, unsigned int newSize)
{
    bool ok = true;
    char fileData[8];
    char *dataPtr = fileData;
    *(unsigned int *)dataPtr = newVersion;
    dataPtr += sizeof(unsigned int);
    *(unsigned int *)dataPtr = newSize;
    ok = writeFile(indexName, fileData, sizeof(fileData));
    if (!ok) {
        errorMsg = "Failed to update the index file";
        return false;
    }
    version = newVersion;
    cacheSize = newSize;
    return ok;
}

// Get the version and size of the cache
//
// In:
// none
//
// Out:
// none
//
// Returns:
// true if successful
// false otherwise
//
bool KernelCache::getCacheInfo()
{
    bool ok = true;
    indexName = rootPath;
    indexName += amd::Os::fileSeparator();
    indexName += "cacheDir";
    // Check for cache index file
    char *contents = NULL;
    size_t size;
    ok = readFile(indexName, &contents, size);
    if (ok) {
        char *tmp = contents;
        if (size >= 8) {
            version = *(unsigned int *)tmp;
            tmp += sizeof(unsigned int);
            cacheSize = *(unsigned int *)tmp;
        } else {
            errorMsg = "Index file truncated";
            ok = false;
        }
        delete[] contents;
    } else {
        ok = setCacheInfo(-1, 0);
    }
    return ok;
}

// Initialize the cache
//
// In:
// compilerVersion - Compiler version
//
// Out:
// none
//
// Returns:
// true if successful
// false otherwise
//
bool KernelCache::cacheInit(unsigned int compilerVersion, const std::string &chipName)
{
    if (!setRootPath(chipName)) {
        return false;
    }
    if (!getCacheInfo()) {
        return false;
    }
    // Limit cache size to default cache size, and wipe out all cache files when it's exceed
    // TODO: need to implement cache eviction policy
    if (version != compilerVersion || cacheSize > DEFAULT_CACHE_SIZE) {
        if (!wipeCacheFolders()) {
            return false;
        }
        if (!setCacheInfo(compilerVersion, 0)) {
            return false;
        }
        if (!setUpCacheFolders()) {
            return false;
        }
    }
    return true;
}

// Compute the hash value for a buffer of data along with the kernelName and buildOpts
//
// In:
// data - Data to hash
// size - Size of data
// buildOpts - Build options
// kernelName - Kernel name
//
// Out:
// none
//
// Returns:
// Hash value computed from the inputs
//
unsigned int KernelCache::computeHash(const KernelCacheData *data, const unsigned int numData,
    const std::string &buildOpts, const std::string &kernelName)
{
    unsigned int hashVal = 0;
    // Two big prime numbers to start the multiplicative hash
    unsigned int seed0 = 500321;
    unsigned int seed1 = 72701;
    for (unsigned int i = 0; i < numData; ++i) {
        for (unsigned int j = 0; j < data[i].dataSize; ++j) {
            hashVal = hashVal * seed0 + data[i].data[j];
            seed0 *= seed1;
        }
    }
    char *strData = (char *)buildOpts.c_str();
    for (unsigned int i = 0; i < buildOpts.size(); ++i) {
        hashVal = hashVal * seed0 + strData[i];
        seed0 *= seed1;
    }
    strData = (char *)kernelName.c_str();
    for (unsigned int i = 0; i < kernelName.size(); ++i) {
        hashVal = hashVal * seed0 + strData[i];
        seed0 *= seed1;
    }
    return hashVal;
}

// Parses a cache file and returns the contents
//
// In:
// fileData - File data we are to parse
// dataSize - Size of file data
//
// Out:
// srcData - Pointer to pointer to source data allocated here
// srcSize - Size of source data
// buildOpts - Build options from file
// kernelName - Kernel name from file
// dstData - Pointer to pointer to destination data allocation here
// dstSize - Size of destination data
// dstHash - Destination hash from file
//
// Returns:
// true if no errors during parsing
// false otherwise
//
bool KernelCache::parseFile(const char *fileData, const unsigned int dataSize,
    KernelCacheData **srcData, unsigned int &srcNum, std::string &buildOpts,
    std::string &kernelName, char **dstData, unsigned int &dstSize, unsigned int &dstHash)
{
    unsigned int dataLeft = dataSize;
    char *data = (char *)fileData;
    errorMsg = "Error parsing file";
    if ((dataLeft > 3) && ((fileData[0] != 'A') || (fileData[1] != 'M') || (fileData[2] != 'D'))) {
        errorMsg = "Not a valid cache file";
        return false;
    }
    data += 3;
    dataLeft -= 3;
    if (dataLeft < sizeof(unsigned int)) {
        errorMsg = "Error in cache file";
        return false;
    }
    dstHash = *(unsigned int*)data;
    data += sizeof(unsigned int);
    dataLeft -= sizeof(unsigned int);
    if (dataLeft-- < 1 || *data++ != ESCAPE) {
        errorMsg = "Error in cache file";
        return false;
    }
    if (dataLeft < sizeof(unsigned int)) {
        errorMsg = "Error in cache file";
        return false;
    }
    srcNum = *(unsigned int*)data;
    data += sizeof(unsigned int);
    dataLeft -= sizeof(unsigned int);
    if (dataLeft-- < 1 || *data++ != ESCAPE) {
        errorMsg = "Error in cache file";
        return false;
    }
    *srcData = new KernelCacheData [srcNum];
    if (*srcData == NULL) {
        errorMsg = "Failed to alloc srcData";
        return false;
    }
    for (unsigned int i = 0; i < srcNum; ++i) {
        if (dataLeft < sizeof(unsigned int)) {
            delete[] *srcData;
            *srcData = NULL;
            srcNum = 0;
            errorMsg = "Error in cache file";
            return false;
        }
        (*srcData)[i].dataSize = *(unsigned int*)data;
        data += sizeof(unsigned int);
        dataLeft -= sizeof(unsigned int);
        if (dataLeft-- < 1 || *data++ != ESCAPE) {
            delete[] *srcData;
            *srcData = NULL;
            srcNum = 0;
            errorMsg = "Error in cache file";
            return false;
        }
        (*srcData)[i].data = data;
        if (dataLeft < (*srcData)[i].dataSize) {
            delete[] *srcData;
            *srcData = NULL;
            srcNum = 0;
            errorMsg = "Error in cache file";
            return false;
        }
        memcpy((*srcData)[i].data, data, (*srcData)[i].dataSize);
        data += (*srcData)[i].dataSize;
        dataLeft -= (*srcData)[i].dataSize;
        if (dataLeft-- < 1 || *data++ != ESCAPE) {
            delete[] *srcData;
            *srcData = NULL;
            srcNum = 0;
            errorMsg = "Error in cache file";
            return false;
        }
    }
    unsigned int buildOptsSize = 0;
    if (dataLeft < sizeof(unsigned int)) {
        delete[] *srcData;
        *srcData = NULL;
        srcNum = 0;
        errorMsg = "Error in cache file";
        return false;
    }
    buildOptsSize = *(unsigned int*)data;
    data += sizeof(unsigned int);
    dataLeft -= sizeof(unsigned int);
    if (dataLeft-- < 1 || *data++ != ESCAPE) {
        delete[] *srcData;
        *srcData = NULL;
        srcNum = 0;
        errorMsg = "Error in cache file";
        return false;
    }
    buildOpts = data;
    if (dataLeft < buildOptsSize) {
        delete[] *srcData;
        *srcData = NULL;
        srcNum = 0;
        errorMsg = "Error in cache file";
        return false;
    }
    data += buildOptsSize;
    dataLeft -= buildOptsSize;
    if (dataLeft-- < 1 || *data++ != ESCAPE) {
        delete[] *srcData;
        *srcData = NULL;
        srcNum = 0;
        errorMsg = "Error in cache file";
        return false;
    }
    unsigned int kernelNameSize;
    if (dataLeft < sizeof(unsigned int)) {
        delete[] *srcData;
        *srcData = NULL;
        srcNum = 0;
        errorMsg = "Error in cache file";
        return false;
    }
    kernelNameSize = *(unsigned int*)data;
    data += sizeof(unsigned int);
    dataLeft -= sizeof(unsigned int);
    if (dataLeft-- < 1 || *data++ != ESCAPE) {
        delete[] *srcData;
        *srcData = NULL;
        srcNum = 0;
        errorMsg = "Error in cache file";
        return false;
    }
    kernelName = data;
    if (dataLeft < kernelNameSize) {
        delete[] *srcData;
        *srcData = NULL;
        srcNum = 0;
        errorMsg = "Error in cache file";
        return false;
    }
    data += kernelNameSize;
    dataLeft -= kernelNameSize;
    if (dataLeft-- < 1 || *data++ != ESCAPE) {
        delete[] *srcData;
        *srcData = NULL;
        srcNum = 0;
        errorMsg = "Error in cache file";
        return false;
    }
    if (dataLeft < sizeof(unsigned int)) {
        delete[] *srcData;
        *srcData = NULL;
        srcNum = 0;
        errorMsg = "Error in cache file";
        return false;
    }
    dstSize = *(unsigned int*)data;
    data += sizeof(unsigned int);
    dataLeft -= sizeof(unsigned int);
    if (dataLeft-- < 1 || *data++ != ESCAPE) {
        delete[] *srcData;
        *srcData = NULL;
        srcNum = 0;
        errorMsg = "Error in cache file";
        return false;
    }
    *dstData = new char [dstSize];
    if (!*dstData || dataLeft != dstSize) {
        if (!*dstData) {
            errorMsg = "Failed to alloc dstData";
        } else {
            errorMsg = "Error in cache file";
        }
        delete[] srcData;
        srcData = NULL;
        srcNum = 0;
        return false;
    }
    memcpy(*dstData, data, dstSize);
    errorMsg.clear();
    return true;
}

// Control kernel cache test
//
// In:
// canUseCache - flag to indicate whether turn on/off cache test
//
// Out:
// none
//
bool KernelCache::internalKCacheTestSwitch(bool &canUseCache) {
#ifndef OPENCL_MAINLINE
    const char *cache_test_switch = getenv("AMD_FORCE_KCACHE_TEST");
    if (!(cache_test_switch && strcmp(cache_test_switch,"1") == 0)) {
        canUseCache = false;
        return false;
    } else {
        return true;
    }
#else
    return false;
#endif
}

// Allocate memory for cache file and build file structure
//
// In:
// srcData - Source data
// srcSize - Source data size
// buildOpts - Build options
// kernelName - Kernel name
// dstData - Destination data
// dstSize - Destination data size
// dstHash - Hash of destination data, buildOpts and kernelName
//
// Out:
// fileData - Pointer to pointer to file data
// dataSize - Size of file data
//
// Returns:
// true if successful
// false otherwise
//
bool KernelCache::buildFile(const KernelCacheData *srcData, const unsigned int srcNum,
    const std::string &buildOpts, const std::string &kernelName, const char *dstData,
    const unsigned int dstSize, const unsigned int dstHash, char **fileData, unsigned int &dataSize)
{
    bool ok = false;
    const unsigned int buildOptsSize = buildOpts.size() + 1;  // Add one for NULL terminator
    const unsigned int kernelNameSize = kernelName.size() + 1;  // Add one for NULL terminator
    dataSize = 3 /* 'AMD' */ + sizeof(dstHash) + 1 /* ESC */ + sizeof(srcNum) + 1 /* ESC */
		+ sizeof(buildOptsSize) + 1 /* ESC */
		+ buildOptsSize /* buildOpts data with NULL termination */ + 1 /* ESC */
		+ sizeof(kernelNameSize) + 1 /* ESC */
		+ kernelNameSize /* kernelName with NULL termination */ + 1 /* ESC */
		+ sizeof(dstSize) + 1 /* ESC */
		+ dstSize /* dst data */;
	for (unsigned int i = 0; i < srcNum; ++i) {
        dataSize += sizeof(srcData[i].dataSize) + 1 + srcData[i].dataSize + 1;
    }
    *fileData = new char [dataSize];
    if (fileData) {
        char *tmp = *fileData;
        tmp[0] = 'A';
        tmp[1] = 'M';
        tmp[2] = 'D';
        tmp += 3;
        *(unsigned int *)tmp = dstHash;
        tmp += sizeof(dstHash);
        *tmp++ = ESCAPE;
        *(unsigned int *)tmp = srcNum;
        tmp += sizeof(srcNum);
        *tmp++ = ESCAPE;
        for (unsigned int i = 0; i < srcNum; ++i) {
            *(unsigned int *)tmp = srcData[i].dataSize;
            tmp += sizeof(srcData[i].dataSize);
            *tmp++ = ESCAPE;
            memcpy(tmp, srcData[i].data, srcData[i].dataSize);
            tmp += srcData[i].dataSize;
            *tmp++ = ESCAPE;
        }
        *(unsigned int *)tmp = (unsigned int)buildOptsSize;
        tmp += sizeof(unsigned int);
        *tmp++ = ESCAPE;
        memcpy(tmp, buildOpts.c_str(), buildOptsSize);
        tmp += buildOptsSize;
        *tmp++ = ESCAPE;
        *(unsigned int *)tmp = (unsigned int)kernelNameSize;
        tmp += sizeof(unsigned int);
        *tmp++ = ESCAPE;
        memcpy(tmp, kernelName.c_str(), kernelNameSize);
        tmp += kernelNameSize;
        *tmp++ = ESCAPE;
        *(unsigned int*)tmp = dstSize;
        tmp += sizeof(dstSize);
        *tmp++ = ESCAPE;
        memcpy(tmp, dstData, dstSize);
        ok = true;
    } else {
        errorMsg = "Out of memory allocating fileData";
    }
    return ok;
}

// Generate file path from a hash value
//
// In:
// hashVal - A hash value
// pathToFile - Path to the file
//
// Returns:
// nothing
//
void KernelCache::getFilePathFromHash(const unsigned int hashVal, std::string &pathToFile)
{
    char textHash[9];
    sprintf(textHash, "%08x", hashVal);
    std::string fileName = textHash;
    pathToFile = rootPath;
    pathToFile += amd::Os::fileSeparator();
    // First char determines first dir level
    pathToFile += fileName[0];
    pathToFile += amd::Os::fileSeparator();
    // Second char determines second dir level
    pathToFile += fileName[1];
    pathToFile += amd::Os::fileSeparator();
    // Rest of file name determines name
    pathToFile += fileName.c_str() + 2;
}

// Use data, buildOpts and kernelName to generate a file name
//
// In:
// data - Pointer to data list
// numData - Size of the list
// buildOpts - Build options
// kernelName - Kernel name
//
// Out:
// pathToFile - Path to the file
//
// Returns:
// nothing
//
void KernelCache::makeFileName(const KernelCacheData *data, const unsigned int numData,
    const std::string &buildOpts, const std::string &kernelName, std::string &pathToFile)
{
    unsigned int hashVal;
    hashVal = computeHash(data, numData, buildOpts, kernelName);
    getFilePathFromHash(hashVal, pathToFile);
}

// Get the path to a cache entry based on the input data
//
// In:
// data - Pointer to data list
// numData - Size of the list
// buildOpts - Build options
// kernelName - Kernel name
//
// Out:
// pathToFile - Path to the file
//
// Returns:
// true if file exists, false otherwise
//
bool KernelCache::findCacheEntry(const KernelCacheData *data, const unsigned int numData,
    const std::string &buildOpts, const std::string &kernelName, std::string &pathToFile)
{
    makeFileName(data, numData, buildOpts, kernelName, pathToFile);
    return fileExists(pathToFile);
}

// Use srcData, buildOpts and kernelName to find the corresponding cache entry, if it exists
//
// In:
// srcData - Source data
// srcNum - Number of source data
// buildOpts - Build options
// kernelName - Kernel name
//
// Out:
// dstData - Destination data
// dstSize - Destination size
// dstHash - Destination hash
//
// Returns:
// true if entry found
// false otherwise, check errorMsg for errors
//
bool KernelCache::getCacheEntry(const KernelCacheData *srcData, const unsigned int srcNum,
    const std::string &buildOpts, const std::string &kernelName, char **dstData, unsigned int &dstSize,
    unsigned int &dstHash)
{
    std::string pathToFile;
    *dstData = NULL;
    dstSize = 0;
    errorMsg.clear();
    bool ok = findCacheEntry(srcData, srcNum, buildOpts, kernelName, pathToFile);
    if (ok) {
        // Found cache entry
        char *contents;
        size_t size;
        ok = readFile(pathToFile, &contents, size);
        if (ok) {
            // Found entry in cache
            KernelCacheData *fileSrcData;
            unsigned int fileSrcNum;
            std::string fileBuildOpts, fileKernelName;
            ok = parseFile(contents, size, &fileSrcData, fileSrcNum, fileBuildOpts, fileKernelName, dstData, dstSize, dstHash);
            if (srcNum == fileSrcNum) {
                for (unsigned int i = 0; i < srcNum; ++i) {
                    if (fileSrcData[i].dataSize == srcData[i].dataSize) {
                        if (!compareData((const char *)fileSrcData[i].data, srcData[i].data, srcData[i].dataSize)) {
                            ok = false;
                            errorMsg = "Cache collision: Size matches, contents do not";
                        } else {
                        }
                    } else {
                        ok = false;
                        errorMsg = "Cache collision: Data size does not match";
                    }
                }
                if (ok == true) {
                    if (buildOpts.size() == fileBuildOpts.size()) {
                        if (memcmp(buildOpts.c_str(), fileBuildOpts.c_str(), buildOpts.size())) {
                            ok = false;
                            errorMsg = "Cache collision: Build opts do not match";
                        } else if (kernelName.size() == fileKernelName.size()) {
                            if (memcmp(kernelName.c_str(), fileKernelName.c_str(), kernelName.size())) {
                                ok = false;
                                errorMsg = "Cache collision: Kernel names do not match";
                            }
                        } else {
                            ok = false;
                            errorMsg = "Cache collision: Kernel name lengths do not match";
                        }
                    } else {
                        ok = false;
                        errorMsg = "Cache collision: Build options lengths do not match";
                    }
                }
            }
            if (fileSrcData) {
                delete[] fileSrcData;
            }
        }
        delete[] contents;
    }
    if (!ok) {
        delete[] *dstData;
        dstData = NULL;
        dstSize = 0;
    }
    return ok;
}

// Use srcdata, buildOpts and kernelName to generate a cache entry
//
// In:
// srcData - Source data
// srcNum - Number of source data
// buildOpts - Build options
// kernelName - Kernel name
// dstData - Destination data
// dstSize - Destination size
//
// Returns:
// true if entry created
// false otherwise, check errorMsg for errors
//
bool KernelCache::makeCacheEntry(const KernelCacheData *srcData, const unsigned int srcNum,
    const std::string &buildOpts, const std::string &kernelName, const char *dstData, const unsigned int dstSize)
{
    bool ok;
    std::string fileName;
    char *fileData = NULL;
    unsigned int fileSize;
    errorMsg.clear();
    makeFileName(srcData, srcNum, buildOpts, kernelName, fileName);
    unsigned int  dstHashVal = 0;
    KernelCacheData cacheData;
    cacheData.data = (char *)dstData;
    cacheData.dataSize = dstSize;
    dstHashVal = computeHash((const KernelCacheData *)&cacheData, 1, buildOpts, kernelName);
    ok = buildFile(srcData, srcNum, buildOpts, kernelName, dstData, dstSize, dstHashVal, &fileData, fileSize);
    if (!ok) {
        // Return an error
        return false;
    }
    ok = writeFile(fileName, fileData, fileSize);
    delete[] fileData;
    if (!ok) {
        // Return an error
        return false;
    }
    ok = setCacheInfo(version, cacheSize + fileSize);
    if (!ok) {
        errorMsg = "Cache version and size is not updated successfully";
        // Return an error
        return false;
    }
    return ok;
}
