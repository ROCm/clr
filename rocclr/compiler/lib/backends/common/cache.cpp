#include "cache.hpp"
// Use srcdata and buildOpts to generate a cache entry
//
// In:
// isCacheReady - Indicate whether cache file strucutre is set up
// srcData - Source data
// buildOpts - Build options
// dstData - Destination data
//
// Out:
// none
//
// Returns:
// true if entry created; false otherwise, check errorMsg for errors
//
bool KernelCache::makeCacheEntry(bool isCacheReady, const KernelCacheData *srcData,
                                 const std::string &buildOpts, const std::string &dstData)
{
  if (!isCacheReady) {
    errorMsg = "makeCacheEntry() failed because cache file structure is not set up successfully";
    appendLogToFile();
    return false;
  }
  errorMsg.clear();
  std::string fileName;
  makeFileName(srcData, buildOpts, fileName);
  /* Write all info to cache file */
  const size_t buildOptsSize = buildOpts.size();
  const size_t dstDataSize = dstData.size();
#if _WIN32
  HANDLE cacheFile = CreateFile(fileName.c_str(), GENERIC_WRITE | WRITE_OWNER | READ_CONTROL,
                                0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
  if (cacheFile == INVALID_HANDLE_VALUE) {
    errorMsg = "Error opening file for writing: " + getLastErrorMsg();
    return false;
  }
#else
  int cacheFile = open(fileName.c_str(), O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
  if (cacheFile == -1) {
    errorMsg = "Error opening file for writing: " + getLastErrorMsg();
    return false;
  }
  // Exclusive write lock for cache file
  struct flock fl = {F_WRLCK, SEEK_SET, 0, 0};
  if (fcntl(cacheFile, F_SETLK, &fl) == -1) {
    logErrorCloseFile("Error setting file write lock: " + getLastErrorMsg(), cacheFile);
    return false;
  }
#endif
  // Write kernel cache file header, build options, source data size, source data
  // and destination data to cache file
  KernelCacheFileHeader H = { {'A', 'M', 'D', '\0'}, buildOptsSize, dstDataSize };
  if (!writeFile(cacheFile, &H, sizeof(KernelCacheFileHeader)) ||
      !writeFile(cacheFile, buildOpts.c_str(), buildOptsSize) ||
      !writeFile(cacheFile, &(srcData->dataSize), sizeof(srcData->dataSize)) ||
      !writeFile(cacheFile, srcData->data, srcData->dataSize) ||
      !writeFile(cacheFile, dstData.c_str(), dstDataSize)) {
    removePartiallyWrittenFile(fileName);
    return false;
  }
#if __linux__
  // Unlock the file
  fl.l_type = F_UNLCK;
  if (fcntl(cacheFile, F_SETLK, &fl) == -1) {
    logErrorCloseFile("Error unlock file write lock: " + getLastErrorMsg(), cacheFile);
    return false;
  }
#endif
  CloseFile(cacheFile);
  // Set file to only owner accessible
  if (!setAccessPermission(fileName, true)) {
    return false;
  }
  // Update cache info
  unsigned int cacheFileSize = sizeof(KernelCacheFileHeader) + buildOptsSize
                             + sizeof(srcData->dataSize) + srcData->dataSize
                             + dstDataSize;
  if (!setCacheInfo(version, cacheSize + cacheFileSize)) {
    errorMsg = "Cache version and size is not updated successfully";
    return false;
  }
  return true;
}
// Use srcData and buildOpts to find the corresponding cache entry, if it exists
//
// In:
// deviceName - chip name
// opts - Options object
// srcData - Source data
// buildOpts - Build options
// Msg - message that need to passed for internal cache testing
//
// Out:
// isCacheReady - Indicate whether cache file strucutre is set up
// dstData - Destination data
//
// Returns:
// true if entry found; false otherwise, check errorMsg for errors
//
bool KernelCache::getCacheEntry(bool &isCacheReady, const std::string &deviceName, amd::option::Options *opts,
                                const KernelCacheData *srcData, const std::string &buildOpts,
                                std::string &dstData, const std::string &msg)
{
  dstData.clear();
  errorMsg.clear();
  bool kernelCached = false;
  if (!opts->oVariables->DisableKernelCaching && opts->oVariables->OptLevel > 0) {
    isCacheReady = cacheInit(SC_BUILD_NUMBER, deviceName);
    if (!isCacheReady) {
      appendLogToFile();
    } else {
      kernelCached = getCacheEntry_helper(srcData, buildOpts, dstData);
      // For internal kernel cache test only
      if (internalKCacheTestSwitch()) {
        std::string cacheMsg = msg;
        if (kernelCached) {
          cacheMsg += " is cached!\n";
        } else {
          cacheMsg += " is not cached!\n";
        }
        fprintf(stdout, cacheMsg.c_str());
        fflush(stdout);
      }
    }
  }
  if (!errorMsg.empty()) {
    appendLogToFile();
  }
  return kernelCached;
}
// Use srcData and buildOpts to find the corresponding cache entry, if it exists
//
// In:
// srcData - Source data
// buildOpts - Build options
//
// Out:
// dstData - Destination data
//
// Returns:
// true if entry found; false otherwise, check errorMsg for errors
//
bool KernelCache::getCacheEntry_helper(const KernelCacheData *srcData, const std::string &buildOpts,
                                       std::string &dstData)
{
  std::string fileName;
  makeFileName(srcData, buildOpts, fileName);
#if _WIN32
  HANDLE cacheFile = CreateFile(fileName.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (cacheFile == INVALID_HANDLE_VALUE) {
    return false;
  }
#else
  int cacheFile = open(fileName.c_str(), O_RDONLY);
  if (cacheFile == -1) {
    return false;
  }
  // Read lock for cache file
  struct flock fl = {F_RDLCK, SEEK_SET, 0, 0};
  if (fcntl(cacheFile, F_SETLK, &fl) == -1) {
    logErrorCloseFile("Error setting file read lock: " + getLastErrorMsg(), cacheFile);
    return false;
  }
#endif
  // Read kernel cache file header
  KernelCacheFileHeader H;
  if (!readFile(cacheFile, &H, sizeof(KernelCacheFileHeader))) {
    return false;
  }
  // Compare kernel cache file header
  if (!verifyKernelCacheFileHeader(H, buildOpts)) {
    CloseFile(cacheFile);
    return false;
  }
  // Read build options
  char *fileBuildOpts = (char *)alloca(H.buildOptSize);
  if (!readFile(cacheFile, fileBuildOpts, H.buildOptSize)) {
    return false;
  }
  // Compare build options
  if (buildOpts.compare(0, H.buildOptSize, fileBuildOpts, H.buildOptSize)) {
    logErrorCloseFile("Cache collision: Build options do not match", cacheFile);
    return false;
  }
  // Get source data size
  size_t fileSrcDataSize = 0;
  if (!readFile(cacheFile, &fileSrcDataSize, sizeof(size_t))) {
    return false;
  }
  // Compare source data size
  if (fileSrcDataSize != srcData->dataSize) {
    logErrorCloseFile("Cache collision: Data size does not match", cacheFile);
    return false;
  }
  // Get source data
  std::unique_ptr<char> fileSrcData(new char [fileSrcDataSize]);
  if (!fileSrcData) {
    logErrorCloseFile("Out of memory: " + getLastErrorMsg(), cacheFile);
    return false;
  }
  if (!readFile(cacheFile, fileSrcData.get(), fileSrcDataSize)) {
    return false;
  }
  // Compare source data
  if (memcmp(fileSrcData.get(), srcData->data, fileSrcDataSize)) {
    logErrorCloseFile("Cache collision: Size matches, contents do not", cacheFile);
    return false;
  }
  // Get cached content
  std::unique_ptr<char> data(new char [H.dstSize]);
  if (!data) {
    logErrorCloseFile("Out of memory: " + getLastErrorMsg(), cacheFile);
    return false;
  }
  if (!readFile(cacheFile, data.get(), H.dstSize)) {
    return false;
  }
  dstData.assign(data.get(), H.dstSize);
#if __linux__
  // Unlock the file
  fl.l_type = F_UNLCK;
  if (fcntl(cacheFile, F_SETLK, &fl) == -1) {
    logErrorCloseFile("Error unlock file read lock: " + getLastErrorMsg(), cacheFile);
    return false;
  }
#endif
  CloseFile(cacheFile);
  return true;
}
#if _WIN32
// Get Sid of account
//
// In:
// userName - accont name
//
// Out:
// none
//
// Return:
// Sid of account if SID is obtained; NULL otherwise
//
std::unique_ptr<SID> KernelCache::getSid(TCHAR *username)
{
  if (username == NULL) {
    errorMsg = "Invalid user name in getSid mehtod";
    return NULL;
  }
  // If a buffer is too small, the count parameter will be set to the size needed.
  const DWORD initialSize = 32;
  SID_NAME_USE sidNameUse;
  DWORD cbSid = initialSize, cchDomainName = initialSize;
  // Create buffers for the SID and the domain name
  std::unique_ptr<SID> sid = std::unique_ptr<SID>((SID*) new BYTE[initialSize]);
  if (!sid) {
    errorMsg = "Failed to allocate space for SID: " + getLastErrorMsg();
    return NULL;
  }
  std::unique_ptr<TCHAR> wszDomainName(new TCHAR[initialSize]);
  if (!wszDomainName) {
    errorMsg = "Failed to allocate space for domain name: " + getLastErrorMsg();
    return NULL;
  }
  // Obtain the SID for the account name passed
  if (LookupAccountName(NULL, username, sid.get(), &cbSid, wszDomainName.get(), &cchDomainName, &sidNameUse)) {
    if (IsValidSid(sid.get()) == FALSE) {
      errorMsg = "The SID for the account is invalid: " + getLastErrorMsg();
      return NULL;
    }
    return sid;
  }
  DWORD dwErrorCode = GetLastError();
  if (dwErrorCode == ERROR_INSUFFICIENT_BUFFER) {
    if (cbSid > initialSize) {
      // Reallocate memory for the SID buffer
      sid = std::unique_ptr<SID>((SID*)new BYTE[cbSid]);
      if (!sid) {
        errorMsg = "Failed to allocate space for SID: " + getLastErrorMsg();
        return NULL;
      }
    }
    if (cchDomainName > initialSize) {
      // Reallocate memory for the domain name buffer
      wszDomainName = std::unique_ptr<TCHAR>(new TCHAR[cchDomainName]);
      if (!wszDomainName) {
        errorMsg = "Failed to allocate space for domain name: " + getLastErrorMsg();
        return NULL;
      }
    }
    // Obtain the SID for the account name passed again
    if (LookupAccountName(NULL, username, sid.get(), &cbSid, wszDomainName.get(), &cchDomainName, &sidNameUse)) {
      if (IsValidSid(sid.get()) == FALSE) {
        errorMsg = "The SID for the account is invalid: " + getLastErrorMsg();
        return NULL;
      }
      return sid;
    }
  } else {
    errorMsg = "Failed to get user security identifier for the account: " + getLastErrorMsg();
    return NULL;
  }
  return sid;
}
#endif
#if defined(__linux__)
inline bool path_is_directory(const std::string &path)
{
  struct stat s_buf;
  if (stat(path.c_str(), &s_buf))
    return false;
  return S_ISDIR(s_buf.st_mode);
}
// Remove all files and subfolders in a dir
//
// In:
// directory_name - folder name
//
// Out:
// none
//
// Returns:
// The number of files that are removed
//
static unsigned long fileCnt = 0;
unsigned long remove_all(const char* directory_name)
{
  DIR *dp;
  struct dirent *ep;
  char p_buf[PATH_MAX] = {0};
  dp = opendir(directory_name);
  while ((ep = readdir(dp)) != NULL) {
    if (strcmp(ep->d_name, "..") && strcmp(ep->d_name, ".")) {
      snprintf(p_buf, PATH_MAX, "%s/%s", directory_name, ep->d_name);
      if (path_is_directory(p_buf)) {
        if (remove_all(p_buf) < 0) {
          return LONG_MIN;
        }
      } else {
        if (unlink(p_buf) != 0) {
          return LONG_MIN;
        } else {
          fileCnt++;
        }
      }
    }
  }
  closedir(dp);
  return (rmdir(directory_name) == 0) ? fileCnt : LONG_MIN;
}
#endif
// Wipe the cache folder structure
//
// In:
// none
//
// Out:
// none
//
// Returns:
// true if folder wipe is ok; false otherwise
//
bool KernelCache::wipeCacheFolders()
{
  for (int i = 0; i < 16; ++i) {
    std::string dir = rootPath;
    std::stringstream ss;
    ss << amd::Os::fileSeparator() << std::hex << i;
    dir += ss.str();
    if (amd::Os::pathExists(dir)) {
#if _WIN32
      std::tr2::sys::path mDir(dir);
      if (remove_all(mDir) < 0) {
#else
      if (remove_all(dir.c_str()) < 0) {
#endif
        errorMsg = "Error deleting cache directory";
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
// true if folders setup is ok; false otherwise
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
// Return detailed error message as string
//
// In:
// None
//
// Out:
// None
//
// Return:
// Error message in string format. Otherwise, an empty string if there is no error
//
std::string KernelCache::getLastErrorMsg()
{
#if _WIN32
  // Get the error message, if any.
  DWORD errorMessageID = GetLastError();
  if (errorMessageID == 0) {
    return std::string(); //No error message has been recorded
  }
  LPSTR messageBuffer = nullptr;
  size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                               NULL, errorMessageID, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL);
  std::string message(messageBuffer, size);
  LocalFree(messageBuffer);
  return message;
#else
  return std::string(strerror(errno));
#endif
}
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
// true if access permission is under control; false otherwise
//
bool KernelCache::setAccessPermission(const std::string &fileName, bool isFile)
{
#if _WIN32
  TCHAR username[UNLEN + 1];
  DWORD username_len = UNLEN + 1;
  if (!GetUserName(username, &username_len)) {
    errorMsg = "Failed to get user name for the account: " + getLastErrorMsg();
    return false;
  }
  std::unique_ptr<SID> sid = getSid(username);
  if (!sid) {
    return false;
  }
  if (SetNamedSecurityInfo((LPTSTR)(fileName.c_str()), SE_FILE_OBJECT, OWNER_SECURITY_INFORMATION,
                           sid.get(), NULL, NULL, NULL) != ERROR_SUCCESS ) {
    errorMsg = "Failed to set user access permission: " + getLastErrorMsg();
    return false;
  }
#else
  if (!isFile) {
    int ret = chmod(fileName.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    if (ret < 0) {
      errorMsg = "Failed to set user access permission: " + getLastErrorMsg();
      return false;
    }
  }
#endif
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
// true if root path of cache is set successfully; false otherwise
//
bool KernelCache::setRootPath(const std::string &chipName)
{
  rootPath.clear();
#if _WIN32
  // Set root path to <USER>\AppData\Local\AMD\CLCache
  TCHAR userLocalAppDir[_MAX_PATH];
  // Get path for user specific and non-roaming data
  if (SUCCEEDED(SHGetFolderPath(NULL, CSIDL_LOCAL_APPDATA, NULL, SHGFP_TYPE_CURRENT, userLocalAppDir))) {
    rootPath = userLocalAppDir;
  } else {
    errorMsg = "User's local app dir is not found: " + getLastErrorMsg();
    return false;
  }
  rootPath += "\\AMD\\CLCache";
#else
  // Set root path to <HOME>/.AMD/CLCache
  struct passwd *pwd = getpwuid(getuid());
  if (pwd == NULL) {
    errorMsg = getLastErrorMsg();
    return false;
  }
  const char *homedir = pwd->pw_dir;
  if (homedir == NULL) {
    errorMsg = "Failed to get HOME directory: " + getLastErrorMsg();
    return false;
  }
  rootPath = homedir;
  // Verify the path exists
  if (!amd::Os::pathExists(rootPath)) {
    errorMsg = "User's home directory is not created: " + getLastErrorMsg();
    return false;
  }
  rootPath += "/.AMD/CLCache";
#endif
  rootPath += amd::Os::fileSeparator() + chipName;
  if (!amd::Os::createPath(rootPath)) {
    errorMsg = "Failed to create cache root directory";
    return false;
  }
  // Set folder to only owner accessible
  return setAccessPermission(rootPath);
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
// true if successful; false otherwise
//
bool KernelCache::setCacheInfo(unsigned int newVersion, unsigned int newSize)
{
  unsigned int fileData[2];
  fileData[0] = newVersion;
  fileData[1] = newSize;
  if (!writeFile(indexName, fileData, sizeof(fileData), false)) {
    removePartiallyWrittenFile(indexName);
    return false;
  }
  version = newVersion;
  cacheSize = newSize;
  return true;
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
// true if successful; false otherwise
//
bool KernelCache::getCacheInfo()
{
  indexName = rootPath;
  indexName += amd::Os::fileSeparator();
  indexName += "cacheDir";
#if _WIN32
  HANDLE cacheFile = CreateFile(indexName.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (cacheFile == INVALID_HANDLE_VALUE) {
    return setCacheInfo(-1, 0);
  }
#else
  int cacheFile = open(indexName.c_str(), O_RDONLY);
  if (cacheFile == -1) {
    return setCacheInfo(-1, 0);
  }
  // Read lock for cache file
  struct flock fl = {F_RDLCK, SEEK_SET, 0, 0};
  if (fcntl(cacheFile, F_SETLK, &fl) == -1) {
    logErrorCloseFile("Error setting file read lock: " + getLastErrorMsg(), cacheFile);
    return false;
  }
#endif
  if (!readFile(cacheFile, &version, sizeof(unsigned int))) {
    return false;
  }
  if (!readFile(cacheFile, &cacheSize, sizeof(unsigned int))) {
    return false;
  }
#if __linux__
  // Unlock the file
  fl.l_type = F_UNLCK;
  if (fcntl(cacheFile, F_SETLK, &fl) == -1) {
    logErrorCloseFile("Error unlock file read lock: " + getLastErrorMsg(), cacheFile);
    return false;
  }
#endif
  CloseFile(cacheFile);
  return true;
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
// true if successful; false otherwise
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
    if (!wipeCacheFolders() || !setCacheInfo(compilerVersion, 0) || !setUpCacheFolders()) {
      return false;
    }
  }
  return true;
}
// Compute the hash value for a buffer of data along with the buildOpts
//
// In:
// data - Data to hash
// buildOpts - Build options
//
// Out:
// none
//
// Returns:
// Hash value computed from the inputs
//
unsigned int KernelCache::computeHash(const KernelCacheData *data, const std::string &buildOpts)
{
  HashType v = { std::string(data->data, data->dataSize), buildOpts };
  std::hash<HashType> hash_fn;
  return hash_fn(v);
}
// Control kernel cache test
//
// In:
// none
//
// Out:
// none
//
// Returns:
// true if kernel cache test is on; false otherwise
//
bool KernelCache::internalKCacheTestSwitch() {

  return false;
}
// Generate file path from a hash value
//
// In:
// hashVal - A hash value
//
// Out:
// pathToFile - Path to the file
//
// Returns:
// none
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
// Use data and buildOpts to generate a file name
//
// In:
// data - Pointer to data
// buildOpts - Build options
//
// Out:
// pathToFile - Path to the file
//
// Returns:
// none
//
void KernelCache::makeFileName(const KernelCacheData *data, const std::string &buildOpts, std::string &pathToFile)
{
  unsigned int hashVal = computeHash(data, buildOpts);
  getFilePathFromHash(hashVal, pathToFile);
}
// Verify whether the file includes the right kernel cache file header
//
// In:
// H - Kernel cache file header
// buildOpts - Build options
//
// Out:
// None
//
// Returns:
// true if the file is the one matched our requirement; false othereise
//
bool KernelCache::verifyKernelCacheFileHeader(KernelCacheFileHeader &H, const std::string &buildOpts)
{
  const char AMD[4] = {'A', 'M', 'D', '\0'};
  if (memcmp(H.AMD, AMD, 4)) {
    errorMsg = "Not a valid cache file";
    return false;
  }
  if (H.buildOptSize != buildOpts.size()) {
    errorMsg = "Cache collision: Build option lengths do not match";
    return false;
  }
  return true;
}
// Read contents in cacheFile
//
// In:
// cacheFile - cache file to be read
// sizeToRead - total bytes to be read
//
// Out:
// buffer - contains file content
//
// Returns:
// true if file reads succeed; false otherwise
//
bool KernelCache::readFile(FileHandle cacheFile, void *buffer, ssize_t sizeToRead)
{
  // Read content to the buffer
#if _WIN32
  DWORD bytesRead = 0;
  if (FALSE == ReadFile(cacheFile, buffer, sizeToRead, &bytesRead, NULL)) {
    logErrorCloseFile("Unable to read cache file: " + getLastErrorMsg(), cacheFile);
    return false;
  }
#else
  ssize_t bytesRead = read(cacheFile, buffer, sizeToRead);
#endif
  // Check if there is any error in file reading
  if (bytesRead != sizeToRead) {
    logErrorCloseFile("Error reading cache file: " + getLastErrorMsg(), cacheFile);
    return false;
  }
  return true;
}
// Write contents to cacheFile
//
// In:
// cacheFile - cache file to be written
// buffer - contains content to be written
// sizeToWriten - total bytes to be written
//
// Out:
// none
//
// Returns:
// true if file writes succeed; false otherwise
//
bool KernelCache::writeFile(FileHandle cacheFile, const void *buffer, ssize_t sizeToWritten)
{
#if _WIN32
  DWORD bytesWritten = 0;
  if (FALSE == WriteFile(cacheFile, buffer, sizeToWritten, &bytesWritten, NULL)) {
    logErrorCloseFile("Unable to write to file: " + getLastErrorMsg(), cacheFile);
    return false;
  }
#else
  ssize_t bytesWritten = write(cacheFile, buffer, sizeToWritten);
#endif
  // Check if there is any error in file reading
  if (bytesWritten != sizeToWritten) {
    logErrorCloseFile("Error writing cache file: " + getLastErrorMsg(), cacheFile);
    return false;
  }
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
// true if the file is written to file successfully; false otherwise
//
bool KernelCache::writeFile(const std::string &fileName, const void *data, size_t size, bool appendable)
{
#if _WIN32
  DWORD appendAccess = 0;
  if (appendable) {
    appendAccess = FILE_APPEND_DATA;
  }
  HANDLE cacheFile = CreateFile(fileName.c_str(), GENERIC_WRITE | WRITE_OWNER | READ_CONTROL | appendAccess,
                            0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
  if (cacheFile == INVALID_HANDLE_VALUE) {
    errorMsg = "Error opening file for writing: " + getLastErrorMsg();
    return false;
  }
#else
  int cacheFile = -1;
  if (appendable) {
    cacheFile = open(fileName.c_str(), O_WRONLY | O_CREAT | O_APPEND, S_IRUSR | S_IWUSR);
  } else {
    cacheFile = open(fileName.c_str(), O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
  }
  if (cacheFile == -1) {
    errorMsg = "Error opening file for writing: " + getLastErrorMsg();
    return false;
  }
  // Exclusive write lock for cache file
  struct flock fl = {F_WRLCK, SEEK_SET, 0, 0};
  if (fcntl(cacheFile, F_SETLK, &fl) == -1) {
    logErrorCloseFile("Error setting file write lock: " + getLastErrorMsg(), cacheFile);
    return false;
  }
#endif
  // Write data to file
  if (!writeFile(cacheFile, data, size)) {
    removePartiallyWrittenFile(fileName);
    return false;
  }
#if __linux__
  // Unlock the file
  fl.l_type = F_UNLCK;
  if (fcntl(cacheFile, F_SETLK, &fl) == -1) {
    logErrorCloseFile("Error unlock file write lock: " + getLastErrorMsg(), cacheFile);
    return false;
  }
#endif
  CloseFile(cacheFile);
  // Set file to only owner accessible
  return setAccessPermission(fileName, true);
}
// Remove file
//
// In:
// fileName - Path to file
//
// Out:
// none
//
// Returns:
// none
//
void KernelCache::removePartiallyWrittenFile(const std::string &fileName)
{
  errorMsg = getLastErrorMsg();
#if _WIN32
  if (!DeleteFile(fileName.c_str())) {
#else
  if (remove(fileName.c_str())) {
#endif
    errorMsg += ", Unable to delete partially written cache file: " + getLastErrorMsg();
  }
}
// Log caching error messages for debugging the cache and/or detecting collisions
//
// In:
// extraMsg - Extra message
//
// Out:
// none
//
// Returns:
// none
//
void KernelCache::appendLogToFile(std::string extraMsg) {
  if (amd::Os::pathExists(rootPath)) {
    std::string fileName = rootPath + amd::Os::fileSeparator() + "cacheError.log";
    errorMsg += extraMsg;
    if ('\n' != errorMsg[errorMsg.size()-1]) {
      errorMsg.append("\n");
    }
    writeFile(fileName, errorMsg.c_str(), errorMsg.length(), true);
  }
}
// Log error message and close the file
//
// In:
// errorMsg - Error message
// file - file handle
//
// Out:
// none
//
// Returns:
// none
//
void KernelCache::logErrorCloseFile(const std::string &errorMsg, const FileHandle file)
{
  appendLogToFile(errorMsg);
  CloseFile(file);
}
