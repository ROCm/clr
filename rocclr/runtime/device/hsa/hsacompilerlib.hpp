#ifndef HSACOMPILERLIB_HPP_
#define HSACOMPILERLIB_HPP_

//
// This file  hsa the code for explicity loading amdoclcl.dll. 
// Exported functions from amdoclcl.dll can be added for usage as need-basis.
// With explicit/dynamic loading oclhsa will not have any linkage to amdoclcl.lib.
//

#include "thread/thread.hpp"
#include "acl.h"
#include "utils/debug.hpp"

using namespace amd;

namespace oclhsa {

//
// To use any new exported function from amdhsacl.dll please add/make that function specific changes 
// in typedef below, struct CompLibApi and in hsacompilerLib.cpp::LoadCompLib() function.
//

//
// Convention: The typedefed function name must be prefixed with pfn_
//
typedef aclCompiler*  (ACL_API_ENTRY *pfn_aclCompilerInit) (aclCompilerOptions *opts, acl_error *error_code);
typedef aclTargetInfo (ACL_API_ENTRY *pfn_aclGetTargetInfo) (const char*, const char*, acl_error*);
typedef aclBinary*    (ACL_API_ENTRY *pfn_aclBinaryInit) (size_t, const aclTargetInfo*, const aclBinaryOptions*, acl_error*);
typedef acl_error     (ACL_API_ENTRY *pfn_aclInsertSection) (aclCompiler *cl, aclBinary *binary, const void *data, size_t data_size, aclSections id);
typedef acl_error     (ACL_API_ENTRY *pfn_aclCompile) (aclCompiler *cl, aclBinary *bin, const char *options, aclType from, aclType to, aclLogFunction compile_callback);
typedef acl_error     (ACL_API_ENTRY *pfn_aclCompilerFini) (aclCompiler *cl);
typedef acl_error     (ACL_API_ENTRY *pfn_aclBinaryFini) (aclBinary *bin);
typedef const void*   (ACL_API_ENTRY *pfn_aclExtractSection) (aclCompiler *cl, const aclBinary *binary, size_t *size, aclSections id, acl_error *error_code);
typedef acl_error     (ACL_API_ENTRY *pfn_aclWriteToMem) (aclBinary *bin,void **mem, size_t *size);
typedef acl_error     (ACL_API_ENTRY *pfn_aclQueryInfo) (aclCompiler *cl, const aclBinary *binary, aclQueryType query, const char *kernel, void *data_ptr, size_t *ptr_size);


typedef const void*   (ACL_API_ENTRY *pfn_aclGetDeviceBinary) (aclCompiler *cl,const aclBinary *bin,const char *kernel,size_t *size,acl_error *error_code);
typedef const void*   (ACL_API_ENTRY *pfn_aclExtractSymbol) (aclCompiler *cl,const aclBinary *binary,size_t *size,aclSections id,const char *symbol,acl_error *error_code);
typedef aclBinary*    (ACL_API_ENTRY *pfn_aclReadFromMem) (void *mem,size_t size, acl_error *error_code);
typedef acl_error     (ACL_API_ENTRY *pfn_aclRemoveSymbol) (aclCompiler *cl, aclBinary *binary, aclSections id, const char *symbol);
typedef acl_error     (ACL_API_ENTRY *pfn_aclInsertSymbol) (aclCompiler *cl, aclBinary *binary, const void *data, size_t data_size, aclSections id, const char *symbol);
typedef acl_error     (ACL_API_ENTRY *pfn_aclWriteToFile) (aclBinary *bin, const char *str);


typedef char*         (ACL_API_ENTRY *pfn_aclGetCompilerLog) (aclCompiler* cl);
typedef aclBinary*    (ACL_API_ENTRY *pfn_aclCreateFromBinary) (const aclBinary *binary,aclBIFVersion version);
typedef aclBIFVersion (ACL_API_ENTRY *pfn_aclBinaryVersion) (const aclBinary *binary);
typedef acl_error     (ACL_API_ENTRY *pfn_aclLink) (aclCompiler* cl, aclBinary *src_bin, unsigned int num_libs, aclBinary **libs, aclType link_mode,const char* options, aclLogFunction link_callback);
//
// Convention: prefix struct member variable with with underscore '_'
// would be nice if there was no underscore prfix, but on Linux the token 
// pasting in the macro is srtict and his is the workaround.
//
struct CompLibApi
{
    pfn_aclCompilerInit     _aclCompilerInit;
    pfn_aclGetTargetInfo    _aclGetTargetInfo;
    pfn_aclBinaryInit       _aclBinaryInit;
    pfn_aclInsertSection    _aclInsertSection;
    pfn_aclCompile          _aclCompile;
    pfn_aclCompilerFini     _aclCompilerFini;
    pfn_aclBinaryFini       _aclBinaryFini;
    pfn_aclExtractSection   _aclExtractSection;
    pfn_aclWriteToMem       _aclWriteToMem;
    pfn_aclQueryInfo        _aclQueryInfo;
    pfn_aclGetDeviceBinary  _aclGetDeviceBinary;
    pfn_aclExtractSymbol    _aclExtractSymbol;
    pfn_aclReadFromMem      _aclReadFromMem;
    pfn_aclRemoveSymbol     _aclRemoveSymbol;
    pfn_aclInsertSymbol     _aclInsertSymbol;
    pfn_aclWriteToFile      _aclWriteToFile;
    pfn_aclGetCompilerLog   _aclGetCompilerLog;
    pfn_aclCreateFromBinary _aclCreateFromBinary;
    pfn_aclBinaryVersion _aclBinaryVersion;
    pfn_aclLink _aclLink;
};


//
// Use g_  prefix for all global variables.
//
extern void* g_complibModule;
extern CompLibApi g_complibApi;

// Note: initializes global variable g_complibApi.
// Not sure what error values we have, for now returning false on failure.
bool LoadCompLib(bool isOfflineDevice=false);
void UnloadCompLib();

} // namespace oclhsa
#endif
