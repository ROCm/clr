#include "roccompilerlib.hpp"
#include "utils/flags.hpp"

#include "acl.h"

namespace roc {

void* g_complibModule = NULL;
struct CompLibApi g_complibApi;

//
// g_complibModule is defined in LoadCompLib(). This macro must be used only in LoadCompLib() function.
//
#define LOADSYMBOL(api) \
    g_complibApi._##api  = (pfn_##api) amd::Os::getSymbol(g_complibModule, #api); \
    if( g_complibApi._##api == NULL ) { \
    LogError ("amd::Os::getSymbol() for exported func " #api " failed."); \
        amd::Os::unloadLibrary(g_complibModule); \
        return false; \
    }

bool LoadCompLib(bool offline)
{
    g_complibModule = amd::Os::loadLibrary("amdhsacl" LP64_SWITCH(LINUX_SWITCH("32",""), "64"));
    if( g_complibModule == NULL ) {
        if (!offline) {
            LogError( "amd::Os::loadLibrary() for loading of amdhsacl.dll failed.");
        }
        return false;
    }

    LOADSYMBOL(aclCompilerInit)
    LOADSYMBOL(aclGetTargetInfo)
    LOADSYMBOL(aclBinaryInit)
    LOADSYMBOL(aclInsertSection)
    LOADSYMBOL(aclCompile)
    LOADSYMBOL(aclCompilerFini)
    LOADSYMBOL(aclBinaryFini)
    LOADSYMBOL(aclWriteToMem)
    LOADSYMBOL(aclQueryInfo)
    LOADSYMBOL(aclExtractSymbol)
    LOADSYMBOL(aclGetCompilerLog)
    LOADSYMBOL(aclCreateFromBinary)
    LOADSYMBOL(aclReadFromMem)
    LOADSYMBOL(aclBinaryVersion)
    LOADSYMBOL(aclLink)

    return true;
}

void UnloadCompLib()
{
    if( g_complibModule )
    {
        amd::Os::unloadLibrary(g_complibModule);
    }
}

} // namespace roc
