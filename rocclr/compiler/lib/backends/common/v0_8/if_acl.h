//
// Copyright (c) 2012 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef _IF_ACL_0_8_H_
#define _IF_ACL_0_8_H_
#include "aclTypes.h"
aclLoaderData* ACL_API_ENTRY
if_aclCompilerInit(aclCompiler *cl, aclBinary *bin, aclLogFunction log, acl_error *error);
acl_error  ACL_API_ENTRY
if_aclCompilerFini(aclLoaderData *ald);

acl_error  ACL_API_ENTRY
if_aclCompile(aclCompiler *cl,
    aclBinary *bin,
    const char *options,
    aclType from,
    aclType to,
    aclLogFunction compile_callback) ACL_API_0_8;

acl_error  ACL_API_ENTRY
if_aclLink(aclCompiler *cl,
    aclBinary *src_bin,
    unsigned int num_libs,
    aclBinary **libs,
    aclType link_mode,
    const char *options,
    aclLogFunction link_callback) ACL_API_0_8;

const char*  ACL_API_ENTRY
if_aclGetCompilerLog(aclCompiler *cl) ACL_API_0_8;

const void*  ACL_API_ENTRY
if_aclRetrieveType(aclCompiler *cl,
    const aclBinary *bin,
    const char *name,
    size_t *data_size,
    aclType type,
    acl_error *error_code) ACL_API_0_8;

acl_error  ACL_API_ENTRY
if_aclSetType(aclCompiler *cl,
    aclBinary *bin,
    const char *name,
    aclType type,
    const void *data,
    size_t size) ACL_API_0_8;

acl_error  ACL_API_ENTRY
if_aclConvertType(aclCompiler *cl,
    aclBinary *bin,
    const char *name,
    aclType type) ACL_API_0_8;

acl_error  ACL_API_ENTRY
if_aclDisassemble(aclCompiler *cl,
    aclBinary *bin,
    const char *kernel,
    aclLogFunction disasm_callback) ACL_API_0_8;

const void*  ACL_API_ENTRY
if_aclGetDeviceBinary(aclCompiler *cl,
    const aclBinary *bin,
    const char *kernel,
    size_t *size,
    acl_error *error_code) ACL_API_0_8;

acl_error  ACL_API_ENTRY
if_aclInsertSection(aclCompiler *cl,
    aclBinary *binary,
    const void *data,
    size_t data_size,
    aclSections id) ACL_API_0_8;

acl_error  ACL_API_ENTRY
if_aclInsertSymbol(aclCompiler *cl,
    aclBinary *binary,
    const void *data,
    size_t data_size,
    aclSections id,
    const char *symbol) ACL_API_0_8;

const void*  ACL_API_ENTRY
if_aclExtractSection(aclCompiler *cl,
    const aclBinary *binary,
    size_t *size,
    aclSections id,
    acl_error *error_code) ACL_API_0_8;

const void*  ACL_API_ENTRY
if_aclExtractSymbol(aclCompiler *cl,
    const aclBinary *binary,
    size_t *size,
    aclSections id,
    const char *symbol,
    acl_error *error_code) ACL_API_0_8;

acl_error  ACL_API_ENTRY
if_aclRemoveSection(aclCompiler *cl,
    aclBinary *binary,
    aclSections id) ACL_API_0_8;

acl_error  ACL_API_ENTRY
if_aclRemoveSymbol(aclCompiler *cl,
    aclBinary *binary,
    aclSections id,
    const char *symbol) ACL_API_0_8;

acl_error  ACL_API_ENTRY
if_aclQueryInfo(aclCompiler *cl,
    const aclBinary *binary,
    aclQueryType query,
    const char *kernel,
    void *data_ptr,
    size_t *ptr_size) ACL_API_0_8;

acl_error  ACL_API_ENTRY
if_aclDbgAddArgument(aclCompiler *cl,
    aclBinary *bin,
    const char *kernel,
    const char *name,
    bool byVal) ACL_API_0_8;

acl_error  ACL_API_ENTRY
if_aclDbgRemoveArgument(aclCompiler *cl,
    aclBinary *bin,
    const char* kernel,
    const char* name) ACL_API_0_8;

acl_error  ACL_API_ENTRY
if_aclSetupLoaderObject(aclCompiler *cl) ACL_API_0_8;

#endif // _IF_ACL_0_8_H_
