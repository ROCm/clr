#ifndef __gl_ATI_Internal_h_
#define __gl_ATI_Internal_h_


/* Copyright (C) 2004 - 2022 Advanced Micro Devices, Inc. */

#ifdef __cplusplus
extern "C" {
#endif


#ifndef APIENTRY
#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>
#endif
#if defined(ATI_OS_WIN)
#include <stddef.h>
#elif (defined(ATI_OS_LINUX) || defined(LINUX))
#include <stdint.h>
#endif

/* EXPORTED GSL INTERFACES */
#ifndef GL_ATI_subset_layer
#define GL_ATI_subset_layer 1

/* GSL CAPS extension */
#define GL_HWCAPS_MAX_VERTEX_COUNT_GSL                                  0x121000
#define GL_HWCAPS_MAX_BYTE_INDEX_COUNT_GSL                              0x121001
#define GL_HWCAPS_MAX_SHORT_INDEX_COUNT_GSL                             0x121002
#define GL_HWCAPS_MAX_INT_INDEX_COUNT_GSL                               0x121003
#define GL_HWCAPS_MAX_CLIP_PLANES_GSL                                   0x121004

#define GL_HWCAPS_BYTE_COMPONENT_COUNT_GSL                              0x121010
#define GL_HWCAPS_UNSIGNED_BYTE_COMPONENT_COUNT_GSL                     0x121011
#define GL_HWCAPS_SHORT_COMPONENT_COUNT_GSL                             0x121012
#define GL_HWCAPS_UNSIGNED_SHORT_COMPONENT_COUNT_GSL                    0x121013
#define GL_HWCAPS_INT_COMPONENT_COUNT_GSL                               0x121014
#define GL_HWCAPS_UNSIGNED_INT_COMPONENT_COUNT_GSL                      0x121015
#define GL_HWCAPS_FLOAT16_COMPONENT_COUNT_GSL                           0x121016
#define GL_HWCAPS_FLOAT_COMPONENT_COUNT_GSL                             0x121017
#define GL_HWCAPS_DOUBLE_COMPONENT_COUNT_GSL                            0x121018

#define GL_HWCAPS_BYTE_COMPONENT_VALUES_GSL                             0x121020
#define GL_HWCAPS_UNSIGNED_BYTE_COMPONENT_VALUES_GSL                    0x121021
#define GL_HWCAPS_SHORT_COMPONENT_VALUES_GSL                            0x121022
#define GL_HWCAPS_UNSIGNED_SHORT_COMPONENT_VALUES_GSL                   0x121023
#define GL_HWCAPS_INT_COMPONENT_VALUES_GSL                              0x121024
#define GL_HWCAPS_UNSIGNED_INT_COMPONENT_VALUES_GSL                     0x121025
#define GL_HWCAPS_FLOAT16_COMPONENT_VALUES_GSL                          0x121026
#define GL_HWCAPS_FLOAT_COMPONENT_VALUES_GSL                            0x121027
#define GL_HWCAPS_DOUBLE_COMPONENT_VALUES_GSL                           0x121028

#define GL_HWCAPS_HIERARCHICAL_DEPTH_AVAILABLE_GSL                      0x121030
#define GL_HWCAPS_COMPRESSED_DEPTH_AVAILABLE_GSL                        0x121031
#define GL_HWCAPS_COMPRESSED_COLOR_AVAILABLE_GSL                        0x121032
#define GL_HWCAPS_FBUFFER_AVAILABLE_GSL                                 0x121033

/* GSL ChromaKey extension */
#define GL_TEXTURE_CHROMAKEY_MODE_GSL                                   0x122000
#define GL_CHROMA_DISABLE_GSL                                           0x122001
#define GL_CHROMA_KILL_GSL                                              0x122002
#define GL_CHROMA_BLEND_GSL                                             0x122003
#define GL_TEXTURE_CHROMAKEY_COLOR_GSL                                  0x122004

/* GSL namespace extension */
#define GL_INTERNAL_NAMESPACE_GSL                                       0x123000
#define GL_GL_NAMESPACE_GSL                                             0x123001

/* GSL uber buffers extension */
#define GL_COLOR_BUFFER_GSL                                             0x124000
#define GL_COLOR_BUFFER1_GSL                                            0x124001
#define GL_COLOR_BUFFER2_GSL                                            0x124002
#define GL_COLOR_BUFFER3_GSL                                            0x124003
#define GL_COLOR_BUFFER4_GSL                                            0x124004
#define GL_COLOR_BUFFER5_GSL                                            0x124005
#define GL_COLOR_BUFFER6_GSL                                            0x124006
#define GL_COLOR_BUFFER7_GSL                                            0x124007
#define GL_COMPRESSED_COLOR_GSL                                         0x124008
#define GL_COMPRESSED_DEPTH_GSL                                         0x124009
#define GL_DEPTH_BUFFER_GSL                                             0x12400a
#define GL_DEPTH_COMPONENT24_STENCIL8_GSL                               0x12400b
#define GL_DEPTH_EXPAND_STATE_GSL                                       0x12400c
#define GL_DRAW_FRAMEBUFFER_GSL                                         0x12400d
#define GL_DRAW_FRAMEBUFFER_BINDING_GSL                                 0x12400e
#define GL_FBUFFER_GSL                                                  0x12400f
#define GL_GENERIC_ARRAY_GSL                                            0x124010
#define GL_HIERARCHICAL_DEPTH_GSL                                       0x124011
#define GL_IMAGES_GSL                                                   0x124012
#define GL_MACRO_TILING_GSL                                             0x124013
#define GL_MAPMEM_NO_SYNC_GSL                                           0x124014
#define GL_MAPMEM_READ_ONLY_GSL                                         0x124015
#define GL_MAPMEM_READ_WRITE_GSL                                        0x124016
#define GL_MEMORY_LOCATION_GSL                                          0x124017
#define GL_MICRO_TILING_GSL                                             0x124018
#define GL_MIPMAP_GSL                                                   0x124019
#define GL_MULTI_WRITE_GSL                                              0x12401a
#define GL_PRIMARY_SURFACE_GSL                                          0x12401b
#define GL_VERTEX_BUFFER_GSL                                            0x12401c
#define GL_WIDTH_GSL                                                    0x12401d
#define GL_HEIGHT_GSL                                                   0x12401e
#define GL_FORMAT_GSL                                                   0x12401f

//EP TestHookAMD
#define GL_EP_TIMMO_ENABLE_AMD_TEST_HOOK                                0x12f000
#define GL_EP_TIMMO_GET_ENABLED_AMD_TEST_HOOK                           0x12f001
#define GL_EP_TIMMO_NEVER_DISABLE_AMD_TEST_HOOK                         0x12f002
#define GL_EP_TIMMO_LIMIT_VIRTUAL_MEMORY_AMD_TEST_HOOK                  0x12f003
#define GL_EP_TIMMO_GET_NEVER_DISABLED_AMD_TEST_HOOK                    0x12f004
#define GL_EP_TIMMO_MATRIX_ENABLE_AMD_TEST_HOOK                         0x12f005
#define GL_EP_TIMMO_MATRIX_GET_ENABLED_AMD_TEST_HOOK                    0x12f006
#define GL_EP_TIMMO_GET_NTOTAL_VERTICES_AMD_TEST_HOOK                   0x12f007
#define GL_EP_TIMMO_GET_STAT_RESUME_COUNT_AMD_TEST_HOOK                 0x12f008
#define GL_EP_TIMMO_GET_STAT_RESUME_VERTICES_AMD_TEST_HOOK              0x12f009
#define GL_EP_TIMMO_GET_STAT_OVERWRITE_COUNT_AMD_TEST_HOOK              0x12f00a
#define GL_EP_TIMMO_GET_STAT_OVERWRITE_TOKEN_AMD_TEST_HOOK              0x12f00b
#define GL_EP_TIMMO_GET_STAT_OVERWRITE_BLITS_AMD_TEST_HOOK              0x12f00c
#define GL_EP_TIMMO_GET_STAT_OVERWRITE_BYTES_AMD_TEST_HOOK              0x12f00d
#define GL_EP_TIMMO_GET_STAT_OVERWRITE_GAP_AMD_TEST_HOOK                0x12f00e

#define GL_EP_DLIST_CACHE_ENABLE_AMD_TEST_HOOK                          0x12f00f
#define GL_EP_DLIST_CACHE_GET_ENABLED_AMD_TEST_HOOK                     0x12f010
#define GL_EP_DLIST_CACHE_STAT_RESET_AMD_TEST_HOOK                      0x12f011
#define GL_EP_DLIST_CACHE_GET_STAT_CACHES_AMD_TEST_HOOK                 0x12f012
#define GL_EP_DLIST_CACHE_GET_STAT_CACHED_AMD_TEST_HOOK                 0x12f013
#define GL_EP_DLIST_CACHE_GET_STAT_REJECTED_AMD_TEST_HOOK               0x12f014
#define GL_EP_DLIST_CACHE_GET_STAT_CALLED_AMD_TEST_HOOK                 0x12f015
#define GL_EP_DLIST_CACHE_GET_STAT_FOUND_IN_SEQUENCE_AMD_TEST_HOOK      0x12f016
#define GL_EP_DLIST_CACHE_GET_STAT_FOUND_OUT_OF_SEQUENCE_AMD_TEST_HOOK  0x12f017

#define GL_EP_SELECT_MODE_AMD_TEST_HOOK                                 0x12f018
#define GL_EP_SELECT_GET_MODE_AMD_TEST_HOOK                             0x12f019

/* GSL guardband extension */
#define GL_CLIP_GUARDBAND_GSL                                           0x125000

/* GSL stipple extension */
#define GL_TEXTURE_COORD_7_RASTERIZER_GEN_GSL                           0x126000

/* GSL texture unit parameter extension */
#define GL_TEXTURE_SWIZZLE_GSL                                          0x127000

/* GSL double rate clear extension */
#define GL_DOUBLE_RATE_CLEAR_GSL                                        0x128000

/* GSL Interfaces */

typedef GLvoid (APIENTRY *PFNGLRESOLVEMVPUPROC)(GLuint name1, GLuint name2);
typedef GLvoid (APIENTRY *PFNGLBINDFRAMEBUFFERGSLPROC)(GLenum target, GLuint buffer);
typedef GLvoid (APIENTRY *PFNGLCREATEFRAMEBUFFERGSLPROC)(GLuint* buffer);
typedef GLvoid (APIENTRY *PFNGLDELETEFRAMEBUFFERGSLPROC)(GLuint buffer);
typedef GLboolean (APIENTRY *PFNGLISFRAMEBUFFERGSLPROC)(GLuint buffer);
typedef GLvoid (APIENTRY *PFNGLFRAMEBUFFERPARAMETERIVGSLPROC)(GLuint fbo,GLenum pname, const GLint* params);
typedef GLvoid (APIENTRY *PFNGLGETFRAMEBUFFERPARAMETERIVGSLPROC)(GLenum pname, GLint* params);
typedef GLvoid (APIENTRY *PFNGLALLOCMEM1DGSLPROC)(GLuint format, GLuint width, GLuint np, const GLint *properties, GLuint* mem);
typedef GLvoid (APIENTRY *PFNGLALLOCMEM2DGSLPROC)(GLuint format, GLuint width, GLuint height, GLuint np, const GLint *properties, GLuint* mem, const GLuint memSrc);
typedef GLvoid (APIENTRY *PFNGLALLOCMEM3DGSLPROC)(GLuint format, GLuint width, GLuint height, GLuint depth, GLuint np, const GLint *properties, GLuint* mem);
typedef GLuint (APIENTRY *PFNGLGETMEMPROPERTYGSLPROC)(GLuint mem, GLuint pname);
typedef GLvoid (APIENTRY *PFNGLDELETEMEMOBJGSLPROC)(GLuint mem);
typedef GLuint (APIENTRY *PFNGLGETTEXTUREMEMGSLPROC)(GLuint textureObject, GLenum pname);
typedef GLuint (APIENTRY *PFNGLGETVERTEXBUFFERMEMGSLPROC)(GLuint vertexBufferObject, GLenum pname);
typedef GLuint (APIENTRY *PFNGLGETFRAMEBUFFERMEMGSLPROC)(GLuint frameBufferObject, GLenum pname);
typedef GLvoid (APIENTRY *PFNGLATTACHTEXTUREMEMGSLPROC)(GLuint textureObject, GLenum pname, GLuint memObj);
typedef GLvoid (APIENTRY *PFNGLATTACHVERTEXBUFFERMEMGSLPROC)(GLuint vertexBufferObject, GLenum pname, GLuint memObj);
typedef GLvoid (APIENTRY *PFNGLATTACHFRAMEBUFFERMEMGSLPROC)(GLuint framebufferObject, GLenum pname, GLuint memObj);
typedef GLvoid* (APIENTRY *PFNGLMAPMEMIMAGEGSLPROC)(GLuint name, GLuint access);
typedef GLvoid (APIENTRY *PFNGLUNMAPMEMIMAGEGSLPROC)(GLuint name);
typedef GLvoid (APIENTRY *PFNGLLOADMEMOBJGSLPROC)(GLuint memObject, GLuint width, GLuint height, const GLvoid* srcAddr, GLenum srcFmt, GLuint srcPitch, GLuint dstX, GLuint dstY, GLuint dstLayer, GLuint dstLvl);
typedef GLvoid (APIENTRY *PFNGLFBUFFERPASSGSLPROC)(GLuint pass);

typedef GLvoid (APIENTRY *PFNGLSETNAMESPACEGSLPROC)(GLenum eNameSpace);

typedef GLenum (APIENTRY *PFNGLGETNAMESPACEGSLPROC)(void);

typedef GLvoid (APIENTRY *PFNGLCLIPPARAMETERFVGSLPROC)(GLenum target, const GLfloat *params);
typedef GLvoid (APIENTRY *PFNGLTEXUNITPARAMETERGSLPROC)(GLenum unit, GLenum pname, const GLvoid* params);

#endif /* GL_ATI_subset_layer */


/* CAL -GL Interop extension  */
#define GL_RESOURCE_ATTACH_TEXTURE_AMD                                  0x12a000
#define GL_RESOURCE_ATTACH_FRAMEBUFFER_AMD                              0x12a001
#define GL_RESOURCE_ATTACH_RENDERBUFFER_AMD                             0x12a002
#define GL_RESOURCE_ATTACH_VERTEXBUFFER_AMD                             0x12a003

/* flags */
#define GL_INTEROP_HSA     0x1      ///< In case of HSA-OGL interop set flag to this.
#define GL_INTEROP_SVM     0x2      ///< In case of OCL-OGL interop on SVM set flag to this.
#define GL_INTEROP_HIP     0x4      ///< In case of HIP-OGL interop set flag to this.

/* miscellaneous */
#define GL_TEXTURE_ALL_LEVELS_AMD -1
#define GL_TEXTURE_ALL_LAYERS_AMD -1

#ifndef GLuintp
typedef uintptr_t GLuintp;
#endif

#ifndef GLlonglong
typedef long long GLlonglong;
#endif

typedef struct GLResourceRec {
       GLenum type;
       GLuint name;
       GLuint flags;
       GLuintp mbResHandle;
       GLuint level;        ///< Start level to attach
       GLuint numLevels;    ///< Number of levels to attach (can be set to GL_TEXTURE_ALL_LEVELS_AMD)
       GLuint layer;        ///< Start layer to attach
       GLuint numLayers;    ///< Number of layers to attach (can be set to GL_TEXTURE_ALL_LAYERS_AMD)
} GLResource;


typedef struct GLResourceDimRec {
        GLuint width;
        GLuint height;
        GLuint depth;
}GLResourceDim;

#define GLRDATA_MAX_LAYERS 8192

#define GL_RESOURCE_DATA_VERSION 7

typedef enum GLResourceDataVersionEnum
{
	GL_RESOURCE_DATA_HAS_OBJECT_ATTRIB_TYPE = 5,
	GL_RESOURCE_DATA_HAS_MIPMAP_LEVELS_INFORMATION = 6,
	GL_RESOURCE_DATA_HAS_SRD_INFORMATION = 7,
} GLResourceDataVersion;

typedef struct GLResourceDataRec {
        GLuint               size;
        GLuint               version;
        GLuint               surfaceSize;                  ///< Size of the base surface.
        GLuint               pad;
        GLuintp              offset;                       ///< Offset pointing to the sub resource's surface.
        GLuintp              mbResHandle;
        GLuint               format;
        GLuint               flags;
        GLuint               tilingMode;
        GLuint               swizzles[GLRDATA_MAX_LAYERS];
        GLResourceDim        paddedDimensions;
        GLResourceDim        rawDimensions;
        GLlonglong           cardAddr;                     ///< Address of the base surface (add offset to get the actual address of the sub surface)
        GLlonglong           p2pAddr;
        GLlonglong           mc_size;
        GLuint               cpuAccess;
        GLuintp              handle;
        GLuint               perSurfTileInfo;
        GLuint               objectAttribType;
        GLuint               sharedBufferID;
        GLuint               levels;
        GLuint               swizzlesMip[GLRDATA_MAX_LAYERS];
        GLuint               textureSRDSize;
        GLuint               samplerSRDSize;
        GLuint               textureSRD[8];
        GLuint               samplerSRD[8];
        GLboolean            isDoppDesktopTexture;
        GLboolean            isDoppPresentTexture;
        GLuint               isTilingRotated;
        GLuint               vidpnSourceId;
        GLboolean            isDisplayable;

} GLResourceData;



#ifndef GL_ATI_internal_debug
#define GL_ATI_internal_debug 1
/*
** float16_input_type
**
**  Support:
**   Rage   128   based : Unsupported
**   Radeon 7000+ based : Unsupported
**   Radeon 8500+ based : Unsupported
**   Radeon 9500+ based : Supported
*/
//#ifndef float16_input_type
//#define float16_input_type 1

#define GL_FLOAT16_ATI 0x140B

//#endif /* float16_input_type */


/* gl_MB_TestHookAMD */
#define GL_MB_EVICT_ALL_TEXTURE_OBJS_AMD_TEST_HOOK                      0x12a000
#define GL_MB_EVICT_TEXTURE_OBJ_AMD_TEST_HOOK                           0x12a001
#define GL_MB_GET_FREE_SIZE_AMD_TEST_HOOK                               0x12a002
#define GL_MB_GET_TEXTURE_OBJ_SIZE_AMD_TEST_HOOK                        0x12a003
#define GL_MB_SET_MAX_VID_MEM_ALLOCS_AMD_TEST_HOOK                      0x12a004

#define GL_APT_SET_SHADER_REPLACEMENT_AMD_TEST_HOOK                     0x12a081
#define GL_APT_GET_CATALYST_AI_SETTING_AMD_TEST_HOOK                    0x12a082
#define GL_APT_SET_CATALYST_AI_SETTINGAMD_TEST_HOOK                     0x12a083
#define GL_APT_GET_ALLOW_TEXTURE_ANALYSE_AMD_TEST_HOOK                  0x12a084
#define GL_APT_SET_ALLOW_TEXTURE_ANALYSE_AMD_TEST_HOOK                  0x12a085

#define GL_SVT_DISABLE_SW_PATH_AMD_TEST_HOOK                            0x12a800
#define GL_SVT_SET_CURRENT_PUNT_KEY_AMD_TEST_HOOK                       0x12a801
#define GL_SVT_GET_CURRENT_PUNT_KEY_AMD_TEST_HOOK                       0x12a802
#define GL_SVT_SET_PUNT_CONDITION_AMD_TEST_HOOK                         0x12a803
#define GL_SVT_GET_PUNT_CONDITION_AMD_TEST_HOOK                         0x12a804
#define GL_SVT_FORCE_SW_PATH_AMD_TEST_HOOK                              0x12a805

#define GL_SVT_PUNT_CAN_NOT_HANDLE_VS_AMD_TEST_HOOK                     0x12b000
#define GL_SVT_PUNT_CAN_NOT_HANDLE_FS_AMD_TEST_HOOK                     0x12b001
#define GL_SVT_PUNT_FS_USES_POSITION_AMD_TEST_HOOK                      0x12b002

#define GL_SVT_PUNT_NON_HW_RENDERABLE_BUFFER_AMD_TEST_HOOK              0x12b010

#define GL_SVT_PUNT_NON_RESIDENT_TEXTURE_AMD_TEST_HOOK                  0x12b020
#define GL_SVT_PUNT_TEXTURE_WITH_BORDER_AMD_TEST_HOOK                   0x12b021
#define GL_SVT_PUNT_LOD_CLAMPING_NEEDED_AMD_TEST_HOOK                   0x12b022
#define GL_SVT_PUNT_UNSUPPORTED_NPOT_TEXTURE_AMD_TEST_HOOK              0x12b023
#define GL_SVT_PUNT_WIDE_FORMAT_IN_USE_AMD_TEST_HOOK                    0x12b024

#define GL_SVT_PUNT_NON_HW_RENDER_MODE_AMD_TEST_HOOK                    0x12b030

#define GL_SVT_PUNT_FRONT_POLYGON_MODE_AMD_TEST_HOOK                    0x12b040
#define GL_SVT_PUNT_BACK_POLYGON_MODE_AMD_TEST_HOOK                     0x12b041
#define GL_SVT_PUNT_SEPARATE_STENCIL_AMD_TEST_HOOK                      0x12b042
#define GL_SVT_PUNT_UNSUPPORTED_2S_STENCIL_AMD_TEST_HOOK                0x12b043
#define GL_SVT_PUNT_WIDE_SMOOTH_POINTS_AMD_TEST_HOOK                    0x12b044
#define GL_SVT_PUNT_WIDE_SMOOTH_LINES_AMD_TEST_HOOK                     0x12b045
#define GL_SVT_PUNT_POLYGON_OFFSET_FILL_AMD_TEST_HOOK                   0x12b046
#define GL_SVT_PUNT_POLYGON_OFFSET_LINE_AMD_TEST_HOOK                   0x12b047
#define GL_SVT_PUNT_POLYGON_OFFSET_POINT_AMD_TEST_HOOK                  0x12b048
#define GL_SVT_PUNT_AA_STIPPLE_TEX_UNIT_AMD_TEST_HOOK                   0x12b049
#define GL_SVT_PUNT_CRIPPLED_STIPPLING_AMD_TEST_HOOK                    0x12b04A
#define GL_SVT_PUNT_CRIPPLED_WIDE_FORMATS_AMD_TEST_HOOK                 0x12b04B

#define GL_CATALYST_AI_DISABLED_AMD_TEST_HOOK                           0
#define GL_CATALYST_AI_STANDARD_AMD_TEST_HOOK                           1
#define GL_CATALYST_AI_ADVANCED_AMD_TEST_HOOK                           2

typedef GLint (APIENTRY *PFNGL__GLLTESTBACKDOORATIPROC)(GLint op, GLint iParamCount, GLint* pParams);

typedef GLvoid* (APIENTRY *PFNGLREGISTERSHADERSTRINGGLLPROC)(GLenum target, GLvoid* string, GLvoid* replaceString);
typedef GLvoid  (APIENTRY *PFNGLUNREGISTERSHADERSTRINGGLLPROC)(GLvoid* shader);

#endif // GL_ATI_internal_debug

#ifdef __cplusplus
}
#endif


#endif /* __gl_ATI_Internal_h_ */
