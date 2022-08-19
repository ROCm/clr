#ifndef __wgl_ATI_h_
#define __wgl_ATI_h_
//
// Copyright (c) 1998 Advanced Micro Devices, Inc.  All rights reserved.
//

#ifdef __cplusplus
extern "C" {
#endif

#ifndef APIENTRY
#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>
#endif

/*
** Notes:
**
**  Listed support is for current drivers and should really only be used
**  as a guideline.  ISV should still use glGetString() and
**  wglGetExtensionsString() to determine the exact set of supported
**  GL and WGL extensions.
**
*/

/*
** WGL_ARB_extensions_string
**
**  Support:
**   Rage 128 * based : Supported
**   Radeon   * based : Supported
*/
#ifndef WGL_ARB_extensions_string
#define WGL_ARB_extensions_string 1

typedef const char * (WINAPI * PFNWGLGETEXTENSIONSSTRINGARBPROC) (HDC hDC);

#endif /* WGL_ARB_extensions_string */

/*
** WGL_ARB_pixel_format
**
**  Support:
**   Rage 128 * based : Supported
**   Radeon   * based : Supported
*/
#ifndef WGL_ARB_pixel_format
#define WGL_ARB_pixel_format 1

#define WGL_NUMBER_PIXEL_FORMATS_ARB        0x2000
#define WGL_DRAW_TO_WINDOW_ARB              0x2001
#define WGL_DRAW_TO_BITMAP_ARB              0x2002
#define WGL_ACCELERATION_ARB                0x2003
#define WGL_NEED_PALETTE_ARB                0x2004
#define WGL_NEED_SYSTEM_PALETTE_ARB         0x2005
#define WGL_SWAP_LAYER_BUFFERS_ARB          0x2006
#define WGL_SWAP_METHOD_ARB                 0x2007
#define WGL_NUMBER_OVERLAYS_ARB             0x2008
#define WGL_NUMBER_UNDERLAYS_ARB            0x2009
#define WGL_TRANSPARENT_ARB                 0x200A
#define WGL_TRANSPARENT_RED_VALUE_ARB       0x2037
#define WGL_TRANSPARENT_GREEN_VALUE_ARB     0x2038
#define WGL_TRANSPARENT_BLUE_VALUE_ARB      0x2039
#define WGL_TRANSPARENT_ALPHA_VALUE_ARB     0x203A
#define WGL_TRANSPARENT_INDEX_VALUE_ARB     0x203B
#define WGL_SHARE_DEPTH_ARB                 0x200C
#define WGL_SHARE_STENCIL_ARB               0x200D
#define WGL_SHARE_ACCUM_ARB                 0x200E
#define WGL_SUPPORT_GDI_ARB                 0x200F
#define WGL_SUPPORT_OPENGL_ARB              0x2010
#define WGL_DOUBLE_BUFFER_ARB               0x2011
#define WGL_STEREO_ARB                      0x2012
#define WGL_PIXEL_TYPE_ARB                  0x2013
#define WGL_COLOR_BITS_ARB                  0x2014
#define WGL_RED_BITS_ARB                    0x2015
#define WGL_RED_SHIFT_ARB                   0x2016
#define WGL_GREEN_BITS_ARB                  0x2017
#define WGL_GREEN_SHIFT_ARB                 0x2018
#define WGL_BLUE_BITS_ARB                   0x2019
#define WGL_BLUE_SHIFT_ARB                  0x201A
#define WGL_ALPHA_BITS_ARB                  0x201B
#define WGL_ALPHA_SHIFT_ARB                 0x201C
#define WGL_ACCUM_BITS_ARB                  0x201D
#define WGL_ACCUM_RED_BITS_ARB              0x201E
#define WGL_ACCUM_GREEN_BITS_ARB            0x201F
#define WGL_ACCUM_BLUE_BITS_ARB             0x2020
#define WGL_ACCUM_ALPHA_BITS_ARB            0x2021
#define WGL_DEPTH_BITS_ARB                  0x2022
#define WGL_STENCIL_BITS_ARB                0x2023
#define WGL_AUX_BUFFERS_ARB                 0x2024
#define WGL_NO_ACCELERATION_ARB             0x2025
#define WGL_GENERIC_ACCELERATION_ARB        0x2026
#define WGL_FULL_ACCELERATION_ARB           0x2027
#define WGL_SWAP_EXCHANGE_ARB               0x2028
#define WGL_SWAP_COPY_ARB                   0x2029
#define WGL_SWAP_UNDEFINED_ARB              0x202A
#define WGL_TYPE_RGBA_ARB                   0x202B
#define WGL_TYPE_COLORINDEX_ARB             0x202C
#define WGL_TYPE_RGBA_FLOAT_ARB             0x21A0

#define WGL_FLOAT_COMPONENTS_NV                         0x20B0
#define WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_R_NV        0x20B1
#define WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RG_NV       0x20B2
#define WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGB_NV      0x20B3
#define WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGBA_NV     0x20B4
#define WGL_TEXTURE_FLOAT_R_NV                          0x20B5
#define WGL_TEXTURE_FLOAT_RG_NV                         0x20B6
#define WGL_TEXTURE_FLOAT_RGB_NV                        0x20B7
#define WGL_TEXTURE_FLOAT_RGBA_NV                       0x20B8
#define WGL_TEXTURE_RECTANGLE_NV                        0x20A2



typedef BOOL (WINAPI * PFNWGLGETPIXELFORMATATTRIBIVARBPROC) (
                                        HDC hDC,
                                        int iPixelFormat,
                                        int iLayerPlane,
                                        UINT nAttributes,
                                        const int *piAttributes,
                                        int *piValues);
typedef BOOL (WINAPI * PFNWGLGETPIXELFORMATATTRIBFVARBPROC) (
                                        HDC hDC,
                                        int iPixelFormat,
                                        int iLayerPlane,
                                        UINT nAttributes,
                                        const int *piAttributes,
                                        FLOAT *pfValues);
typedef BOOL (WINAPI * PFNWGLCHOOSEPIXELFORMATARBPROC) (
                                        HDC hDC,
                                        const int *piAttribIList,
                                        const FLOAT *pfAttribFList,
                                        UINT nMaxFormats,
                                        int *piFormats,
                                        UINT *nNumFormats);

#endif /* WGL_ARB_pixel_format */

/*
** WGL_ARB_make_current_read
**
**  Support:
**   Rage 128 * based : Supported
**   Radeon   * based : Supported
*/
#ifndef WGL_ARB_make_current_read
#define WGL_ARB_make_current_read 1

typedef BOOL (WINAPI * PFNWGLMAKECONTEXTCURRENTARBPROC) (HDC hDrawDC,
                                                         HDC hReadDC,
                                                         HGLRC hGLRC);
typedef HDC (WINAPI * PFNWGLGETCURRENTREADDCARBPROC) (VOID);

#endif /* WGL_ARB_make_current_read */

/*
** WGL_ARB_multisample
**
**  Support:
**   Rage 128 * based : Not Supported
**   Radeon   * based : Not Supported
*/
#ifndef WGL_ARB_multisample
#define WGL_ARB_multisample 1

#define WGL_SAMPLE_BUFFERS_ARB              0x2041
#define WGL_SAMPLES_ARB                     0x2042

#endif /* WGL_ARB_multisample */

/*
** WGL_ARB_pbuffer
**
**  Support:
**   Rage 128 * based : Supported
**   Radeon   * based : Supported
*/
#ifndef WGL_ARB_pbuffer
#define WGL_ARB_pbuffer 1

#define WGL_DRAW_TO_PBUFFER_ARB              0x202D
#define WGL_MAX_PBUFFER_PIXELS_ARB           0x202E
#define WGL_MAX_PBUFFER_WIDTH_ARB            0x202F
#define WGL_MAX_PBUFFER_HEIGHT_ARB           0x2030
#define WGL_PBUFFER_LARGEST_ARB              0x2033
#define WGL_PBUFFER_WIDTH_ARB                0x2034
#define WGL_PBUFFER_HEIGHT_ARB               0x2035
#define WGL_PBUFFER_LOST_ARB                 0x2036

DECLARE_HANDLE(HPBUFFERARB);

typedef HPBUFFERARB (WINAPI * PFNWGLCREATEPBUFFERARBPROC) (
                                        HDC hDC,
                                        int iPixelFormat,
                                        int iWidth,
                                        int iHeight,
                                        const int *piAttribList);
typedef HDC (WINAPI * PFNWGLGETPBUFFERDCARBPROC) (HPBUFFERARB hPbuffer);
typedef int (WINAPI * PFNWGLRELEASEPBUFFERDCARBPROC) (
                                        HPBUFFERARB hPbuffer,
                                        HDC hDC);
typedef BOOL (WINAPI * PFNWGLDESTROYPBUFFERARBPROC) (HPBUFFERARB hPbuffer);
typedef BOOL (WINAPI * PFNWGLQUERYPBUFFERARBPROC) (
                                        HPBUFFERARB hPbuffer,
                                        int iAttribute,
                                        int *piValue);

#endif /* WGL_ARB_pbuffer */


/*
** WGL_ARB_render_texture
**
**  Support:
**   Rage 128 * based : Supported
**   Radeon   * based : Supported
*/
#ifndef WGL_ARB_render_texture
#define WGL_ARB_render_texture 1

#define WGL_BIND_TO_TEXTURE_RGB_ARB         0x2070
#define WGL_BIND_TO_TEXTURE_RGBA_ARB        0x2071
#define WGL_TEXTURE_FORMAT_ARB              0x2072
#define WGL_TEXTURE_TARGET_ARB              0x2073
#define WGL_MIPMAP_TEXTURE_ARB              0x2074
#define WGL_TEXTURE_RGB_ARB                 0x2075
#define WGL_TEXTURE_RGBA_ARB                0x2076
#define WGL_NO_TEXTURE_ARB                  0x2077
#define WGL_TEXTURE_CUBE_MAP_ARB            0x2078
#define WGL_TEXTURE_1D_ARB                  0x2079
#define WGL_TEXTURE_2D_ARB                  0x207A
#define WGL_MIPMAP_LEVEL_ARB                0x207B
#define WGL_CUBE_MAP_FACE_ARB               0x207C
#define WGL_TEXTURE_CUBE_MAP_POSITIVE_X_ARB 0x207D
#define WGL_TEXTURE_CUBE_MAP_NEGATIVE_X_ARB 0x207E
#define WGL_TEXTURE_CUBE_MAP_POSITIVE_Y_ARB 0x207F
#define WGL_TEXTURE_CUBE_MAP_NEGATIVE_Y_ARB 0x2080
#define WGL_TEXTURE_CUBE_MAP_POSITIVE_Z_ARB 0x2081
#define WGL_TEXTURE_CUBE_MAP_NEGATIVE_Z_ARB 0x2082
#define WGL_FRONT_LEFT_ARB                  0x2083
#define WGL_FRONT_RIGHT_ARB                 0x2084
#define WGL_BACK_LEFT_ARB                   0x2085
#define WGL_BACK_RIGHT_ARB                  0x2086
#define WGL_AUX0_ARB                        0x2087
#define WGL_AUX1_ARB                        0x2088
#define WGL_AUX2_ARB                        0x2089
#define WGL_AUX3_ARB                        0x208A
#define WGL_AUX4_ARB                        0x208B
#define WGL_AUX5_ARB                        0x208C
#define WGL_AUX6_ARB                        0x208D
#define WGL_AUX7_ARB                        0x208E
#define WGL_AUX8_ARB                        0x208F
#define WGL_AUX9_ARB                        0x2090

typedef BOOL (WINAPI * PFNWGLBINDTEXIMAGEARBPROC)(HPBUFFERARB hPbuffer, int iBuffer);
typedef BOOL (WINAPI * PFNWGLRELEASETEXIMAGEARBPROC)(HPBUFFERARB hPbuffer, int iBuffer);
typedef BOOL (WINAPI * PFNWGLSETPBUFFERATTRIBARBPROC)(HPBUFFERARB hPbuffer,
                                                       const int *piAttribList);
#endif


/*
** WGL_EXT_extensions_string
**
**  Support:
**   Rage 128 * based : Supported
**   Radeon   * based : Supported
*/
#ifndef WGL_EXT_extensions_string
#define WGL_EXT_extensions_string 1

typedef const char * (WINAPI * PFNWGLGETEXTENSIONSSTRINGEXTPROC) (VOID);

#endif /* WGL_EXT_extensions_string */

/*
** WGL_EXT_swap_control
**
**  Support:
**   Rage 128 * based : Supported
**   Radeon   * based : Supported
*/
#ifndef WGL_EXT_swap_control
#define WGL_EXT_swap_control 1

typedef BOOL (WINAPI * PFNWGLSWAPINTERVALEXTPROC) (int interval);
typedef int (WINAPI * PFNWGLGETSWAPINTERVALEXTPROC) (VOID);

#endif /* WGL_EXT_swap_control */

#ifndef WGL_ATI_pixel_format_float
#define WGL_ATI_pixel_format_float  1

#define WGL_TYPE_RGBA_FLOAT_ATI             0x21A0
#define GL_TYPE_RGBA_FLOAT_ATI              0x8820
#define GL_COLOR_CLEAR_UNCLAMPED_VALUE_ATI  0x8835
#endif

/* EXT_packed_float*/
#ifndef WGL_EXT_pixel_format_unsigned_float
#define WGL_EXT_pixel_format_unsigned_float 1
#define WGL_TYPE_RGBA_UNSIGNED_FLOAT_EXT    0x20A8
#endif
/*
** HPV_transmit_buffer
**
**  Support:
**   under development, currently r300 only
**   disabled unless define HPV_transmit_buffer in build.
**
*/
#ifdef HPV_transmit_buffer

#define WGL_TRANSMITTABLE_HPV               0x70000001
#define WGL_TRANSMIT_FORMAT_HPV             0x70000002

#define GLX_TRANSMITTABLE_HPV               0x70000001
#define GLX_TRANSMIT_FORMAT_HPV             0x70000002

#define WGL_RGB_HPV                         0x70000003
#define WGL_RGBA_HPV                        0x70000004
#define WGL_DEPTH_HPV                       0x70000005
#define WGL_DEPTH_STENCIL_HPV               0x70000006
#define WGL_RGB_DEPTH_HPV                   0x70000007
#define WGL_RGBA_DEPTH_STENCIL_HPV          0x70000008

#define GLX_RGB_HPV                         0x70000003
#define GLX_RGBA_HPV                        0x70000004
#define GLX_DEPTH_HPV                       0x70000005
#define GLX_DEPTH_STENCIL_HPV               0x70000006
#define GLX_RGB_DEPTH_HPV                   0x70000007
#define GLX_RGBA_DEPTH_STENCIL_HPV          0x70000008


#define WGL_PBUFFER_PITCH_HPV               0x70000009

#define GLX_PBUFFER_PITCH_HPV    	        0x70000009

#define WGL_TRANSMIT_HOR_ADDR_TIME_HPV      0x7000000A
#define WGL_TRANSMIT_HOR_FP_TIME_HPV        0x7000000B
#define WGL_TRANSMIT_HOR_SYNC_TIME_HPV      0x7000000C
#define WGL_TRANSMIT_HOR_BP_TIME_HPV        0x7000000D
#define WGL_TRANSMIT_HOR_POLARITY_HPV		0x7000000E
#define WGL_TRANSMIT_VER_ADDR_TIME_HPV      0x7000000F
#define WGL_TRANSMIT_VER_FP_TIME_HPV        0x70000010
#define WGL_TRANSMIT_VER_SYNC_TIME_HPV      0x70000011
#define WGL_TRANSMIT_VER_BP_TIME_HPV        0x70000012
#define WGL_TRANSMIT_VER_POLARITY_HPV		0x70000013
#define WGL_TRANSMIT_PIXELCLOCK_HPV         0x70000014

#define GLX_TRANSMIT_HOR_ADDR_TIME_HPV      0x7000000A
#define GLX_TRANSMIT_HOR_FP_TIME_HPV        0x7000000B
#define GLX_TRANSMIT_HOR_SYNC_TIME_HPV      0x7000000C
#define GLX_TRANSMIT_HOR_BP_TIME_HPV        0x7000000D
#define GLX_TRANSMIT_HOR_POLARITY_HPV		0x7000000E
#define GLX_TRANSMIT_VER_ADDR_TIME_HPV      0x7000000F
#define GLX_TRANSMIT_VER_FP_TIME_HPV        0x70000010
#define GLX_TRANSMIT_VER_SYNC_TIME_HPV      0x70000011
#define GLX_TRANSMIT_VER_BP_TIME_HPV        0x70000012
#define GLX_TRANSMIT_VER_POLARITY_HPV		0x70000013
#define GLX_TRANSMIT_PIXELCLOCK_HPV         0x70000014


typedef void (WINAPI * PFNWGLTRANSMITPBUFFERHPVPROC) (
										HPBUFFERARB hPbuffer,
										unsigned int id);
typedef void (WINAPI * PFNWGLTRANSMITSTOPHPVPROC) (VOID);

#endif /* HPV_transmit_buffer */


/*
**  WGL_EXT_framebuffer_sRGB
**
**  Support:
**   Rage 128 * based : Not Supported
**   Radeon   * based : Not Supported
*/
#ifndef WGL_EXT_framebuffer_sRGB
#define WGL_EXT_framebuffer_sRGB 1

#define WGL_FRAMEBUFFER_SRGB_CAPABLE_EXT       0x20A9

#endif /* EXT_framebuffer_sRGB */

/*
** WGL_ATI_render_texture_rectangle
**
**  Support:
**   Rage 128 * based : Not Supported
**   Radeon   * based : Supported
*/
#ifndef WGL_ATI_render_texture_rectangle
#define WGL_ATI_render_texture_rectangle 1

#define WGL_TEXTURE_RECTANGLE_ATI           0x21A5

#endif /* WGL_ATI_render_texture_rectangle */

/*
** WGL_ARB_buffer_region
**
**  Support:
**   Rage 128 * based : Not Supported
**   Radeon   * based : Supported
*/
#ifndef WGL_ARB_buffer_region
#define WGL_ARB_buffer_region 1

typedef HANDLE (WINAPI* PFNWGLCREATEBUFFERREGIONARB)(HDC hDC, int iLayerPlane, UINT uType);
typedef VOID (WINAPI* PFNWGLDELETEBUFFERREGIONARB)(HANDLE hRegion);
typedef BOOL (WINAPI* PFNWGLSAVEBUFFERREGIONARB)(HANDLE hRegion, int x, int y, int width, int height);
typedef BOOL (WINAPI* PFNWGLRESTOREBUFFERREGIONARB)(HANDLE hRegion, int x, int y, int width, int height, int xSrc, int ySrc);

#define WGL_FRONT_COLOR_BUFFER_BIT_ARB  0x00000001
#define WGL_BACK_COLOR_BUFFER_BIT_ARB   0x00000002
#define WGL_DEPTH_BUFFER_BIT_ARB        0x00000004
#define WGL_STENCIL_BUFFER_BIT_ARB      0x00000008

#endif /* WGL_ARB_buffer_region */


#ifndef WGL_I3D_genlock
#define WGL_I3D_genlock 1

#define WGL_GENLOCK_SOURCE_MULTIVIEW_I3D        0x2044
#define WGL_GENLOCK_SOURCE_EXTERNAL_SYNC_I3D    0x2045
#define WGL_GENLOCK_SOURCE_EXTERNAL_FIELD_I3D   0x2046
#define WGL_GENLOCK_SOURCE_EXTERNAL_TTL_I3D     0x2047
#define WGL_GENLOCK_SOURCE_DIGITAL_SYNC_I3D     0x2048
#define WGL_GENLOCK_SOURCE_DIGITAL_FIELD_I3D    0x2049
#define WGL_GENLOCK_SOURCE_EDGE_FALLING_I3D     0x204A
#define WGL_GENLOCK_SOURCE_EDGE_RISING_I3D      0x204B
#define WGL_GENLOCK_SOURCE_EDGE_BOTH_I3D        0x204C

typedef BOOL (WINAPI* PFNWGLENABLEGENLOCKI3DPROC)(HDC hDC);
typedef BOOL (WINAPI* PFNWGLDISABLEGENLOCKI3DPROC)(HDC hDC);
typedef BOOL (WINAPI* PFNWGLISENABLEDGENLOCKI3DPROC)(HDC hDC, BOOL *pFlag);
typedef BOOL (WINAPI* PFNWGLGENLOCKSOURCEI3DPROC)(HDC hDC, UINT uSource);
typedef BOOL (WINAPI* PFNWGLGETGENLOCKSOURCEI3DPROC)(HDC hDC, UINT *uSource);
typedef BOOL (WINAPI* PFNWGLGENLOCKSOURCEEDGEI3DPROC)(HDC hDC, UINT uEdge);
typedef BOOL (WINAPI* PFNWGLGETGENLOCKSOURCEEDGEI3DPROC)(HDC hDC, UINT *uEdge);
typedef BOOL (WINAPI* PFNWGLGENLOCKSAMPLERATEI3DPROC)(HDC hDC, UINT uRate);
typedef BOOL (WINAPI* PFNWGLGETGENLOCKSAMPLERATEI3DPROC)(HDC hDC, UINT *uRate);
typedef BOOL (WINAPI* PFNWGLGENLOCKSOURCEDELAYI3DPROC)(HDC hDC, UINT uDelay);
typedef BOOL (WINAPI* PFNWGLGETGENLOCKSOURCEDELAYI3DPROC)(HDC hDC, UINT *uDelay);
typedef BOOL (WINAPI* PFNWGLQUERYGENLOCKMAXSOURCEDELAYI3DPROC)(HDC hDC, UINT *uMaxLineDelay, UINT *uMaxPixelDelay);

#endif /* WGL_I3D_genlock */


#ifndef WGL_NV_swap_group
#define WGL_NV_swap_group 1

typedef BOOL (WINAPI* PFNWGLJOINSWAPGROUPNVPROC)(HDC hDC, GLuint group);
typedef BOOL (WINAPI* PFNWGLBINDSWAPBARRIERNVPROC)(GLuint group, GLuint barrier);
typedef BOOL (WINAPI* PFNWGLQUERYSWAPGROUPNVPROC)(HDC hDC, GLuint *group, GLuint *barrier);
typedef BOOL (WINAPI* PFNWGLQUERYMAXSWAPGROUPSNVPROC)(HDC hDC, GLuint *maxGroups, GLuint *maxBarriers);
typedef BOOL (WINAPI* PFNWGLQUERYFRAMECOUNTNVPROC)(HDC hDC, GLuint *count);
typedef BOOL (WINAPI* PFNWGLRESETFRAMECOUNTNVPROC)(HDC hDC);

#endif /* WGL_NV_swap_group */

#ifndef WGL_ARB_create_context
#define WGL_ARB_create_context 1

#define WGL_CONTEXT_MAJOR_VERSION_ARB             0x2091
#define WGL_CONTEXT_MINOR_VERSION_ARB             0x2092
#define WGL_CONTEXT_LAYER_PLANE_ARB               0x2093
#define WGL_CONTEXT_FLAGS_ARB                     0x2094
#define WGL_CONTEXT_DEBUG_BIT_ARB                 0x0001
#define WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB    0x0002
#define WGL_CONTEXT_ROBUST_ACCESS_BIT_ARB         0x0004
#define WGL_CONTEXT_NO_ERROR_BIT                  0x0008
#define WGL_CONTEXT_PROFILE_MASK_ARB              0x9126
#define WGL_CONTEXT_OPENGL_NO_ERROR_ARB           0x31B3

#define WGL_CONTEXT_ASIC_ID_AMD                   0x91c4 //chip id
#define WGL_CONTEXT_ASIC_FAMILY_AMD               0x91c5 //asic family
#define WGL_CONTEXT_ASIC_REV_AMD                  0x91c6 //asic revision
#define WGL_CONTEXT_DEVICE_MEMORY_SIZE_AMD        0x91c7 //asic device memory

#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB          0x00000001
#define WGL_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB 0x00000002

#define ERROR_INCOMPATIBLE_DEVICE_CONTEXTS_ARB    0x2054

#define ERROR_INVALID_VERSION_ARB                 0x2095
#define ERROR_INVALID_PROFILE_ARB                 0x2096
#define ERROR_INVALID_ATTRIBS_ARB                 0xBAAD // This enum doesn't exist in any WGL spec

typedef HGLRC (WINAPI* PFNWGLCREATECONTEXTATTRIBSARB)(HDC hDC, HGLRC hShareContext, const int *attribList);
#endif

#ifndef WGL_AMD_gpu_association
#define WGL_AMD_gpu_association 1

#define WGL_GPU_VENDOR                 0x1F00
#define WGL_GPU_RENDERER_STRING        0x1F01
#define WGL_GPU_OPENGL_VERSION_STRING  0x1F02
#define WGL_GPU_FASTEST_TARGET_GPUS    0x21A2
#define WGL_GPU_RAM                    0x21A3
#define WGL_GPU_CLOCK                  0x21A4
#define WGL_GPU_NUM_PIPES              0x21A5
#define WGL_GPU_NUM_SIMD               0x21A6
#define WGL_GPU_NUM_RB                 0x21A7
#define WGL_GPU_NUM_SPI                0x21A8

#define WGL_CONTEXT_DUMMY_DEVICE_AMD   ~0UL //used to define the dummy context

typedef UINT  (WINAPI *PFNWGLGETGPUIDSAMDXPROC) (UINT maxcount, UINT *ids);
typedef HGLRC (WINAPI *PFNWGLCREATEASSOCIATEDCONTEXTAMDXPORC) (UINT ids);
typedef BOOL  (WINAPI *PFNWGLMAKEASSOCIATEDCONTEXTCURRENTAMDXPORC) (HGLRC hglrc);
typedef BOOL  (WINAPI *PFNWGLDELETEASSOCIATEDCONTEXTAMDXPORC) (HGLRC hglrc);
typedef HGLRC (WINAPI *PFNWGLGETCURRENTASSOCIATEDCONTEXTAMDXPORC) (void);
typedef UINT  (WINAPI *PFNWGLGETCONTEXTGPUIDAMDXPORC) (HGLRC hglrc);
typedef INT   (WINAPI *PFNWGLGETGPUINFOAMDXPROC) (UINT id, GLenum infoType, GLenum dataType, UINT size, void *data);
typedef VOID  (WINAPI *PFNWGLBLITCONTEXTFRAMEBUFFERAMDXPORC) (HGLRC hglrc,
                                                              int srcX0, int srcY0, int srcX1, int srcY1,
                                                              int dstX0, int dstY0, int dstX1, int dstY1,
                                                              GLbitfield mask, GLenum filter);
#endif /* WGL_AMD_gpu_association */

/*
 *  WGL_ARB_context_flush_control
 */
#ifndef WGL_ARB_context_flush_control
#define WGL_ARB_context_flush_control 1

#define WGL_CONTEXT_RELEASE_BEHAVIOR_ARB        0x2097
#define WGL_CONTEXT_RELEASE_BEHAVIOR_NONE_ARB   0x0000
#define WGL_CONTEXT_RELEASE_BEHAVIOR_FLUSH_ARB  0x2098

#endif /* WGL_ARB_context_flush_control */


#ifndef WGL_EXT_colorspace
#define WGL_EXT_colorspace 1

#define WGL_COLORSPACE_EXT                0x309D
#define WGL_COLORSPACE_SRGB_EXT           0x3089
#define WGL_COLORSPACE_LINEAR_EXT         0x308A

#endif /* WGL_EXT_colorspace */


#ifdef __cplusplus
}
#endif

#endif /* __wgl_ATI_h_ */
