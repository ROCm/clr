#ifndef __glx_ATI_h_
#define __glx_ATI_h_
//
// Copyright (c) 2002 Advanced Micro Devices, Inc.  All rights reserved.
//

#ifdef __cplusplus
extern "C" {
#endif

/*
** Notes:
**
**  Listed support is for current drivers and should really only be used
**  as a guideline.  ISV should still use glGetString() and
**  glXGetClientString() to determine the exact set of supported
**  GL and GLX extensions.
**
*/

#ifndef GLX_ATI_pixel_format_float
#define GLX_ATI_pixel_format_float  1

#define GLX_RGBA_FLOAT_ATI_BIT				0x00000100

#endif // GLX_ATI_pixel_format_float

#ifndef GLX_RGBA_FLOAT_BIT
#define GLX_RGBA_FLOAT_BIT                  0x00000004
#endif //GLX_RGBA_FLOAT_BIT

#ifndef GLX_RGBA_UNSIGNED_FLOAT_BIT_EXT
#define GLX_RGBA_UNSIGNED_FLOAT_BIT_EXT     0x00000008
#endif //GLX_RGBA_UNSIGNED_FLOAT_BIT_EXT

#ifndef GLX_ATI_render_texture
#define GLX_ATI_render_texture  1

#define GLX_BIND_TO_TEXTURE_RGB_ATI         0x9800 // need real tokens here
#define GLX_BIND_TO_TEXTURE_RGBA_ATI        0x9801
#define GLX_TEXTURE_FORMAT_ATI              0x9802
#define GLX_TEXTURE_TARGET_ATI              0x9803
#define GLX_MIPMAP_TEXTURE_ATI              0x9804
#define GLX_TEXTURE_RGB_ATI                 0x9805
#define GLX_TEXTURE_RGBA_ATI                0x9806
#define GLX_NO_TEXTURE_ATI                  0x9807
#define GLX_TEXTURE_CUBE_MAP_ATI            0x9808
#define GLX_TEXTURE_1D_ATI                  0x9809
#define GLX_TEXTURE_2D_ATI                  0x980A
#define GLX_MIPMAP_LEVEL_ATI                0x980B
#define GLX_CUBE_MAP_FACE_ATI               0x980C
#define GLX_TEXTURE_CUBE_MAP_POSITIVE_X_ATI 0x980D
#define GLX_TEXTURE_CUBE_MAP_NEGATIVE_X_ATI 0x980E
#define GLX_TEXTURE_CUBE_MAP_POSITIVE_Y_ATI 0x980F
#define GLX_TEXTURE_CUBE_MAP_NEGATIVE_Y_ATI 0x9810
#define GLX_TEXTURE_CUBE_MAP_POSITIVE_Z_ATI 0x9811
#define GLX_TEXTURE_CUBE_MAP_NEGATIVE_Z_ATI 0x9812
#define GLX_FRONT_LEFT_ATI                  0x9813
#define GLX_FRONT_RIGHT_ATI                 0x9814
#define GLX_BACK_LEFT_ATI                   0x9815
#define GLX_BACK_RIGHT_ATI                  0x9816
#define GLX_AUX0_ATI                        0x9817
#define GLX_AUX1_ATI                        0x9818
#define GLX_AUX2_ATI                        0x9819
#define GLX_AUX3_ATI                        0x981A
#define GLX_AUX4_ATI                        0x981B
#define GLX_AUX5_ATI                        0x981C
#define GLX_AUX6_ATI                        0x981D
#define GLX_AUX7_ATI                        0x981E
#define GLX_AUX8_ATI                        0x981F
#define GLX_AUX9_ATI                        0x9820
#define GLX_BIND_TO_TEXTURE_LUMINANCE_ATI   0x9821
#define GLX_BIND_TO_TEXTURE_INTENSITY_ATI   0x9822

typedef void (* PFNGLXBINDTEXIMAGEATIPROC)(Display *dpy, GLXPbuffer pbuf, int buffer);
typedef void (* PFNGLXRELEASETEXIMAGEATIPROC)(Display *dpy, GLXPbuffer pbuf, int buffer);
typedef void (* PFNGLXDRAWABLEATTRIBATIPROC)(Display *dpy, GLXDrawable draw, const int *attrib_list);

#endif // GLX_ATI_render_texture

#ifndef GLX_ARB_multisample
#define GLX_ARB_multisample                 1

#define GLX_SAMPLE_BUFFERS_ARB              100000
#define GLX_SAMPLES_ARB                     100001

// put GL interface here for convenience
#ifndef GL_ARB_multisample
#define GL_ARB_multisample                  1
#define GL_MULTISAMPLE_ARB                  0x809D
#define GL_SAMPLE_ALPHA_TO_COVERAGE_ARB     0x809E
#define GL_SAMPLE_ALPHA_TO_ONE_ARB          0x809F
#define GL_SAMPLE_COVERAGE_ARB              0x80A0
#define GL_SAMPLE_BUFFERS_ARB               0x80A8
#define GL_SAMPLES_ARB                      0x80A9
#define GL_SAMPLE_COVERAGE_VALUE_ARB        0x80AA
#define GL_SAMPLE_COVERAGE_INVERT_ARB       0x80AB
#define GL_MULTISAMPLE_BIT_ARB              0x20000000

typedef GLvoid (APIENTRY * PFNGLSAMPLECOVERAGEARBPROC)(GLclampf value, GLboolean invert);
#endif /* GL_ARB_multisample */

#endif // GLX_ARB_multisample


#ifndef HPV_transmit_buffer
//#define HPV_transmit_buffer 1
#endif

#ifdef HPV_transmit_buffer

#define GLX_TRANSMITTABLE_HPV               0x70000001
#define GLX_TRANSMIT_FORMAT_HPV             0x70000002

#define GLX_RGB_HPV                         0x70000003
#define GLX_RGBA_HPV                        0x70000004
#define GLX_DEPTH_HPV                       0x70000005
#define GLX_DEPTH_STENCIL_HPV               0x70000006
#define GLX_RGB_DEPTH_HPV                   0x70000007
#define GLX_RGBA_DEPTH_STENCIL_HPV          0x70000008


#define GLX_PBUFFER_PITCH_HPV    	        0x70000009

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

#define GLX_TRANSMIT_STEREO_ID_NONE         0x70000020
#define GLX_TRANSMIT_STEREO_ID_1            0x70000021
#define GLX_TRANSMIT_STEREO_ID_2            0x70000022
#define GLX_TRANSMIT_STEREO_ID_3            0x70000023

typedef BOOL (APIENTRY * PFNGLXTRANSMITPBUFFERHPVPROC) (
										Display *dpy,
										GLXPbuffer pBuffer,
										unsigned int id,
										unsigned int idStereo,
										BOOL wait);
typedef BOOL (APIENTRY * PFNGLXTRANSMITSTOPHPVPROC) (Display *dpy);

#endif /* HPV_transmit_buffer */


#ifndef GLX_NV_swap_group
#define GLX_NV_swap_group  1
#ifdef GLX_GLXEXT_PROTOTYPES
extern Bool glXJoinSwapGroupNV(Display *dpy, GLXDrawable drawable, GLuint group);
extern Bool glXQuerySwapGroupNV(Display *dpy, GLXDrawable drawable, GLuint *group, GLuint *barrier);
extern Bool glXBindSwapBarrierNV(Display *dpy, GLuint group, GLuint barrier);
extern Bool glXQueryMaxSwapGroupsNV(Display *dpy, int screen, GLuint *maxGroups, GLuint *maxBarriers);
extern Bool glXQueryFrameCountNV(Display *dpy, int screen, GLuint *count);
extern Bool glXResetFrameCountNV(Display *dpy, int screen);
#endif /* GLX_GLXEXT_PROTOTYPES */
typedef Bool ( * PFNGLXJOINSWAPGROUPNVPROC) (Display *dpy, GLXDrawable drawable, GLuint group);
typedef Bool ( * PFNGLXQUERYSWAPGROUPNVPROC) (Display *dpy, GLXDrawable drawable, GLuint *group, GLuint *barrier);
typedef Bool ( * PFNGLXBINDSWAPBARRIERNVPROC) (Display *dpy, GLuint group, GLuint barrier);
typedef Bool ( * PFNGLXQUERYMAXSWAPGROUPSNVPROC) (Display *dpy, int screen, GLuint *maxGroups, GLuint *maxBarriers);
typedef Bool ( * PFNGLXQUERYFRAMECOUNTNVPROC)  (Display *dpy, int screen, GLuint *count);
typedef Bool ( * PFNGLXRESETFRAMECOUNTNVPROC) (Display *dpy, int screen);
#endif /* GLX_NV_swap_group */

/*
 * GLX_ARB_create_context
 */
#ifndef GLX_ARB_create_context
#define GLX_ARB_create_context 1

#define GLX_CONTEXT_MAJOR_VERSION_ARB             0x2091
#define GLX_CONTEXT_MINOR_VERSION_ARB             0x2092
#define GLX_CONTEXT_FLAGS_ARB                     0x2094
#define GLX_CONTEXT_DEBUG_BIT_ARB                 0x0001
#define GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB    0x0002
#define GLX_CONTEXT_PROFILE_MASK_ARB              0x9126
#define GLX_CONTEXT_CORE_PROFILE_BIT_ARB          0x00000001
#define GLX_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB 0x00000002

extern GLXContext glXCreateContextAttribsARB(Display *dpy, GLXFBConfig config, GLXContext share_context, Bool direct, const int *attrib_list);
#endif /* GLX_ARB_create_context */

/*
 *  GLX_ARB_context_flush_control
 */
#ifndef GLX_ARB_context_flush_control
#define GLX_ARB_context_flush_control 1

#define GLX_CONTEXT_RELEASE_BEHAVIOR_ARB        0x2097
#define GLX_CONTEXT_RELEASE_BEHAVIOR_NONE_ARB   0x0000
#define GLX_CONTEXT_RELEASE_BEHAVIOR_FLUSH_ARB  0x2098

#endif /* GLX_ARB_context_flush_control */

#ifdef __cplusplus
}
#endif

#endif /* __glx_ATI_h_ */
