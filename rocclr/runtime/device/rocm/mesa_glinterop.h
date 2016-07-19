/*
 * Mesa 3-D graphics library
 *
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

/* Mesa OpenGL inter-driver interoperability interface designed for but not
 * limited to OpenCL.
 *
 * This is a driver-agnostic, backward-compatible interface. The structures
 * are only allowed to grow. They can never shrink and their members can
 * never be removed, renamed, or redefined.
 *
 * The interface doesn't return a lot of static texture parameters like
 * width, height, etc. It mainly returns mutable buffer and texture view
 * parameters that can't be part of the texture allocation (because they are
 * mutable). If drivers want to return more data or want to return static
 * allocation parameters, they can do it in one of these two ways:
 * - attaching the data to the DMABUF handle in a driver-specific way
 * - passing the data via "out_driver_data" in the "in" structure.
 *
 * Mesa is expected to do a lot of error checking on behalf of OpenCL, such
 * as checking the target, miplevel, and texture completeness.
 *
 * OpenCL, on the other hand, needs to check if the display+context combo
 * is compatible with the OpenCL driver by querying the device information.
 * It also needs to check if the texture internal format and channel ordering
 * (returned in a driver-specific way) is supported by OpenCL, among other
 * things.
 */

#ifndef MESA_GLINTEROP_H
#define MESA_GLINTEROP_H

#include <stdint.h>

#if !defined(MESA_GLINTEROP_NO_GLX)
#include <GL/glx.h>
#include <EGL/egl.h>
#else
#include <GL/gl.h>
#include <EGL/egl.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MESA_GLINTEROP_VERSION 1

/** Returned error codes. */
enum {
   MESA_GLINTEROP_SUCCESS = 0,
   MESA_GLINTEROP_OUT_OF_RESOURCES,
   MESA_GLINTEROP_OUT_OF_HOST_MEMORY,
   MESA_GLINTEROP_INVALID_OPERATION,
   MESA_GLINTEROP_INVALID_VALUE,
   MESA_GLINTEROP_INVALID_DISPLAY,
   MESA_GLINTEROP_INVALID_CONTEXT,
   MESA_GLINTEROP_INVALID_TARGET,
   MESA_GLINTEROP_INVALID_OBJECT,
   MESA_GLINTEROP_INVALID_MIP_LEVEL,
   MESA_GLINTEROP_UNSUPPORTED
};

/** Access flags. */
enum {
   MESA_GLINTEROP_ACCESS_READ_WRITE = 0,
   MESA_GLINTEROP_ACCESS_READ_ONLY,
   MESA_GLINTEROP_ACCESS_WRITE_ONLY
};


/**
 * Device information returned by Mesa.
 */
typedef struct _mesa_glinterop_device_info {
   uint32_t size; /* size of this structure */

   /* PCI location */
   uint32_t pci_segment_group;
   uint32_t pci_bus;
   uint32_t pci_device;
   uint32_t pci_function;

   /* Device identification */
   uint32_t vendor_id;
   uint32_t device_id;
} mesa_glinterop_device_info;


/**
 * Input parameters to Mesa interop export functions.
 */
typedef struct _mesa_glinterop_export_in {
   uint32_t size; /* size of this structure */

   /* One of the following:
    * - GL_TEXTURE_BUFFER
    * - GL_TEXTURE_1D
    * - GL_TEXTURE_2D
    * - GL_TEXTURE_3D
    * - GL_TEXTURE_RECTANGLE
    * - GL_TEXTURE_1D_ARRAY
    * - GL_TEXTURE_2D_ARRAY
    * - GL_TEXTURE_CUBE_MAP_ARRAY
    * - GL_TEXTURE_CUBE_MAP
    * - GL_TEXTURE_CUBE_MAP_POSITIVE_X
    * - GL_TEXTURE_CUBE_MAP_NEGATIVE_X
    * - GL_TEXTURE_CUBE_MAP_POSITIVE_Y
    * - GL_TEXTURE_CUBE_MAP_NEGATIVE_Y
    * - GL_TEXTURE_CUBE_MAP_POSITIVE_Z
    * - GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
    * - GL_TEXTURE_2D_MULTISAMPLE
    * - GL_TEXTURE_2D_MULTISAMPLE_ARRAY
    * - GL_TEXTURE_EXTERNAL_OES
    * - GL_RENDERBUFFER
    * - GL_ARRAY_BUFFER
    */
   GLenum target;

   /* If target is GL_ARRAY_BUFFER, it's a buffer object.
    * If target is GL_RENDERBUFFER, it's a renderbuffer object.
    * If target is GL_TEXTURE_*, it's a texture object.
    */
   GLuint obj;

   /* Mipmap level. Ignored for non-texture objects. */
   GLuint miplevel;

   /* One of MESA_GLINTEROP_ACCESS_* flags. This describes how the exported
    * object is going to be used.
    */
   uint32_t access;

   /* Size of memory pointed to by out_driver_data. */
   uint32_t out_driver_data_size;

   /* If the caller wants to query driver-specific data about the OpenGL
    * object, this should point to the memory where that data will be stored.
    */
   void *out_driver_data;
} mesa_glinterop_export_in;


/**
 * Outputs of Mesa interop export functions.
 */
typedef struct _mesa_glinterop_export_out {
   uint32_t size; /* size of this structure */

   /* The DMABUF handle. It must be closed by the caller using the POSIX
    * close() function when it's not needed anymore. Mesa is not responsible
    * for closing the handle.
    *
    * Not closing the handle by the caller will lead to a resource leak,
    * prevents releasing the GPU buffer, and may prevent creating new DMABUF
    * handles until the process termination.
    */
   int dmabuf_fd;

   /* The mutable OpenGL internal format specified by glTextureView or
    * glTexBuffer. If the object is not one of those, the original internal
    * format specified by glTexStorage, glTexImage, or glRenderbufferStorage
    * will be returned.
    */
   GLenum internalformat;

   /* Parameters specified by glTexBufferRange for GL_TEXTURE_BUFFER. */
   GLintptr buf_offset;
   GLsizeiptr buf_size;

   /* Parameters specified by glTextureView. If the object is not a texture
    * view, default parameters covering the whole texture will be returned.
    */
   GLuint view_minlevel;
   GLuint view_numlevels;
   GLuint view_minlayer;
   GLuint view_numlayers;
} mesa_glinterop_export_out;

#if !defined(MESA_GLINTEROP_NO_GLX)
/**
 * Query device information.
 *
 * \param dpy        GLX display
 * \param context    GLX context
 * \param out        where to return the information
 *
 * \return MESA_GLINTEROP_SUCCESS or MESA_GLINTEROP_* != 0 on error
 */
GLAPI int GLAPIENTRY
MesaGLInteropGLXQueryDeviceInfo(Display *dpy, GLXContext context,
                                mesa_glinterop_device_info *out);
#endif

/**
 * Same as MesaGLInteropGLXQueryDeviceInfo except that it accepts EGLDisplay
 * and EGLContext.
 */
GLAPI int GLAPIENTRY
MesaGLInteropEGLQueryDeviceInfo(EGLDisplay dpy, EGLContext context,
                                mesa_glinterop_device_info *out);


#if !defined(MESA_GLINTEROP_NO_GLX)
/**
 * Create and return a DMABUF handle corresponding to the given OpenGL
 * object, and return other parameters about the OpenGL object.
 *
 * \param dpy        GLX display
 * \param context    GLX context
 * \param in         input parameters
 * \param out        return values
 *
 * \return MESA_GLINTEROP_SUCCESS or MESA_GLINTEROP_* != 0 on error
 */
GLAPI int GLAPIENTRY
MesaGLInteropGLXExportObject(Display *dpy, GLXContext context,
                             mesa_glinterop_export_in *in,
                             mesa_glinterop_export_out *out);
#endif

/**
 * Same as MesaGLInteropGLXExportObject except that it accepts
 * EGLDisplay and EGLContext.
 */
GLAPI int GLAPIENTRY
MesaGLInteropEGLExportObject(EGLDisplay dpy, EGLContext context,
                             mesa_glinterop_export_in *in,
                             mesa_glinterop_export_out *out);


#if !defined(MESA_GLINTEROP_NO_GLX)
typedef int (APIENTRYP PFNMESAGLINTEROPGLXQUERYDEVICEINFOPROC)(Display *dpy, GLXContext context,
                                                               mesa_glinterop_device_info *out);
#endif
typedef int (APIENTRYP PFNMESAGLINTEROPEGLQUERYDEVICEINFOPROC)(EGLDisplay dpy, EGLContext context,
                                                               mesa_glinterop_device_info *out);
#if !defined(MESA_GLINTEROP_NO_GLX)
typedef int (APIENTRYP PFNMESAGLINTEROPGLXEXPORTOBJECTPROC)(Display *dpy, GLXContext context,
                                                            mesa_glinterop_export_in *in,
                                                            mesa_glinterop_export_out *out);
#endif
typedef int (APIENTRYP PFNMESAGLINTEROPEGLEXPORTOBJECTPROC)(EGLDisplay dpy, EGLContext context,
                                                            mesa_glinterop_export_in *in,
                                                            mesa_glinterop_export_out *out);

#ifdef __cplusplus
}
#endif

#endif /* MESA_GLINTEROP_H */
