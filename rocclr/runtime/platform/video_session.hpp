//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef VIDEO_SESSION_HPP_
#define VIDEO_SESSION_HPP_

#include "top.hpp"
#include "platform/object.hpp"
#include "device/device.hpp"
#include "platform/commandqueue.hpp"

#if cl_amd_open_video

namespace amd
{
//! VideoSession class
class VideoSession : public RuntimeObject
{
private:
    Context&                    context_;               //!< OpenCL context
    Device&                     device_;                //!< OpenCL device
    HostQueue*                  queue_;                 //!< Open CL video command queue
    cl_bitfield&                video_session_flags_;   //!< Creation flags
    cl_video_config_type_amd    type_;                  //!< config buffer type
    cl_uint                     size_;                  //!< config buffer size
    void*                       buffer_;                //!< pointer to config buffer

public:
    VideoSession(
        Context&                    context,    //!< OpenCL context
        Device&                     device,     //!< OpenCL device
        HostQueue*                  queue,      //!< OpenCL command queue
        cl_bitfield                 flags,      //!< Video Session flags
        cl_video_config_type_amd    type,       //!< config buffer type
        cl_uint                     size,       //!< config buffer size
        void*                       buffer      //!< pointer to config buffer
        )
        : context_(context)
        , device_(device)
        , queue_(queue)
        , video_session_flags_(flags)
        , type_(type)
        , size_(size)
        , buffer_(NULL)
    {
        if (size > 0 && buffer) {
            buffer_ = new char[size];
            memcpy(buffer_, buffer, size);
        }
    }

    virtual ~VideoSession()
    {
        if (queue_) {
            queue_->release();
        }
        if (buffer_) {
            delete[] static_cast<char*>(buffer_);
        }
    }

    //! Accessor functions
    Context&                context() const { return context_; }
    Device&                 device() const { return device_; }
    HostQueue&              queue() const { return *queue_; }
    cl_bitfield&            flags() const { return video_session_flags_; }
    cl_video_config_type_amd    type() const { return type_; }
    void * configbuffer() const {return buffer_;}

    //! RTTI internal implementation
   virtual ObjectType objectType() const {return ObjectTypeVideoSession;}
};

} // namespace amd

#endif // cl_amd_open_video

#endif /*VIDEO_SESSION_HPP_*/
