//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//

#include "cl_common.hpp"
#include "platform/command.hpp"
#include <cstring>

#if cl_amd_open_video

#include "platform/video_session.hpp"

#include "cal.h"
#include "calcl.h"

amd::HostQueue* staticQueue; //@todo fix the interface and remove hardcoded queue
/* Helper function GetCalDecodeProfile()
*/
static bool
GetCalVideoProfile(cl_video_decode_profile_amd clVideoProfile,
    CALdecodeProfile& calVideoProfile)
{
    switch(clVideoProfile) {
    case CL_VIDEO_DECODE_PROFILE_H264_BASELINE_AMD:
        calVideoProfile = CAL_VID_H264_BASELINE;
        break;
    case CL_VIDEO_DECODE_PROFILE_H264_MAIN_AMD:
        calVideoProfile = CAL_VID_H264_MAIN;
        break;
    case CL_VIDEO_DECODE_PROFILE_H264_HIGH_AMD:
        calVideoProfile = CAL_VID_H264_HIGH;
        break;
    case CL_VIDEO_DECODE_PROFILE_VC1_SIMPLE_AMD:
        calVideoProfile = CAL_VID_VC1_SIMPLE;
        break;
    case CL_VIDEO_DECODE_PROFILE_VC1_MAIN_AMD:
        calVideoProfile = CAL_VID_VC1_MAIN;
        break;
    case CL_VIDEO_DECODE_PROFILE_VC1_ADVANCED_AMD:
        calVideoProfile = CAL_VID_VC1_ADVANCED;
        break;
    case CL_VIDEO_DECODE_PROFILE_MPEG2_VLD_AMD:
        calVideoProfile = CAL_VID_MPEG2_VLD;
        break;
    //case CL_VIDEO_DECODE_PROFILE_MPEG2_SIMPLE_AMD:
    //case CL_VIDEO_DECODE_PROFILE_MPEG2_MAIN_AMD:
    default:
        return false;
        break;
    }
    return true;
}
/* Helper function GetCalDecodeProfile()
*/
static bool
GetCalVideoFormat(cl_video_format_amd clVideoFormat,
    CALdecodeFormat& calVideoFormat)
{
    switch(clVideoFormat) {
    case CL_VIDEO_NV12_INTERLEAVED_AMD:
        calVideoFormat = CAL_VID_NV12_INTERLEAVED;
        break;
    case CL_VIDEO_YV12_INTERLEAVED_AMD:
        calVideoFormat = CAL_VID_YV12_INTERLEAVED;
        break;
    default:
        return false;
    }
    return true;
}
/*! \addtogroup API
 *  @{
 *
 *  \addtogroup cl_amd_open_video
 *
 *  This section provides OpenCL extension functions that allow applications
 * to use kernel decoding of video streams using corresponding HW capabilities.
 *
 *  @}
 *  \addtogroup clQueryVidDecoderCapsAMD
 *  @{
 */


/*! \brief Creates a video session object.
 *
 *  \param context is a valid OpenCL context on which the image object
 * is to be created.
 *
 *  \param device must be a device associated with context. It can either be in
 * the list of devices specified when context is created using
 * clCreateContext or have the same device type as the device type
 * specified when the context is created using clCreateContextFromType
 *
 *  \param flags is a bit-field that is used to specify the video session
 * operation mode, available flags are as follows:
 *  - CL_VIDEO_DECODE_ACCELERATION_AMD - the video session supports HW
 * acceleration of video decoding operation.
 *
 *  \param errcode_ret will return CL_SUCCESS if no error occurs;
 * or an appropriate error code:
 *  - CL_INVALID_CONTEXT if \a context is not a valid context;
 *  - CL_INVALID_DEVICE if \a device is not a valid device;
 *  - CL_INVALID_VIDEO_SESSION_FLAGS_AMD if flags is NULL or contains invalid
 *    configuration;
 *  - CL_INVALID_OPERATION if video decode extension is not supported by
 *    any device associated with context;
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.1r25
 */
RUNTIME_ENTRY_RET(cl_video_session_amd, clCreateVideoSessionAMD, (
    cl_context                  context,
    cl_device_id                device,
    cl_video_session_flags_amd  flags,
    cl_video_config_type_amd    config_buffer_type,
    cl_uint                     config_buffer_size,
    void*                       config_buffer,
    cl_int*                     errcode_ret))
{
    *not_null(errcode_ret) = CL_SUCCESS;

    if (!is_valid(context)) {
        *not_null(errcode_ret) = CL_INVALID_CONTEXT;
        LogWarning("invalid parameter \"context\"");
        return (cl_video_session_amd) 0;
    }
    if (!is_valid(device)) {
        *not_null(errcode_ret) = CL_INVALID_DEVICE;
        LogWarning("invalid parameter \"device\"");
        return (cl_video_session_amd) 0;
    }
    amd::Device* amdDevice = as_amd(device);
    if (!amdDevice->info().openVideo_) {
        *not_null(errcode_ret) = CL_INVALID_OPERATION;
        LogWarning("Device or CAL does not support Open Video extension");
        return (cl_video_session_amd) 0;
    }

    // Create video command queue
    cl_video_encode_desc_amd ovSessionProperties;
    CALvideoProperties calVideoProperties;
    calVideoProperties.size = sizeof(calVideoProperties);
    calVideoProperties.flags = static_cast<CALuint>(flags);
    switch(config_buffer_type) {
    case CL_VIDEO_DECODE_CONFIGURATION_AMD:
        {
            cl_video_decode_desc_amd* videoDecodeDesc =
                static_cast<cl_video_decode_desc_amd*>(config_buffer);
            if (!GetCalVideoProfile(videoDecodeDesc->attrib.profile,
                calVideoProperties.profile)) {
                *not_null(errcode_ret) = CL_INVALID_OPERATION;
                LogWarning("Profile is not supported or invalid");
                return (cl_video_session_amd) 0;
            }
            if (!GetCalVideoFormat(videoDecodeDesc->attrib.format,
                calVideoProperties.format)) {
                *not_null(errcode_ret) = CL_INVALID_OPERATION;
                LogWarning("Format is not supported or invalid");
                return (cl_video_session_amd) 0;
            }
            calVideoProperties.width = videoDecodeDesc->image_width;
            calVideoProperties.height = videoDecodeDesc->image_height;
            calVideoProperties.VideoEngine_name = CAL_CONTEXT_VIDEO;
        }
        break;
    default:
        *not_null(errcode_ret) = CL_INVALID_VIDEO_CONFIG_TYPE_AMD;
        LogWarning("invalid parameter \"config_buffer_type\"");
        return (cl_video_session_amd) 0;
        break;
    }

    ovSessionProperties.calVideoProperties = &calVideoProperties;

    amd::HostQueue* queue = new amd::HostQueue(
        *as_amd(context), *as_amd(device), 0, &ovSessionProperties);

    if (queue == NULL || !queue->create()) {
        *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
        LogWarning("not enough host memory");
        return (cl_video_session_amd) 0;
    }

    staticQueue = queue;
    amd::VideoSession* video_session = new amd::VideoSession(
        *as_amd(context), *as_amd(device), queue, flags, config_buffer_type,
        config_buffer_size, config_buffer);
    if (!video_session) {
        queue->release();
        *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
        LogWarning("not enough host memory");
        return (cl_video_session_amd) 0;
    }

    return as_cl<amd::VideoSession>(video_session);
}
RUNTIME_EXIT


RUNTIME_ENTRY_RET(cl_video_session_amd, clCreateVideoEncSessionAMD, (
    cl_context                  context,
    cl_device_id                device,
    cl_video_session_flags_amd  flags,
    cl_video_config_type_amd    config_buffer_type,
    cl_uint                     config_buffer_size,
    void*                       config_buffer,
    cl_int*                     errcode_ret))
{
    *not_null(errcode_ret) = CL_SUCCESS;

    // Make sure the context is valid
    if (!is_valid(context)) {
        *not_null(errcode_ret) = CL_INVALID_CONTEXT;
        LogWarning("invalid parameter \"context\"");
        return (cl_video_session_amd) 0;
    }

    // Make sure the device is valid
    if (!is_valid(device)) {
        *not_null(errcode_ret) = CL_INVALID_DEVICE;
        LogWarning("invalid parameter \"device\"");
        return (cl_video_session_amd) 0;
    }

    // Make sure the device supports Open Video extensions
    amd::Device* amdDevice = as_amd(device);
    if (!amdDevice->info().openVideo_) {
        *not_null(errcode_ret) = CL_INVALID_OPERATION;
        LogWarning("Device or CAL does not support Open Video extension");
        return (cl_video_session_amd) 0;
    }

    cl_video_encode_desc_amd *ovSessionProperties = (cl_video_encode_desc_amd *)config_buffer;

    // Create video command queue
    CALvideoProperties calVideoProperties;

    switch(config_buffer_type)
    {
    case CL_VIDEO_ENCODE_CONFIGURATION_AMD:
        calVideoProperties.size                 = sizeof(calVideoProperties);
        calVideoProperties.flags                = static_cast<CALuint>(flags);
        calVideoProperties.profile              = (CALdecodeProfile)ovSessionProperties->attrib.codec_profile;
        calVideoProperties.format               = (CALdecodeFormat)ovSessionProperties->attrib.format;
        calVideoProperties.width                = ovSessionProperties->image_width;
        calVideoProperties.height               = ovSessionProperties->image_height;
        calVideoProperties.VideoEngine_name     = CAL_CONTEXT_VIDEO_VCE;
        ovSessionProperties->calVideoProperties = &calVideoProperties;
        break;
    default:
        *not_null(errcode_ret) = CL_INVALID_VIDEO_CONFIG_TYPE_AMD;
        LogWarning("invalid parameter \"config_buffer_type\"");
        return (cl_video_session_amd) 0;
        break;
    }

    amd::HostQueue* queue = new amd::HostQueue(
        *as_amd(context), *as_amd(device), 0, ovSessionProperties);

    if (queue == NULL || !queue->create()) {
        *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
        LogWarning("not enough host memory");
        return (cl_video_session_amd) 0;
    }
    staticQueue = queue;
#if 0 // this doesn't work here due to context value...
    CAL_VID_PROFILE_LEVEL encode_profile_level;
    encode_profile_level.profile = buffer->attrib.profile;
    encode_profile_level.level = buffer->attrib.level;
    CALuint frameRate = 0;
    CALuint YUVwidth = buffer->image_width;
    CALuint YUVhigh = buffer->image_height;

    cal::details::extensions_.extEncodeCreateSession_((CALcontext)as_amd(context), 0, CAL_VID_encode_AVC_FULL, encode_profile_level,
                            CAL_VID_PICTURE_NV12, YUVwidth, YUVhigh, frameRate, CAL_VID_ENCODE_JOB_PRIORITY_LEVEL1);
#endif

    amd::VideoSession* video_session = new amd::VideoSession(
        *as_amd(context), *as_amd(device), queue, flags, config_buffer_type,
        config_buffer_size, config_buffer);
    if (!video_session)
    {
        queue->release();
        *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
        LogWarning("not enough host memory");
        return (cl_video_session_amd) 0;
    }

    return as_cl<amd::VideoSession>(video_session);
}
RUNTIME_EXIT

RUNTIME_ENTRY(cl_int, clDestroyVideoEncSessionAMD,
    (cl_video_session_amd video_session))
{
    if (!is_valid(video_session)) {
        LogWarning("invalid parameter \"video_session\"");
        return CL_INVALID_VIDEO_SESSION_AMD;
    }

    cl_int errcode_ret = CL_SUCCESS;
    amd::VideoSession* session = as_amd(video_session);

    amd::Command::EventWaitList eventWaitList;
    cl_int err = amd::clSetEventWaitList(eventWaitList,
        session->context(), 0, NULL);
    amd::Command* command = new amd::SetVideoSessionCommand(
        session->queue(), eventWaitList,
        amd::SetVideoSessionCommand::CloseSession, NULL);
    if (command == NULL) {
        return CL_OUT_OF_HOST_MEMORY;
    }
    command->enqueue();
    command->release();

    session->queue().finish();
    session->release();

    return errcode_ret;
}
RUNTIME_EXIT

/*! \brief Increments the video session reference count.
 *
 *  \param video_session is a valid OpenCL Video Session.
 *
 *  \return CL_SUCCESS if the function executes successfully;
 * Otherwise an error code is returned:
 *  - CL_INVALID_VIDEO_SESSION_AMD if video_session is not a valid
 * video session;
 *
 *  \version 1.1r25
 */
RUNTIME_ENTRY(cl_int, clRetainVideoSessionAMD,
    (cl_video_session_amd video_session))
{
    if (!is_valid(video_session)) {
        LogWarning("invalid parameter \"video_session\"");
        return CL_INVALID_VIDEO_SESSION_AMD;
    }
    as_amd(video_session)->retain();
    return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Decrements the video session reference count.
 *
 *  \param video_session is a valid OpenCL Video Session.
 *
 *  \return CL_SUCCESS if the function executes successfully;
 * Otherwise an error code is returned:
 *  - CL_INVALID_VIDEO_SESSION_AMD if video_session is not a valid
 * video session;
 *
 *  \version 1.1r25
 */
RUNTIME_ENTRY(cl_int, clReleaseVideoSessionAMD,
    (cl_video_session_amd video_session))
{
    if (!is_valid(video_session)) {
        LogWarning("invalid parameter \"video_session\"");
        return CL_INVALID_VIDEO_SESSION_AMD;
    }
    as_amd(video_session)->release();
    return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Queries the configuration of a specific video session.
 *
 *  \param video_session is a valid OpenCL Video Session.
 *
 *  \param config_buffer_type is an enumeration constant that identifies the type
 * of the configuration buffer. It can be one of the following values:
 *  - CL_VIDEO_DECODE_CONFIGURATION_AMD - The buffer contains decoder configuration,
 *    according to the cl_video_decode_desc_amd structure definition.
 *
 *  \param config_buffer_size is a pointer to size of config_buffer in bytes.
 * The implementation returns the size of the config_buffer in this argument.
 *
 *  \param config_buffer is a pointer to a configuration buffer. The
 * implementation writes the configuration query data into this
 * buffer. If this pointer is NULL, it is ignored (which enables the
 * application to query the required buffer size for this configuration type).
 *
 *  \returns CL_SUCCESS if the video session configuration was set properly;
 * Otherwise an appropriate error code is returned:
 *  - CL_INVALID_VIDEO_SESSION_AMD if video_session is not a valid video session;
 *  - CL_INVALID_VIDEO_CONFIG_TYPE_AMD if config_buffer_type is NULL or contains a
 *    configuration type which is not supported by the specific implementation;
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.1r25
 */
RUNTIME_ENTRY(cl_int, clGetVideoSessionInfoAMD, (
    cl_video_session_amd        video_session,
    cl_video_session_info_amd   param_name,
    size_t                      param_value_size,
    void*                       param_value,
    size_t*                     param_value_size_ret))
{
    if (!is_valid(video_session)) {
        LogWarning("invalid parameter \"video_session\"");
        return CL_INVALID_VIDEO_SESSION_AMD;
    }
    //switch(config_buffer_type) {
    //case CL_VIDEO_DECODE_CONFIGURATION_AMD:
    //    break;
    //default:
    //    return CL_INVALID_VIDEO_CONFIG_TYPE_AMD;
    //}
/*
    return amd::clGetVideoSessionInfoAMD(*as_amd(video_session),
        config_buffer_type, config_buffer_size, config_buffer);
*/
    return CL_INVALID_VALUE;
//    return CL_SUCCESS;
}
RUNTIME_EXIT


/*! \brief Queries the configuration of a specific video session.
 *
 *  \param video_session is a valid OpenCL Video Session.
 *
 *  \param config_buffer_type is an enumeration constant that identifies the type
 * of the configuration buffer. It can be one of the following values:
 *  - CL_VIDEO_ENCODE_CONFIGURATION_AMD - The buffer contains encoder configuration,
 *    according to the cl_video_decode_desc_amd structure definition.
 *
 *  \param config_buffer_size is a pointer to size of config_buffer in bytes.
 * The implementation returns the size of the config_buffer in this argument.
 *
 *  \param config_buffer is a pointer to a configuration buffer. The
 * implementation writes the configuration query data into this
 * buffer. If this pointer is NULL, it is ignored (which enables the
 * application to query the required buffer size for this configuration type).
 *
 *  \returns CL_SUCCESS if the video session configuration was set properly;
 * Otherwise an appropriate error code is returned:
 *  - CL_INVALID_VIDEO_SESSION_AMD if video_session is not a valid video session;
 *  - CL_INVALID_VIDEO_CONFIG_TYPE_AMD if config_buffer_type is NULL or contains a
 *    configuration type which is not supported by the specific implementation;
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.1r25
 */
RUNTIME_ENTRY(cl_int, clGetVideoSessionEncInfoAMD, (
    cl_video_session_amd            video_session,
    cl_video_session_enc_info_amd   param_name,
    size_t                          param_value_size,
    void*                           param_value,
    size_t*                         param_value_size_ret))
{
    cl_uint errcode_ret = CL_SUCCESS;

    if (!is_valid(video_session)) {
        LogWarning("invalid parameter \"video_session\"");
        return CL_INVALID_VIDEO_SESSION_AMD;
    }

    amd::VideoSession& session = *as_amd(video_session);
    amd::Command::EventWaitList eventWaitList;
    cl_int err = amd::clSetEventWaitList(eventWaitList, session.context(), 0, NULL);
    amd::Command* command = NULL;

    switch(param_name) {
    case CL_CONFIG_TYPE_PICTURE_CONTROL :
        command = new amd::SetVideoSessionCommand(
            session.queue(), eventWaitList,
            amd::SetVideoSessionCommand::ConfigTypePictureControl, param_value);
        break;
    case CL_CONFIG_TYPE_RATE_CONTROL :
        command = new amd::SetVideoSessionCommand(
            session.queue(), eventWaitList,
            amd::SetVideoSessionCommand::ConfigTypeRateControl, param_value);
        break;
    case CL_CONFIG_TYPE_MOTION_ESTIMATION :
        command = new amd::SetVideoSessionCommand(
            session.queue(), eventWaitList,
            amd::SetVideoSessionCommand::ConfigTypeMotionEstimation, param_value);
        break;
    case CL_CONFIG_TYPE_RDO :
        command = new amd::SetVideoSessionCommand(
            session.queue(), eventWaitList,
            amd::SetVideoSessionCommand::ConfigTypeRDO, param_value);
        break;
    default:
        errcode_ret = CL_INVALID_VIDEO_CONFIG_TYPE_AMD;
        break;
    }
    if (command == NULL) {
        return CL_OUT_OF_HOST_MEMORY;
    }
    command->enqueue();
    command->release();

    session.queue().finish();
    * param_value_size_ret = sizeof(param_value);

    return errcode_ret;
}
RUNTIME_EXIT


RUNTIME_ENTRY(cl_int, clSendEncodeConfigInfoAMD, (
    cl_video_session_amd            video_session,
    size_t                          numBuffers,
    void*                           pConfigBuffers))
{
    cl_uint errcode_ret = CL_SUCCESS;

    if (!is_valid(video_session)) {
        LogWarning("invalid parameter \"video_session\"");
        return CL_INVALID_VIDEO_SESSION_AMD;
    }

    amd::VideoSession& session = *as_amd(video_session);
    amd::Command::EventWaitList eventWaitList;
    cl_int err = amd::clSetEventWaitList(eventWaitList, session.context(), 0, NULL);
    amd::Command* command = new amd::SetVideoSessionCommand(
            session.queue(), eventWaitList,
            amd::SetVideoSessionCommand::SendEncodeConfig, pConfigBuffers, numBuffers);
    if (command == NULL) {
        return CL_OUT_OF_HOST_MEMORY;
    }
    command->enqueue();
    command->release();

    session.queue().finish();

    return errcode_ret;
}
RUNTIME_EXIT

/*! \brief Enqueues execution of a decode/encode command on UVD unit of the device
 * specified at creation of video session.
 *
 *  \param video_session is a valid OpenCL video session.
 *
 *  \param video_data_structure is a valid pointer to a structure
 * containing control and video stream data for decoding/encoding.
 *
 *  \param config_buffer_size is a pointer to size of config_buffer in bytes.
 * The implementation returns the size of the config_buffer in this argument.
 *
 *  \param event_wait_list and \param num_events_in_wait_list specify events
 * that need to complete before this particular command can be executed.
 *
 *  \param event returns an event object that identifies this particular
 * video program execution instance.
 *
 *  \returns CL_SUCCESS if the video program execution was
 * successfully passed to the CAL for execution and CAL returned CAL_RESULT_OK.
 * Otherwise, it returnes one of the following errors:
 *
 *  - CL_INVALID_VIDEO_SESSION_AMD if the parameter video_session is not
 * a valid video session object returned by a call to function
 * clCreateVideoSessionAMD;
 *  - CL_IVALID_VIDEO_DATA_AMD if pointer video_data_struct is NULL, points
 * to invalid memory, or any of the pointers/objects provided in the
 * video_data_structure is NULL or invalid;
 *  - CL_INVALID_CONTEXT if context associated with video_session is not
 * the same as context associated with event_wait_list;
 *  - CL_INVALID_EVENT_WAIT_LIST if event_wait_list is NULL and
 * num_events_in_wait_list > 0, or event_wait_list is not NULL and
 * num_events_in_wait_list is 0, or if event objects in event_wait_list
 * are not valid events;
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources
 * required by the Open CL implementation on the host.
 *
 *  \version 1.1r33
 */
RUNTIME_ENTRY(cl_int, clEnqueueRunVideoProgramAMD, (
    cl_video_session_amd        video_session,
    void*                       video_data,
    cl_uint                     num_events_in_wait_list,
    const cl_event*             event_wait_list,
    cl_event*                   event))
{
    if (!is_valid(video_session)) {
        LogWarning("invalid parameter \"video_session\"");
        return CL_INVALID_VIDEO_SESSION_AMD;
    }
    cl_int errcode = CL_SUCCESS;

    amd::VideoSession& session = *as_amd(video_session);

    switch(session.type()) {
    case CL_VIDEO_DECODE_CONFIGURATION_AMD:
        errcode = amd::clEnqueueVideoDecodeAMD(session,
            (cl_video_decode_data_amd*) video_data,
            num_events_in_wait_list, event_wait_list, event);
        break;
    case CL_VIDEO_ENCODE_CONFIGURATION_AMD:
        errcode = amd::clEnqueueVideoEncodeAMD(session,
            (cl_video_encode_data_amd*) video_data,
            num_events_in_wait_list, event_wait_list, event);
        break;

    default:
        errcode = CL_INVALID_VIDEO_CONFIG_TYPE_AMD;
    }
    return errcode;
}
RUNTIME_EXIT

RUNTIME_ENTRY(cl_int, clEncodeGetDeviceCapAMD, (
    cl_device_id    device_id,
    cl_uint         encode_mode,
    cl_uint         encode_cap_total_size,
    cl_uint         *num_encode_cap,
    void            *pEncodeCAP))
{
    // Make sure the device supports Open Video extensions
    amd::Device* device = as_amd(device_id);
    if (!device->info().openVideo_) {
        return CL_DEVICE_NOT_FOUND;
    }

//    amd::VideoSession& session = *as_amd(video_session);
    amd::Command::EventWaitList eventWaitList;
    cl_int err = amd::clSetEventWaitList(eventWaitList, staticQueue->context(), 0, NULL);
    amd::Command* command = new amd::SetVideoSessionCommand(
            *staticQueue, eventWaitList,
            amd::SetVideoSessionCommand::GetDeviceCapVCE, pEncodeCAP, encode_cap_total_size);
    if (command == NULL) {
        return CL_OUT_OF_HOST_MEMORY;
    }
    command->enqueue();
    command->release();
    staticQueue->finish();

    return CL_SUCCESS;
}
RUNTIME_EXIT

#if 1
RUNTIME_ENTRY(cl_int, clEncodePictureAMD, (
    cl_video_session_amd        video_session,
    cl_uint                     number_of_encode_task_input_buffers,
    void*                       encode_task_input_buffer_list,
    void*                       picture_parameter,
    cl_uint*                    pTaskID))

{
    cl_uint errcode_ret = CL_SUCCESS;

    if (!is_valid(video_session)) {
        LogWarning("invalid parameter \"video_session\"");
        return CL_INVALID_VIDEO_SESSION_AMD;
    }

    return errcode_ret;
}
RUNTIME_EXIT
#endif

RUNTIME_ENTRY(cl_int, clEncodeQueryTaskDescriptionAMD, (
    cl_video_session_amd        video_session,
    cl_uint                        num_of_task_description_request,
    cl_uint*                    num_of_task_description_return,
    void *                      task_description_list))
{
    cl_uint errcode_ret = CL_SUCCESS;

    if (!is_valid(video_session)) {
        LogWarning("invalid parameter \"video_session\"");
        return CL_INVALID_VIDEO_SESSION_AMD;
    }

    amd::VideoSession& session = *as_amd(video_session);
    amd::Command::EventWaitList eventWaitList;
    cl_int err = amd::clSetEventWaitList(eventWaitList, session.context(), 0, NULL);
    amd::Command* command = new amd::SetVideoSessionCommand(
            session.queue(), eventWaitList,
            amd::SetVideoSessionCommand::EncodeQueryTaskDescription,
            num_of_task_description_request, task_description_list, num_of_task_description_return);
    if (command == NULL) {
        return CL_OUT_OF_HOST_MEMORY;
    }
    command->enqueue();
    command->release();
    session.queue().finish();

    return errcode_ret;
}
RUNTIME_EXIT


RUNTIME_ENTRY(cl_int, clEncodeReleaseOutputResourceAMD, (
    cl_video_session_amd        video_session,
    cl_uint                        task_id))
{
    cl_uint errcode_ret = CL_SUCCESS;

    if (!is_valid(video_session)) {
        LogWarning("invalid parameter \"video_session\"");
        return CL_INVALID_VIDEO_SESSION_AMD;
    }

    amd::VideoSession& session = *as_amd(video_session);
    amd::Command::EventWaitList eventWaitList;
    cl_int err = amd::clSetEventWaitList(eventWaitList, session.context(), 0, NULL);
    amd::Command* command = new amd::SetVideoSessionCommand(
            session.queue(), eventWaitList,
            amd::SetVideoSessionCommand::ReleaseOutputResource,
            NULL, task_id);
    if (command == NULL) {
        return CL_OUT_OF_HOST_MEMORY;
    }
    command->enqueue();
    command->release();
    session.queue().finish();

    return errcode_ret;
}
RUNTIME_EXIT

namespace amd {

cl_int
clEnqueueVideoDecodeAMD(VideoSession& session,
    cl_video_decode_data_amd* data,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    if ((NULL == data) || (data->video_type.type != CL_VIDEO_DECODE)) {
        return CL_INVALID_OPERATION;
    }

    if (!is_valid(data->output_surface)) {
        return CL_INVALID_MEM_OBJECT;
    }

    amd::Command::EventWaitList eventWaitList;
    cl_int errcode = amd::clSetEventWaitList(eventWaitList,
        session.context(), num_events_in_wait_list, event_wait_list);
    if (errcode != CL_SUCCESS){
        return errcode;
    }

    //! Now create command and enqueue
    amd::RunVideoProgramCommand* command =
        new amd::RunVideoProgramCommand(session.queue(),
        eventWaitList, data, *as_amd(data->output_surface),
        CL_COMMAND_VIDEO_DECODE_AMD);
    if (command == NULL) {
        LogError("Cannot create new RunVideoProgramCommand");
        return CL_OUT_OF_HOST_MEMORY;
    }

    // Make sure we have memory for the command execution
    if (!command->validateMemory()) {
        delete command;
        return CL_MEM_OBJECT_ALLOCATION_FAILURE;
    }

    command->enqueue();

    *not_null(event) = as_cl(&command->event());
    if (event == NULL) {
        command->release();
    }
    return CL_SUCCESS;
}

cl_int
clEnqueueVideoEncodeAMD(VideoSession& session,
    cl_video_encode_data_amd* data,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    if ((NULL == data) || (data->video_type.type != CL_VIDEO_ENCODE)) {
        return CL_INVALID_OPERATION;
    }

    CAL_VID_BUFFER_DESCRIPTION *bufferList =
        reinterpret_cast<CAL_VID_BUFFER_DESCRIPTION *>(data->pictureParam1);
    cl_mem memory = static_cast<cl_mem>(bufferList[0].buffer.pPicture);

    if (!is_valid(memory)) {
        return CL_INVALID_MEM_OBJECT;
    }

    // Start up the event queue
    amd::Command::EventWaitList eventWaitList;
    cl_int errcode = amd::clSetEventWaitList(eventWaitList,
        session.context(), num_events_in_wait_list, event_wait_list);
    if (errcode != CL_SUCCESS){
        return errcode;
    }

    //! Now create command and enqueue
    amd::RunVideoProgramCommand* command =
        new amd::RunVideoProgramCommand(session.queue(),
        eventWaitList, data, *as_amd(memory),
        CL_COMMAND_VIDEO_ENCODE_AMD);
    if (command == NULL) {
        LogError("Cannot create new RunVideoProgramCommand");
        return CL_OUT_OF_HOST_MEMORY;
    }

    // Make sure we have memory for the command execution
    if (!command->validateMemory()) {
        delete command;
        return CL_MEM_OBJECT_ALLOCATION_FAILURE;
    }

    // Issue the command via the queue system.
    command->enqueue();

    *not_null(event) = as_cl(&command->event());
    if (event == NULL) {
        command->release();
    }
    return CL_SUCCESS;
}


} // namespace amd

#endif // cl_amd_video_session
