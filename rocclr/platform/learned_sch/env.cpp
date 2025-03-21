#include "env.hpp"
#include <cmath>

AMDresult GetIPUUtilization(int* averageUtil, int* coreUtil, AMDdev dev)
{
    AMDInit(0);
    mluDev_t *device = devGetDeviceHandle(dev);
    AMDresult res = monitorGetDeviceUtilizationInfo(device, averageUtil, coreUtil);
    if (res) {};
    return AMD_SUCCESS;
}

AMDresult ObserveState(Observe_st *Obs)
{
    AMDresult res;
    AMDdev dev;
    mluDev_t *device;
    AMDContextManager *ctx_mgr;
    ContextHandler *Context, *CtxTemp;
    AMDqueue hqueue, temp;
    struct queueRLSCapturer_st *capturer;
    struct slCapturerInvokeData_st *rls, *n;
    struct sbtsInvokeKernelCaptureData_st *data = NULL;
    int coreUtil[80];
    Kernel_st kernel_param;
    Queue_st queue_param;

    Obs->queueNum = 0;
    Obs->kernelNum = 0;
    Obs->info.Param_list.clear();

    for (int i = 0; i < 1; i++) { //遍历每个设备 现仅只考虑dev0
        res = AMDDeviceGet(&dev, i);
        res = GetIPUUtilization(&(Obs->info.averageCoreUtilization), coreUtil, dev); //获取ipu利用率
        if (res) return AMD_ERROR_UNKNOWN;

        //遍历每个设备的每个context的每个queue，获取queue task信息
        device = devGetDeviceHandle(dev);
        if (!device) {
            return AMD_ERROR_UNKNOWN;
        }
        ctx_mgr = GetContextManager(device);
        if (!ctx_mgr) {
            return AMD_ERROR_UNKNOWN;
        }
        camb_mutex_lock(&ctx_mgr->context_lock);

        list_for_each_entry_safe (Context, CtxTemp, &ctx_mgr->context, entry) { //遍历context
            struct SbtsPrivData *pcambpd = (struct SbtsPrivData *)Context->priv_data_sbts;

            camb_mutex_lock(&pcambpd->queueListDevice.lock);

            list_for_each_entry_safe (hqueue, temp, &pcambpd->queueListDevice.list,
                                      list_node) { //遍历queue
                std::tuple<Queue_st, std::vector<Kernel_st>> tp;
                std::vector<Kernel_st> kernel_list;

                capturer = hqueue->rls_capturer;
                if (!capturer) {
                    return AMD_ERROR_UNKNOWN;
                }

                Obs->queueNum++;
                queue_param.queue_priority = hqueue->priority;
                queue_param.hqueue = hqueue;
                queue_param.queue_sparsity = capturer->totalInvokeNum;

                camb_mutex_lock(&capturer->rls_lock);

                list_for_each_entry_safe (rls, n, &capturer->capturing_list, entry) { //遍历kernel

                    Obs->kernelNum++;
                    if (rls->type != rlsCapturerTaskType::RLS_CAPTURER_TASK_TYPE_InvokeKernel) {
                        kernel_param.hkernel = 0;
                        kernel_param.dimx = 1;
                        kernel_param.dimy = 1;
                        kernel_param.dimz = 1;
                        kernel_param.c = (KernelClass)1;
                        kernel_list.emplace_back(kernel_param);

                        continue;
                    }

                    data = (struct sbtsInvokeKernelCaptureData_st *)rls->data;
                    kernel_param.hkernel = data->hkernel;
                    kernel_param.dimx = data->dimx;
                    kernel_param.dimy = data->dimy;
		    // temporal modify
                    kernel_param.dimz = 1;
                    kernel_param.predict_time = data->dimz;
                    kernel_param.c = data->c;
                    kernel_list.emplace_back(kernel_param);
                }
		queue_param.kernel_num = kernel_list.size();
                camb_mutex_unlock(&capturer->rls_lock);

                tp = std::make_tuple(queue_param, kernel_list);
                Obs->info.Param_list.emplace_back(tp);
            }
            camb_mutex_unlock(&pcambpd->queueListDevice.lock);
        }
        camb_mutex_unlock(&ctx_mgr->context_lock);
    }

    if (Obs->queueNum != 0)
        Obs->current_queue = ((++Obs->current_queue) % Obs->queueNum);
    else
        Obs->current_queue = -1;

    return AMD_SUCCESS;
}

static inline float calc_elapsed_of_timeval(struct timeval *start, struct timeval *end)
{
    return ((end->tv_sec * 1000000.0 + end->tv_usec) -
            (start->tv_sec * 1000000.0 + start->tv_usec)) /
           1e3;
}

AMDresult AssignCaptureKernel(AMDqueue hqueue, int num)
{ // do action
    struct queueRLSCapturer_st *capturer = hqueue->rls_capturer;
    struct slCapturerInvokeData_st *rls, *n;
    struct timeval waitEnd;
    mluDev_t *device = ctxGetDeviceHandle(hqueue->context);
    int count = 0;
    AMDctxConfigParam ctxParam;
    AMDdev dev;
    int max_cluster_num;
    AMDDeviceGet(&dev, 0);

    if (!capturer || list_empty(&capturer->capturing_list) || num == 0) return AMD_ERROR_UNKNOWN;
    uint32_t default_block_dimx = __AMDGetDefaultBlockDimX();
    uint32_t default_block_dimy = __AMDGetDefaultBlockDimY();
    uint32_t default_block_dimz = __AMDGetDefaultBlockDimZ();

    list_for_each_entry_safe (rls, n, &capturer->capturing_list, entry) {
        if (rls->type == rlsCapturerTaskType::RLS_CAPTURER_TASK_TYPE_InvokeKernel) {
            struct sbtsInvokeKernelCaptureData_st *invoke_data = NULL;
            invoke_data = (struct sbtsInvokeKernelCaptureData_st *)rls->data;

            AMDCtxSetCurrent(hqueue->context);
            AMDDeviceGetAttribute(&max_cluster_num, AMD_DEVICE_ATTRIBUTE_MAX_CLUSTER_COUNT, dev);
            unsigned int bitmap_mask = (0x1U << max_cluster_num) - 1;
            // ctxParam.unionLimit = AMD_KERNEL_CLASS_UNION;
            ctxParam.unionLimit = AMD_KERNEL_CLASS_BLOCK;
            AMDSetCtxConfigParam(hqueue->context, AMD_CTX_CONFIG_UNION_LIMIT, &ctxParam);
            ctxParam.visibleCluster = invoke_data->strong_affinity;

            // ctxParam.visibleCluster = bitmap_mask & ((0x1U << VISIBLE_CLUSTER) - 1);
            AMDSetCtxConfigParam(hqueue->context, AMD_CTX_CONFIG_VISIBLE_CLUSTER, &ctxParam);
            // AMDGetCtxConfigParam(hqueue->context, AMD_CTX_CONFIG_VISIBLE_CLUSTER, &ctxParam);
            // printf("current context visible cluster is %#lx\n", ctxParam.visibleCluster);

            void *extra[] = {AMD_INVOKE_PARAM_BUFFER_POINTER, (void *)invoke_data->param_data,
                             AMD_INVOKE_PARAM_BUFFER_SIZE, (void *)invoke_data->param_size,
                             AMD_INVOKE_PARAM_END};

	    // temporal modify
            device->ops_sbts->InvokeKernel(invoke_data->hkernel, invoke_data->dimx,
                                           invoke_data->dimy, 1, default_block_dimx,
                                           default_block_dimy, default_block_dimz, invoke_data->c,
                                           invoke_data->reserve, hqueue, NULL, extra, invoke_data->tls);
            gettimeofday(&waitEnd, NULL);
            float waitTime = calc_elapsed_of_timeval(&invoke_data->waitStart, &waitEnd);
            capturer->totalWaitTime += waitTime;
            capturer->totalInvokeNum++;
            __list_del_entry(&rls->entry);
            count++;
        } else if (rls->type == rlsCapturerTaskType::RLS_CAPTURER_TASK_TYPE_Notifier) {
            struct invoke_extra_info extra_info;
	    extra_info.topo_info = 0;
            extra_info.dev_topo_id = 0;
            extra_info.dev_topo_node_index = 0;
            extra_info.dev_topo_cmd = DEV_TOPO_TASK_TYPE_NORMAL;
            extra_info.perf_disable = false;
            struct sbtsNotifierCaptureData_st *invoke_data = NULL;
            invoke_data = (struct sbtsNotifierCaptureData_st *)rls->data;
            sbtsPushPlaceNotifierTaskRLS(hqueue, invoke_data->hnotifier, &extra_info,
                                         invoke_data->inter_info);
            gettimeofday(&waitEnd, NULL);
            float waitTime = calc_elapsed_of_timeval(&invoke_data->waitStart, &waitEnd);
            capturer->totalWaitTime += waitTime;
            capturer->totalInvokeNum++;
            __list_del_entry(&rls->entry);
            count++;
        } else if (rls->type == rlsCapturerTaskType::RLS_CAPTURER_TASK_TYPE_MemcpyAsync) {
            struct sbtsMemcpyAsyncCaptureData_st *invoke_data = NULL;
            invoke_data = (struct sbtsMemcpyAsyncCaptureData_st *)rls->data;
            sbtsPushMemcpyAsyncRLS(hqueue, invoke_data->dst, invoke_data->src, invoke_data->bytes,
                                   invoke_data->dir);
            gettimeofday(&waitEnd, NULL);
            float waitTime = calc_elapsed_of_timeval(&invoke_data->waitStart, &waitEnd);
            capturer->totalWaitTime += waitTime;
            capturer->totalInvokeNum++;
            __list_del_entry(&rls->entry);
            count++;
        }

        if (count >= num || list_empty(&capturer->capturing_list)) break;
    }
    return AMD_SUCCESS;
}

AMDresult calcWaitTimeVariance(float *timeVariance)
{
    *timeVariance = 0;
    AMDresult res;
    AMDdev dev;
    mluDev_t *device;
    AMDContextManager *ctx_mgr;
    ContextHandler *Context, *CtxTemp;
    AMDqueue hqueue, temp;
    struct queueRLSCapturer_st *capturer;
    std::vector<float> queueWaitTime;
    float meanTime = 0.0f;
    float variance = 0.0f;

    for (int i = 0; i < 1; i++) {
        res = AMDDeviceGet(&dev, i);

        device = devGetDeviceHandle(dev);
        if (!device) {
            return AMD_ERROR_UNKNOWN;
        }
        ctx_mgr = GetContextManager(device);
        if (!ctx_mgr) {
            return AMD_ERROR_UNKNOWN;
        }

        list_for_each_entry_safe (Context, CtxTemp, &ctx_mgr->context, entry) {
            struct SbtsPrivData *pcambpd = (struct SbtsPrivData *)Context->priv_data_sbts;

            list_for_each_entry_safe (hqueue, temp, &pcambpd->queueListDevice.list, list_node) {
                capturer = hqueue->rls_capturer;
                if (!capturer) {
                    return AMD_ERROR_UNKNOWN;
                }
                if (capturer->totalWaitTime == 0 || capturer->totalInvokeNum == 0) continue;
                queueWaitTime.push_back(capturer->totalWaitTime / capturer->totalInvokeNum);
            }
        }
    }
    if (queueWaitTime.size() <=
        1) { // when there is only one sample, calculating variance is meaningless
        return AMD_SUCCESS;
    }

    for (float value : queueWaitTime) {
        meanTime += value;
    }
    meanTime /= queueWaitTime.size();

    for (float value : queueWaitTime) {
        float difference = value - meanTime;
        variance += difference * difference;
    }

    *timeVariance = variance / (queueWaitTime.size() - 1);

    return AMD_SUCCESS;
}

AMDresult calcWaitTimeCV(float *timeCV)
{
    *timeCV = 0;
    AMDresult res;
    AMDdev dev;
    mluDev_t *device;
    AMDContextManager *ctx_mgr;
    ContextHandler *Context, *CtxTemp;
    AMDqueue hqueue, temp;
    struct queueRLSCapturer_st *capturer;
    std::vector<float> queueWaitTime;
    float meanTime = 0.0f;
    float variance = 0.0f;

    for (int i = 0; i < 1; i++) {
        res = AMDDeviceGet(&dev, i);

        device = devGetDeviceHandle(dev);
        if (!device) {
            return AMD_ERROR_UNKNOWN;
        }
        ctx_mgr = GetContextManager(device);
        if (!ctx_mgr) {
            return AMD_ERROR_UNKNOWN;
        }

        list_for_each_entry_safe (Context, CtxTemp, &ctx_mgr->context, entry) {
            struct SbtsPrivData *pcambpd = (struct SbtsPrivData *)Context->priv_data_sbts;

            list_for_each_entry_safe (hqueue, temp, &pcambpd->queueListDevice.list, list_node) {
                capturer = hqueue->rls_capturer;
                if (!capturer) {
                    return AMD_ERROR_UNKNOWN;
                }
                if (capturer->totalWaitTime == 0 || capturer->totalInvokeNum == 0) continue;
                queueWaitTime.push_back(capturer->totalWaitTime / capturer->totalInvokeNum);
            }
        }
    }
    if (queueWaitTime.size() <=
        1) { // when there is only one sample, calculating variance is meaningless
        return AMD_SUCCESS;
    }

    for (float value : queueWaitTime) {
        meanTime += value;
    }
    meanTime /= queueWaitTime.size();

    for (float value : queueWaitTime) {
        float difference = value - meanTime;
        variance += difference * difference;
    }
    float epsilon = 1e-6;
    variance = variance / (queueWaitTime.size() - 1);
    *timeCV = std::sqrt(variance) / std::max(meanTime, epsilon);

    return AMD_SUCCESS;
}

AMDresult AMDGetIPUUtilization(int *averageUtil, int *coreUtilization, AMDdev dev)
{
    return GetIPUUtilization(averageUtil, coreUtilization, dev);
}
AMDresult AMDObserveState(void *RawObs)
{
    return ObserveState(static_cast<Observe_st *>(RawObs));
}
AMDresult AMDAssignCaptureKernel(AMDqueue hqueue, int num)
{
    return AssignCaptureKernel(hqueue, num);
}
