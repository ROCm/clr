#ifndef __SBTS__LEARNED_SCH_ENV_H__
#define __SBTS__LEARNED_SCH_ENV_H__

#include <vector>
#include <tuple>
#include <map>
#include <iostream>
#include "AMD_api.h"

#define TOTAL_CLUSTER 6
#define VISIBLE_CLUSTER 6

typedef struct {
    AMDqueue hqueue;
    int kernel_num;
    int queue_sparsity;
    int queue_priority;
} Queue_st;

typedef struct {
    AMDkernel hkernel;
    unsigned int dimx;
    unsigned int dimy;
    unsigned int dimz;
    int predict_time;  //actually size
    KernelClass c;
    Queue_st queueInfo;
} Kernel_st;

typedef std::vector<std::tuple<Queue_st, std::vector<Kernel_st>>> Param_vec;

typedef struct raw_info {
    int averageCoreUtilization;
    Param_vec Param_list;
} Raw_Info_st;

typedef struct observation {
    Raw_Info_st info;
    int kernelNum;
    int queueNum;
    int current_queue;
} Observe_st;

AMDresult GetIPUUtilization(int *averageUtil, int *coreUtil, AMDdev dev);
AMDresult ObserveState(Observe_st *Obs);
AMDresult AssignCaptureKernel(AMDqueue hqueue, int num);
AMDresult calcWaitTimeVariance(float *timeVariance);
AMDresult calcWaitTimeCV(float *timeCV);

#endif /*__SBTS__LEARNED_SCH_ENV_H__*/
