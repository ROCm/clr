#ifndef __SBTS__LEARNED_SCH_MODEL_H__
#define __SBTS__LEARNED_SCH_MODEL_H__

#include <cstddef>
#include <cstring>
#include <mlpack/core.hpp>
#include <ensmallen.hpp>
#include <mlpack/methods/ann/ann.hpp>
#include <vector>
#include <stdlib.h>
#include <string>
#include "AMD_api.h"

class LschModel {
  private:
    // 用来进行推理的FFN
    mlpack::FFN<mlpack::EmptyLoss, mlpack::GlorotInitializationType<false>> ffn;
    // 更新FFN的optimizer
    ens::AdamUpdate networkUpdater;
    // optimizer的UpdatePolicy
    typename ens::AdamUpdate::template Policy<arma::mat, arma::mat> *networkUpdatePolicy;
    // 折扣因子
    double discountFactor;
    // 步长
    double opStepSize;
    // 输入的state Embedding大小
    size_t stateInputSize;
    // 输出的action vector的大小
    size_t actionOutputSize;
    double epsilon;
    double entropyWeight;
    bool copyTheData;

  public:
    LschModel(double discount, double stepSize, double epsilon, double entropyWeight,
              bool copyTheData)
        : ffn((mlpack::EmptyLoss()), (mlpack::GlorotInitializationType<false>())),
          networkUpdater(),
          discountFactor(discount),
          opStepSize(stepSize),
          epsilon(epsilon),
          entropyWeight(entropyWeight),
          copyTheData(copyTheData)
    {
    }
    ~LschModel()
    {
        delete networkUpdatePolicy;
    }
    void InitModel(size_t stateEmbeddingSize, int hiddenNum, int *hiddenOutSize);
    void predict(double *stateEmbeddings, std::vector<double> &result);
    void update(double *stateEmbeddings, int *actionIdx, double *advantages, int batchSize);
};

#endif /*__SBTS__LEARNED_SCH_MODEL_H__*/

