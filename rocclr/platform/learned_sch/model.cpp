#include "model.hpp"

using namespace mlpack;

/*
 * @param stateEmbeddingSize
 * 输入网络的state embedding的大小，比如输入的是一个64维的state
 * embedding，则inputSize=64。
 * @param hiddenNum 隐藏层数量。
 * @param hiddenOutSize
 * 隐藏层输出大小数组，比如hiddenNum==3，hiddenOutSize={64, 32,
 * 16}，则隐藏层输出的大小分别为64, 32,
 * 16，需要注意的是，模型最终的输出，即动作概率向量（action
 * vector)的大小等于hiddenOutSize[hiddenNum-1]。
 */
void LschModel::InitModel(size_t stateEmbeddingSize, int hiddenNum, int *hiddenOutSize)
{
    stateInputSize = stateEmbeddingSize;
    actionOutputSize = hiddenOutSize[hiddenNum - 1];
    // 初始化网络结构，每层网络使用LeakyReLU激活函数
    for (int i = 0; i < hiddenNum; i++) {
        ffn.Add(new mlpack::Linear(hiddenOutSize[i]));
        // ffn.Add<LeakyReLU>();
        ffn.Add<ReLU>();
    }
    // 最后一层网络使用SoftMax
    ffn.Add<Softmax>();
    ffn.Reset(stateEmbeddingSize);
    // 初始化Adam优化器的UpdatePolicy，需要注意的是，UpdatePolicy初始化时，需要知道FFN的具体参数大小，因此在之前调用ffn.Reset(...)。
    networkUpdatePolicy = new ens::AdamUpdate::template Policy<arma::mat, arma::mat>(
        networkUpdater, ffn.Parameters().n_rows, ffn.Parameters().n_cols);
}

/**
 * @brief 根据输入的state embedding，预测输出的动作概率向量（action vector）。
 *
 * @param stateEmbeddings 输入的state embeddings，state
 * embeddings是一个数组，由多个state
 * embedding拼接而成，其大小为(this->stateInputSize)*batchSize。
 * @param batchSize 输入的state embedding（列向量）的个数，也就是进行预测的数据点的个数。
 * @param copyTheData
 * 是否拷贝输入的stateEmbeddings，如果为true，则拷贝输入的stateEmbeddings，否则直接利用stateEmbeddings的存储空间执行计算。
 * @return 动作概率向量的数组，由多个动作概率向量
 */
void LschModel::predict(double *stateEmbeddings, std::vector<double> &result)
{
    arma::mat inputStates(stateEmbeddings, this->stateInputSize, 1, this->copyTheData, false);
    arma::mat output;
    ffn.Predict(inputStates, output);
    result.resize(output.n_elem);
    double *source = output.memptr();
    for (int i = 0; i < result.size(); i++) {
        result[i] = source[i];
    }
}

/**
 * @brief 更新模型参数。
 *
 * @param epsilon epsilon值，防止因为概率太小导致的梯度过大，可以设置为1e-6。
 * @param entropyWeight
 * 熵权重，权重越大，熵越大，越有利于探索，随着训练的进行，可以逐渐减小该值。
 * @param stateEmbeddings
 * 输入的state
 * embeddings数组，大小为(this->stateInputSize)*batchSize。
 * @param actionProps
 * 动作概率向量（action
 * vector）的数组，大小为(this->actionOutputSize)*batchSize。
 * @param advantages
 * 优势值向量的数组，大小为(this->actionOutputSize)*batchSize。
 * 优势值向量的每一个分量，代表着对应动作的优势值的相反数（-advantage=-Q(s,a)+V(s)），
 * 其数值上应该等于-1*（该动作的累计奖励-该状态的平均累计奖励）。
 * @param batchSize 等于state embedding的数量，等于动作概率向量的数量，等于优势值数组的数量，代表了此次更新使用的数据点的个量。
 * @param copyTheData
 * 是否复制数据，如果为true，则复制数据，否则直接利用输入的数据的内存空间进行计算。
 */
void LschModel::update(double *stateEmbeddings, int *actionIdx, double *advantages, int batchSize)
{
    arma::mat avg_lossGradient;

    for (int i = 0; i < batchSize; i++) {
        int act = actionIdx[i];
        arma::mat lossGradient(actionOutputSize, 1);
        arma::mat result_prob;
        arma::mat ffn_grad;
        arma::mat ffn_input =
            arma::mat(stateEmbeddings + i * stateInputSize, stateInputSize, 1, copyTheData, false);
        ffn.Forward(ffn_input, result_prob);
        lossGradient(act, 0) = -1 * advantages[i] / (epsilon + result_prob(act, 0));
	/*
        for (int j = 0; j < actionOutputSize; j++) {
           lossGradient(j, 0) = lossGradient(j, 0) - entropyWeight * (1.0 + arma::log(result_prob(j,
        0) + this->epsilon)); // -=? or +=?
        }
	*/
        ffn.Backward(ffn_input, lossGradient, ffn_grad);
        avg_lossGradient = (i == 0 ? ffn_grad : avg_lossGradient + ffn_grad);
    }
    avg_lossGradient /= batchSize;

    networkUpdatePolicy->Update(ffn.Parameters(), opStepSize, avg_lossGradient);
}

AMDresult AMDLschModelTest(int input_size)
{
    LschModel model(0.8, 0.01, 1e-6, 0.4, true);
    int hiddenOutSize[] = {64, 32, 16, 8};
    int hiddenNum = 4;
    int batch_size = 6;
    int action_dim = hiddenOutSize[hiddenNum - 1];
    int tmp1 = input_size * batch_size;
    int tmp2 = action_dim * batch_size;
    double *train_embedding = new double[tmp1];
    double *test_embedding = new double[tmp1];
    int *idx = new int[tmp2];
    double *adv = new double[tmp2];
    std::vector<double> result;
    int tmp;

    model.InitModel(input_size, hiddenNum, hiddenOutSize);

    for (int i = 0; i < tmp1; i++) {
        train_embedding[i] = static_cast<double>(rand() / RAND_MAX);
        test_embedding[i] = static_cast<double>(rand() / RAND_MAX);
    }

    for (int i = 0; i < tmp2; i++) {
        idx[i] = static_cast<int>(rand() / RAND_MAX);
        adv[i] = static_cast<double>(rand() / RAND_MAX);
    }

    model.update(train_embedding, idx, adv, batch_size);

    model.predict(test_embedding, result);

    delete[] train_embedding;
    delete[] test_embedding;
    delete[] idx;
    delete[] adv;

    return AMD_SUCCESS;
}

