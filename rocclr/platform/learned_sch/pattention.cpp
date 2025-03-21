#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/linear.hpp>
#include <mlpack/methods/ann/layer/relu.hpp>
#include <mlpack/methods/ann/layer/softmax.hpp>
#include <mlpack/methods/ann/layer/sigmoid.hpp>
#include <mlpack/methods/ann/layer/gelu.hpp>
#include <armadillo>
#include <iostream>

using namespace mlpack;
using namespace mlpack::ann;

// 定义Pattention层
class PAttention
{
public:
    PAttention(size_t inputSize, size_t outputSize, size_t paramTokenNum, const std::string& normType)
        : inputSize(inputSize),
          outputSize(outputSize),
          paramTokenNum(paramTokenNum),
          normalizationType(normType)
    {
        // 初始化参数token，key和value的线性变换
        keyParamTokens = arma::randn<arma::mat>(paramTokenNum, inputSize);
        valueParamTokens = arma::randn<arma::mat>(paramTokenNum, outputSize);
    }

    // 前向传播
    arma::mat Forward(const arma::mat& input)
    {
        // 计算注意力权重，基于输入和key的点积
        arma::mat attnWeights = input * keyParamTokens.t();

        // 应用归一化和非线性激活函数
        attnWeights = ApplyNormalization(attnWeights);

        // 计算加权和
        arma::mat output = attnWeights * valueParamTokens;

        return output;
    }

private:
    size_t inputSize;
    size_t outputSize;
    size_t paramTokenNum;
    std::string normalizationType;

    arma::mat keyParamTokens;
    arma::mat valueParamTokens;

    // 应用归一化和激活函数
    arma::mat ApplyNormalization(const arma::mat& attnWeights)
    {
        if (normalizationType == "gelu_l2_norm")
        {
            // GELU激活函数 + L2范数归一化
            arma::mat nonLinearOutputs = attnWeights;
            nonLinearOutputs = Gelu(nonLinearOutputs);  // GELU激活
            arma::mat normOutputs = nonLinearOutputs / arma::norm(nonLinearOutputs, "fro");
            return normOutputs;
        }
        else if (normalizationType == "l2_norm_gelu")
        {
            // L2范数归一化 + GELU激活函数
            arma::mat normOutputs = attnWeights / arma::norm(attnWeights, "fro");
            arma::mat nonLinearOutputs = Gelu(normOutputs);  // GELU激活
            return nonLinearOutputs;
        }
        else
        {
            // 默认返回输入
            return attnWeights;
        }
    }

    // GELU激活函数
    arma::mat Gelu(const arma::mat& input)
    {
        return 0.5 * input % (1.0 + arma::tanh(std::sqrt(2.0 / 3.1416) * (input + 0.044715 * arma::pow(input, 3))));
    }
};

int main()
{
    // 输入大小和输出大小
    size_t inputSize = 128;
    size_t outputSize = 128;
    size_t paramTokenNum = 8;  // 假设有8个参数token
    std::string normalizationType = "gelu_l2_norm";  // 归一化和激活类型

    // 初始化PAttention层
    PAttention patten(inputSize, outputSize, paramTokenNum, normalizationType);

    // 创建一个假输入
    arma::mat input = arma::randn<arma::mat>(10, inputSize);  // 10个样本，每个样本128维

    // 通过PAttention层进行前向传播
    arma::mat output = patten.Forward(input);

    // 输出结果
    std::cout << "Output: " << std::endl << output << std::endl;

    return 0;
}
