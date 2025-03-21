#include "preprocess.hpp"

void generate_weight_attension(int idx, int length, std::vector<float> &weight)
{
    weight.resize(length);
    fill(weight.begin(), weight.end(), 1);

    for (int i = 0; i < length; i++) {
        int id = (idx + i < length ? idx + i : (idx + i) % length);
        weight[id] = pow(2, -i);
    }
}

void leak_filling(Param_vec &Param_list, int queue_vision, int kernel_vision, int idx)
{
    Param_vec tmp_list;
    Kernel_st kernel_fill_sample;
    Queue_st queue_fill;
    std::tuple<Queue_st, std::vector<Kernel_st>> queue_fill_sample;
    Param_vec queues_fill;
    if (Param_list.size() <= queue_vision) {
        queues_fill.resize(queue_vision - Param_list.size() + 1);
    }

    for (auto &element : Param_list) {
        std::vector<Kernel_st> *kernels_info = &(std::get<1>(element));
        if (kernels_info->size() <= kernel_vision) {
            std::vector<Kernel_st> kernels_fill(kernel_vision - kernels_info->size() + 1);
            memset(&kernel_fill_sample, 0, sizeof(Kernel_st));
            fill(kernels_fill.begin(), kernels_fill.end(), kernel_fill_sample);
            kernels_info->insert(kernels_info->end(), kernels_fill.begin(), kernels_fill.end());
        }
    }

    for (int i = 0; i < Param_list.size(); i++) {
        int id = (idx + i < Param_list.size() ? idx + i : (idx + i) % Param_list.size());
        tmp_list.emplace_back(Param_list[id]);
    }

    Param_list = tmp_list;

    if (Param_list.size() <= queue_vision) {
        std::vector<Kernel_st> kernels_fill(kernel_vision + 1);
        memset(&kernel_fill_sample, 0, sizeof(Kernel_st));
        memset(&queue_fill, 0, sizeof(Queue_st));

        fill(kernels_fill.begin(), kernels_fill.end(), kernel_fill_sample);

        queue_fill_sample = make_tuple(queue_fill, kernels_fill);

        fill(queues_fill.begin(), queues_fill.end(), queue_fill_sample);

        Param_list.insert(Param_list.end(), queues_fill.begin(), queues_fill.end());
    }
}

AMDresult StreamEmbedding(Observe_st Obs, std::vector<float> &state, int queue_vision,
                         int kernel_vision)
{
    std::vector<float> queue_weight;
    std::vector<float> queue_feature;
    std::vector<float> kernel_weight;
    std::vector<float> kernel_feature;
    int tmp;

    if (Obs.kernelNum == 0) {
        return AMD_ERROR_UNKNOWN;
    }

    Param_vec Param_list = Obs.info.Param_list;

    generate_weight_attension(Obs.current_queue, Obs.queueNum, queue_weight);

    leak_filling(Param_list, queue_vision, kernel_vision, Obs.current_queue);

    if (Obs.queueNum <= queue_vision) {
        std::fill(queue_weight.end(), queue_weight.end() + queue_vision - Obs.queueNum + 1, 0);
    }

    state.push_back(static_cast<float>(Obs.kernelNum));
    state.push_back(static_cast<float>(Obs.queueNum));
    state.push_back(static_cast<float>(Obs.info.averageCoreUtilization));

    for (const auto &element : Param_list) {
        int idx = &element - &Param_list[0];
        queue_feature.clear();
        queue_feature.push_back(static_cast<float>((std::get<0>(element)).queue_priority));
        queue_feature.push_back(static_cast<float>((std::get<0>(element)).queue_sparsity));
        queue_feature.push_back(static_cast<float>((std::get<0>(element)).kernel_num));
        std::vector<Kernel_st> kernels_info = std::get<1>(element);
        kernel_weight.clear();
        generate_weight_attension(0, kernels_info.size(), kernel_weight);

        for (int i = 0; i < kernels_info.size(); i++) {
            kernel_feature.clear();
            kernel_feature.push_back(static_cast<float>(kernels_info[i].dimx *
		kernels_info[i].dimy * kernels_info[i].dimz));
            kernel_feature.push_back(static_cast<float>(kernels_info[i].c));
	    kernel_feature.push_back(static_cast<float>(kernels_info[i].predict_time));

            for (int j = 0; j < kernel_feature.size(); j++) {
                kernel_feature[j] *= kernel_weight[i];
            }

            if (i > kernel_vision) {
                tmp = queue_feature.size() - kernel_feature.size();

                for (int j = 0; j < kernel_feature.size(); j++) {
                    queue_feature[tmp + j] += kernel_feature[j];
                }
            } else {
                queue_feature.insert(queue_feature.end(), kernel_feature.begin(),
                                     kernel_feature.end());
            }
        }

        for (int i = 0; i < queue_feature.size(); i++) {
            queue_feature[i] *= queue_weight[idx];
        }

        if (idx > queue_vision) {
            tmp = state.size() - queue_feature.size();

            for (int i = 0; i < queue_feature.size(); i++) {
                state[tmp + i] += queue_feature[i];
            }
        } else {
            state.insert(state.end(), queue_feature.begin(), queue_feature.end());
        }
    }

    return AMD_SUCCESS;
}

