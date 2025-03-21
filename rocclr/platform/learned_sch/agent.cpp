#include "agent.hpp"

static inline float calc_elapsed_of_timeval(struct timeval *start, struct timeval *end)
{
    return ((end->tv_sec * 1000000.0 + end->tv_usec) -
            (start->tv_sec * 1000000.0 + start->tv_usec)) /
           1e3;
}

AMDresult Agent::preprocess(Observe_st raw_info, std::vector<float> &state)
{
    return StreamEmbedding(raw_info, state, queue_vision, kernel_vision);
}

void Agent::record_traj(Traj_Step_st traj_step)
{
    trajectorys.push_back(traj_step);
}

int Agent::random_hit()
{
    double epsilon;
    const int SOFT_MAX_STEP = 100;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    double random_rate = dis(gen);
    step++;
    epsilon = (step < 90 ? (step * 1.0 / SOFT_MAX_STEP) : 0.9);
    return (random_rate < epsilon ? 1 : 0);
}

void Agent::train_traj()
{
    step = 0;
    int batchSize = trajectorys.size();
    double *stateEmbeddings = (double *)malloc(batchSize * state_dim * sizeof(double));
    int *actionIdx = (int *)malloc(batchSize * sizeof(int));
    double *advantage = (double *)malloc(batchSize * sizeof(double));
    double R = 0;
    double ave_adv = 0;
    // double std_adv = 0;
    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < state_dim; j++) {
            stateEmbeddings[i * state_dim + j] = trajectorys[i].state[j];
        }
        actionIdx[i] = trajectorys[i].idx;
	/*
        R = trajectorys[batchSize - i - 1].reward + discount * R;
        advantage[batchSize - i - 1] = R;
	*/
        ave_adv += trajectorys[batchSize - i - 1].reward;
        // std_adv += (R * R);
    }
    ave_adv /= batchSize;
    // std_adv /= batchSize;
    // std_adv = sqrt(std_adv - ave_adv * ave_adv) + 1e-6;
    for (int i = 0; i < batchSize; i++) {
        // advantage[i] = (advantage[i] - ave_adv) / std_adv;
        advantage[i] = ave_adv;
    }

    model.update(stateEmbeddings, actionIdx, advantage, batchSize);

    free(stateEmbeddings);
    free(actionIdx);
    free(advantage);

    trajectorys.clear();
}

void Agent::get_action(std::vector<float> state, std::vector<int> &action, double &prob)
{
    action.resize(action_dim);
    fill(action.begin(), action.end(), 0);
    std::vector<double> result(action_dim);
    std::vector<double> embeddingstate(state.begin(), state.end());
    double *stateEmbeddings = &embeddingstate[0];
    model.predict(stateEmbeddings, result);
    // 该部分需要进一步适应mlpack的网络输出数据结构arma，有待进一步调整
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    double random_rate = dis(gen);
    double cumulate_prob = 0;

/*
    double summary = 0;
    for (int i = 0; i < action_dim; i++) {
        std::cout << result[i] << " ";
	summary+=result[i];
    }
    std::cout << summary << std::endl;
*/

    for (int i = 0; i < action_dim; i++) {
        cumulate_prob += result[i];
        if (random_rate <= cumulate_prob) {
            prob = result[i];
            action[i] = 1;
            break;
        }
    }
}

void Agent::do_action(AMDqueue hqueue, int num)
{
    AMDresult res = AssignCaptureKernel(hqueue, num);
}

void Agent::schedule(Observe_st raw_info)
{
    Traj_Step_st traj_step;
    int averageUtilization;
    int coreUtil[80];
    AMDresult res;
    float waitTimeVar;
    float epsilon = 1e-6;

    res = GetIPUUtilization(&averageUtilization, coreUtil, dev);

    averageUtilization *= (TOTAL_CLUSTER * 1.0 / VISIBLE_CLUSTER);
    if (averageUtilization < 0) {
        int launch_depth = std::get<1>(raw_info.info.Param_list[raw_info.current_queue]).size();
        // launch_depth = action_dim * 2;

        do_action((std::get<0>(raw_info.info.Param_list[raw_info.current_queue])).hqueue,
                  launch_depth);
        res = GetIPUUtilization(&averageUtilization, coreUtil, dev); //获取ipu利用率
        res = calcWaitTimeCV(&waitTimeVar);
        averageUtilization *= (TOTAL_CLUSTER * 1.0 / VISIBLE_CLUSTER);
        //std::cout << "this time ipuutil  is " << averageUtilization << std::endl;
        //std::cout << "wait time variance is " << waitTimeVar << std::endl;

    } else {
        struct timeval waitBegin, waitEnd0, waitEnd1, waitEnd2, waitEnd3;
	float waitTime0, waitTime1, waitTime2, waitTime3;
        gettimeofday(&waitBegin, NULL);
        res = preprocess(raw_info, traj_step.state);
        if (res) {
            return;
        }
        gettimeofday(&waitEnd0, NULL);

        get_action(traj_step.state, traj_step.action, traj_step.prob);
        gettimeofday(&waitEnd1, NULL);

        auto it = find(traj_step.action.begin(), traj_step.action.end(), 1);

        traj_step.idx = (it != traj_step.action.end() ? distance(traj_step.action.begin(), it) : 0);

        int launch_depth = traj_step.idx * 2;

        do_action((std::get<0>(raw_info.info.Param_list[raw_info.current_queue])).hqueue,
                  launch_depth);
        gettimeofday(&waitEnd2, NULL);

        int queue_kernel_num =
            (std::get<1>(raw_info.info.Param_list[raw_info.current_queue])).size();

        // todo: get utilization again

        res = GetIPUUtilization(&averageUtilization, coreUtil, dev); //获取ipu利用率

        res = calcWaitTimeCV(&waitTimeVar);
        waitTimeVar = (isnan(waitTimeVar) ? 1 : waitTimeVar);

        averageUtilization *= (TOTAL_CLUSTER * 1.0 / VISIBLE_CLUSTER);
        double ipu_reward = 0;
	if (averageUtilization < 0) {
	    ipu_reward = 10 * (averageUtilization / 100.0);
	} else {
            ipu_reward = 10 * std::log(averageUtilization / 100.0 + 1e-6);
	}
        double var_reward = 0;
	if (waitTimeVar < 0.01) {
	    var_reward = 5 + (0.01 - waitTimeVar) * 100;
	} else if (waitTimeVar < 0.15) {
	    var_reward = 9.0 / 140 / waitTimeVar - 10.0 / 7;
	} else {
	    var_reward = 10 * (0 - std::exp(waitTimeVar));
	}

        // traj_step.reward = static_cast<double>(averageUtilization / 100.0 - launch_depth +
        // 1/(waitTimeVar + epsilon));
	/*
        traj_step.reward =
            (queue_kernel_num < launch_depth
                 ? static_cast<double>(1 * (queue_kernel_num - launch_depth) + ipu_reward +
                                       0 * (1 / (waitTimeVar + epsilon) - 0))
                 : static_cast<double>(ipu_reward + 0 * (1 / (waitTimeVar + epsilon) - 0)));
	*/
        traj_step.reward =
            (queue_kernel_num < launch_depth
                 ? static_cast<double>((queue_kernel_num - launch_depth) + ipu_reward + var_reward)
                 : static_cast<double>(ipu_reward + var_reward));
        traj_step.done = random_hit();

        record_traj(traj_step);
        gettimeofday(&waitEnd3, NULL);

        std::cout << "this time depth   is " << launch_depth << std::endl;
        std::cout << "this time reward   is " << traj_step.reward << std::endl;
        std::cout << "this time ipu reward   is " << ipu_reward << std::endl;
        std::cout << "this time var reward   is " << var_reward << std::endl;
        std::cout << "this time ipuutil  is " << averageUtilization << std::endl;
        std::cout << "wait time variance is " << waitTimeVar << std::endl;
	waitTime0 = calc_elapsed_of_timeval(&waitBegin, &waitEnd0);
	std::cout << "OverHead: preprocess time:" << waitTime0 << std::endl;
	waitTime1 = calc_elapsed_of_timeval(&waitBegin, &waitEnd1);
	std::cout << "OverHead: inference time:" << waitTime1 - waitTime0 << std::endl;
	waitTime2 = calc_elapsed_of_timeval(&waitBegin, &waitEnd2);
	std::cout << "OverHead: invoke time:" << waitTime2 - waitTime1 << std::endl;
	waitTime3 = calc_elapsed_of_timeval(&waitBegin, &waitEnd3);
	std::cout << "OverHead: obtainReward time:" << waitTime3 - waitTime2 << std::endl;
        // todo:if cpu support, multithread
        if (traj_step.done) {
            gettimeofday(&waitBegin, NULL);
            train_traj();
            gettimeofday(&waitEnd1, NULL);
	    waitTime0 = calc_elapsed_of_timeval(&waitBegin, &waitEnd0);
	    std::cout << "OverHead: train time:" << waitTime0 << std::endl;
        }
    }
}

