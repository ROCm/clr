#ifndef __SBTS__LEARNED_SCH_AGENT_H__
#define __SBTS__LEARNED_SCH_AGENT_H__

#include <vector>
#include <mlpack/core.hpp>
#include <random>
#include <cmath>
#include "preprocess.hpp"
#include "env.hpp"
#include "model.hpp"
#include "rls_capturer.hpp"
#include "AMD_api.h"

struct Traj_Step_st {
    int done;
    double reward;
    std::vector<float> state;
    std::vector<int> action;
    double prob;
    int idx;
};

class Agent {
  private:
    std::vector<Traj_Step_st> trajectorys;
    LschModel model;
    int step;
    int action_dim;
    int state_dim;
    float discount;

    int queue_vision;
    int kernel_vision;
    AMDdev dev;

  public:
    Agent(Observe_st raw_info)
        : step(0),
          discount(0.8),
          queue_vision(2),
          kernel_vision(10),
          model(0.8, 0.001, 1e-6, 0.4, true)
    {
        std::vector<float> state;
        int hiddenOutSize[] = {128, 64, 32, 16, 8};
        int hiddenNum = 4;
        preprocess(raw_info, state); // TODO: deal with state dim when kenrelNum = 0
        // state_dim = 10; //init state size
        state_dim = 3 + (queue_vision + 1) * (3 + (kernel_vision + 1) * 3); // init state size
        action_dim = hiddenOutSize[hiddenNum - 1];
        model.InitModel(state_dim, hiddenNum, hiddenOutSize);
        AMDDeviceGet(&dev, 0);
    }
    ~Agent()
    {
    }

  private:
    AMDresult preprocess(Observe_st raw_info, std::vector<float>& state);

    void record_traj(Traj_Step_st traj_step);

    int random_hit();

    void train_traj();

    void get_action(std::vector<float> state, std::vector<int>& action, double& prob);

    void do_action(AMDqueue hqueue, int num);

  public:
    void schedule(Observe_st raw_info);
};

#endif /*__SBTS__LEARNED_SCH_AGENT_H__*/
