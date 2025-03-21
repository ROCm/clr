#ifndef __SBTS__LEARNED_SCH_PREPROCESS_H__
#define __SBTS__LEARNED_SCH_PREPROCESS_H__

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <cstring>
#include <algorithm>
#include "env.hpp"
#include "AMD_api.h"

AMDresult StreamEmbedding(Observe_st Obs, std::vector<float> &state, int queue_vision,
                         int kernel_vision);

#endif /*__SBTS__LEARNED_SCH_PREPROCESS_H__*/

