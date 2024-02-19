#ifndef GIM_MODEL_
#define GIM_MODEL_

#include "ap_fixed.h"
#include <cmath>
#include <algorithm>

#define ARRAY_SIZE 2
#define NUM_ITERATIONS 10000

typedef ap_fixed<16,7> fixed_16;

std::vector<fixed_16> weights_pe(fixed_16 delta_k, fixed_16 output_kmin1, fixed_16 partial_sum_out_k, fixed_16 partial_sum_delta_k, fixed_16 init_weight, fixed_16 eta, fixed_16 training);

std::vector<fixed_16> bias_pe(fixed_16 delta_k, fixed_16 sum_in, fixed_16 init_bias, fixed_16 eta, fixed_16 training);

fixed_16 act_pe(fixed_16 net_in, char model, fixed_16 alpha);
fixed_16 error_pe(fixed_16 output_kmin1, fixed_16 partial_sum_delta_k, char model, fixed_16 alpha);

std::vector<fixed_16> array(vector<fixed_16> weights, vector<fixed_16> biases, vector<fixed_16> output_kmin1,
                    vector<fixed_16> delta_k, fixed_16 eta, char model, fixed_16 alpha, fixed_16 training);

void accelerator(vector<fixed_16> w1, vector<fixed_16>  w2, vector<fixed_16>  bias_1, vector<fixed_16> bias_2, 
                fixed_16 training, vector<fixed_16> return_array);
#endif // GIM_MODEL_