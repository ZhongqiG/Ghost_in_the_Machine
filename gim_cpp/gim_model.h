#ifndef GIM_MODEL_
#define GIM_MODEL_

#include "ap_fixed.h"
#include <cmath>
#include <algorithm>

#define ARRAY_SIZE 2
#define NUM_ITERATIONS 10000

typedef ap_fixed<16,7> fixed_16;

fixed_16* weights_pe(fixed_16 delta_k, fixed_16 output_kmin1, fixed_16 partial_sum_out_k, fixed_16 partial_sum_delta_k, fixed_16 init_weight, fixed_16 eta, fixed_16 training);

fixed_16* bias_pe(fixed_16 delta_k, fixed_16 sum_in, fixed_16 init_bias, fixed_16 eta, fixed_16 training);

fixed_16 act_pe(fixed_16 net_in, char model, fixed_16 alpha);
fixed_16 error_pe(fixed_16 output_kmin1, fixed_16 partial_sum_delta_k, char model, fixed_16 alpha);

fixed_16* array(fixed_16* weights, fixed_16* biases, fixed_16* delta_k, fixed_16 eta, char model, fixed_16 alpha, fixed_16 training);

fixed_16* accelerator(fixed_16 *w1, fixed_16 *w2, fixed_16 *bias_1, fixed_16 *bias_2, fixed_16 training);

#endif // GIM_MODEL_