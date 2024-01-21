#ifndef GIM_MODEL_
#define GIM_MODEL_

#include "ap_fixed.h"

#define ARRAY_SIZE 2
#define NUM_ITERATIONS 500

typedef ap_fixed<16,3> fixed_16;

fixed_16* weights_pe(fixed_16 delta_k, fixed_16 output_kmin1, fixed_16 partial_sum_out_k, fixed_16 partial_sum_delta_k, fixed_16 init_weight, fixed_16 eta);

fixed_16 bias_pe_net_sum(float init_bias, float sum_in);
fixed_16 bias_pe_bias_change(float init_bias, float delta_k, float eta);

fixed_16 act_pe(float net_in, char model, float alpha);
fixed_16 error_pe(float output_kmin1, float partial_sum_delta_k, char model, float alpha);

fixed_16* array_output_k(fixed_16* weights, fixed_16* biases, fixed_16* delta_k, fixed_16 eta, char model, fixed_16 alpha);

#endif // GIM_MODEL_