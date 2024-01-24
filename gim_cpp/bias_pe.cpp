#include "gim_model.h"

// bias pe definition

fixed_16* bias_pe(fixed_16 delta_k, fixed_16 sum_in, fixed_16 init_bias, fixed_16 eta, fixed_16 training) {
    fixed_16 net_sum = init_bias + sum_in;
    fixed_16 bias_change = init_bias - (delta_k * eta);
    fixed_16 return_output[2];
    if (training == 0) {
        return_output[0] = 0;
        return_output[1] = 0;
    }
    else {
        return_output[0] = net_sum;
        return_output[1] = bias_change;
    }
    return return_output;
}