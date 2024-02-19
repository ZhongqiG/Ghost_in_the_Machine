#include "gim_model.h"

// weight pe function, returns as vector instead of a list in the python code
std::vector<fixed_16> weights_pe(fixed_16 delta_k, fixed_16 output_kmin1, fixed_16 partial_sum_out_k, fixed_16 partial_sum_delta_k, fixed_16 init_weight, fixed_16 eta, fixed_16 training) {
    std::vector<fixed_16> return_output;
    fixed_16 sum_delta_out = partial_sum_delta_k + (delta_k * init_weight);
    fixed_16 sum_output_out = partial_sum_out_k + (output_kmin1 * init_weight);
    fixed_16 weight_change = init_weight - (output_kmin1 * delta_k * eta);
    if (training == 0) {
        return_output.push_back(0);
        return_output.push_back(0);
        return_output.push_back(0);
    }
    else {
        return_output.push_back(sum_delta_out);
        return_output.push_back(sum_output_out);
        return_output.push_back(weight_change);
    }
    return return_output;
}