#include <stdio.h>
#include <math.h>
#include "gim_model.h"

// block of bias pe functions (split up to reduce issues with returning arrays)
float bias_pe_net_sum(float init_bias, float sum_in){
    float net_sum;
    return net_sum = init_bias + sum_in;
}

float bias_pe_bias_change(float init_bias, float delta_k, float eta){
    float bias_change;
    return bias_change = bias_change = init_bias - delta_k * eta;
}