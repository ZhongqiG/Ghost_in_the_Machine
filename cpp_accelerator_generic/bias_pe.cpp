#include "gim_model.h"

// bias pe definition

Bias bias_pe(fixed_16 delta_k,
				fixed_16 sum_in,
				fixed_16 init_bias,
				fixed_16 eta,
				fixed_16 training) {

	Bias return_array;

	// perform the operations of the bias pe as presented by Ray Simar
    return_array.net_sum = init_bias + sum_in;
    return_array.bias_change = init_bias - (delta_k * eta);

	return return_array;

}
