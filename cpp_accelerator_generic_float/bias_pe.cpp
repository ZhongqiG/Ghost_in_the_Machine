#include "gim_model.h"

// bias pe definition

Bias bias_pe(float delta_k,
				float sum_in,
				float init_bias,
				float eta,
				float training) {

	Bias return_array;

	// perform the operations of the bias pe as presented by Ray Simar
    return_array.net_sum = init_bias + sum_in;
    return_array.bias_change = init_bias - (delta_k * eta);

	return return_array;

}
