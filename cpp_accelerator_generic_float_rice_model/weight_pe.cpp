#include "gim_model.h"

// weight pe function

Weight weights_pe(float delta_k, float output_kmin1, float partial_sum_out_k,
				float partial_sum_delta_k, float init_weight,
				float eta, float training) {

	Weight return_array;

	// perform the operations of the weight pe as presented by Ray Simar
    return_array.sum_delta_out = partial_sum_delta_k + (delta_k * init_weight);
    return_array.sum_output_out = partial_sum_out_k + (output_kmin1 * init_weight);
    return_array.weight_change = init_weight - (output_kmin1 * delta_k * eta);

	return return_array;
}
