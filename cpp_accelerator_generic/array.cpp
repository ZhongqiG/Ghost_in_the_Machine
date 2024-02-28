#include "gim_model.h"
using namespace std;

Array model_array(fixed_16 weights[ARRAY_SIZE][ARRAY_SIZE],
			fixed_16 biases[ARRAY_SIZE],
			fixed_16 output_kmin1[ARRAY_SIZE],
			fixed_16 delta_k[ARRAY_SIZE], fixed_16 eta,
			char model, fixed_16 alpha, fixed_16 training) {

    Array return_array;
    
    // initialize internal array with zeros
    fixed_16 partial_delta_sum[ARRAY_SIZE] = {};

    // setting up an additional net array for softmax
    SoftMax net;

    // iterate through the neurons in the layer
    int n = 0;
    for (n = 0; n < ARRAY_SIZE; n++) {
        // initialize the running output sum
        fixed_16 partial_output_sum = 0;
        int c = 0;
        // iterate through the columns of the current layer
        for (c = 0; c < ARRAY_SIZE; c++) {
            // get the running sums for the output and the delta from the current weight pe
            Weight weight_out = weights_pe(delta_k[n], output_kmin1[c], partial_output_sum,
            		partial_delta_sum[c], weights[n][c], eta, training);
            partial_output_sum = weight_out.sum_output_out;
            partial_delta_sum[c] = weight_out.sum_delta_out;
            return_array.weight_changes[n][c] = weight_out.weight_change;
        }
    
        // get the output for the current neuron in the layer
        Bias bias_out = bias_pe(delta_k[n], partial_output_sum, biases[n], eta, training);
        return_array.bias_change[n] = bias_out.bias_change;
        net.out_vector[n] = bias_out.net_sum;
        if (model != 'm')
            return_array.output_k[n] = act_pe(net.out_vector[n], model, alpha);
    }
    if (model == 'm')
        SoftMax prediction_vector = softmax(net);

    for (int m; m < ARRAY_SIZE; m++) {
        return_array.output_k[m] = prediction_vector.out_vector[n];
    }
    // get the delta signal for the previous layer using the error pe
    int j = 0;
    for (j = 0; j < ARRAY_SIZE; j++) {
        if (training == 0) 
            return_array.delta_kmin1[j] = 0;
        else
            return_array.delta_kmin1[j] = error_pe(output_kmin1[j], partial_delta_sum[j], model, alpha);
    }

    return return_array;
}
