#include "gim_model.h"
using namespace std;

// block of array functions (split up to reduce issues with returning arrays)
vector<fixed_16> array(vector<fixed_16> weights, vector<fixed_16> biases, vector<fixed_16> output_kmin1,
                    vector<fixed_16> delta_k, fixed_16 eta, char model, fixed_16 alpha, fixed_16 training) {
    
    vector<fixed_16> return_output;
    vector<fixed_16> output_k(ARRAY_SIZE);
    vector<fixed_16> delta_kmin1(ARRAY_SIZE);
    vector<fixed_16> weight_changes(ARRAY_SIZE, vector<fixed_16>(ARRAY_SIZE));
    vector<fixed_16> bias_changes(ARRAY_SIZE);
    vector<fixed_16> partial_delta_sum(ARRAY_SIZE);

    int n = 0;
    for (n = 0; n < ARRAY_SIZE; n++) {
        fixed_16 partial_output_sum = 0;
        int c = 0;
        for (c = 0; c < ARRAY_SIZE; c++) {
            vector<fixed_16> weight_result = weights_pe(delta_k[n], output_kmin1[c], partial_output_sum, partial_delta_sum[c], weights[n][c], eta, training);
            partial_delta_sum[c] = weight_result[0];
            partial_output_sum = weight_result[1];
            weight_changes[n][c] = weight_result[2];
        }
        vector<fixed_16> bias_result = bias_pe(delta_k[n], partial_output_sum, biases[n], eta, training);
        fixed_16 net_sum = bias_result[0];
        bias_changes[n] = bias_result[1];

        output_k[n] = act_pe(net_sum, model, alpha);
    }
    int j = 0;
    for (j = 0; j < ARRAY_SIZE; j++) {
        if (training == 0) 
            delta_kmin1[j] = 0;
        else
            delta_kmin1[j] = error_pe(output_kmin1[j], partial_delta_sum[j], model, alpha);
    }
    
    return_output.push_back(output_k);
    return_output.push_back(delta_kmin1);
    return_output.push_back(weight_changes);
    return_output.push_back(bias_changes);

    return return_array;
}