#include "gim_model.h"

// error pe function

float error_pe(float output_kmin1, float partial_sum_delta_k, char model, float alpha) {
    float error = 0;
    // if (model == 's'){
    //     error = output_kmin1 * (1 - output_kmin1) * partial_sum_delta_k;
    // }
    if (model == 'r'){
        if (output_kmin1 > 0){
            error = partial_sum_delta_k;
        }
        else {
            error = 0;
        }
    }
    else if (model == 'l'){
        if (output_kmin1 > 0){
            error = partial_sum_delta_k;
        }
        else {
            error = alpha * partial_sum_delta_k;
        }
    }
    else {
        //std::cout << "Error: Invalid Model Name" << std::endl;
        error = 0;
    }
    return error;
}
