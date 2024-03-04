// defines the softmax function for large classification problems
#include "gim_model.h"

SoftMax softmax(SoftMax output_array) {
    SoftMax e;
    float sum = 0;
    float max = std::max_element(output_array.out_vector[0], output_array.out_vector[ARRAY_SIZE]);
    for (int i = 0; i < ARRAY_SIZE; i++) {
        e.out_vector[i] = std::exp(output_array[i] - max);
        sum = sum + e.out_vector[i];
    }
    for (int j = 0; j < ARRAY_SIZE; j++) {
        e.out_vector[j] = e.out_vector[j] / sum;
    }
    return e
}