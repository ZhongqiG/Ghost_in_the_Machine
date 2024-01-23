#include "gim_model.h"
#include <iostream>
#include <fstream>
using namespace std;

// now, we actually run the full model
fixed_16* accelerator(fixed_16 *w1, fixed_16 *w2, fixed_16 *bias_1, fixed_16 *bias_2, fixed_16 training) { 

    fixed_16 return_array[4];

    // initializing the data for the XOR problem
    fixed_16 x1[4] = {0, 0, 1, 1};
    fixed_16 x2[4] = {0, 1, 0, 1};
    fixed_16 y[4] = {0, 1, 1, 0};

    // setting up initial values for signals between layers
    fixed_16 output_kmin1[2] = {0, 0};
    fixed_16 delta_1[2] = {0, 0};
    fixed_16 delta_2[2] = {0, 0};

    // number of iterations defined in the header file

    // store actual and predicted difference in vector, set other params
    std::vector<fixed_16> inaccuracies;
    char model = 'r'; // s = sigmoid, r = relu, l = leaky relu
    fixed_16 alpha = 0.1; // for leaky relu
    fixed_16 lr = 0.1; // learning rate

    // iterate through the alloted epochs
    int i;
    for (i = 0; i < NUM_ITERATIONS; i++) {

        // initialize inaccuracy and error for this epoch
        fixed_16 inaccuracy = 0;
        fixed_16 square_error = 0;

        // iterate through all the data points
        int j;
        for (j = 0; j < 4; j++) {

            // setup the initial data input
            fixed_16 output_0[ARRAY_SIZE] = {x1[j], x2[j]};

            // initialize the error backpropagation
            delta_1 = {0, 0};
            delta_2 = {0, 0};

            // run the forward propagation
            fixed_16 *layer_1_forward = array(w1, bias_1, output_0, delta_1, lr, model, alpha, training);
            fixed_16 *output_1 = layer_1_forward[0];
            fixed_16 *layer_2_forward = array(w2, bias_2, output_1, delta_2, lr, model, alpha, training);
            fixed_16 *output_2 = layer_2_forward[0];

            if (output_2[0] > 0.5) {
                return_array[j] = 1;
            }
            else if (output_2[0] <= 0.5) {
                return_array[j] = 0;
            }
            
            // calculate the final error with mse' after the last output
            if (model == 's') {
                delta_2[0] = -(y[j] - output_2[0]) * output_2[0] * (1 - output_2[0]);
            }
            else if (model == "r") {
                if (output_2[0] > 0)
                    delta_2[0] = -(y[j] - output_2[0]);
                else
                    delta_2[0] = 0;
            }
            else if (model == "l") {
                if (output_2[0] > 0)
                    delta_2[0] = -(y[j] - output_2[0]);
                else
                    delta_2[0] = -(y[j] - output_2[0]) * alpha;
            }
            else {
                std::cout << "model invalid" << std::endl;
                break;
            }

            // calculate inaccuracy and error
            inaccuracy += abs(y[j] - output_2[0]);
            square_error += pow((y[j] - output_2[0]), 2);

            // run the backpropagation and update the array
            fixed_16 *layer_2_backward = array(w2, bias_2, output_1, delta_2, lr, model, alpha, training);
            fixed_16 *delta_1 = layer_2_backward[1];
            fixed_16 *weight_changes_2 = layer_2_backward[2];
            fixed_16 *bias_2_update = layer_2_backward[3];
            w2 = weight_changes_2;
            bias_2 = bias_2_update;
            fixed_16 *layer_1_backward = array(w1, bias_1, output_0, delta_1, lr, model, alpha, training);
            fixed_16 *delta_0 = layer_1_backward[1];
            fixed_16 *weight_changes_1 = layer_1_backward[2];
            fixed_16 *bias_1_update = layer_1_backward[3];
            w1 = weight_changes_1;
            bias_1 = bias_1_update;

            if ((training == 0) && (j == 3)) {
                break; // only run this for all 4 data points once if infering
            }
        }
        // store inaccuracy for model training reference
        innacuracies.push_back(inaccuracy);

        if (training == 0) {
            break; // only run this once if we are infering
        }
    }
    return return_array;
}