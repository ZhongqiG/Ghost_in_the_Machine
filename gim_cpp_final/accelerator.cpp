#include "gim_model.h"
#include <iostream>
#include <fstream>
using namespace std;

// now, we actually run the full model
Inference accelerator(fixed_16 w1[ARRAY_SIZE][ARRAY_SIZE], fixed_16 w2[ARRAY_SIZE][ARRAY_SIZE],
				fixed_16  bias_1[ARRAY_SIZE], fixed_16 bias_2[ARRAY_SIZE],
                fixed_16 training) {

    // array for the final output
    Inference output_array;

    // initializing the data for the XOR problem
    fixed_16 x1[4] = {0, 0, 1, 1};
    fixed_16 x2[4] = {0, 1, 0, 1};
    fixed_16 y[4] = {0, 1, 1, 0};
    // setting up initial values for signals between layers
    fixed_16 output_kmin1[2] = {0, 0};

    // initializing internal arrays with zeros
    fixed_16 delta_2[ARRAY_SIZE] = {0, 0};
    fixed_16 output_back1[ARRAY_SIZE] = {0, 0};
    fixed_16 delta_1[ARRAY_SIZE] = {0, 0};
    fixed_16 weight_changes_2[ARRAY_SIZE][ARRAY_SIZE] = {{0, 0}, {0, 0}};
    fixed_16 bias_2_update[ARRAY_SIZE] = {0, 0};

    fixed_16 output_back2[ARRAY_SIZE] = {0, 0};
    fixed_16 delta_0[ARRAY_SIZE] = {0, 0};
    fixed_16 weight_changes_1[ARRAY_SIZE][ARRAY_SIZE] = {{0, 0}, {0, 0}};
    fixed_16 bias_1_update[ARRAY_SIZE] = {0, 0};

    fixed_16 output_0[ARRAY_SIZE] = {0, 0};
    fixed_16 output_1[ARRAY_SIZE] = {0, 0};
    fixed_16 output_2[ARRAY_SIZE] = {0, 0};

    // dummy arrays used to capture unused outputs
    fixed_16 dummy1[ARRAY_SIZE];
    fixed_16 dummy2[ARRAY_SIZE][ARRAY_SIZE];
    fixed_16 dummy3[ARRAY_SIZE];

    // make local versions of the weights/biases
    fixed_16 w1_local[ARRAY_SIZE][ARRAY_SIZE] = {{0, 0}, {0, 0}};
    fixed_16 w2_local[ARRAY_SIZE][ARRAY_SIZE] = {{0, 0}, {0, 0}};
    fixed_16 bias_1_local[ARRAY_SIZE] = {0, 0};
    fixed_16 bias_2_local[ARRAY_SIZE] = {0, 0};
    for (int n = 0; n<ARRAY_SIZE; n++) {
        bias_1_local[n] = bias_1[n];
        bias_2_local[n] = bias_2[n];
        for (int m = 0;m<ARRAY_SIZE; m++) {
            w1_local[n][m] = w1[n][m];
            w2_local[n][m] = w2[n][m];
        }
    }

    // number of iterations defined in the header file

    // store actual and predicted difference in vector, set other params
    char model = 'r'; // s = sigmoid, r = relu, l = leaky relu NOTE: SIGMOID CANNOT BE USED ON HARDWARE
    fixed_16 alpha = 0.1; // for leaky relu
    fixed_16 lr = 0.1; // learning rate

    // iterate through the alloted epochs
    int i;
    for (i = 0; i < NUM_ITERATIONS; i++) {

        // iterate through all the data points
        int j;
        for (j = 0; j < 4; j++) {

            // setup the initial data input
            output_0[0] = x1[j];
            output_0[1] = x2[j];

            // initialize the error backpropagationcout
            delta_1[0] = 0;
            delta_1[1] = 0; 
            delta_2[0] = 0; 
            delta_2[1] = 0;

            // run the forward propagation
            // start with layer 1
            Array array_out1 = model_array(w1_local, bias_1_local, output_0, delta_1, lr, model, alpha, training);
            output_1[0] = array_out1.output_k[0];
            output_1[1] = array_out1.output_k[1];

            // then layer two
            Array array_out2 = model_array(w2_local, bias_2_local, output_1, delta_2, lr, model, alpha, training);
            output_2[0] = array_out2.output_k[0];
            output_2[1] = array_out2.output_k[1];

            // make inferences for the return array if training has completed
            if (output_2[0] > 0.5) {
                output_array.inference[j] = 1;
            }
            else if (output_2[0] <= 0.5) {
                output_array.inference[j] = 0;
            }
            
            // lastly calculate the final error with the derivative of mse after the last output
            if (model == 's') {
                delta_2[0] = -(y[j] - output_2[0]) * output_2[0] * (1 - output_2[0]);
            }
            else if (model == 'r') {
                if (output_2[0] > 0)
                    delta_2[0] = -(y[j] - output_2[0]);
                else
                    delta_2[0] = 0;
            }
            else if (model == 'l') {
                if (output_2[0] > 0)
                    delta_2[0] = -(y[j] - output_2[0]);
                else
                    delta_2[0] = -(y[j] - output_2[0]) * alpha;
            }
            else {
                // std::cout << "model invalid" << std::endl;
                break;
            }

            // run the backpropagation and update the array
            // start with layer 2
            Array array_back2 = model_array(w2_local, bias_2_local, output_1, delta_2, lr, model, alpha, training);
            delta_1[0] = array_back2.delta_kmin1[0];
            delta_1[1] = array_back2.delta_kmin1[1];
            // update the weights and biases
            for (int n = 0; n<ARRAY_SIZE; n++) {
                bias_2_local[n] = array_back2.bias_change[n];
                for (int m = 0;m<ARRAY_SIZE; m++) {
                    w2_local[n][m] = array_back2.weight_changes[n][m];
                }
            }
            // end with layer 1
            Array array_back1 = model_array(w1_local, bias_1_local, output_0, delta_1, lr, model, alpha, training);
            // update the weights and biases
            for (int n = 0; n<ARRAY_SIZE; n++) {
                bias_1_local[n] = array_back1.bias_change[n];
                for (int m = 0; m<ARRAY_SIZE; m++) {
                    w1_local[n][m] = array_back1.weight_changes[n][m];
                }
            }

            if ((training == 0) && (j == 3)) {
                break; // only run this for all 4 data points once if infering
            }
        }
        // store inaccuracy for model training reference
        // cout << "i" << inaccuracy << endl;

        if (training == 0) {
            break; // only run this once if we are infering
        }
    }

    // produce the final weights to be used in inference
    for (int n = 0; n<ARRAY_SIZE; n++) {
        output_array.new_b1[n] = bias_1_local[n];
        output_array.new_b2[n] = bias_2_local[n];
        for (int m = 0;m<ARRAY_SIZE; m++) {
            output_array.new_w1[n][m] = w1_local[n][m];
            output_array.new_w2[n][m] = w2_local[n][m];
        }
    }

    return output_array;
}
