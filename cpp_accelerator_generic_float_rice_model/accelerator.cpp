#include "gim_model.h"
#include <iostream>
#include <fstream>
using namespace std;

// now, we actually run the full model
Inference accelerator(float w1[ARRAY_SIZE][ARRAY_SIZE], float w2[ARRAY_SIZE][ARRAY_SIZE],
				float  bias_1[ARRAY_SIZE], float bias_2[ARRAY_SIZE], float output_inference[DATA_SIZE], Data input_data,
                float training) {

    // array for the final output
    Inference output_array;

    // initializing the data for the XOR problem
    // this is where some work could be done to input larger datasets
    // float data[2][4] = {{0, 0, 1, 1},
    //                     {0, 1, 0, 1}};
    // float y[4] = {0, 1, 1, 0};

    // initialize data here for the rice model using Zhongqi's code
    float data[DATA_SIZE][7] = intput_data.data;
    float y[DATA_SIZE] = input_data.labels;

    // this is used for softmax
    // float train_labels_one_hot[];

    // setting up initial values for signals between layers
    float output_kmin1[ARRAY_SIZE] = {};

    // initializing internal arrays with zeros
    float delta_2[ARRAY_SIZE] = {};
    float output_back1[ARRAY_SIZE] = {};
    float delta_1[ARRAY_SIZE] = {};
    float weight_changes_2[ARRAY_SIZE][ARRAY_SIZE] = {};
    float bias_2_update[ARRAY_SIZE] = {};

    float output_back2[ARRAY_SIZE] = {};
    float delta_0[ARRAY_SIZE] = {};
    float weight_changes_1[ARRAY_SIZE][ARRAY_SIZE] = {};
    float bias_1_update[ARRAY_SIZE] = {};

    float output_0[ARRAY_SIZE] = {};
    float output_1[ARRAY_SIZE] = {};
    float output_2[ARRAY_SIZE] = {};

    // dummy arrays used to capture unused outputs
    float dummy1[ARRAY_SIZE];
    float dummy2[ARRAY_SIZE][ARRAY_SIZE];
    float dummy3[ARRAY_SIZE];

    // make local versions of the weights/biases
    float w1_local[ARRAY_SIZE][ARRAY_SIZE] = {};
    float w2_local[ARRAY_SIZE][ARRAY_SIZE] = {};
    float bias_1_local[ARRAY_SIZE] = {};
    float bias_2_local[ARRAY_SIZE] = {};
    for (int n = 0; n < ARRAY_SIZE; n++) {
        bias_1_local[n] = bias_1[n];
        bias_2_local[n] = bias_2[n];
        for (int m = 0; m < ARRAY_SIZE; m++) {
            w1_local[n][m] = w1[n][m];
            w2_local[n][m] = w2[n][m];
        }
    }

    // number of iterations defined in the header file

    // store actual and predicted difference in vector, set other params
    char model1 = 'r'; // s = sigmoid, r = relu, l = leaky relu NOTE: SIGMOID CANNOT BE USED ON HARDWARE
    char model2 = 'r'; // model used for the second layer, currently set to softmax for MNIST data
    float alpha = 0.1; // for leaky relu
    float lr = 0.1; // learning rate

    // iterate through the alloted epochs
    int i;
    for (i = 0; i < NUM_ITERATIONS; i++) {

        // iterate through all the data points
        int j;
        for (j = 0; j < DATA_SIZE; j++) {

            int p;
            for (p = 0; p < ARRAY_SIZE; p++) {
                // setup the initial data input
                output_0[p] = data[p][j];
                // initialize the error backpropagation
                delta_1[p] = 0;
                delta_2[p] = 0;
            }

            // run the forward propagation
            // start with layer 1
            Array array_out1 = model_array(w1_local, bias_1_local, output_0, delta_1, lr, model1, alpha, training);
            int o;
            for (o = 0; o < ARRAY_SIZE; o++){
                output_1[o] = array_out1.output_k[o];
            }

            // then layer two
            Array array_out2 = model_array(w2_local, bias_2_local, output_1, delta_2, lr, model2, alpha, training);
            for (o = 0; o < ARRAY_SIZE; o++){
                output_2[o] = array_out2.output_k[o];
            }

            // make inferences for the return array if training has completed
            // this part is hard to generalize depending on the dataset.
            if (output_2[0] > 0.5) {
                output_inference[j] = 1;
            }
            else if (output_2[0] <= 0.5) {
                output_inference[j] = 0;
            }

            // uncomment the following for softmax
            // complete the inference (please check this comes from ChatGPT)
            // float max_element = std::max_element(array_out2.output_k[0], array_out1.output_k[ARRAY_SIZE]);
            // output_inference[j] = std::distance(array_out2.output_k, max_element);
            
            // lastly calculate the final error with the derivative of mse after the last output
            // if (model == 's') {
            //     delta_2[0] = -(y[j] - output_2[0]) * output_2[0] * (1 - output_2[0]);
            // }
            int e;
            for (e = 0; e < ARRAY_SIZE; e++) {
                if (model2 == 'r') {
                    if (output_2[e] > 0)
                        delta_2[e] = -(y[j] - output_2[e]);
                    else
                        delta_2[e] = 0;
                    }
                else if (model2 == 'l') {
                    if (output_2[e] > 0)
                        delta_2[e] = -(y[j] - output_2[e]);
                    else
                        delta_2[e] = -(y[j] - output_2[e]) * alpha;
                }
                else if (model2 == 'm') {
                    // find the error signal if the model of the output layer is softmax
                    // delta_2[p] = array_out2.output_k[p] - train_labels_one_hot[j][p];
                }
                else {
                    // std::cout << "model invalid" << std::endl;
                    break;
                }
            }

            // run the backpropagation and update the array
            // start with layer 2
            Array array_back2 = model_array(w2_local, bias_2_local, output_1, delta_2, lr, model2, alpha, training);
            for (e = 0; e < ARRAY_SIZE; e++) {
                delta_1[e] = array_back2.delta_kmin1[e];
            }
            // update the weights and biases
            for (int n = 0; n < ARRAY_SIZE; n++) {
                bias_2_local[n] = array_back2.bias_change[n];
                for (int m = 0; m < ARRAY_SIZE; m++) {
                    w2_local[n][m] = array_back2.weight_changes[n][m];
                }
            }
            // end with layer 1
            Array array_back1 = model_array(w1_local, bias_1_local, output_0, delta_1, lr, model1, alpha, training);
            // update the weights and biases
            for (int n = 0; n < ARRAY_SIZE; n++) {
                bias_1_local[n] = array_back1.bias_change[n];
                for (int m = 0; m < ARRAY_SIZE; m++) {
                    w1_local[n][m] = array_back1.weight_changes[n][m];
                }
            }

            if ((training == 0) && (j == (DATA_SIZE-1))) {
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
        for (int m = 0; m < ARRAY_SIZE; m++) {
            output_array.new_w1[n][m] = w1_local[n][m];
            output_array.new_w2[n][m] = w2_local[n][m];
        }
    }

    output_array.inference = output_inference;

    return output_array;
}
