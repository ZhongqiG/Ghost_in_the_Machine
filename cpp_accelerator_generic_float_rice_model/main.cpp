// This is the test bench file for the GIM@Rice accelerator in HLS
#include "gim_model.h"
using namespace std;

int main() {

    // matrices initialized with random values from Python, known to converge
    float w1[ARRAY_SIZE][ARRAY_SIZE] = {{2.6171875e-01, 1.5625e-02, 3.28125e-01, 2.6171875e-01, 2.5390625e-01, 1.9921875e-01, 1.2109375e-01}, 
                                        {3.7109375e-01, 1.796875e-01, 1.6015625e-01, 3.7109375e-01, 3.1640625e-01, 8.203125e-02, 3.0859375e-01},
                                        {1.09375e-01, 4.7265625e-01, 5.1171875e-01, 2.96875e-01, 5.078125e-01, 4.6875e-02, 3.046875e-01}};
    float w2[ARRAY_SIZE][ARRAY_SIZE] = {{3.90625e-02, 2.578125e-01, 5.859375e-02, 0, 0, 0, 0}, 
                                        {0, 0, 0, 0, 0, 0, 0},
                                        {0, 0, 0, 0, 0, 0, 0}};
    float bias_1[ARRAY_SIZE] = {7.8125e-02, 3.59375e-01, 1.3671875e-01, 0, 0, 0, 0};
    float bias_2[ARRAY_SIZE] = {6.25e-02, 0, 0, 0, 0, 0, 0};

    float test_data[TEST_DATA_SIZE] = {};

    // the following code produces random values for all weights/biases,
    // and this intialization is generalized to any square array.
    // however, training success is not guaranteed every time

    // std::random_device rd{};
    // std::mt19937 gen{rd()};
 
    // // values near the mean are the most likely
    // // standard deviation affects the dispersion of generated values from the mean
    // std::normal_distribution<float> d{0.0, 1.0};

    // float w1[ARRAY_SIZE][ARRAY_SIZE];
    // float w2[ARRAY_SIZE][ARRAY_SIZE];
    // float bias_1[ARRAY_SIZE];
    // float bias_2[ARRAY_SIZE];

    // int row;
    // int col;
    // for (row = 0; row < ARRAY_SIZE; row++) {
    //     // set the biases to be random values from a normal distribution between -1 and 1
    //     do {
    //         bias_1[row] = d(gen);
    //     } while (bias_1[row] < -1.0 || bias_1[row] > 1.0);

    //     do {
    //         bias_2[row] = d(gen);
    //     } while (bias_2[row] < -1.0 || bias_2[row] > 1.0);

    //     for (col = 0; col < ARRAY_SIZE; col++) {
    //         // set the weights to be random values from a normal distribution between -1 and 1
    //         do {
    //             w1[row][col] = d(gen);
    //         } while (w1[row][col] < -1.0 || w1[row][col] > 1.0);

    //         do {
    //             w2[row][col] = d(gen);
    //         } while (w2[row][col] < -1.0 || w2[row][col] > 1.0);
    //     }
    // }

    // used to keep the inference separate from the struct
    float output_inference[DATA_SIZE] = {};

    // training the array
    Inference training = accelerator(w1, w2, bias_1, bias_2, output_inference, 1);

    // running inference using the trained accelerator
    Inference output = accelerator(training.new_w1, training.new_w2, training.new_b1, training.new_b2, output_inference, 0);

    // capture the outputs of the accelerator
    std::cout << "The test accuracy of the network in classification is:" << endl;
    float total_correct = 0;
    for (int infer = 0; infer < DATA_SIZE; infer++) {
        if (output.inference[infer] == test_data[infer])
            total_correct++;
    }
    float accuracy = total_correct/TEST_DATA_SIZE;
    std::cout << accuracy * 100 << "%" << std::endl;

    return 0;

}
