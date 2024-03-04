// This is the test bench file for the GIM@Rice accelerator in HLS
#include "gim_model.h"
using namespace std;

int main() {

    // matrices initialized with random values from Python, known to converge
    float w1[ARRAY_SIZE][ARRAY_SIZE] = {{0.13457995, 0.51357812}, {0.18443987, 0.78533515}};
    float w2[ARRAY_SIZE][ARRAY_SIZE] = {{0.85397529, 0.49423684}, {0, 0}};
    float bias_1[ARRAY_SIZE] = {0.50524609, 0.0652865};
    float bias_2[ARRAY_SIZE] = {0.42812233, 0};

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
    std::cout << "The following are the predictions of the DNN:" << endl;
    for (int infer = 0; infer < DATA_SIZE; infer++) {
        std::cout << output.inference[infer] << endl;
    }

    return 0;

}
