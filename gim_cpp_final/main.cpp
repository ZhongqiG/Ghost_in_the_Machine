// This is the test bench file for the GIM@Rice accelerator in HLS
#include "gim_model.h"
using namespace std;

int main() {

    // matrices initialized with random values from Python, known to converge
    fixed_16 w1[ARRAY_SIZE][ARRAY_SIZE] = {{0.13457995, 0.51357812}, {0.18443987, 0.78533515}};
    fixed_16 w2[ARRAY_SIZE][ARRAY_SIZE] = {{0.85397529, 0.49423684}, {0, 0}};
    fixed_16 bias_1[ARRAY_SIZE] = {0.50524609, 0.0652865};
    fixed_16 bias_2[ARRAY_SIZE] = {0.42812233, 0};

    // training the array
    Inference training = accelerator(w1, w2, bias_1, bias_2, 1);

    // running inference using the trained accelerator
    Inference output = accelerator(training.new_w1, training.new_w2, training.new_b1, training.new_b2, 0);

    // capture the outputs of the accelerator
    cout << "The following are the predictions of the DNN:" << endl;
    cout << output.inference[0] << endl;
    cout << output.inference[1] << endl;
    cout << output.inference[2] << endl;
    cout << output.inference[3] << endl;
    return 0;

}
