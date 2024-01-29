// This is the test bench file for the GIM@Rice accelerator in HLS
#include "gim_model.h"
using namespace std;

void main() {

    // matrices initialized with random values from Python, known to converge
    vector<fixed_16> w1 = {{0.13457995, 0.51357812}, {0.18443987, 0.78533515}};
    vector<fixed_16> w2 = {{0.85397529, 0.49423684}, {0, 0}};
    vector<fixed_16> bias_1 = {0.50524609, 0.0652865};
    vector<fixed_16> bias_2 = {0.42812233, 0};

    // setting up the return arrays for the benchmark
    vector<fixed_16> dummy_array, inference;

    // training the array, return_array not used
    accelerator(w1, w2, bias_1, bias_2, 1, dummy_array);

    // running inference using the trained accelerator
    accelerator(w1, w2, bias_1, bias_2, 0, inference);

    // capture the outputs of the accelerator
    cout << "The following are the predictions of DNN" << endl;
    cout << inference[0] << endl;
    cout << inference[1] << endl;
    cout << inference[2] << endl;
    cout << inference[3] << endl;

}