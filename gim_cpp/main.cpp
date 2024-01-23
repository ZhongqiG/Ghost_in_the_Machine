// This is the test bench file for the GIM@Rice accelerator in HLS
#include "gim_model.h"
using namespace std;

int main() {

    // matrices initialized with random values from Python, known to converge
    fixed_16 w1[2][2] = {{0.13457995, 0.51357812}, {0.18443987, 0.78533515}};
    fixed_16 w2[2][2] = {{0.85397529, 0.49423684}, {0, 0}};
    fixed_16 bias_1[2] = {0.50524609, 0.0652865};
    fixed_16 bias_2[2] = {0.42812233, 0};

    int ret = 0; // for pass/fail purposes

    fixed_16 *dummy_output = accelerator(w1, w2, bias_1, bias_2, 1); // training the array, no return

    fixed_16 *inference = accelerator(w1, w2, bias_1, bias_2, 0); // running inference using the trained accelerator

    // capture the outputs of the accelerator
    fixed_16 out1 = inference[0];
    fixed_16 out2 = inference[1];
    fixed_16 out3 = inference[2];
    fixed_16 out4 = inference[3];

    // write these outputs to the output file for comparison with the golden standard
    ofstream myfile;
    myfile.open ("output.txt");
    myfile << out1;
    myfile << out2;
    myfile << out3;
    myfile << out4;
    myfile.close();
    
    // Compare the results of the function against expected results
    ret = system("diff --brief -w output.txt output_golden.txt"); // should there be two spaces between --brief and -w ?
  
    if (ret != 0) {
        cout << "Test failed  !!!\n" << endl; 
        ret=1;
    } else {
        cout << "Test passed !\n" << endl; 
    }
    return ret;
}