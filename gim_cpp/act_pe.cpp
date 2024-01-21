#include <stdio.h>
#include <math.h>
#include "gim_model.h"

using namespace std

// activation pe function
float act_pe(float net_in, char model, float alpha){
    float omega = net_in;
    float output = 0;
    if (model == 's'){
        output = 1 / (1 + exp(-omega));
    }
    else if (model == 'r'){
        output = fmax(0, omega);
    }
    else if (model == 'l'){
        if (omega >= 0){
            output = omega;
        }
        else {
            output = alpha * omega;
        }
    }
    else {
        std::cout << "Error: Invalid Model Name" << std::endl;
        output = 0;
    }
    return output;
}