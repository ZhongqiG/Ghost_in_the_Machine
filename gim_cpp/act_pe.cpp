#include "gim_model.h"

using namespace cmath;

// activation pe function
fixed_16 act_pe(fixed_16 net_in, char model, fixed_16 alpha) {
    fixed_16 omega = net_in;
    fixed_16 output = 0;
    if (model == 's'){
        output = 1 / (1 + exp(-omega));
    }
    else if (model == 'r'){
        output = std::max(0, omega);
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