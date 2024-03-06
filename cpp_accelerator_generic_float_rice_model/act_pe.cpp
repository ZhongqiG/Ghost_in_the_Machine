#include "gim_model.h"

// activation pe function
float act_pe(float net_in, char model, float alpha) {
    float omega = net_in;
    float output = 0;
    // sigmoid not implemented due to difficulty in hardware
    //if (model == 's'){
    //    output = 1 / (1 + exp(-omega));
    //}
    if (model == 'r'){
    	if (omega > 0)
    		output = omega;
    	else
    		output = 0;
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
        //std::cout << "Error: Invalid Model Name" << std::endl;  // Comment this out when doing synthesis
        output = 0;
    }
    return output;
}
