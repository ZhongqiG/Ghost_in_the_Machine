#ifndef GIM_MODEL_
#define GIM_MODEL_

// #include "ap_fixed.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>
#include <random>

#define ARRAY_SIZE 7
#define DATA_SIZE 200 // change this number to match the number of data points
#define TEST_DATA_SIZE 200
#define NUM_ITERATIONS 10000

// typedef ap_fixed<16,7> float;
using namespace std;


// these structs are used to hold return values/arrays for simplicity
// of return statements (i.e. no pointers used)
struct Weight{
	// members     
	float sum_delta_out;         
	float sum_output_out;
	float weight_change;
	// constructor
	Weight(){}
}; 

struct Bias {
	// members
	float net_sum;
	float bias_change;
	// constructor
	Bias(){}
};

struct Array {
	// members
    float output_k[ARRAY_SIZE];
    float delta_kmin1[ARRAY_SIZE];
    float weight_changes[ARRAY_SIZE][ARRAY_SIZE];
    float bias_change[ARRAY_SIZE];
	// constructor
	Array(){}
};

struct Inference {
	// members
	float *inference;
	float new_w1[ARRAY_SIZE][ARRAY_SIZE];
	float new_w2[ARRAY_SIZE][ARRAY_SIZE];
	float new_b1[ARRAY_SIZE];
	float new_b2[ARRAY_SIZE];
	// constructor
	Inference(){}
};

struct SoftMax {
	//members
	float out_vector[ARRAY_SIZE];
	// constructor
	SoftMax(){}
};

// processing elements, array, and accelerator function prototypes
Weight weights_pe(float delta_k, float output_kmin1, float partial_sum_out_k,
				float partial_sum_delta_k, float init_weight,
				float eta, float training);

Bias bias_pe(float delta_k,
				float sum_in,
				float init_bias,
				float eta,
				float training);

float act_pe(float net_in, char model, float alpha);

SoftMax softmax(SoftMax output_array); 

float error_pe(float output_kmin1, float partial_sum_delta_k,
				char model, float alpha);

Array model_array(float weights[ARRAY_SIZE][ARRAY_SIZE],
			float biases[ARRAY_SIZE],
			float output_kmin1[ARRAY_SIZE],
			float delta_k[ARRAY_SIZE], float eta,
			char model, float alpha, float training);

Inference accelerator(float w1[ARRAY_SIZE][ARRAY_SIZE], float w2[ARRAY_SIZE][ARRAY_SIZE],
				 float  bias_1[ARRAY_SIZE], float bias_2[ARRAY_SIZE], float output_inference[DATA_SIZE],
                 float training);

#endif // GIM_MODEL_
