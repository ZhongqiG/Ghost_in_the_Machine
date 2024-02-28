%% Code to run the XOR hardware model
num_bits = 16; % size of the data
frac_bits = 8; % fraction for fixed point interpretation
num_layer = 2; % number of layers in the model
eta = 0.1; % learning rate

% initialize weights and biases
initial_weight_layer_1 = mod(randn([2 2]), 1);
initial_weight_layer_2 = mod(randn([2 2]), 1);
initial_bias_layer_1 = mod(randn([2 1]), 1);
initial_bias_layer_2 = mod(randn([2 1]), 1);

% set the weights and biases of the unused neuron to zero for XOR model
initial_weight_layer_2(2,:) = 0;
initial_bias_layer_2(2, 1) = 0;

%% Code to input the data and control signals
% define the data inputs and labels based on the hardware timing
% input for first weight
% forward prop has 2 clock cycles after input for one layer, so double for
% two layers of forward prop (4)
% back prop takes two per weight pe (update weight on third) but since
% output is on the first clock edge we'll just count 2 cycles plus error
% for one layer, so 5 total (double - 1 because first layer error doesn't
% matter)
% grand total of 9 cycles for one data point
% assumes that the backprop occurs for each input
o0_in = timeseries([zeros(9,1); ones(9,1); zeros(9,1); ones(9,1)]);
% input for second weight
o1_in = timeseries([zeros(9,1); zeros(9,1); ones(9,1); ones(9,1)]);
% label to compare with first neuron output
label_0 = timeseries([zeros(9,1); ones(9,1); ones(9,1); zeros(9,1)]);
% label to compare with second neuron output
label_1 = timeseries(zeros(36,1)); % always zero

% define the control signals
% selects input for layer (is this specific to the layer?)
input_control = timeseries([ones(2,1); 2*ones(2,1); 3*ones(5,1); ...
                            ones(2,1); 2*ones(2,1); 3*ones(5,1); ...
                            ones(2,1); 2*ones(2,1); 3*ones(5,1); ...
                            ones(2,1); 2*ones(2,1); 3*ones(5,1)]);
% which layer are we on
layer = timeseries([zeros(2,1); ones(5,1); zeros(2,1); ...
                    zeros(2,1); ones(5,1); zeros(2,1); ...
                    zeros(2,1); ones(5,1); zeros(2,1); ...
                    zeros(2,1); ones(5,1); zeros(2,1)]); 

% signal for backprop (is this specific to layer?)
backprop_on = timeseries([zeros(4,1); ones(5,1); ...
                        zeros(4,1); ones(5,1); ...
                        zeros(4,1); ones(5,1); ...
                        zeros(4,1); ones(5,1)]);

% tells blocks to compute MSE when forward prop finishes
MSE_on = timeseries([zeros(3,1); 1; zeros(5,1); ...
                    zeros(3,1); 1; zeros(5,1); ...
                    zeros(3,1); 1; zeros(5,1); ...
                    zeros(3,1); 1; zeros(5,1)]);





