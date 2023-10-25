% This code utilizes the 2x2 array to perform test calculations

%% Setting Up the Data

% Initial Weights
weights = [1 1 
           1 1];
biases = [1 1];

% Retrieve the Output from the Previous Layer
% Retrieve Error from Current Layer
output_kmin1 = [1 1 1 1 1 1 1 1 1 1]; %10 clock cycles right now, 1 is placeholder
delta_k = [1 1 1 1 1 1 1 1 1 1];

% Set the Constants and Activation Function
eta = 0.1; %need to replace with a standard value for eta
alpha = 0.1; %need to replace with a standard value for alpha
model = "elu";

% Create Arrays to Hold the Outputs
output_k = zeros(2, 1); %this array should be as long as there are neurons in the layer
delta_kmin1 = zeros(2, 1); %I think this should be as long as there are weights in a neuron
weight_changes = zeros(2, 2); %should be as large as the PE array
bias_changes = zeros(2, 1); %should be as large as the number of neurons in the layer

%% Using the PE for 1 Layer
output_k, delta_kmin1, weight_changes, bias_changes = pe_array(weights, biases, output_kmin1, delta_k, eta, model, alpha);







