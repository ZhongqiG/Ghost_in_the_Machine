num_bits =16;
frac_bits=8;
num_layer = 4;
eta = 0.001;
% Test case 1: 

o0_in = timeseries([ 1; zeros(2,1); 2; zeros(2,1);]);
o1_in = timeseries([ 2; zeros(2,1); 2; zeros(2,1);]);
label_0 = timeseries(zeros(8,1)*1); %consistently outputs 1 for now 
label_1 = timeseries(zeros(8,1)*1);

input_control = timeseries([0;1;1;0;1;1]);
layer = timeseries([0;1;0;1;0;1]); 
backprop_on = timeseries([0]);
initial_weight = [ 1 2 ];
initial_bias = [ 1 1 ];
MSE_on = timeseries([0; 0; 0; 0; 1; 0; 0]); %need to match when forward prop finishes. 



